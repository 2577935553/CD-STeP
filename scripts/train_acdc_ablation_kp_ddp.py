"""
DDP variant of train_MIA_config_abls_p.py, dedicated to the K=5 and K=6
ablation experiments. Effective batch size = args.batch_size, distributed
across `world_size` GPUs (so per-rank batch = args.batch_size / world_size).

Key differences vs the single-GPU script:
  1. Removed the silent  `if K_sample > 4: batch_size //= 2`  hack.
     The effective batch is exactly `args.batch_size` regardless of K.
  2. SyncBatchNorm wraps both models, so BN statistics are computed over
     the global batch -- numerically identical to single-GPU bs=args.batch_size BN.
  3. Both models are wrapped in DDP. The contrastive loss is fully per-sample
     (compute_temporal_similarity loops over the batch dim, compute_global_similarity
     uses bmm; no cross-batch negatives), so DDP gradient averaging is
     mathematically identical to single-GPU training.
  4. Logging, checkpointing, validation: rank-0 only.
  5. find_unused_parameters=True for the temporal model, since loss_dste only
     uses one of the dste_module outputs.

Launch:
    torchrun --nproc_per_node=2 --master_port=29501 \
        train_MIA_config_K56_ddp.py [args...]
"""
import argparse
from abc import ABC
import importlib
from models import TempSeg_Mem_New_ALL, SingleUnet
import torch
import os
import random
import numpy as np
import torch.optim as optim
from data import SemiSegDataset_VT, SemiSegValidDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from data.io import get_image_list, augment_data_batch_frames, crop_batch_data
from utils.image_utils import LR_Scheduler, crop_image
import nibabel as nib
from losses.metrics import *
from losses.loss import *
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import losses
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from skimage import measure, morphology
from scipy import ndimage


torch.autograd.set_detect_anomaly(True)


# ============================================================
# DDP helpers
# ============================================================
def setup_ddp():
    """Initialize the process group from torchrun env vars."""
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ============================================================
# Scheduler / utils (unchanged from original)
# ============================================================
def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=0):
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return scheduler


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Training (DDP-aware)
# ============================================================
def training(args, local_rank, world_size):
    seed_torch(args.seed)
    train_output_dir = args.train_output_dir
    model_dir = train_output_dir + '/model'
    training_graph = train_output_dir + '/graph'
    train_csv = train_output_dir + '/csv'

    # ---- per-rank batch size (NO silent K_sample-based halving) ----
    assert args.batch_size % world_size == 0, (
        f"--batch_size ({args.batch_size}) must be divisible by "
        f"world_size ({world_size}). Effective batch is preserved this way."
    )
    per_rank_bs = args.batch_size // world_size

    learning_rate = args.learning_rate
    epochs = int(args.epochs)
    image_size = args.image_size
    num_classes = args.num_classes

    if is_main_process():
        mkdir(train_output_dir)
        mkdir(model_dir)
        mkdir(training_graph)
        mkdir(train_csv)
    barrier()  # ensure dirs exist before any rank tries to write

    # ---- datasets ----
    train_dataset = SemiSegDataset_VT(
        data_dir=args.data_dir,
        train_data_csv=args.train_data_csv,
        train_data_gt=args.train_data_gt,
        image_size=image_size,
        frame_number=args.frame_number,
        label_index=args.label_index,
        mode='train',
        rand_frame=False,
    )
    valid_dataset = SemiSegValidDataset(
        data_dir=args.data_dir,
        valid_data_csv=args.valid_data_csv,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_bs,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
    )
    # validation only runs on rank 0; a plain DataLoader is fine.
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=per_rank_bs,
        shuffle=False,
        num_workers=0,
    )

    if is_main_process():
        print(f'Data Initialized! world_size={world_size}, '
              f'per_rank_bs={per_rank_bs}, effective_bs={args.batch_size}, '
              f'K_sample={args.K_sample}, bkg_sample_size={args.bkg_sample_size}',
              flush=True)
        writer = SummaryWriter(training_graph)
    else:
        writer = None

    save_path_time = os.path.join(model_dir, 'segmodel.pth')
    save_path_agnostic = os.path.join(model_dir, 'agnostic.pth')

    # ---- build models ----
    model_agnostic = SingleUnet('resnet50', None, classes=4, deep_stem=32)
    model = TempSeg_Mem_New_ALL(
        'resnet50', None,
        frame_number=args.frame_number,
        classes=4,
        reduction_rate=args.reduction_rate,
        deep_stem=32,
        in_channels=2,
        reduc_time=8,
        K_sample=args.K_sample,
        bkg_sample_size=args.bkg_sample_size,
        tau_bkg=args.tau_bkg,
    )

    if args.resume is not None:
        model.load_state_dict(
            torch.load(args.resume, map_location=f'cuda:{local_rank}')
        )

    # ---- SyncBN: BN stats are all-reduced across ranks  ----
    # ==> equivalent to single-GPU BN over the global batch.
    model_agnostic = nn.SyncBatchNorm.convert_sync_batchnorm(model_agnostic)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_agnostic = model_agnostic.cuda(local_rank)
    model = model.cuda(local_rank)

    # ---- DDP wrap ----
    # model_agnostic: every param contributes to loss_agnostic, so
    #   find_unused_parameters=False is fine and slightly faster.
    # model: loss_dste uses only `loss_temporal_consistency` from dste_module
    #   and not loss_spatial_contrast/loss_rpp/etc, so some sub-modules may
    #   produce activations without receiving gradients. Set True to be safe.
    model_agnostic = DDP(
        model_agnostic, device_ids=[local_rank], find_unused_parameters=False
    )
    model = DDP(
        model, device_ids=[local_rank], find_unused_parameters=True
    )

    max_iteration = args.max_iteration
    iter_nums = 0
    entropy_loss = CELoss()
    dice_loss = losses.DiceLoss(num_classes)

    params = [p for p in model.parameters() if p.requires_grad]
    params_agnostic = [p for p in model_agnostic.parameters() if p.requires_grad]

    optimizer_t = optim.AdamW(params, lr=args.learning_rate * 0.04, weight_decay=0.0001)
    optimizer_agnostic = optim.AdamW(
        params_agnostic, lr=args.learning_rate * 0.04, weight_decay=0.0001
    )
    scheduler_t = create_warmup_cosine_scheduler(
        optimizer_t, 0.1 * args.max_iteration, args.max_iteration
    )
    scheduler_agnostic = create_warmup_cosine_scheduler(
        optimizer_agnostic, 0.1 * args.max_iteration, args.max_iteration
    )

    best_valid_dsc = 0.0
    final_train_dsc = 0.0
    avloss_agnostic_train = 0.0
    avloss_agnostic_train_unsup = 0.0
    avloss_time_train = 0.0
    avloss_time_train_unsup = 0.0

    for epoch in range(epochs):
        # critical: different shuffle each epoch on each rank
        train_sampler.set_epoch(epoch)

        # tqdm only on rank 0
        if is_main_process():
            train_bar = tqdm(train_loader)
        else:
            train_bar = train_loader

        model_agnostic.train()
        model.train()

        for step, (images, labels, tov_target) in enumerate(train_bar):
            tov_target = tov_target.cuda(local_rank)

            stride = images.size(0)
            aug_images, aug_labels = images.numpy(), labels.numpy()

            aug_images, aug_labels = augment_data_batch_frames(
                aug_images, aug_labels, shift=0, rotate=60, scale=0.5, flip=False
            )
            aug_images = aug_images.reshape(
                stride * args.frame_number, image_size[0], image_size[1]
            )
            shift_max = 60
            shift = int(shift_max * np.random.uniform(-1, 1))
            aug_images = crop_batch_data(
                aug_images, (image_size[0], image_size[1]), shift_value=shift
            )
            aug_labels = crop_batch_data(
                aug_labels, (image_size[0], image_size[1]), shift_value=shift
            )
            aug_images = aug_images.reshape(
                stride, args.frame_number, image_size[0], image_size[1]
            )
            _aug_images = aug_images.reshape(
                stride * args.frame_number, image_size[0], image_size[1]
            )
            aug_images = torch.Tensor(aug_images)
            _aug_images = torch.Tensor(np.expand_dims(_aug_images, axis=1))
            aug_labels = torch.Tensor(aug_labels).long()
            aug_images_list = [
                aug_images[:, i:i + 1, :, :] for i in range(args.frame_number)
            ]

            # ---- step 1: train model_agnostic ----
            for p in model_agnostic.parameters():
                p.requires_grad = True
            for p in model.parameters():
                p.requires_grad = False

            aug_pred_list = [model_agnostic(img.cuda(local_rank)) for img in aug_images_list]
            dice_loss_1 = dice_loss(
                aug_pred_list[args.label_index], aug_labels.cuda(local_rank), softmax=True
            )
            sup_loss_1 = 0.5 * (
                F.cross_entropy(aug_pred_list[args.label_index], aug_labels.cuda(local_rank))
                + dice_loss_1
            )
            pseudo_labels_aug_list = [
                torch.argmax(F.softmax(pred, dim=1), dim=1).detach()
                for pred in aug_pred_list
            ]
            prior_list = [
                torch.cat([img.cuda(local_rank), prediction.detach()], dim=1)
                for img, prediction in zip(aug_images_list, aug_pred_list)
            ]
            x_prev = torch.stack(prior_list, dim=1).flatten(start_dim=0, end_dim=1)

            aug_images_diff = torch.diff(
                torch.cat([aug_images[:, 0:1], aug_images], dim=1), dim=1
            )
            aug_images_temp = torch.stack(
                (aug_images, aug_images_diff), dim=2
            ).flatten(0, 1).cuda(local_rank)

            # ---- step 2: forward through `model` (now grads on) ----
            for p in model_agnostic.parameters():
                p.requires_grad = False
            for p in model.parameters():
                p.requires_grad = True

            masks, losses_contrast, losses_dste = model(
                aug_images_temp, x_prev.cuda(local_rank),
                gt_first_frame=aug_labels.cuda(local_rank),
            )
            loss_contrast = (
                losses_contrast['temporal_contrastive']
                + losses_contrast['global_contrastive']
            )
            loss_dste = losses_dste['loss_temporal_consistency']

            if is_main_process():
                # keep behaviour: original printed losses_contrast every step
                print(losses_contrast, flush=True)

            masks_reshape = masks.reshape(
                stride, args.frame_number, num_classes, image_size[0], image_size[1]
            )
            masks_split_list = [
                masks_reshape[:, i, :, :, :] for i in range(args.frame_number)
            ]

            dice_loss_2 = dice_loss(
                masks_split_list[args.label_index], aug_labels.cuda(local_rank), softmax=True
            )
            sup_loss_2 = 0.5 * (
                F.cross_entropy(masks_split_list[args.label_index], aug_labels.cuda(local_rank))
                + dice_loss_2
            )
            pseudo_labels_series_list = [
                torch.argmax(F.softmax(pred, dim=1), dim=1).detach()
                for pred in masks_split_list
            ]

            unsup_dice_loss_1 = sum(
                [dice_loss(preds, pseudo_lab, softmax=True)
                 for (preds, pseudo_lab) in zip(aug_pred_list, pseudo_labels_series_list)]
            ) / len(pseudo_labels_series_list)
            unsup_dice_loss_2 = sum(
                [dice_loss(preds, pseudo_lab, softmax=True)
                 for (preds, pseudo_lab) in zip(masks_split_list, pseudo_labels_aug_list)]
            ) / len(pseudo_labels_aug_list)
            unsup_loss_1 = 0.5 * (
                sum([F.cross_entropy(preds, pseudo_lab)
                     for (preds, pseudo_lab) in zip(aug_pred_list, pseudo_labels_series_list)])
                / len(pseudo_labels_series_list)
                + unsup_dice_loss_1
            )
            unsup_loss_2 = 0.5 * (
                sum([F.cross_entropy(preds, pseudo_lab)
                     for (preds, pseudo_lab) in zip(masks_split_list, pseudo_labels_aug_list)])
                / len(pseudo_labels_aug_list)
                + unsup_dice_loss_2
            )

            # ---- backward 1: update model_agnostic only ----
            for p in model_agnostic.parameters():
                p.requires_grad = True
            for p in model.parameters():
                p.requires_grad = False

            loss_agnostic = sup_loss_1 + unsup_loss_1
            loss_agnostic.backward()
            optimizer_agnostic.step()
            scheduler_agnostic.step()
            optimizer_agnostic.zero_grad()

            # ---- backward 2: update model only ----
            for p in model_agnostic.parameters():
                p.requires_grad = False
            for p in model.parameters():
                p.requires_grad = True

            tau = (1 + np.cos((iter_nums / args.max_iteration) * np.pi)) / 2
            loss_t = sup_loss_2 + unsup_loss_2 + tau * loss_dste + 0.2 * tau * loss_contrast
            loss_t.backward()
            optimizer_t.step()
            scheduler_t.step()
            optimizer_t.zero_grad()

            iter_nums += 1

            avloss_agnostic_train += sup_loss_1.item()
            avloss_agnostic_train_unsup += unsup_loss_1.item()
            avloss_time_train += sup_loss_2.item()
            avloss_time_train_unsup += unsup_loss_2.item()

            if is_main_process():
                train_bar.desc = "train epoch[{}/{}] sup_loss:{:.3f} unsup_loss:{:.3f}".format(
                    epoch + 1, epochs, sup_loss_1, unsup_loss_1
                )
                writer.add_scalar("Model Train/Loss", sup_loss_1.item(), iter_nums)
                writer.add_scalar("Model Train/DiceLoss", unsup_loss_1.item(), iter_nums)
                writer.add_scalar("Model Train/GContrast", losses_contrast['global_contrastive'].item(), iter_nums)
                writer.add_scalar("Model Train/TContrast", losses_contrast['temporal_contrastive'].item(), iter_nums)
                writer.add_scalar("Model Train/loss_spatial_contrast", losses_dste['loss_spatial_contrast'].item(), iter_nums)
                writer.add_scalar("Model Train/loss_temporal_consistency", losses_dste['loss_temporal_consistency'].item(), iter_nums)

            if iter_nums >= args.max_iteration:
                break  # break inner loop; outer break handled below

        # ---- validation: rank 0 only, but switch eval mode on all ranks ----
        avdsc_valid = 0.0
        model.eval()
        model_agnostic.eval()

        if is_main_process():
            dsc_val = []
            with torch.no_grad():
                for val_data in valid_loader:
                    val_images, val_labels = val_data
                    # bypass DDP wrapper at inference: cleaner & avoids any
                    # cross-rank synchronization expectations.
                    masks_val = model_agnostic.module(val_images.cuda(local_rank))
                    dsc = DSC_average(masks_val, val_labels.cuda(local_rank), average=False)
                    dsc_val.append(dsc)
                    avdsc_valid += dsc.mean().item()
            if len(dsc_val) > 0:
                dsc_1, dsc_2, dsc_3 = sum(dsc_val) / len(dsc_val)
            else:
                dsc_1 = dsc_2 = dsc_3 = torch.tensor(0.0)

            avloss_agnostic_train /= max(len(train_loader), 1)
            avloss_time_train /= max(len(train_loader), 1)
            avloss_agnostic_train_unsup /= max(len(train_loader), 1)
            avloss_time_train_unsup /= max(len(train_loader), 1)
            avdsc_valid /= max(len(valid_loader), 1)

            print(
                "train epoch[{}/{}] Training average sup_loss_1:{:.3f}, unsup_loss_1:{:.3f},"
                "sup_loss_2:{:.3f}, unsup_loss_2:{:.3f} "
                "Validation average DSC:{:.3f}".format(
                    epoch + 1, epochs,
                    avloss_agnostic_train, avloss_agnostic_train_unsup,
                    avloss_time_train, avloss_time_train_unsup,
                    avdsc_valid,
                ), flush=True
            )

            if avdsc_valid > best_valid_dsc:
                best_valid_dsc = avdsc_valid
                final_train_dsc = avdsc_valid
                print("model saved !", flush=True)
                # IMPORTANT: save .module.state_dict() to strip DDP prefix,
                # so the saved file is loadable by plain (non-DDP) testing code.
                torch.save(model.module.state_dict(), save_path_time)
                torch.save(model_agnostic.module.state_dict(), save_path_agnostic)

            writer.add_scalar("Model Train/EpochLoss", avloss_agnostic_train, epoch)
            writer.add_scalar("Model Train/EpochDiceLoss", avloss_agnostic_train_unsup, epoch)
            writer.add_scalar("Model Validation/DSC_average", avdsc_valid, epoch)
            writer.add_scalar("Model Validation/DSC_1", dsc_1.item(), epoch)
            writer.add_scalar("Model Validation/DSC_2", dsc_2.item(), epoch)
            writer.add_scalar("Model Validation/DSC_3", dsc_3.item(), epoch)

        # reset accumulators on all ranks for next epoch
        avloss_agnostic_train = 0.0
        avloss_agnostic_train_unsup = 0.0
        avloss_time_train = 0.0
        avloss_time_train_unsup = 0.0

        # sync ranks before next epoch
        barrier()

        if iter_nums >= args.max_iteration:
            break

    if writer is not None:
        writer.close()
    if is_main_process():
        print("Finished Training !", flush=True)


# ============================================================
# Testing (unchanged from original; wrapped by rank-0 guard at call site)
# ============================================================
def testing(args):
    image_size = (args.image_size[0], args.image_size[1])
    num_classes = args.num_classes
    model_dir = args.train_output_dir + '/model'
    test_output_dir = args.test_output_dir
    pred_dir = test_output_dir + '/predictions'

    model_path = os.path.join(model_dir, 'agnostic.pth')
    model_agnostic = SingleUnet('resnet50', None, classes=4, deep_stem=32)

    if torch.cuda.is_available():
        model_agnostic.cuda()
    model_agnostic.load_state_dict(torch.load(model_path, map_location=device))
    model_agnostic.eval()
    mkdir(pred_dir)

    seed_torch(args.seed)
    data_list = get_image_list(args.test_data_list)

    print("using {} images for testing.".format(len(data_list['image_filenames'])))

    with torch.no_grad():
        for index in range(len(data_list['image_filenames'])):
            gt_name = data_list['label_filenames'][index]
            nib_gt = nib.load(os.path.join(args.test_data_dir, gt_name))

            gt = nib_gt.get_fdata()

            img_name = data_list['image_filenames'][index]
            nib_img = nib.load(os.path.join(args.test_data_dir, img_name))

            img = np.squeeze(nib_img.get_fdata().astype('float32'))

            clip_min = np.percentile(img, 1)
            clip_max = np.percentile(img, 99)
            img = np.clip(img, clip_min, clip_max)
            img = (img - img.min()) / float(img.max() - img.min())
            x, y, z = img.shape
            x_centre, y_centre = int(x / 2), int(y / 2)
            img = crop_image(img, x_centre, y_centre, image_size, constant_values=0)

            pred_res = torch.zeros(img.shape, dtype=torch.int8)
            for i in range(img.shape[2]):
                tmp_image = torch.from_numpy(img[:, :, i]).unsqueeze(dim=0).unsqueeze(dim=0)
                outputs = model_agnostic(tmp_image.to(device))
                tmp_prob = F.softmax(outputs, dim=1)
                pred_res[:, :, i] = torch.argmax(tmp_prob[0, :, :, :], dim=0)
            

            pred_res = crop_image(pred_res, image_size[0] // 2, image_size[1] // 2, (x, y), constant_values=0)
            pred_res = pred_res.astype('int16')
            nii_pred = nib.Nifti1Image(pred_res, None, header=nib_gt.header)
            mkdir(pred_dir)
            loc_end = img_name.find('.')
            loc_start = img_name.rfind('/')
            savedirname = os.path.join(pred_dir, img_name[:loc_start])

            mkdir(savedirname)
            pred_name = savedirname + img_name[loc_start:loc_end] + '_Pred.nii.gz'
            nib.save(nii_pred, pred_name)

    print('Finished Testing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_output_dir", type=str, default=None)
    parser.add_argument("--max_iteration", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="EFFECTIVE (global) batch size; per-rank batch is "
                             "this divided by world_size.")
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image_size", nargs='+', type=int, default=[224, 224, 18])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--valid_data_csv", type=str, default=None)
    parser.add_argument("--train_data_csv", type=str, default=None)
    parser.add_argument("--train_data_gt", type=str, default=None)
    parser.add_argument("--frame_number", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr_scheduler", type=str, default='poly')
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_index", type=int, default=0)
    parser.add_argument("--test_output_dir", type=str, default=None)
    parser.add_argument("--test_data_list", type=str, default=None)
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--reduction_rate", type=int, default=256)
    parser.add_argument('--model_type', type=str, default=TempSeg_Mem_New_ALL)
    # NOTE: --device is intentionally removed; with DDP, the GPU each process
    # uses is determined by torchrun's LOCAL_RANK env var (and by
    # CUDA_VISIBLE_DEVICES set in the launch script).
    parser.add_argument('--K_sample', type=int, default=4)
    parser.add_argument('--bkg_sample_size', type=int, default=100)
    parser.add_argument('--tau_bkg', type=float, default=0.5)
    args = parser.parse_args()

    # ---- DDP init ----
    local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')  # used by testing()

    mode = args.mode

    try:
        if mode == 'train':
            training(args, local_rank, world_size)
            barrier()  # ensure all ranks finish before rank 0 starts testing
            if is_main_process():
                testing(args)
        elif mode == 'test':
            if is_main_process():
                testing(args)
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
    finally:
        cleanup_ddp()
