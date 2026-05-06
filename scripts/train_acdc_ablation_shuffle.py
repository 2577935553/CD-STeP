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
from tqdm import tqdm
from tensorboardX import SummaryWriter
from data.io import get_image_list, augment_data_batch_frames, crop_batch_data
from utils.image_utils import LR_Scheduler, crop_image
import nibabel as nib
from losses.metrics import *
from losses.loss import *
import torch.nn as nn
from utils import losses
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from skimage import measure, morphology
from scipy import ndimage


import torch
torch.autograd.set_detect_anomaly(True)



# device = torch.device('cuda:0')

def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=0):
    """
    Create a scheduler with linear warmup followed by cosine annealing using PyTorch official schedulers
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        eta_min: Minimum learning rate for cosine annealing
    """
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01,  # Start at 1% of base LR
        end_factor=1.0,     # End at 100% of base LR
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,  
        eta_min=eta_min
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]  
    )
    
    return scheduler

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Frame-shuffle helpers (added for the shuffle ablation)
# ============================================================
def shuffle_frames_per_sample(images, label_index):
    """Per-sample random permutation of the temporal (frame) dimension.

    Args:
        images:      [B, T, H, W] float tensor.
        label_index: int. The position in the ORIGINAL frame ordering whose
                     GT mask the dataset returned (default 0 for this codebase).

    Returns:
        shuffled:    [B, T, H, W] tensor with frames independently permuted
                     within each batch element.
        gt_pos:      [B] long tensor.  gt_pos[b] gives the new position (in
                     the shuffled ordering) of the originally-labeled frame
                     for sample b, so that the supervised loss can pick the
                     correct prediction afterwards.
    """
    B, T, H, W = images.shape
    perm = torch.stack([torch.randperm(T) for _ in range(B)], dim=0)  # [B, T]
    idx = perm.view(B, T, 1, 1).expand(B, T, H, W)
    shuffled = torch.gather(images, dim=1, index=idx)
    # gt_pos[b] = t such that perm[b, t] == label_index
    gt_pos = (perm == label_index).long().argmax(dim=1)
    return shuffled, gt_pos


def gather_gt_frame_pred(pred_list, gt_pos):
    """Pick the prediction at the (per-sample) GT-frame position.

    Args:
        pred_list: list of T tensors, each [B, C, H, W].
        gt_pos:    [B] long tensor, values in [0, T-1].

    Returns:
        [B, C, H, W] tensor where row b is pred_list[gt_pos[b]][b].
    """
    pred_stack = torch.stack(pred_list, dim=1)  # [B, T, C, H, W]
    B, T, C, H, W = pred_stack.shape
    idx = gt_pos.to(pred_stack.device).view(B, 1, 1, 1, 1).expand(B, 1, C, H, W)
    return pred_stack.gather(dim=1, index=idx).squeeze(1)  # [B, C, H, W]


def training(args):
    seed_torch(args.seed)
    train_output_dir = args.train_output_dir
    model_dir = train_output_dir + '/model'
    training_graph = train_output_dir + '/graph'
    train_csv = train_output_dir + '/csv'

    if args.K_sample>4:
        batch_size = args.batch_size//2
    else:
        batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = int(args.epochs)
    image_size = args.image_size
    num_classes = args.num_classes

    mkdir(train_output_dir)
    mkdir(model_dir)
    mkdir(training_graph)
    mkdir(train_csv)

    train_dataset = SemiSegDataset_VT(data_dir=args.data_dir,
                                   train_data_csv=args.train_data_csv,
                                   train_data_gt=args.train_data_gt,
                                   image_size=image_size,
                                   frame_number=args.frame_number,
                                   label_index=args.label_index,
                                   mode='train',rand_frame=False)
    valid_dataset = SemiSegValidDataset(
        data_dir=args.data_dir,
        valid_data_csv=args.valid_data_csv,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    print('Data Initialized!')
    writer = SummaryWriter(training_graph)

    save_path_time = os.path.join(model_dir, 'segmodel.pth')
    save_path_agnostic = os.path.join(model_dir, 'agnostic.pth')
    model_agnostic = SingleUnet('resnet50', None, classes=4, deep_stem=32)
    segmodel = importlib.import_module("SegModel")
    # if hasattr(segmodel, args.model_type):
    #     model = getattr(segmodel, args.model_type)('resnet50', None, frame_number=args.frame_number, classes=4,
    #                     reduction_rate=args.reduction_rate, deep_stem=32)
    # else:
    #     raise ValueError(f'{args.model_type} not exists')
    model = TempSeg_Mem_New_ALL('resnet50', None, frame_number=args.frame_number, classes=4,
                        reduction_rate=args.reduction_rate, deep_stem=32,in_channels=2,reduc_time=8,
                        K_sample=args.K_sample,bkg_sample_size=args.bkg_sample_size,tau_bkg=args.tau_bkg)
    max_iteration = args.max_iteration
    iter_nums = 0
    entropy_loss = CELoss()
    dice_loss = losses.DiceLoss(num_classes)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    model_agnostic = model_agnostic.cuda()
    model = model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    params_agnostic = [p for p in model_agnostic.parameters() if p.requires_grad]
    # scheduler = LR_Scheduler(args.lr_scheduler, learning_rate, epochs, len(train_loader))
    # optimizer_t = optim.SGD(params, lr=learning_rate, momentum=0.9,
                            # weight_decay=0.0001)
    # optimizer_agnostic = optim.SGD(params_agnostic, lr=learning_rate, momentum=0.9,
                                #    weight_decay=0.0001)
    # 
    optimizer_t = optim.AdamW(params, lr=args.learning_rate*0.04, weight_decay=0.0001)
    optimizer_agnostic = optim.AdamW(params_agnostic, lr=args.learning_rate*0.04, weight_decay=0.0001)
    scheduler_t = create_warmup_cosine_scheduler(optimizer_t, 0.1*args.max_iteration, args.max_iteration)
    scheduler_agnostic = create_warmup_cosine_scheduler(optimizer_agnostic, 0.1*args.max_iteration, args.max_iteration)
    
    best_valid_dsc = 0.0
    final_train_dsc = 0.0
    avloss_agnostic_train = 0.0
    avloss_agnostic_train_unsup = 0.0
    avloss_time_train = 0.0
    avloss_time_train_unsup = 0.0

    for epoch in range(epochs):

        train_bar = tqdm(train_loader)

        model_agnostic.train()
        model.train()
        for step, (images, labels,tov_target) in enumerate(train_bar):
            tov_target=tov_target.cuda()

            # ============================================================
            # FRAME SHUFFLE (this is the only training-loop change vs the
            # default script).  We permute the T frames per sample, and
            # remember where the originally-labeled frame ended up so the
            # supervised losses below can pick the right prediction.
            # All downstream tensors (aug_images, aug_pred_list, masks, ...)
            # are produced from this shuffled `images`, so they are all
            # consistent in the new ordering.  `aug_labels` is the one
            # tensor whose meaning is *not* a frame index -- it's the GT
            # mask of whichever frame originally had label_index, which
            # is now at gt_pos[b].
            # ============================================================
            images, gt_pos = shuffle_frames_per_sample(images, args.label_index)
            gt_pos = gt_pos.cuda()

            stride = images.size(0)
            aug_images, aug_labels = images.numpy(), labels.numpy()

            aug_images, aug_labels = augment_data_batch_frames(aug_images, aug_labels, shift=0, rotate=60,
                                                               scale=0.5,
                                                               flip=False)
            aug_images = aug_images.reshape(stride * args.frame_number,
                                            image_size[0], image_size[1])
            shift_max = 60
            shift = int(shift_max * np.random.uniform(-1, 1))
            aug_images = crop_batch_data(aug_images, (image_size[0], image_size[1]), shift_value=shift)

            aug_labels = crop_batch_data(aug_labels, (image_size[0], image_size[1]), shift_value=shift)
            aug_images = aug_images.reshape(stride, args.frame_number,
                                            image_size[0], image_size[1])
            _aug_images = aug_images.reshape(stride * args.frame_number,
                                             image_size[0], image_size[1])
            aug_images = torch.Tensor(aug_images)
            _aug_images = torch.Tensor(np.expand_dims(_aug_images, axis=1))
            aug_labels = torch.Tensor(aug_labels).long()
            aug_images_list = [aug_images[:, i:i + 1, :, :] for i in range(args.frame_number)]

            # debugging_aug_img = aug_images_list[args.label_index].squeeze().permute([1, 2, 0]).cpu().numpy()
            # nii_debugging_aug_img = nib.Nifti1Image(debugging_aug_img, None)
            # nib.save(nii_debugging_aug_img, './debugging_aug_img.nii.gz')

            for p in model_agnostic.parameters():
                p.requires_grad = True
            for p in model.parameters():
                p.requires_grad = False

            aug_pred_list = [model_agnostic(img.cuda()) for img in aug_images_list]
            # Pick prediction at GT-frame's *new* position for each sample.
            gt_frame_pred_agnostic = gather_gt_frame_pred(aug_pred_list, gt_pos)
            dice_loss_1 = dice_loss(gt_frame_pred_agnostic, aug_labels.cuda(), softmax=True)
            sup_loss_1 = 0.5 * (F.cross_entropy(gt_frame_pred_agnostic, aug_labels.cuda())
                                + dice_loss_1)
            pseudo_labels_aug_list = [torch.argmax(F.softmax(pred, dim=1), dim=1).detach() for pred in
                                      aug_pred_list]
            prior_list = [torch.cat([img.cuda(), prediction.detach()], dim=1) for img, prediction in zip(aug_images_list, aug_pred_list)]
            x_prev = torch.stack(prior_list, dim=1).flatten(start_dim=0, end_dim=1)
            
            aug_images_diff=torch.diff(torch.cat([aug_images[:,0:1],aug_images],dim=1),dim=1)
            aug_images_temp=torch.stack((aug_images,aug_images_diff),dim=2).flatten(0,1).cuda()
            # outs_temp=model(aug_images_temp)
            
            for p in model_agnostic.parameters():
                p.requires_grad = False
            for p in model.parameters():
                p.requires_grad = True
                
            masks,losses_contrast,losses_dste = model(aug_images_temp, x_prev.cuda(),gt_first_frame=aug_labels.cuda())
            # masks,losses_contrast,losses_dste = model(_aug_images.cuda(), x_prev.cuda(),gt_first_frame=aug_labels.cuda(),tov_target=tov_target)
            loss_contrast=losses_contrast['temporal_contrastive']+losses_contrast['global_contrastive']
            # loss_dste=losses_dste['loss_tov']+losses_dste['loss_ad']+losses_dste['loss_rpp']
            loss_dste=losses_dste['loss_temporal_consistency']
            print(losses_contrast)
            
            masks_reshape = masks.reshape(stride, args.frame_number, num_classes,
                                          image_size[0], image_size[1])
            masks_split_list = [masks_reshape[:, i, :, :, :] for i in range(args.frame_number)]

            gt_frame_pred_temp = gather_gt_frame_pred(masks_split_list, gt_pos)
            dice_loss_2 = dice_loss(gt_frame_pred_temp, aug_labels.cuda(), softmax=True)

            sup_loss_2 = 0.5 * (F.cross_entropy(gt_frame_pred_temp, aug_labels.cuda())
                                + dice_loss_2)
            pseudo_labels_series_list = [torch.argmax(F.softmax(pred, dim=1), dim=1).detach() for pred in
                                         masks_split_list]

            unsup_dice_loss_1 = sum([dice_loss(preds, pseudo_lab, softmax=True) for (preds, pseudo_lab) in
                                     zip(aug_pred_list, pseudo_labels_series_list)]) / len(
                pseudo_labels_series_list)
            unsup_dice_loss_2 = sum([dice_loss(preds, pseudo_lab, softmax=True) for (preds, pseudo_lab) in
                                     zip(masks_split_list, pseudo_labels_aug_list)]) / len(pseudo_labels_aug_list)
            unsup_loss_1 = 0.5 * (sum([F.cross_entropy(preds, pseudo_lab) for (preds, pseudo_lab) in
                                       zip(aug_pred_list, pseudo_labels_series_list)]) / len(
                pseudo_labels_series_list)
                                  + unsup_dice_loss_1)
            unsup_loss_2 = 0.5 * (sum([F.cross_entropy(preds, pseudo_lab) for (preds, pseudo_lab) in
                                       zip(masks_split_list, pseudo_labels_aug_list)]) / len(
                pseudo_labels_aug_list) +
                                  unsup_dice_loss_2)

            for p in model_agnostic.parameters():
                p.requires_grad = True
            for p in model.parameters():
                p.requires_grad = False

            loss_agnostic = sup_loss_1 + unsup_loss_1
            loss_agnostic.backward()
            optimizer_agnostic.step()
            scheduler_agnostic.step()
            optimizer_agnostic.zero_grad()

            for p in model_agnostic.parameters():
                p.requires_grad = False
            for p in model.parameters():
                p.requires_grad = True
                
            tau=(1+np.cos((iter_nums/args.max_iteration)*np.pi))/2
            loss_t = sup_loss_2 + unsup_loss_2+tau*loss_dste+0.2*tau*loss_contrast
            loss_t.backward()
            optimizer_t.step()
            scheduler_t.step()
            optimizer_t.zero_grad()

            iter_nums += 1
            # lr_ = learning_rate * (1.0 - iter_nums / max_iteration) ** 0.9
            # for param_group in optimizer_agnostic.param_groups:
                # param_group['lr'] = lr_
            # for param_group in optimizer_t.param_groups:
                # param_group['lr'] = lr_
            
            avloss_agnostic_train += sup_loss_1.item()
            avloss_agnostic_train_unsup += unsup_loss_1.item()
            avloss_time_train += sup_loss_2.item()
            avloss_time_train_unsup += unsup_loss_2.item()

            train_bar.desc = "train epoch[{}/{}] sup_loss:{:.3f} unsup_loss:{:.3f}".format(epoch + 1, epochs,
                                                                                           sup_loss_1,
                                                                                           unsup_loss_1)
            
            writer.add_scalar("Model Train/Loss", sup_loss_1.item(), iter_nums)
            writer.add_scalar("Model Train/DiceLoss", unsup_loss_1.item(), iter_nums)
            writer.add_scalar("Model Train/GContrast", losses_contrast['global_contrastive'].item(), iter_nums)
            writer.add_scalar("Model Train/TContrast", losses_contrast['temporal_contrastive'].item(), iter_nums)
            writer.add_scalar("Model Train/loss_spatial_contrast", losses_dste['loss_spatial_contrast'].item(), iter_nums)
            writer.add_scalar("Model Train/loss_temporal_consistency", losses_dste['loss_temporal_consistency'].item(), iter_nums)
            # writer.add_scalar("Model Train/loss_rpp", losses_dste['loss_rpp'].item(), iter_nums)

        avdsc_valid = 0.0
        model.eval()
        model_agnostic.eval()

        dsc_val=[]
        with torch.no_grad():
            for val_data in valid_loader:
                val_images, val_labels = val_data

                masks = model_agnostic(val_images.cuda())

                dsc = DSC_average(masks, val_labels.cuda(),average=False)
                dsc_val.append(dsc)

                avdsc_valid += dsc.mean().item()
        dsc_1,dsc_2,dsc_3=sum(dsc_val)/len(dsc_val)

        avloss_agnostic_train /= len(train_loader)
        avloss_time_train /= len(train_loader)
        avloss_agnostic_train_unsup /= len(train_loader)
        avloss_time_train_unsup /= len(train_loader)
        avdsc_valid /= len(valid_loader)

        print(
            "train epoch[{}/{}] Training average sup_loss_1:{:.3f}, unsup_loss_1:{:.3f},"
            "sup_loss_2:{:.3f}, unsup_loss_2:{:.3f} "
            "Validation average DSC:{:.3f}".format(
                epoch + 1, epochs,
                avloss_agnostic_train, avloss_agnostic_train_unsup,
                avloss_time_train, avloss_time_train_unsup,
                avdsc_valid
            )
        )

        if avdsc_valid > best_valid_dsc:
            best_valid_dsc = avdsc_valid
            final_train_dsc = avdsc_valid
            print("model saved !")
            torch.save(model.state_dict(), save_path_time)
            torch.save(model_agnostic.state_dict(), save_path_agnostic)

        writer.add_scalar("Model Train/EpochLoss", avloss_agnostic_train, epoch)
        writer.add_scalar("Model Train/EpochDiceLoss", avloss_agnostic_train_unsup, epoch)
        writer.add_scalar("Model Validation/DSC_average", avdsc_valid, epoch)
        writer.add_scalar("Model Validation/DSC_1", dsc_1.item(), epoch)
        writer.add_scalar("Model Validation/DSC_2", dsc_2.item(), epoch)
        writer.add_scalar("Model Validation/DSC_3", dsc_3.item(), epoch)
        if iter_nums>=args.max_iteration:
            break

    writer.close()
    print("Finished Training !")



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
            # print(f"Post-processing prediction for {img_name}...")
            # pred_res = post_process_segmentation(pred_res)
            
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
def series_features():
    image_size = (args.image_size[0], args.image_size[1])
    num_classes = args.num_classes
    model_dir = args.train_output_dir + '/model'
    test_output_dir = args.test_output_dir
    pred_dir = test_output_dir + '/feature_viz'
    model_path = os.path.join(model_dir, 'agnostic.pth')
    model_agnostic = SingleUnet('resnet50', None, classes=4, deep_stem=32)
    series_model_path = os.path.join(model_dir,'segmodel.pth')
    model_series = TempSeg_Mem_New_ALL('resnet50', None, frame_number=args.frame_number, classes=4,
                        reduction_rate=args.reduction_rate, deep_stem=32)
    if torch.cuda.is_available():
        model_agnostic.cuda()
        model_series.cuda()
    model_agnostic.load_state_dict(torch.load(model_path, map_location=device))
    model_agnostic.eval()
    model_series.load_state_dict(torch.load(series_model_path,map_location=device))
    model_series.cuda()
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

            img = np.squeeze(nib_img.get_data().astype('float32'))

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
    parser.add_argument("--batch_size", type=int, default=2)
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
    parser.add_argument('--model_type',type=str, default=TempSeg_Mem_New_ALL)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--K_sample',type=int,default=4)
    parser.add_argument('--bkg_sample_size',type=int,default=100)
    parser.add_argument('--tau_bkg',type=float,default=0.5)
    args = parser.parse_args()
    
    torch.cuda.set_device(args.device)
    device=torch.device('cuda')

    mode = args.mode

    if mode == 'train':
        training(args)
        testing(args)

    if mode == 'test':
        testing(args)
    else:
        raise NotImplementedError
