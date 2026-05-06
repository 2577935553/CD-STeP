# CD-STeP

Code for "Single-frame Semi-supervised Cine MRI and Echocardiography Sequence Segmentation".

## Paper-to-code mapping

| Paper | File | Class |
|---|---|---|
| SegSlc | `models/seg_models.py` | `SingleUnet` (alias `SegSlc`) |
| SegSeq | `models/seg_models.py` | `SegSeq` (alias `TempSeg_Mem_New_ALL`) |
| CSTC   | `models/seg_models.py` | `SegSeq.compute_temporal_similarity()` |
| CGGC   | `models/seg_models.py` | `SegSeq.compute_global_similarity()` |
| DSTE   | `models/dste.py` | `DSTE` |
| CDE    | `models/dste.py` | `CDE` (alias `ImprovedTemporalModule`) |
| SCA    | `models/dste.py` | `SCA` (alias `LightweightSpatialModule`) |
| DMBF   | `models/dste.py` | `DMBF` (alias `AdaptiveFusion`) |

## Layout

```
CD-STeP/
├── models/                   SegSlc, SegSeq, DSTE, CDE/SCA/DMBF, ContrastiveLoss
├── networks/                 ResNet encoders, U-Net decoder, CBAM
├── data/                     ACDC / CAMUS / EchoNet datasets + I/O
├── losses/                   CE / Dice losses + DSC metrics
├── utils/                    LR scheduler, image ops, ramps
├── scripts/
│   ├── train_acdc.py
│   ├── train_camus.py
│   ├── train_acdc_ablation_*.py
│   └── shell/                paper-table launchers
├── csv/                      train/val/test split CSVs (see csv/README.md)
└── requirements.txt
```

## Install

```bash
conda create -n cdstep python=3.10 -y
conda activate cdstep
pip install -r requirements.txt
```

Tested with PyTorch 1.7.1 + CUDA 11.0 (also works with PyTorch 2.0 + CUDA 11.8).

## Datasets

Download and place under your own paths:

- **ACDC** — <https://www.creatis.insa-lyon.fr/Challenge/acdc/>
  ```
  acdc/training/patientXXX/patientXXX_4d.nii.gz
                          /patientXXX_4d_manu.nii.gz
  acdc/testing/...
  ```
- **CAMUS** — <https://www.creatis.insa-lyon.fr/Challenge/camus/>

CSV split files are not included; create them following `csv/README.md`.

## Run

All shell scripts read `DATA_DIR`, `TEST_DIR`, `DEVICE` env vars.

```bash
cd scripts/shell

DATA_DIR=/path/to/acdc/training \
TEST_DIR=/path/to/acdc/testing \
DEVICE=0  bash run_acdc_20subj.sh

DATA_DIR=/path/to/camus  DEVICE=0  bash run_camus_2ch_100subj.sh
```

## Reproducing paper tables

| Paper | Launcher |
|---|---|
| Table 1  — ACDC 10 subj             | `run_acdc_10subj.sh` |
| Table 2  — ACDC 20 subj             | `run_acdc_20subj.sh` |
| Table 3  — CAMUS-2CH 50 subj        | `run_camus_2ch_50subj.sh` |
| Table 4  — CAMUS-2CH 100 subj       | `run_camus_2ch_100subj.sh` |
| Table 5  — SSTCR / DSTE             | `abl_table5_modules.sh` |
| Table 6  — CSTC / CGGC              | `abl_table6_sstcr.sh` |
| Table 7  — CDE / SCA / DMBF / L_dyn | `abl_table7_dste.sh` |
| Table 8  — frame interval (ACDC)    | `abl_table8_frame_interval_acdc.sh` |
| Table 9  — frame interval (CAMUS)   | `abl_table9_frame_interval_camus.sh` |
| Table 10 — label position (ACDC)    | `abl_table10_label_position_acdc.sh` |
| Table 11 — label position (CAMUS)   | `abl_table11_label_position_camus.sh` |
| Figure 3 — sequence length          | `abl_fig3_seq_length_{acdc,camus}.sh` |

Additional hyper-parameter sweeps:

| Study | Launcher |
|---|---|
| `K_sample` ∈ {2..6}, `bkg_sample_size` ∈ {64,128,256,512,1024} | `abl_hyperparam_K_bkg.sh` |
| `tau_bkg` ∈ {0.3..0.7}                                         | `abl_hyperparam_tau.sh` |
| per-step random frame shuffle                                  | `abl_frame_shuffle.sh` |

Multi-GPU `K=5/6` (per-rank batch = 2, effective batch = 4 via DDP+SyncBN):

```bash
torchrun --nproc_per_node=2 --master_port=29501 \
    scripts/train_acdc_ablation_kp_ddp.py [args...]
```

## Default hyper-parameters

| Setting | Value |
|---|---|
| Optimizer            | AdamW, weight-decay 1e-4 |
| LR schedule          | linear warm-up (10 %) → cosine annealing |
| Peak LR              | 4e-4 |
| Effective batch size | 4 |
| Total iterations     | 30 000 |
| Input resolution     | 224 × 224 |
| Frame number         | 6 |
| `K_sample`           | 4 |
| `bkg_sample_size`    | 100 |
| `tau_bkg`            | 0.5 |
| `λ` (Eq. 23)         | 0.2 |
| Augmentation         | rotation ±60°, scale 0.5–1.5× |

## CLI arguments

Common to all training scripts:

| Argument | Description |
|---|---|
| `--data_dir`         | Training images root |
| `--test_data_dir`    | Test images root |
| `--train_data_csv`   | CSV listing training subjects |
| `--train_data_gt`    | CSV listing the labeled-frame index per subject |
| `--valid_data_csv`   | CSV for validation |
| `--test_data_list`   | CSV for test |
| `--train_output_dir` | Where to save the checkpoints / TB logs |
| `--test_output_dir`  | Where to save predictions |
| `--frame_number`     | T, sequence length |
| `--label_index`      | Index of the labeled frame inside each T-frame window |
| `--max_iteration`    | Total iterations |
| `--learning_rate`    | Base LR (effective = `0.04 × this` after the internal scaler) |
| `--batch_size`       | Batch size |
| `--reduction_rate`   | Bottleneck channel reduction in SegSeq |
| `--mode`             | `train` or `test` |
| `--device`           | CUDA device id |

Ablation-specific flags:

| Script | Flag | Effect |
|---|---|---|
| `train_acdc_ablation_modules.py` | `--use_contrast`     | enable SSTCR (CSTC + CGGC) |
| `train_acdc_ablation_modules.py` | `--use_dste`         | enable DSTE bottleneck |
| `train_acdc_ablation_sstcr.py`   | `--TC`               | enable CSTC sub-loss |
| `train_acdc_ablation_sstcr.py`   | `--GC`               | enable CGGC sub-loss |
| `train_acdc_ablation_dste.py`    | `--TB`               | enable CDE branch |
| `train_acdc_ablation_dste.py`    | `--SB`               | enable SCA branch |
| `train_acdc_ablation_dste.py`    | `--Gate`             | enable DMBF gating fusion |
| `train_acdc_ablation_dste.py`    | `--wtloss`           | turn OFF the self-supervised L_dyn |
| `train_acdc_ablation_kp.py`      | `--K_sample`         | frames sparsely sampled per side in CSTC |
| `train_acdc_ablation_kp.py`      | `--bkg_sample_size`  | background pixels sampled in CSTC fallback |
| `train_acdc_ablation_kp.py`      | `--tau_bkg`          | foreground / background threshold |

## Inference

After training, the slice checkpoint is at `<train_output_dir>/model/agnostic.pth`.
Set `--mode test` on any train script to dump `*_Pred.nii.gz` under
`<test_output_dir>/predictions/`.

## Contact

Corresponding author: Prof. Xin Zhou (<xinzhou@wipm.ac.cn>).
Corresponding for code problem: Siyang Zhang (<zhsy2577935553@gmail.com>)
