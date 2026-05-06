# CSV split files

This directory holds the train / validation / test split CSVs that the
training scripts reference. They were excluded from the public repo
release; place your own dataset-split CSVs here following the layout below.

## ACDC

| File | What it lists |
|------|---------------|
| `train_subj_T6.csv` | 20 ACDC subject IDs, 6-frame sequences (default) |
| `train_subj_10_T6.csv` | 10-subject subset for the low-data run |
| `train_label.csv` | label-frame index (= 0 by default) |
| `valid.csv` | held-out validation IDs |
| `test_all_phases.csv` | all-phase test list |
| `train_subj_T6_interval_{2,3,4,5,6}.csv` | inter-frame interval ablation |
| `train_label_idx{1..6}.csv` | label-position ablation |
| `train_subj_T{2..30}.csv` | sequence-length ablation |

## CAMUS

| File | What it lists |
|------|---------------|
| `train_Echo.csv` / `train_Echo_label.csv` | 100-subject CAMUS-2CH training |
| `train_Echo_50.csv` / `train_Echo_label_50.csv` | 50-subject low-data CAMUS-2CH |
| `train_Echo_4CH.csv` / ..._4CH_50.csv` | CAMUS-4CH variants |
| `valid_Echo.csv` | validation list |
| `test_Echo.csv` | test list |

Each row is one subject ID. See the dataset-loading code in `data/datasets.py`
for the expected on-disk layout.
