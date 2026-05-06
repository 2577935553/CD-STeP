#!/bin/bash
# CD-STeP main run on CAMUS-2CH, 50 training subjects.
# Reproduces Table 3 of the paper (single-frame supervision, 6-frame input).

DATA_DIR=${DATA_DIR:-/path/to/camus}
OUT_DIR=${OUT_DIR:-./outputs/CD-STeP_CAMUS_2CH_50subj}
DEVICE=${DEVICE:-0}

python ../train_camus.py \
    --data_dir         ${DATA_DIR} \
    --test_data_dir    ${DATA_DIR} \
    --train_data_csv   ../../csv/train_Echo_50.csv \
    --train_data_gt    ../../csv/train_Echo_label_50.csv \
    --valid_data_csv   ../../csv/valid_Echo_50.csv \
    --test_data_list   ../../csv/test_Echo.csv \
    --train_output_dir ${OUT_DIR} \
    --test_output_dir  ${OUT_DIR} \
    --frame_number     6 \
    --label_index      0 \
    --epochs           10000 \
    --max_iteration    30000 \
    --learning_rate    0.01 \
    --batch_size       4 \
    --reduction_rate   128 \
    --mode             train \
    --device           ${DEVICE}
