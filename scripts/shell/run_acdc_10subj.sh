#!/bin/bash
# CD-STeP main run on ACDC, 10 training subjects.
# Reproduces Table 1 of the paper (single-frame supervision, 6-frame input).

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_DIR=${OUT_DIR:-./outputs/CD-STeP_ACDC_10subj}
DEVICE=${DEVICE:-0}

python ../train_acdc.py \
    --data_dir         ${DATA_DIR} \
    --test_data_dir    ${TEST_DIR} \
    --train_data_csv   ../../csv/train_subj_10_T6.csv \
    --train_data_gt    ../../csv/train_label.csv \
    --valid_data_csv   ../../csv/valid_10.csv \
    --test_data_list   ../../csv/test_all_phases.csv \
    --train_output_dir ${OUT_DIR} \
    --test_output_dir  ${OUT_DIR} \
    --frame_number     6 \
    --label_index      0 \
    --epochs           1000 \
    --max_iteration    30000 \
    --learning_rate    0.01 \
    --batch_size       4 \
    --reduction_rate   128 \
    --mode             train \
    --device           ${DEVICE}
