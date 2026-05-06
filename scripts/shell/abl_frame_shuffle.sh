#!/bin/bash
# Frame-shuffle ablation: at each step, randomly permute the T input frames
# per sample.  Tests whether the model relies on chronological ordering for
# DSTE causal modeling.
#
# Single-GPU; the train script picks the prediction at the labeled-frame's
# new position via torch.gather.

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_DIR=${OUT_DIR:-./outputs/abl_shuffle}
DEVICE=${DEVICE:-0}

python ../train_acdc_ablation_shuffle.py \
    --data_dir         ${DATA_DIR} \
    --test_data_dir    ${TEST_DIR} \
    --train_data_csv   ../../csv/train_subj_T6.csv \
    --train_data_gt    ../../csv/train_label.csv \
    --valid_data_csv   ../../csv/valid.csv \
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
    --K_sample         4 \
    --bkg_sample_size  100 \
    --mode             train \
    --device           ${DEVICE}
