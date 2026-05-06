#!/bin/bash
# Ablation study (Paper Table 10): label position on ACDC.
#
# Within a 6-frame sequence, varies which frame index (1st .. 6th)
# carries the manual label.  --label_index is 0..5 (0 = 1st frame).

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_label_position_acdc}
DEVICE=${DEVICE:-0}

run_one () {
    local LB=$1                                   # 0..5
    python ../train_acdc.py \
        --data_dir         ${DATA_DIR} \
        --test_data_dir    ${TEST_DIR} \
        --train_data_csv   ../../csv/train_subj_T6.csv \
        --train_data_gt    ../../csv/train_label.csv \
        --valid_data_csv   ../../csv/valid.csv \
        --test_data_list   ../../csv/test_all_phases.csv \
        --train_output_dir ${OUT_ROOT}/lb_$((LB+1)) \
        --test_output_dir  ${OUT_ROOT}/lb_$((LB+1)) \
        --frame_number     6 \
        --label_index      ${LB} \
        --epochs           1000 \
        --max_iteration    30000 \
        --learning_rate    0.01 \
        --batch_size       4 \
        --reduction_rate   128 \
        --mode             train \
        --device           ${DEVICE}
}

for LB in 0 1 2 3 4 5; do
    run_one ${LB}
done
