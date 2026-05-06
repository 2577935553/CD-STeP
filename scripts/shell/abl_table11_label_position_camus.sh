#!/bin/bash
# Ablation (Paper Table 11): label position on CAMUS-2CH (50 subjects).

DATA_DIR=${DATA_DIR:-/path/to/camus}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_label_position_camus}
DEVICE=${DEVICE:-0}

run_one () {
    local LB=$1                                   # 0..5
    python ../train_camus.py \
        --data_dir         ${DATA_DIR} \
        --test_data_dir    ${DATA_DIR} \
        --train_data_csv   ../../csv/train_Echo_50.csv \
        --train_data_gt    ../../csv/train_Echo_label_50.csv \
        --valid_data_csv   ../../csv/valid_Echo_50.csv \
        --test_data_list   ../../csv/test_Echo.csv \
        --train_output_dir ${OUT_ROOT}/lb_$((LB+1)) \
        --test_output_dir  ${OUT_ROOT}/lb_$((LB+1)) \
        --frame_number     6 \
        --label_index      ${LB} \
        --epochs           10000 \
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
