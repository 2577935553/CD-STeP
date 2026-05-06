#!/bin/bash
# Sequence-length ablation on CAMUS-2CH (Paper Figure 3, right panel).

DATA_DIR=${DATA_DIR:-/path/to/camus}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_seq_length_camus}
DEVICE=${DEVICE:-0}

run_one () {
    local T=$1
    python ../train_camus.py \
        --data_dir         ${DATA_DIR} \
        --test_data_dir    ${DATA_DIR} \
        --train_data_csv   ../../csv/train_Echo_50.csv \
        --train_data_gt    ../../csv/train_Echo_label_50.csv \
        --valid_data_csv   ../../csv/valid_Echo_50.csv \
        --test_data_list   ../../csv/test_Echo.csv \
        --train_output_dir ${OUT_ROOT}/T${T} \
        --test_output_dir  ${OUT_ROOT}/T${T} \
        --frame_number     ${T} \
        --label_index      0 \
        --epochs           10000 \
        --max_iteration    30000 \
        --learning_rate    0.01 \
        --batch_size       4 \
        --reduction_rate   128 \
        --mode             train \
        --device           ${DEVICE}
}

for T in 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26; do
    run_one ${T}
done
