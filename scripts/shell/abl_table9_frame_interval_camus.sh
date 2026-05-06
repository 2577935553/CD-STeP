#!/bin/bash
# Ablation (Paper Table 9): inter-frame interval on CAMUS-2CH (50 subjects).

DATA_DIR=${DATA_DIR:-/path/to/camus}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_frame_interval_camus}
DEVICE=${DEVICE:-0}

run_one () {
    local ITV=$1
    local CSV
    if [ "${ITV}" = "0" ]; then
        CSV=../../csv/train_Echo_50.csv
    else
        CSV=../../csv/train_Echo_50_interval_${ITV}.csv
    fi
    python ../train_camus.py \
        --data_dir         ${DATA_DIR} \
        --test_data_dir    ${DATA_DIR} \
        --train_data_csv   ${CSV} \
        --train_data_gt    ../../csv/train_Echo_label_50.csv \
        --valid_data_csv   ../../csv/valid_Echo_50.csv \
        --test_data_list   ../../csv/test_Echo.csv \
        --train_output_dir ${OUT_ROOT}/itv_${ITV} \
        --test_output_dir  ${OUT_ROOT}/itv_${ITV} \
        --frame_number     6 \
        --label_index      0 \
        --epochs           10000 \
        --max_iteration    30000 \
        --learning_rate    0.01 \
        --batch_size       4 \
        --reduction_rate   128 \
        --mode             train \
        --device           ${DEVICE}
}

for ITV in 0 1 2 3 4; do
    run_one ${ITV}
done
