#!/bin/bash
# Ablation study (Paper Table 8 / Table 9): inter-frame interval on ACDC / CAMUS.
#
# At training time we sample 6 frames starting from frame 0 with a stride
# (interval) of  i  --  i.e. the frame indices are 0, 1+i, 2+(2i), ... for i=0..5.
# CSV files train_subj_T6_interval_{2..6}.csv encode the resulting indices.
#
# Required flags below match the original code paths:
#   * train_acdc.py uses SemiSegDataset_VT (default sampling)
#   * For interval > 1, the CSV directly lists the chosen frame indices,
#     and the dataset class handles the rest.

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_frame_interval_acdc}
DEVICE=${DEVICE:-0}

run_one () {
    local ITV=$1                                  # 0,1,2,3,4,5
    local CSV
    if [ "${ITV}" = "0" ]; then
        CSV=../../csv/train_subj_T6.csv
    else
        CSV=../../csv/train_subj_T6_interval_${ITV}.csv
    fi
    python ../train_acdc.py \
        --data_dir         ${DATA_DIR} \
        --test_data_dir    ${TEST_DIR} \
        --train_data_csv   ${CSV} \
        --train_data_gt    ../../csv/train_label.csv \
        --valid_data_csv   ../../csv/valid.csv \
        --test_data_list   ../../csv/test_all_phases.csv \
        --train_output_dir ${OUT_ROOT}/itv_${ITV} \
        --test_output_dir  ${OUT_ROOT}/itv_${ITV} \
        --frame_number     6 \
        --label_index      0 \
        --epochs           1000 \
        --max_iteration    30000 \
        --learning_rate    0.01 \
        --batch_size       4 \
        --reduction_rate   128 \
        --mode             train \
        --device           ${DEVICE}
}

for ITV in 0 1 2 3 4 5; do
    run_one ${ITV}
done
