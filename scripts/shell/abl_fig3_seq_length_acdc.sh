#!/bin/bash
# Sequence-length ablation (Paper Figure 3): train with N consecutive frames
# starting from the first (labeled) frame.  N varies from 2 to 30 on ACDC and
# from 2 to 26 on CAMUS.  Plots peak around N=6 (default).

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_seq_length_acdc}
DEVICE=${DEVICE:-0}

run_one () {
    local T=$1
    python ../train_acdc.py \
        --data_dir         ${DATA_DIR} \
        --test_data_dir    ${TEST_DIR} \
        --train_data_csv   ../../csv/train_subj_T${T}.csv \
        --train_data_gt    ../../csv/train_label.csv \
        --valid_data_csv   ../../csv/valid.csv \
        --test_data_list   ../../csv/test_all_phases.csv \
        --train_output_dir ${OUT_ROOT}/T${T} \
        --test_output_dir  ${OUT_ROOT}/T${T} \
        --frame_number     ${T} \
        --label_index      0 \
        --epochs           1000 \
        --max_iteration    30000 \
        --learning_rate    0.01 \
        --batch_size       4 \
        --reduction_rate   128 \
        --mode             train \
        --device           ${DEVICE}
}

# Sequence lengths used in the paper (Figure 3, ACDC).
for T in 2 3 4 5 6 7 8 9 10 11 12 13 14 16 18 20 22 24 26 28 30; do
    run_one ${T}
done
