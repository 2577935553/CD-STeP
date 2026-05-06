#!/bin/bash
# Hyper-parameter ablation: foreground/background threshold tau_bkg in CSTC.
# Default value used in the paper is 0.5; here we sweep {0.3, 0.4, 0.6, 0.7}.

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_hyperparam_tau}
DEVICE_BASE=${DEVICE_BASE:-0}

COMMON="\
 --data_dir         ${DATA_DIR} \
 --test_data_dir    ${TEST_DIR} \
 --train_data_csv   ../../csv/train_subj_T6.csv \
 --train_data_gt    ../../csv/train_label.csv \
 --valid_data_csv   ../../csv/valid.csv \
 --test_data_list   ../../csv/test_all_phases.csv \
 --frame_number     6 \
 --label_index      0 \
 --epochs           1000 \
 --max_iteration    30000 \
 --learning_rate    0.01 \
 --batch_size       4 \
 --reduction_rate   128 \
 --K_sample         4 \
 --bkg_sample_size  100 \
 --mode             train"

i=0
for TAU in 0.3 0.4 0.6 0.7; do
    DEVICE=$((DEVICE_BASE + i))
    TAU_INT=$(awk "BEGIN {print int(${TAU} * 10)}")
    nohup python ../train_acdc_ablation_kp.py ${COMMON} \
        --tau_bkg ${TAU} \
        --train_output_dir ${OUT_ROOT}/tau${TAU_INT} \
        --test_output_dir  ${OUT_ROOT}/tau${TAU_INT} \
        --device   ${DEVICE} \
        > ${OUT_ROOT}/tau${TAU_INT}.log 2>&1 &
    i=$((i+1))
done

echo "Launched 4 tau ablation runs.  Logs: ${OUT_ROOT}/tau*.log"
