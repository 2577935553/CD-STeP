#!/bin/bash
# Hyper-parameter ablation: K_sample and bkg_sample_size for SSTCR.
#
# K_sample        : number of frames sparsely sampled per side in CSTC.
# bkg_sample_size : number of background pixels sampled when a frame has
#                   no foreground prediction (fallback path of CSTC).
# tau_bkg         : foreground/background threshold.
#
# Group 1: vary K_sample in {2,3,4,5,6}, fix bkg_sample_size=100.
# Group 2: vary bkg_sample_size in {64,128,256,512,1024}, fix K_sample=4.
#
# Default uses 8 GPUs (device 0..7).  Edit DEVICE_BASE if you have fewer.

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_hyperparam_kp}
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
 --mode             train"

# Group 1: K_sample ablation (bkg fixed to 100)
i=0
for K in 2 3 5 6; do
    DEVICE=$((DEVICE_BASE + i))
    nohup python ../train_acdc_ablation_kp.py ${COMMON} \
        --K_sample        ${K} \
        --bkg_sample_size 100 \
        --train_output_dir ${OUT_ROOT}/K${K}_bkgs100 \
        --test_output_dir  ${OUT_ROOT}/K${K}_bkgs100 \
        --device          ${DEVICE} \
        > ${OUT_ROOT}/K${K}_bkgs100.log 2>&1 &
    i=$((i+1))
done

# Group 2: bkg_sample_size ablation (K_sample fixed to 4)
for BKG in 64 128 512 1024; do
    DEVICE=$((DEVICE_BASE + i))
    nohup python ../train_acdc_ablation_kp.py ${COMMON} \
        --K_sample        4 \
        --bkg_sample_size ${BKG} \
        --train_output_dir ${OUT_ROOT}/K4_bkgs${BKG} \
        --test_output_dir  ${OUT_ROOT}/K4_bkgs${BKG} \
        --device          ${DEVICE} \
        > ${OUT_ROOT}/K4_bkgs${BKG}.log 2>&1 &
    i=$((i+1))
done

echo "Launched 8 K/bkg ablation runs.  Logs: ${OUT_ROOT}/*.log"
