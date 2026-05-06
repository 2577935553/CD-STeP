#!/bin/bash
# Ablation study (Paper Table 5): SSTCR vs DSTE module on ACDC 20-subject setting.
#
# 4 settings:
#   no_sstcr_no_dste  : pure baseline (cross-pseudo supervision only)
#   only_sstcr        : + SSTCR
#   only_dste         : + DSTE
#   full              : SSTCR + DSTE  (== run_acdc_20subj.sh)
#
# Use --use_contrast / --use_dste flags to enable each module.

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_modules}
DEVICE=${DEVICE:-0}

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
 --mode             train \
 --device           ${DEVICE}"

# 1) baseline (no SSTCR, no DSTE)
python ../train_acdc_ablation_modules.py ${COMMON} \
    --train_output_dir ${OUT_ROOT}/baseline \
    --test_output_dir  ${OUT_ROOT}/baseline

# 2) SSTCR only
python ../train_acdc_ablation_modules.py ${COMMON} --use_contrast \
    --train_output_dir ${OUT_ROOT}/only_sstcr \
    --test_output_dir  ${OUT_ROOT}/only_sstcr

# 3) DSTE only
python ../train_acdc_ablation_modules.py ${COMMON} --use_dste \
    --train_output_dir ${OUT_ROOT}/only_dste \
    --test_output_dir  ${OUT_ROOT}/only_dste

# 4) full (== main run)
python ../train_acdc_ablation_modules.py ${COMMON} --use_contrast --use_dste \
    --train_output_dir ${OUT_ROOT}/full \
    --test_output_dir  ${OUT_ROOT}/full
