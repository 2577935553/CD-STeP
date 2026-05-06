#!/bin/bash
# Ablation study (Paper Table 6): CSTC vs CGGC components inside SSTCR.
#
# Uses train_acdc_ablation_sstcr.py whose flags enable individual sub-losses:
#   --TC : add CSTC (cross-net spatial-temporal contrastive, sparse pseudo-label)
#   --GC : add CGGC (cross-net GT-guided global contrastive)

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_sstcr}
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

# 1) baseline (no contrastive)
python ../train_acdc_ablation_sstcr.py ${COMMON} \
    --train_output_dir ${OUT_ROOT}/baseline \
    --test_output_dir  ${OUT_ROOT}/baseline

# 2) CSTC only
python ../train_acdc_ablation_sstcr.py ${COMMON} --TC \
    --train_output_dir ${OUT_ROOT}/only_cstc \
    --test_output_dir  ${OUT_ROOT}/only_cstc

# 3) CGGC only
python ../train_acdc_ablation_sstcr.py ${COMMON} --GC \
    --train_output_dir ${OUT_ROOT}/only_cggc \
    --test_output_dir  ${OUT_ROOT}/only_cggc

# 4) CSTC + CGGC (full SSTCR)
python ../train_acdc_ablation_sstcr.py ${COMMON} --TC --GC \
    --train_output_dir ${OUT_ROOT}/full_sstcr \
    --test_output_dir  ${OUT_ROOT}/full_sstcr
