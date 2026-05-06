#!/bin/bash
# Ablation study (Paper Table 7): CDE / SCA / DMBF inside the DSTE module.
#
# DSTE-internal flags exposed by train_acdc_ablation_dste.py:
#   --TB     : enable CDE (Causal Dynamics Estimation, temporal branch)
#   --SB     : enable SCA (Spatial Context Aggregator, spatial branch)
#   --Gate   : enable DMBF (Dynamic Multi-Branch Fusion gating; if off, branches sum)
#   --wtloss : turn OFF the self-supervised dynamics loss L_dyn  (CDE w/o L_dyn)
#
# Reproduces the 9 rows of Table 7 by toggling these flags.

DATA_DIR=${DATA_DIR:-/path/to/acdc/training}
TEST_DIR=${TEST_DIR:-/path/to/acdc/testing}
OUT_ROOT=${OUT_ROOT:-./outputs/abl_dste}
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

# Single branches
python ../train_acdc_ablation_dste.py ${COMMON} --TB \
    --train_output_dir ${OUT_ROOT}/only_cde   --test_output_dir ${OUT_ROOT}/only_cde
python ../train_acdc_ablation_dste.py ${COMMON} --SB \
    --train_output_dir ${OUT_ROOT}/only_sca   --test_output_dir ${OUT_ROOT}/only_sca
python ../train_acdc_ablation_dste.py ${COMMON} --Gate \
    --train_output_dir ${OUT_ROOT}/only_dmbf  --test_output_dir ${OUT_ROOT}/only_dmbf

# Pairs
python ../train_acdc_ablation_dste.py ${COMMON} --TB --SB \
    --train_output_dir ${OUT_ROOT}/cde_sca    --test_output_dir ${OUT_ROOT}/cde_sca
python ../train_acdc_ablation_dste.py ${COMMON} --TB --Gate \
    --train_output_dir ${OUT_ROOT}/cde_dmbf   --test_output_dir ${OUT_ROOT}/cde_dmbf
python ../train_acdc_ablation_dste.py ${COMMON} --SB --Gate \
    --train_output_dir ${OUT_ROOT}/sca_dmbf   --test_output_dir ${OUT_ROOT}/sca_dmbf

# CDE w/o L_dyn (full architecture, but no self-supervised dynamics loss)
python ../train_acdc_ablation_dste.py ${COMMON} --TB --SB --Gate --wtloss \
    --train_output_dir ${OUT_ROOT}/full_wo_ldyn \
    --test_output_dir  ${OUT_ROOT}/full_wo_ldyn

# Full DSTE
python ../train_acdc_ablation_dste.py ${COMMON} --TB --SB --Gate \
    --train_output_dir ${OUT_ROOT}/full       --test_output_dir ${OUT_ROOT}/full
