#!/usr/bin/env bash
# Metadata shared by the new full-train entry wrappers only.
DRIVOR_FULL_ID="${DRIVOR_FULL_ID^^}"
case "${DRIVOR_FULL_ID}" in
    T07) DRIVOR_FULL_STEM=query_human_diverse8_l1; DRIVOR_FULL_SUFFIX=query_dinov3_vitl_human_diverse8_l1 ;;
    T10) DRIVOR_FULL_STEM=query_single_gt; DRIVOR_FULL_SUFFIX=query_dinov3_vitl_single_gt ;;
    T05) DRIVOR_FULL_STEM=query_single_gt_long; DRIVOR_FULL_SUFFIX=query_dinov3_vitl_single_gt_long ;;
    A23) DRIVOR_FULL_STEM=action_dit_dinov3_vitl_human_xl1_diverse8_l1; DRIVOR_FULL_SUFFIX="${DRIVOR_FULL_STEM}" ;;
    A21) DRIVOR_FULL_STEM=action_dit_dinov3_vitl_human_diverse8_l1; DRIVOR_FULL_SUFFIX="${DRIVOR_FULL_STEM}" ;;
    A27) DRIVOR_FULL_STEM=action_dit_dinov3_vitl_human_xl1_diverse8_l1_register; DRIVOR_FULL_SUFFIX="${DRIVOR_FULL_STEM}" ;;
    A28) DRIVOR_FULL_STEM=action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer; DRIVOR_FULL_SUFFIX="${DRIVOR_FULL_STEM}" ;;
    *) printf 'Expected T07, T10, T05, A23, A21, A27 or A28; got %s\n' "${DRIVOR_FULL_ID}" >&2; exit 2 ;;
esac
export DRIVOR_TRAINING_ENTRY=run_training_full.py
export DRIVOR_TRAINING_CONFIG="full_${DRIVOR_FULL_ID,,}"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_${DRIVOR_FULL_SUFFIX}_trainval_full}"
export DRIVOR_TRAIN_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT:-${DRIVOR_EXPERIMENT}}"
export DRIVOR_EVAL_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_${DRIVOR_FULL_SUFFIX}_trainval_full}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-${DRIVOR_FULL_ID,,}_trainval_full_seed2_run1}}"
export DRIVOR_AUTO_RESUME="${DRIVOR_AUTO_RESUME:-false}"
export DRIVOR_PRECISION="${DRIVOR_PRECISION:-bf16-mixed}"
export DRIVOR_EVAL_PRECISION=32-true
