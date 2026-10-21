#!/usr/bin/env bash
set -euo pipefail

DRIVOR_XL1_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_TRAIN_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_xl1}"
export DRIVOR_EVAL_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_action_dit_dinov3_vitl_single_gt_long_xl1}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-action_dit_dinov3_vitl_single_gt_long_xl1_seed2_train_eval_run1}}"
export DRIVOR_EVAL_PRECISION=32-true
exec bash "${DRIVOR_XL1_DIR}/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh" \
    single_gt_long agent=drivoR_action_dit_dinov3_vitl_single_gt_long_xl1 "$@"
