#!/usr/bin/env bash
set -euo pipefail

# Reuse the existing single_gt_long pipeline with the new agent override.
# Keep shared launchers intact while other DLC tasks are still reading them.
DRIVOR_LONG_PE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_TRAIN_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_pseudo_expert}"
export DRIVOR_EVAL_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_action_dit_dinov3_vitl_single_gt_long_pseudo_expert}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-action_dit_dinov3_vitl_single_gt_long_pseudo_expert_seed2_train_eval_run1}}"
export DRIVOR_EVAL_PRECISION=32-true
exec bash "${DRIVOR_LONG_PE_DIR}/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh" \
    single_gt_long \
    agent=drivoR_action_dit_dinov3_vitl_single_gt_long_pseudo_expert \
    "$@"
