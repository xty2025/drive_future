#!/usr/bin/env bash
set -euo pipefail

# Reuse the complete legacy pipeline without editing its live Bash contents.
# Checkpoints retain the original Action-DiT inference parameter structure.
DRIVOR_XL1_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_TRAIN_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1}"
export DRIVOR_EVAL_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-action_dit_dinov3_vitl_human_xl1_diverse8_l1_seed2_train_eval_run1}}"
export DRIVOR_EVAL_PRECISION=32-true
exec bash "${DRIVOR_XL1_DIR}/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh" \
    human_diverse8_l1 agent=drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1 "$@"
