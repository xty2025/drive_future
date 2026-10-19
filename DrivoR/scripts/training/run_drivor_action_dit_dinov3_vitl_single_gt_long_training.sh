#!/usr/bin/env bash
set -euo pipefail

# Ordinary human GT + long GT: two 64-candidate groups packed into one DiT pass.
DRIVOR_GT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long}"
exec bash "${DRIVOR_GT_DIR}/run_drivor_action_dit_dinov3_vitl_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_single_gt_long \
    agent.config.long_trajectory_additional_poses=2 \
    "$@"
