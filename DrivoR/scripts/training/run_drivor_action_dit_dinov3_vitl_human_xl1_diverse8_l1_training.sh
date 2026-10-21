#!/usr/bin/env bash
set -euo pipefail

DRIVOR_XL1_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1}"
exec bash "${DRIVOR_XL1_DIR}/run_drivor_action_dit_dinov3_vitl_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1 \
    agent.config.long_trajectory_additional_poses=0 "$@"
