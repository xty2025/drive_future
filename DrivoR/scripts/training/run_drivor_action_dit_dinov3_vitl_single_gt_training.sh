#!/usr/bin/env bash
set -euo pipefail

# Original human GT only; no long GT, exported multi-GT or auxiliary objectives.
DRIVOR_GT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt}"
exec bash "${DRIVOR_GT_DIR}/run_drivor_action_dit_dinov3_vitl_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_single_gt \
    "$@"
