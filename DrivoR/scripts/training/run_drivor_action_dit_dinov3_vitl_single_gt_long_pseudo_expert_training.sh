#!/usr/bin/env bash
set -euo pipefail

# Ordinary human + long human v loss, plus 0.5 * CLOVER L1 coverage.
DRIVOR_LONG_PE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_pseudo_expert}"
exec bash "${DRIVOR_LONG_PE_DIR}/run_drivor_action_dit_dinov3_vitl_single_gt_long_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_single_gt_long_pseudo_expert \
    "$@"
