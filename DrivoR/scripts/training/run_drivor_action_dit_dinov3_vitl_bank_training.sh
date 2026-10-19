#!/usr/bin/env bash
set -euo pipefail

# DINOv3-L Action-DiT multi-GT baseline + World4Scorer Bank loss only.
DRIVOR_AUX_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_bank}"
exec bash "${DRIVOR_AUX_DIR}/run_drivor_action_dit_dinov3_vitl_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_bank \
    "$@"
