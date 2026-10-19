#!/usr/bin/env bash
set -euo pipefail

# Original top4 + valid poor4 trajectories, equally weighted within each scene.
DRIVOR_GT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_poor}"
exec bash "${DRIVOR_GT_DIR}/run_drivor_action_dit_dinov3_vitl_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_multigt_poor \
    "$@"
