#!/usr/bin/env bash
set -euo pipefail

# Four-camera query features; equal-weight Hungarian multi-GT regression.
DRIVOR_HUNGARIAN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_multiview_navtrain_multigt_hungarian}"

exec bash "${DRIVOR_HUNGARIAN_SCRIPT_DIR}/run_drivor_action_dit_multiview_multigt_training.sh" \
    agent=drivoR_action_dit_multiview_multigt_hungarian \
    "$@"
