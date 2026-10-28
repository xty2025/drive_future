#!/usr/bin/env bash
set -euo pipefail

# Four-camera/16-query visual context; same shared 64-proposal multi-GT objective.
DRIVOR_MULTIGT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_multiview_navtrain_multigt}"

exec bash "${DRIVOR_MULTIGT_SCRIPT_DIR}/run_drivor_action_dit_multiview_training.sh" \
    agent=drivoR_action_dit_multiview_multigt \
    agent.config.long_trajectory_additional_poses=0 \
    "$@"
