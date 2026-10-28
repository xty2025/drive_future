#!/usr/bin/env bash
set -euo pipefail

# Original DrivoR: human loss 1.0 + other-GT mean loss 0.5, following Drive-JEPA.
DRIVOR_MULTIGT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_Nav1_navtrain_multigt_jepa_weighted}"

exec bash "${DRIVOR_MULTIGT_SCRIPT_DIR}/run_drivor_training.sh" \
    agent=drivoR_multigt \
    agent.config.long_trajectory_additional_poses=0 \
    agent.loss.inter_weight=0.0 \
    "$@"
