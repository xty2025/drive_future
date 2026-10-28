#!/usr/bin/env bash
set -euo pipefail

# Front-only/native patches; one 64-proposal X-prediction batch supervised by all GTs.
DRIVOR_MULTIGT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_navtrain_multigt}"

exec bash "${DRIVOR_MULTIGT_SCRIPT_DIR}/run_drivor_action_dit_training.sh" \
    agent=drivoR_action_dit_multigt \
    agent.config.long_trajectory_additional_poses=0 \
    "$@"
