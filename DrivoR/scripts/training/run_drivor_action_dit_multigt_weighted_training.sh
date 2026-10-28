#!/usr/bin/env bash
set -euo pipefail

# Same 64 X predictions: human v loss 1.0 + mean(non-human GT v loss) 0.5.
DRIVOR_WEIGHTED_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_navtrain_multigt_weighted}"

exec bash "${DRIVOR_WEIGHTED_SCRIPT_DIR}/run_drivor_action_dit_multigt_training.sh" \
    agent=drivoR_action_dit_multigt_weighted \
    "$@"
