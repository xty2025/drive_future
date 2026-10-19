#!/usr/bin/env bash
set -euo pipefail

# DrivoR multi-view queries with the same configurable human/non-human v-loss weights.
DRIVOR_WEIGHTED_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_multiview_navtrain_multigt_weighted}"

exec bash "${DRIVOR_WEIGHTED_SCRIPT_DIR}/run_drivor_action_dit_multiview_multigt_training.sh" \
    agent=drivoR_action_dit_multiview_multigt_weighted \
    "$@"
