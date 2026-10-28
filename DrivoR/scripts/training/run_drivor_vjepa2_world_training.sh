#!/usr/bin/env bash
set -euo pipefail

# Original DrivoR generator; independent World4Scorer / visual-backbone experiment.
DRIVOR_VARIANT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_vjepa2_world_navtrain}"
exec bash "${DRIVOR_VARIANT_DIR}/run_drivor_training.sh" \
    agent=drivoR_vjepa2_world \
    "$@"
