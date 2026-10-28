#!/usr/bin/env bash
set -euo pipefail

# Original DrivoR + CLOVER Stage-1 final-proposal pseudo-expert coverage.
DRIVOR_VARIANT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_pseudo_expert_navtrain}"
exec bash "${DRIVOR_VARIANT_DIR}/run_drivor_training.sh" \
    agent=drivoR_pseudo_expert \
    "$@"
