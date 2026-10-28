#!/usr/bin/env bash
set -euo pipefail

# One distinct proposal per valid GT; average all matched L1 costs equally.
DRIVOR_HUNGARIAN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_Nav1_navtrain_multigt_hungarian}"

exec bash "${DRIVOR_HUNGARIAN_SCRIPT_DIR}/run_drivor_multigt_training.sh" \
    agent=drivoR_multigt_hungarian \
    "$@"
