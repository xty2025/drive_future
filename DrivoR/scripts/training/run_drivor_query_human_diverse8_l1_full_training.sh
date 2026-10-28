#!/usr/bin/env bash
set -euo pipefail
DRIVOR_FULL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DRIVOR_FULL_DIR}/run_drivor_trainval_full_training.sh" T07 "$@"
