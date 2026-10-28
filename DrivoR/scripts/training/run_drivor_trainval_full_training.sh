#!/usr/bin/env bash
set -euo pipefail
DRIVOR_FULL_ID="${1:?Specify T07, T10, T05, A23, A21, A27 or A28}"
shift
DRIVOR_FULL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DRIVOR_FULL_DIR}/drivor_trainval_full_common.sh"
exec bash "${DRIVOR_FULL_DIR}/run_drivor_${DRIVOR_FULL_STEM}_training.sh" "$@"
