#!/usr/bin/env bash
set -euo pipefail
DRIVOR_FULL_ID="${1:?Specify T07, T10, T05, A23, A21, A27 or A28}"
shift
DRIVOR_FULL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DRIVOR_FULL_DIR}/drivor_trainval_full_common.sh"
if [[ "${DRIVOR_FULL_ID}" == A21 ]]; then
    exec bash "${DRIVOR_FULL_DIR}/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh" human_diverse8_l1 "$@"
fi
exec bash "${DRIVOR_FULL_DIR}/run_drivor_${DRIVOR_FULL_STEM}_train_eval.sh" "$@"
