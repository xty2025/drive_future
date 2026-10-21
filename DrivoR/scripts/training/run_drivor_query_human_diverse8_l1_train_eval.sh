#!/usr/bin/env bash
set -euo pipefail
DRIVOR_QUERY_GT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DRIVOR_QUERY_GT_DIR}/run_drivor_query_gt_train_eval.sh" human_diverse8_l1 "$@"
