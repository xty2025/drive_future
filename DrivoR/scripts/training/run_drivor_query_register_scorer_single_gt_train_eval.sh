#!/usr/bin/env bash
set -euo pipefail

DRIVOR_QUERY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DRIVOR_QUERY_DIR}/run_drivor_query_register_train_eval.sh" scorer_single_gt "$@"
