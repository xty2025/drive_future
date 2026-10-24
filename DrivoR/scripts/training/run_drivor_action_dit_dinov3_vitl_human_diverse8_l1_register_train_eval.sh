#!/usr/bin/env bash
set -euo pipefail
DRIVOR_DIVERSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DRIVOR_DIVERSE_DIR}/run_drivor_diverse8_register_train_eval.sh" action_dit_human_diverse8_l1_register "$@"
