#!/usr/bin/env bash
set -euo pipefail
# A27: trajectory H + D8; future-register supervision H only.
DRIVOR_DIVERSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_TRAINING_ENTRY=run_training.py
export DRIVOR_TRAINING_CONFIG=default_training
exec bash "${DRIVOR_DIVERSE_DIR}/run_drivor_diverse8_register_training.sh" action_dit_human_xl1_diverse8_l1_register_human_only "$@"
