#!/usr/bin/env bash
set -euo pipefail
DRIVOR_DIVERSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_TRAINING_ENTRY=run_training.py
export DRIVOR_TRAINING_CONFIG=default_training
exec bash "${DRIVOR_DIVERSE_DIR}/run_drivor_diverse8_register_train_eval.sh" action_dit_human_xl1_diverse8_l1_register_scorer_human_only "$@"
