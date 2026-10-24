#!/usr/bin/env bash
set -euo pipefail
DRIVOR_HALF_ENTRY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DRIVOR_HALF_ENTRY_DIR}/run_drivor_half_ablation_training.sh" a28_wan21_1p3b "$@"
