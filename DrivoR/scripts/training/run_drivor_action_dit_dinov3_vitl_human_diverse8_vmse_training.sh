#!/usr/bin/env bash
set -euo pipefail

# Original human v MSE + 0.5 * new diverse8 v MSE, same shared training forward.
DRIVOR_DIVERSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_vmse}"
exec bash "${DRIVOR_DIVERSE_DIR}/run_drivor_action_dit_dinov3_vitl_human_diverse8_l1_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_human_diverse8_vmse \
    "$@"
