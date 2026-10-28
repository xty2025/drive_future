#!/usr/bin/env bash
set -euo pipefail

# Original human GT only; same DINOv3-L, Action/Register DiT and fused scorer.
DRIVOR_SINGLE_REGISTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_register_scorer_single_gt}"
exec bash "${DRIVOR_SINGLE_REGISTER_DIR}/run_drivor_action_dit_dinov3_vitl_register_scorer_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_register_scorer_single_gt \
    "$@"
