#!/usr/bin/env bash
set -euo pipefail

# Joint Action/Register denoising, original DrivoR scorer. Same DLC 8x8 launcher.
DRIVOR_REGISTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_register}"
exec bash "${DRIVOR_REGISTER_DIR}/run_drivor_action_dit_dinov3_vitl_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_register \
    agent.config.multi_gt.input_gt_assignment=original \
    "agent.config.future_register.loss_type=${DRIVOR_REGISTER_LOSS:-velocity}" \
    "agent.config.future_register.scorer.fusion=${DRIVOR_REGISTER_FUSION:-project}" \
    "$@"
