#!/usr/bin/env bash
set -euo pipefail

# Action-DiT with original DrivoR visual features:
# four current cameras, 16 Scene queries per camera, 64 visual tokens for DiT/scorer.
# All other training, inference, loss, optimizer and DLC settings use the existing wrapper.
# New run:
# DRIVOR_RUN_ID=action_dit_multiview_q16_seed2_run1 bash DrivoR/scripts/training/run_drivor_action_dit_multiview_training.sh auto_resume=false train_ckpt_path=null runtime.require_resume=false
# Inspect:
# bash DrivoR/scripts/training/run_drivor_action_dit_multiview_training.sh --cfg job --package agent --resolve

DRIVOR_MULTIVIEW_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_multiview_navtrain}"

exec bash "${DRIVOR_MULTIVIEW_SCRIPT_DIR}/run_drivor_action_dit_training.sh" \
    agent=drivoR_action_dit_multiview \
    agent.config.num_scene_tokens=16 \
    "$@"
