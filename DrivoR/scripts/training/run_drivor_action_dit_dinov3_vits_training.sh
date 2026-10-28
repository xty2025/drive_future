#!/usr/bin/env bash
set -euo pipefail

# Official DINOv3 ViT-S/16 + frozen base + Q/V LoRA; parent training recipe unchanged.
# DRIVOR_SCENE_QUERIES=0: native patches; e.g. 16: learned queries per enabled camera.
DRIVOR_DINO_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_DINO_QUERY_COUNT="${DRIVOR_SCENE_QUERIES:-0}"
if ! [[ "${DRIVOR_DINO_QUERY_COUNT}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf '%s\n' 'DRIVOR_SCENE_QUERIES 必须是非负整数。' >&2
    exit 2
fi
DRIVOR_DINO_VISUAL_MODE=native_patches
DRIVOR_DINO_EXPERIMENT=training_drivoR_action_dit_dinov3_vits_navtrain_multigt
if (( DRIVOR_DINO_QUERY_COUNT > 0 )); then
    DRIVOR_DINO_VISUAL_MODE=drivor_queries
    DRIVOR_DINO_EXPERIMENT="${DRIVOR_DINO_EXPERIMENT}_q${DRIVOR_DINO_QUERY_COUNT}"
fi
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-${DRIVOR_DINO_EXPERIMENT}}"

exec bash "${DRIVOR_DINO_SCRIPT_DIR}/run_drivor_action_dit_multigt_training.sh" \
    agent=drivoR_action_dit_dinov3_vits \
    "agent.config.visual_feature_mode=${DRIVOR_DINO_VISUAL_MODE}" \
    "agent.config.num_scene_tokens=${DRIVOR_DINO_QUERY_COUNT}" \
    "$@"
