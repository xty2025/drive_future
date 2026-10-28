#!/usr/bin/env bash
set -euo pipefail

# V-JEPA 2 ViT-L: frozen pretrained target encoder, Q/V LoRA and optional queries.
DRIVOR_VJEPA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_VJEPA_QUERY_COUNT="${DRIVOR_SCENE_QUERIES:-0}"
if ! [[ "${DRIVOR_VJEPA_QUERY_COUNT}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf '%s\n' 'DRIVOR_SCENE_QUERIES 必须是非负整数。' >&2
    exit 2
fi
DRIVOR_VJEPA_VISUAL_MODE=native_patches
DRIVOR_VJEPA_EXPERIMENT=training_drivoR_action_dit_vjepa2_vitl_navtrain_multigt
if (( DRIVOR_VJEPA_QUERY_COUNT > 0 )); then
    DRIVOR_VJEPA_VISUAL_MODE=drivor_queries
    DRIVOR_VJEPA_EXPERIMENT="${DRIVOR_VJEPA_EXPERIMENT}_q${DRIVOR_VJEPA_QUERY_COUNT}"
fi
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-${DRIVOR_VJEPA_EXPERIMENT}}"

exec bash "${DRIVOR_VJEPA_SCRIPT_DIR}/run_drivor_action_dit_multigt_training.sh" \
    agent=drivoR_action_dit_vjepa2_vitl \
    "agent.config.visual_feature_mode=${DRIVOR_VJEPA_VISUAL_MODE}" \
    "agent.config.num_scene_tokens=${DRIVOR_VJEPA_QUERY_COUNT}" \
    "$@"
