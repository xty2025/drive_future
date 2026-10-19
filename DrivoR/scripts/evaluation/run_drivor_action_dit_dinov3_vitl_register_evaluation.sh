#!/usr/bin/env bash
set -euo pipefail

DRIVOR_REGISTER_VARIANT="${1:?Specify register, register_scorer, register_scorer_single_gt, register_staged, pseudo_expert_vmse_register or pseudo_expert_vmse_register_scorer}"
DRIVOR_REGISTER_CHECKPOINT="${2:?Specify a complete trained checkpoint}"
shift 2
case "${DRIVOR_REGISTER_VARIANT}" in
    # Pseudo-experts add no parameters and are needed only for training.
    register|pseudo_expert_vmse_register) DRIVOR_REGISTER_AGENT=drivoR_action_dit_dinov3_vitl_register ;;
    register_scorer|register_scorer_single_gt|register_staged|pseudo_expert_vmse_register_scorer)
        DRIVOR_REGISTER_AGENT=drivoR_action_dit_dinov3_vitl_register_scorer ;;
    *) printf 'Unknown register variant: %s\n' "${DRIVOR_REGISTER_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_REGISTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-eval_navtest_action_dit_dinov3_vitl_${DRIVOR_REGISTER_VARIANT}}"
export DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}"
export DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}"
DRIVOR_REGISTER_QUERIES="${DRIVOR_SCENE_QUERIES:-0}"
DRIVOR_REGISTER_VISUAL_MODE=native_patches
if (( DRIVOR_REGISTER_QUERIES > 0 )); then
    DRIVOR_REGISTER_VISUAL_MODE=drivor_queries
fi
exec bash "${DRIVOR_REGISTER_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_REGISTER_CHECKPOINT}" \
    "agent=${DRIVOR_REGISTER_AGENT}" \
    agent.config.multi_gt.enabled=false \
    agent.config.future_register.init_checkpoint=null \
    agent.config.action_dit.pretrained_path=null \
    "agent.config.visual_feature_mode=${DRIVOR_REGISTER_VISUAL_MODE}" \
    "agent.config.num_scene_tokens=${DRIVOR_REGISTER_QUERIES}" \
    "agent.config.future_register.scorer.fusion=${DRIVOR_REGISTER_FUSION:-project}" \
    "trainer.params.precision=${DRIVOR_EVAL_PRECISION:-32-true}" \
    dataloader.params.batch_size=1 "$@"
