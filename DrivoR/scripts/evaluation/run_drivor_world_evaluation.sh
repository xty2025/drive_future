#!/usr/bin/env bash
set -euo pipefail

# Usage: run_drivor_world_evaluation.sh <bank|vjepa2_vitl|vjepa2_1_vitl|vjepa2_world|vjepa2_world_multigt> <checkpoint> [overrides]
if (( $# < 2 )); then
    printf '%s\n' '需要模型版本和完整训练 checkpoint 路径。' >&2
    exit 2
fi
DRIVOR_VARIANT="$1"
DRIVOR_EVAL_CKPT="$2"
shift 2
case "${DRIVOR_VARIANT}" in
    bank|vjepa2_vitl|vjepa2_1_vitl|vjepa2_world|vjepa2_world_multigt) ;;
    *) printf '未知模型版本：%s\n' "${DRIVOR_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-eval_navtest_drivoR_${DRIVOR_VARIANT}}"
export DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}"
export DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}"
DRIVOR_EVAL_ARGS=()
if [[ "${DRIVOR_VARIANT}" == vjepa2_world_multigt ]]; then
    DRIVOR_EVAL_ARGS+=(agent.config.multi_gt.enabled=false)
fi
exec bash "${DRIVOR_EVAL_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_EVAL_CKPT}" \
    "agent=drivoR_${DRIVOR_VARIANT}" \
    trainer.params.precision=bf16-mixed dataloader.params.batch_size=1 \
    "${DRIVOR_EVAL_ARGS[@]}" "$@"
