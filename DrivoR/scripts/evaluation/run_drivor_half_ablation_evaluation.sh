#!/usr/bin/env bash
set -euo pipefail
DRIVOR_HALF_VARIANT="${1:?Specify the half-data experiment}"
DRIVOR_HALF_CHECKPOINT="${2:?Specify a checkpoint from this experiment}"
shift 2
DRIVOR_HALF_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DRIVOR_HALF_EVAL_DIR}/../training/drivor_half_ablation_common.sh"
if [[ -d "${DRIVOR_HALF_CHECKPOINT}" ]]; then
    DRIVOR_HALF_CHECKPOINT="$("${DRIVOR_PYTHON}" "${DRIVOR_HALF_ROOT}/tools/prepare_half_eval_checkpoint.py" \
        --checkpoint "${DRIVOR_HALF_CHECKPOINT}")"
fi
export DRIVOR_EXPERIMENT="${DRIVOR_HALF_EVAL_EXP}"
export DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}"
export DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}"
DRIVOR_HALF_EVAL_ARGS=()
for DRIVOR_HALF_ARG in "$@"; do
    DRIVOR_HALF_KEY="${DRIVOR_HALF_ARG%%=*}"
    DRIVOR_HALF_KEY="${DRIVOR_HALF_KEY#+}"
    DRIVOR_HALF_KEY="${DRIVOR_HALF_KEY#+}"
    case "${DRIVOR_HALF_KEY}" in
        agent|agent.config.action_dit.pretrained_path|trainer.params.precision) continue ;;
    esac
    DRIVOR_HALF_EVAL_ARGS+=("${DRIVOR_HALF_ARG}")
done
exec bash "${DRIVOR_HALF_EVAL_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_HALF_CHECKPOINT}" \
    "agent=${DRIVOR_HALF_AGENT}" \
    agent.config.action_dit.pretrained_path=null \
    trainer.params.precision=32-true dataloader.params.batch_size=1 "${DRIVOR_HALF_EVAL_ARGS[@]}"
