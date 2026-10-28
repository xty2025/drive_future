#!/usr/bin/env bash
set -euo pipefail
DRIVOR_HALF_VARIANT="${1:?Specify a23, a27, a28, or an a28_* ablation}"
shift
DRIVOR_HALF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DRIVOR_HALF_DIR}/drivor_half_ablation_common.sh"
DRIVOR_HALF_EVAL_ARGS=()
for DRIVOR_HALF_ARG in "$@"; do
    case "${DRIVOR_HALF_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            exec bash "${DRIVOR_HALF_DIR}/run_drivor_half_ablation_training.sh" "${DRIVOR_HALF_VARIANT}" "$@" ;;
    esac
    DRIVOR_HALF_KEY="${DRIVOR_HALF_ARG#+}"
    DRIVOR_HALF_KEY="${DRIVOR_HALF_KEY#+}"
    case "${DRIVOR_HALF_KEY}" in
        agent.config.*) DRIVOR_HALF_EVAL_ARGS+=("${DRIVOR_HALF_ARG}") ;;
    esac
done
bash "${DRIVOR_HALF_DIR}/run_drivor_half_ablation_training.sh" "${DRIVOR_HALF_VARIANT}" \
    auto_resume=false train_ckpt_path=null runtime.require_resume=false "$@"

if [[ "${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}" == 0 ]]; then
    DRIVOR_HALF_CHECKPOINT="$("${DRIVOR_PYTHON}" "${DRIVOR_HALF_ROOT}/tools/prepare_half_eval_checkpoint.py" \
        --run-dir "${NAVSIM_EXP_ROOT}/ke/${DRIVOR_HALF_TRAIN_EXP}/${DRIVOR_RUN_ID}")"
    NPROC_PER_NODE="${DRIVOR_EVAL_GPUS:-${NPROC_PER_NODE:-${GPU_NUM:-8}}}" \
    bash "${DRIVOR_HALF_ROOT}/scripts/evaluation/run_drivor_half_ablation_evaluation.sh" \
        "${DRIVOR_HALF_VARIANT}" "${DRIVOR_HALF_CHECKPOINT}" "${DRIVOR_HALF_EVAL_ARGS[@]}"
fi
