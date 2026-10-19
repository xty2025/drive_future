#!/usr/bin/env bash
set -euo pipefail

# One DLC command: prepare missing frozen targets (groups3/4), train, then
# evaluate on the head node. Resources: 8 nodes x 8 GPUs, global batch64.
DRIVOR_VARIANT="${1:?Specify bank, vjepa2_vitl, vjepa2_1_vitl, vjepa2_world or vjepa2_world_multigt}"
shift
case "${DRIVOR_VARIANT}" in
    bank|vjepa2_vitl|vjepa2_1_vitl|vjepa2_world|vjepa2_world_multigt) ;;
    *) printf '未知模型版本：%s\n' "${DRIVOR_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_PIPELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd -- "${DRIVOR_PIPELINE_DIR}/../.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_ROOT}/exp}"
export DRIVOR_WORLD_DATA_ROOT="${DRIVOR_WORLD_DATA_ROOT:-${NAVSIM_EXP_ROOT}/world4scorer_data}"
export DRIVOR_BANK_PACK="${DRIVOR_BANK_PACK:-${DRIVOR_WORLD_DATA_ROOT}/bank_clover}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${DRIVOR_VARIANT}_seed2_train_eval_run1}"
export DRIVOR_PRECISION="${DRIVOR_PRECISION:-bf16-mixed}"
DRIVOR_TRAIN_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_${DRIVOR_VARIANT}_navtrain}"

for DRIVOR_ARG in "$@"; do
    case "${DRIVOR_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            DRIVOR_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT}" \
            exec bash "${DRIVOR_PIPELINE_DIR}/run_drivor_${DRIVOR_VARIANT}_training.sh" "$@" ;;
    esac
done

if [[ "${DRIVOR_VARIANT}" == vjepa2_world || "${DRIVOR_VARIANT}" == vjepa2_world_multigt ]]; then
    if [[ "${DRIVOR_VARIANT}" == vjepa2_world ]]; then
        DRIVOR_FUTURE_MODE=real
    else
        DRIVOR_FUTURE_MODE=cosmos
    fi
    export DRIVOR_FUTURE_FEATURES="${DRIVOR_FUTURE_FEATURES:-${DRIVOR_WORLD_DATA_ROOT}/future_${DRIVOR_FUTURE_MODE}_vjepa21}"
    if [[ "${DRIVOR_PREPARE_FUTURE:-true}" == true ]]; then
        bash "${DRIVOR_PIPELINE_DIR}/run_drivor_world_data_preparation.sh" "${DRIVOR_FUTURE_MODE}" \
            --output "${DRIVOR_FUTURE_FEATURES}"
    fi
fi

DRIVOR_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT}" \
bash "${DRIVOR_PIPELINE_DIR}/run_drivor_${DRIVOR_VARIANT}_training.sh" \
    auto_resume=false train_ckpt_path=null runtime.require_resume=false \
    runtime.console_log_interval=25 "$@"

if [[ "${NODE_RANK:-${RANK:-0}}" == 0 ]]; then
    DRIVOR_RUN_DIR="${NAVSIM_EXP_ROOT}/ke/${DRIVOR_TRAIN_EXPERIMENT}/${DRIVOR_RUN_ID}"
    DRIVOR_LAST_CKPT="$(/usr/bin/python3 - "${DRIVOR_RUN_DIR}" <<'PY'
from pathlib import Path
import sys
files = list(Path(sys.argv[1]).glob('lightning_logs/version_*/checkpoints/last.ckpt'))
if not files:
    raise FileNotFoundError('Training completed without a last.ckpt in ' + sys.argv[1])
print(max(files, key=lambda path: path.stat().st_mtime_ns))
PY
    )"
    DRIVOR_EXPERIMENT="eval_navtest_drivoR_${DRIVOR_VARIANT}" NPROC_PER_NODE=8 \
    bash "${DRIVOR_ROOT}/scripts/evaluation/run_drivor_world_evaluation.sh" \
        "${DRIVOR_VARIANT}" "${DRIVOR_LAST_CKPT}"
fi
