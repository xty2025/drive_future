#!/usr/bin/env bash
set -euo pipefail

# Independent register experiments, each followed by head-node FP32 navtest.
DRIVOR_REGISTER_VARIANT="${1:?Specify register, register_scorer, register_scorer_single_gt, register_staged, pseudo_expert_vmse_register or pseudo_expert_vmse_register_scorer}"
shift
case "${DRIVOR_REGISTER_VARIANT}" in
    register|register_scorer|register_scorer_single_gt|register_staged) DRIVOR_REGISTER_EXP_SUFFIX="${DRIVOR_REGISTER_VARIANT}" ;;
    pseudo_expert_vmse_register|pseudo_expert_vmse_register_scorer)
        DRIVOR_REGISTER_EXP_SUFFIX="multigt_${DRIVOR_REGISTER_VARIANT}" ;;
    *) printf 'Unknown register variant: %s\n' "${DRIVOR_REGISTER_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_REGISTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_REGISTER_ROOT="$(cd -- "${DRIVOR_REGISTER_DIR}/../.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_REGISTER_ROOT}/exp}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-action_dit_dinov3_vitl_${DRIVOR_REGISTER_VARIANT}_seed2_train_eval_run1}}"
export DRIVOR_PRECISION="${DRIVOR_PRECISION:-bf16-mixed}"
export DRIVOR_EVAL_PRECISION="${DRIVOR_EVAL_PRECISION:-32-true}"
export DRIVOR_TRAIN_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_${DRIVOR_REGISTER_EXP_SUFFIX}}"
DRIVOR_REGISTER_FINAL_EXP="${DRIVOR_TRAIN_EXPERIMENT}"
if [[ "${DRIVOR_REGISTER_VARIANT}" == register_staged ]]; then
    DRIVOR_REGISTER_FINAL_EXP="${DRIVOR_TRAIN_EXPERIMENT}_stage2"
fi
DRIVOR_REGISTER_SCRIPT="${DRIVOR_REGISTER_DIR}/run_drivor_action_dit_dinov3_vitl_${DRIVOR_REGISTER_VARIANT}_training.sh"
for DRIVOR_REGISTER_ARG in "$@"; do
    case "${DRIVOR_REGISTER_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            DRIVOR_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT}" exec bash "${DRIVOR_REGISTER_SCRIPT}" "$@" ;;
    esac
done
DRIVOR_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT}" \
bash "${DRIVOR_REGISTER_SCRIPT}" \
    auto_resume=false train_ckpt_path=null runtime.require_resume=false runtime.console_log_interval=25 "$@"

if [[ "${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}" == 0 ]]; then
    DRIVOR_REGISTER_CKPT="$(/usr/bin/python3 - "${NAVSIM_EXP_ROOT}/ke/${DRIVOR_REGISTER_FINAL_EXP}/${DRIVOR_RUN_ID}" <<'PY'
from pathlib import Path
import sys
files=list(Path(sys.argv[1]).glob('lightning_logs/version_*/checkpoints/last.ckpt'))
if not files:
    raise FileNotFoundError('Training completed without last.ckpt in '+sys.argv[1])
print(max(files,key=lambda value:value.stat().st_mtime_ns))
PY
    )"
    DRIVOR_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_action_dit_dinov3_vitl_${DRIVOR_REGISTER_VARIANT}}" \
    NPROC_PER_NODE="${DRIVOR_EVAL_GPUS:-${NPROC_PER_NODE:-${GPU_NUM:-8}}}" \
    bash "${DRIVOR_REGISTER_ROOT}/scripts/evaluation/run_drivor_action_dit_dinov3_vitl_register_evaluation.sh" \
        "${DRIVOR_REGISTER_VARIANT}" "${DRIVOR_REGISTER_CKPT}"
fi
