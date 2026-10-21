#!/usr/bin/env bash
set -euo pipefail

# DLC: train a loss/GT variant on all nodes, then evaluate on node 0 in FP32.
# Usage: bash run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh <variant> [training overrides]
DRIVOR_AUX_VARIANT="${1:?Specify bank, pseudo_expert, pseudo_expert_vmse, multigt_poor, single_gt, single_gt_long, human_diverse8_l1 or human_diverse8_vmse}"
shift
case "${DRIVOR_AUX_VARIANT}" in
    bank|pseudo_expert|pseudo_expert_vmse) DRIVOR_AUX_SUFFIX="multigt_${DRIVOR_AUX_VARIANT}" ;;
    multigt_poor|single_gt|single_gt_long|human_diverse8_l1|human_diverse8_vmse) DRIVOR_AUX_SUFFIX="${DRIVOR_AUX_VARIANT}" ;;
    *) printf 'Unknown auxiliary variant: %s\n' "${DRIVOR_AUX_VARIANT}" >&2; exit 2 ;;
esac

DRIVOR_AUX_PIPELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_AUX_ROOT="$(cd -- "${DRIVOR_AUX_PIPELINE_DIR}/../.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_AUX_ROOT}/exp}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-action_dit_dinov3_vitl_${DRIVOR_AUX_SUFFIX}_seed2_train_eval_run1}}"
export DRIVOR_PRECISION="${DRIVOR_PRECISION:-bf16-mixed}"
export DRIVOR_EVAL_PRECISION="${DRIVOR_EVAL_PRECISION:-32-true}"
export DRIVOR_SCENE_QUERIES="${DRIVOR_SCENE_QUERIES:-0}"
DRIVOR_AUX_TRAIN_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_${DRIVOR_AUX_SUFFIX}}"
DRIVOR_AUX_RUN_DIR="${NAVSIM_EXP_ROOT}/ke/${DRIVOR_AUX_TRAIN_EXPERIMENT}/${DRIVOR_RUN_ID}"
DRIVOR_AUX_TRAIN_SCRIPT="${DRIVOR_AUX_PIPELINE_DIR}/run_drivor_action_dit_dinov3_vitl_${DRIVOR_AUX_VARIANT}_training.sh"

for DRIVOR_AUX_ARG in "$@"; do
    case "${DRIVOR_AUX_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            DRIVOR_EXPERIMENT="${DRIVOR_AUX_TRAIN_EXPERIMENT}" \
            exec bash "${DRIVOR_AUX_TRAIN_SCRIPT}" "$@" ;;
    esac
done

DRIVOR_EXPERIMENT="${DRIVOR_AUX_TRAIN_EXPERIMENT}" \
bash "${DRIVOR_AUX_TRAIN_SCRIPT}" \
    auto_resume=false train_ckpt_path=null runtime.require_resume=false \
    runtime.console_log_interval=25 "$@"

if [[ "${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}" == 0 ]]; then
    DRIVOR_AUX_LAST_CKPT="$(/usr/bin/python3 - "${DRIVOR_AUX_RUN_DIR}" <<'PY'
from pathlib import Path
import sys
checkpoints = list(Path(sys.argv[1]).glob('lightning_logs/version_*/checkpoints/last.ckpt'))
if not checkpoints:
    raise FileNotFoundError('Training completed without last.ckpt in ' + sys.argv[1])
print(max(checkpoints, key=lambda path: path.stat().st_mtime_ns))
PY
    )"
    DRIVOR_AUX_VISUAL_MODE=native_patches
    if (( DRIVOR_SCENE_QUERIES > 0 )); then
        DRIVOR_AUX_VISUAL_MODE=drivor_queries
    fi
    DRIVOR_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_action_dit_dinov3_vitl_${DRIVOR_AUX_SUFFIX}}" \
    DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}" \
    DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}" \
    NPROC_PER_NODE="${DRIVOR_EVAL_GPUS:-${NPROC_PER_NODE:-${GPU_NUM:-8}}}" \
    bash "${DRIVOR_AUX_ROOT}/scripts/evaluation/run_drivor_pdm_score_evaluation.sh" \
        "${DRIVOR_AUX_LAST_CKPT}" \
        agent=drivoR_action_dit_dinov3_vitl \
        agent.config.multi_gt.enabled=false \
        agent.config.action_dit.pretrained_path=null \
        agent.config.action_dit.num_inference_steps=10 \
        "agent.config.visual_feature_mode=${DRIVOR_AUX_VISUAL_MODE}" \
        "agent.config.num_scene_tokens=${DRIVOR_SCENE_QUERIES}" \
        "trainer.params.precision=${DRIVOR_EVAL_PRECISION}" \
        dataloader.params.batch_size=1
fi

# DLC_POST_RUN_RETENTION_20261020: opt-in only for the four verified running jobs.
case "${DRIVOR_RUN_ID:-}" in
    action_dit_human_diverse8_l1_seed2_dlc_20261020_095054|action_dit_human_diverse8_vmse_seed2_dlc_20261020_095054|dinov3_register_scorer_single_gt_seed2_train_eval_run1|action_dit_pe_vmse_register_scorer_seed2_train_eval_run1)
        /usr/bin/python3 "/cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_retention.py" \
            --control "/cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/retention_probe_20261020_210605/control.json"
        ;;
esac

# DLC_QUERY_GT_CHAIN_chain_query_gt_20261021_164710: append only, exact predecessor RunIDs.
case "${DRIVOR_RUN_ID:-}" in
    action_dit_dinov3_vitl_single_gt_long_pseudo_expert_seed2_dlc8hwmx72tdbjoc_chain1)
        /usr/bin/python3 /cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_once.py --control /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/chain_query_gt_20261021_164710/control.json
        ;;
esac
