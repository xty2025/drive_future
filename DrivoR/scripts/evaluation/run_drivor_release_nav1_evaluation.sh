#!/usr/bin/env bash
set -euo pipefail

# 依次评测 DrivoR 官方发布的 NAVSIM-v1 权重（navtest PDMS），参数与 README 一致：
#   train85k      weights/releases/model_weights/drivor_Nav1_train_only_85k_25epochs.ckpt  README：93.1
#   trainval103k  weights/releases/model_weights/drivor_Nav1_25epochs.pth                  README：93.7
# 用法（任意目录执行，默认 all）：
#   bash DrivoR/scripts/evaluation/run_drivor_release_nav1_evaluation.sh [all|train85k|trainval103k] [Hydra 参数...]
# 环境变量和额外 Hydra 参数与 run_drivor_pdm_score_evaluation.sh 相同，例如：
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash DrivoR/scripts/evaluation/run_drivor_release_nav1_evaluation.sh train85k

DRIVOR_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd -- "${DRIVOR_EVAL_DIR}/../.." && pwd)"
DRIVOR_WEIGHTS_DIR="${DRIVOR_ROOT}/weights/releases/model_weights"

DRIVOR_SELECTION=all
if (( $# > 0 )) && [[ "$1" != -* && "$1" != *=* ]]; then
    DRIVOR_SELECTION="$1"
    shift
fi
case "${DRIVOR_SELECTION}" in
    all) DRIVOR_CHECKPOINTS=(drivor_Nav1_train_only_85k_25epochs.ckpt drivor_Nav1_25epochs.pth) ;;
    train85k) DRIVOR_CHECKPOINTS=(drivor_Nav1_train_only_85k_25epochs.ckpt) ;;
    trainval103k) DRIVOR_CHECKPOINTS=(drivor_Nav1_25epochs.pth) ;;
    *)
        printf '未知选项：%s（可选 all / train85k / trainval103k）\n' "${DRIVOR_SELECTION}" >&2
        exit 2 ;;
esac

# 与 run_drivor_pdm_score_evaluation.sh 的默认值一致，用于汇总各次的 pdms_summary.txt。
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_ROOT}/exp}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

DRIVOR_SUMMARIES=()
for DRIVOR_NAME in "${DRIVOR_CHECKPOINTS[@]}"; do
    # 每个权重使用独立实验名，避免沿用外部 DRIVOR_EXPERIMENT 导致结果写进同一目录。
    DRIVOR_EXPERIMENT="eval_navtest_${DRIVOR_NAME%.*}"
    DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT}" bash "${DRIVOR_EVAL_DIR}/run_drivor_pdm_score_evaluation.sh" \
        "${DRIVOR_WEIGHTS_DIR}/${DRIVOR_NAME}" "$@"
    DRIVOR_SUMMARY="${NAVSIM_EXP_ROOT}/ke/${DRIVOR_EXPERIMENT}/${DRIVOR_RUN_ID}/pdms_summary.txt"
    if [[ -f "${DRIVOR_SUMMARY}" ]]; then
        DRIVOR_SUMMARIES+=("$(cat -- "${DRIVOR_SUMMARY}")")
    fi
done

if (( ${#DRIVOR_SUMMARIES[@]} > 0 )); then
    printf '\n===== navtest PDMS（README：train85k 93.1，trainval103k 93.7）=====\n'
    printf '%s\n' "${DRIVOR_SUMMARIES[@]}"
fi
