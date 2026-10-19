#!/usr/bin/env bash
set -euo pipefail

# 在 navtest 上评测 run_drivor_multigt_training.sh 训练得到的多 GT 监督 DrivoR 权重（PDMS）。
# 默认权重为 training_drivoR_Nav1_navtrain_multigt/drivor_multigt_seed2_run1 的最后一个 epoch：
#   exp/ke/training_drivoR_Nav1_navtrain_multigt/drivor_multigt_seed2_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
# 多 GT 只替换训练时的轨迹 loss，网络结构和推理与原版相同，所以按原版 agent=drivoR 评测。
# drivoR_multigt 的多 GT 目标只在训练时读取（navtest 没有多 GT 文件），评测打分也不用 targets。
# 训练同样只用 train logs（85k）、25 epochs，可与官方 train85k（93.13）及单 GT 训练的权重对比。
# 用法（任意目录执行）：
#   bash DrivoR/scripts/evaluation/run_drivor_multigt_nav1_evaluation.sh [Hydra 参数...]
# 换成同一训练的其他权重，例如验证集最优的 epoch：
#   DRIVOR_CHECKPOINT=DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt/drivor_multigt_seed2_run1/lightning_logs/version_0/checkpoints/best-epoch=17-step=23940.ckpt \
#     bash DrivoR/scripts/evaluation/run_drivor_multigt_nav1_evaluation.sh
# 环境变量和额外 Hydra 参数与 run_drivor_pdm_score_evaluation.sh 相同。

DRIVOR_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd -- "${DRIVOR_EVAL_DIR}/../.." && pwd)"
DRIVOR_TRAIN_RUN_DIR="${DRIVOR_ROOT}/exp/ke/training_drivoR_Nav1_navtrain_multigt/drivor_multigt_seed2_run1"
DRIVOR_CHECKPOINT="${DRIVOR_CHECKPOINT:-${DRIVOR_TRAIN_RUN_DIR}/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt}"

exec bash "${DRIVOR_EVAL_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_CHECKPOINT}" "$@"
