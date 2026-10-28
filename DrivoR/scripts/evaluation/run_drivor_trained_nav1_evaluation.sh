#!/usr/bin/env bash
set -euo pipefail

# 在 navtest 上评测 run_drivor_training.sh 训练得到的 DrivoR NAVSIM-v1 权重（PDMS）。
# 默认权重为 training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode 的最后一个 epoch：
#   exp/ke/training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/lightning_logs/version_2/checkpoints/epoch=24-step=33250.ckpt
# 训练只用 train logs（85k）、25 epochs，与官方 train85k 权重（README 93.1）设置相同，可直接对比。
# 用法（任意目录执行）：
#   bash DrivoR/scripts/evaluation/run_drivor_trained_nav1_evaluation.sh [Hydra 参数...]
# 换成同一训练的其他权重，例如验证集最优的 epoch：
#   DRIVOR_CHECKPOINT=DrivoR/exp/ke/training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/lightning_logs/version_2/checkpoints/best-epoch=23-step=31920.ckpt \
#     bash DrivoR/scripts/evaluation/run_drivor_trained_nav1_evaluation.sh
# 环境变量和额外 Hydra 参数与 run_drivor_pdm_score_evaluation.sh 相同。

DRIVOR_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd -- "${DRIVOR_EVAL_DIR}/../.." && pwd)"
DRIVOR_TRAIN_RUN_DIR="${DRIVOR_ROOT}/exp/ke/training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode"
DRIVOR_CHECKPOINT="${DRIVOR_CHECKPOINT:-${DRIVOR_TRAIN_RUN_DIR}/lightning_logs/version_2/checkpoints/epoch=24-step=33250.ckpt}"

exec bash "${DRIVOR_EVAL_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_CHECKPOINT}" "$@"
