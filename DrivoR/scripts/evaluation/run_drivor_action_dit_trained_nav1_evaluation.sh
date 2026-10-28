#!/usr/bin/env bash
set -euo pipefail

# 单节点 navtest 评测：Action-DiT 单前视原生 patches，64 proposals，10 步 Euler。
# 默认加载用户指定的完整训练 checkpoint（version_3 / epoch 24 / step 33250）。
# 单节点 DLC 分配 8 张 GPU 后，直接 bash 本脚本；内部使用 torchrun --standalone。
# DRIVOR_CHECKPOINT 可覆盖为同一模型/视觉结构的其他完整训练 checkpoint。
# DRIVOR_EVAL_BATCH_SIZE / DRIVOR_EVAL_PRECISION 可覆盖默认每卡 batch 1 / BF16。
# 其余参数透传 Hydra；例：train_test_split.scene_filter.max_scenes=64。

DRIVOR_ACTION_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ACTION_ROOT="$(cd -- "${DRIVOR_ACTION_EVAL_DIR}/../.." && pwd)"
DRIVOR_CHECKPOINT="${DRIVOR_CHECKPOINT:-${DRIVOR_ACTION_ROOT}/exp/ke/training_drivoR_action_dit_navtrain/multinode/lightning_logs/version_3/checkpoints/epoch=24-step=33250.ckpt}"

export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-eval_navtest_drivor_action_dit_navtrain_v3_epoch24}"
export DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}"
export DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}"

# 完整 checkpoint 会严格恢复全部网络权重，无需先加载 Action-DiT 初始化文件。
# 评测只预测轨迹，不构造训练用的额外长 GT。
exec bash "${DRIVOR_ACTION_EVAL_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_CHECKPOINT}" \
    agent=drivoR_action_dit \
    agent.config.action_dit.pretrained_path=null \
    agent.config.long_trajectory_additional_poses=0 \
    agent.config.action_dit.num_inference_steps=10 \
    "trainer.params.precision=${DRIVOR_EVAL_PRECISION:-bf16-mixed}" \
    "dataloader.params.batch_size=${DRIVOR_EVAL_BATCH_SIZE:-1}" \
    "dataloader.params.num_workers=${DRIVOR_NUM_WORKERS:-4}" \
    "$@"
