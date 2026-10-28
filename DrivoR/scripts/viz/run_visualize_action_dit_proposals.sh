#!/usr/bin/env bash
set -euo pipefail

# 在 BEV 上对比 DrivoR（单 GT / multi-GT）和 4 个 Action-DiT 权重（单前视 / 单前视 multi-GT / multiview /
# multiview multi-GT），均为 epoch=24：每个场景 64 条候选的分布、打分器选中的轨迹和真实最优轨迹，以及两者的各子指标。
# 推理配置与各自的 navtest 评测相同（DrivoR FP32，确定性；Action-DiT BF16、10 步 Euler）。
# Action-DiT 每个场景用同一份固定的初始噪声（与评测时的噪声不同）。
# navtest 和训练集各先跑一个场景池（默认 160 个）并逐条打分，再按 5 组各挑 2 个场景画图：
# DrivoR 好而 Action-DiT 多数失败 / Action-DiT 好而 DrivoR 失败 / 打分器漏掉好候选 / 某模型 64 条里没有好轨迹 / 全部都好。
# 用法（任意目录执行，单卡即可）：
#   bash DrivoR/scripts/viz/run_visualize_action_dit_proposals.sh [--models drivor front ...] [--per-group 2] [--test-tokens t1 ...]
# 只重画已有结果：... --reuse DrivoR/exp/viz/action_dit_proposals_<时间戳>
# 全部参数见：bash DrivoR/scripts/viz/run_visualize_action_dit_proposals.sh --help
# 输出：DrivoR/exp/viz/action_dit_proposals_<时间戳>/{test,train}/*.png、overview_*.png、summary.md、index.html。

DRIVOR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_VIZ_SCRIPT=visualize_action_dit_proposals.py exec bash "${DRIVOR_SCRIPT_DIR}/run_visualize_proposals_bev.sh" "$@"
