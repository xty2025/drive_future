#!/usr/bin/env bash
set -euo pipefail

# 可视化 run_drivor_action_dit_dinov3_vitl_pseudo_expert_training.sh 使用的 CLOVER pseudo expert：
# 候选池的分布（按 PDMS 着色，低于 0.8 的灰色、不参与选择），以及按训练时的策略选出的轨迹：
# PDMS ≥ 0.8 → 按分数排序取最好的一条 → 最远点采样到 8 条；训练时没有 expert 靠近人类轨迹且有空位时再加入人类轨迹。
# 同时画出同一训练的 top4 多 GT 作对比，每条轨迹算官方 PDMS。
# 第一次运行读取 4 GB 的 CLOVER 源文件（几分钟），统计全部场景并抽取随机场景池；只用 CPU。
# 用法（任意目录执行）：
#   bash DrivoR/scripts/viz/run_visualize_pseudo_experts.sh [--pool 400] [--per-group 2] [--tokens t1 t2 ...]
# 只重画：... --reuse DrivoR/exp/viz/pseudo_experts_<时间戳>
# 输出：DrivoR/exp/viz/pseudo_experts_<时间戳>/overview.png、summary.md、每个场景一张 png、index.html。

DRIVOR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_VIZ_SCRIPT=visualize_pseudo_experts.py exec bash "${DRIVOR_SCRIPT_DIR}/run_visualize_proposals_bev.sh" "$@"
