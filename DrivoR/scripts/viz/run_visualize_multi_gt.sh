#!/usr/bin/env bash
set -euo pipefail

# 可视化 NAVSIM 多 GT（top4 高分 GT 与 poor4 负样本）：每个场景两套 GT 画在同一视野的 BEV 上，
# 每条 GT 用官方 NAVSIM-v1 PDMS 打分（含各子指标、碰撞/TTC/驶出可行驶区域的位置），并与数据自带的分数对比。
# 先随机打分一个场景池（默认 300 个）写 summary.md，再按左转/直行/右转挑场景画图（默认 9 个）。只用 CPU。
# 用法（任意目录执行）：
#   bash DrivoR/scripts/viz/run_visualize_multi_gt.sh [--scenes 9] [--pool 300] [--tokens t1 t2 ...]
# 输出：DrivoR/exp/viz/multi_gt_<时间戳>/*.png、summary.md、index.html。

DRIVOR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VISIBLE_DEVICES="" DRIVOR_VIZ_SCRIPT=visualize_multi_gt.py exec bash "${DRIVOR_SCRIPT_DIR}/run_visualize_proposals_bev.sh" "$@"
