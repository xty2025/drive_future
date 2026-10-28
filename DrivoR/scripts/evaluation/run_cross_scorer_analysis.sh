#!/usr/bin/env bash
set -euo pipefail

# 用 DrivoR（单 GT / multi-GT）的打分器给 4 个 Action-DiT 权重的 64 条候选排序，与各自自带打分器在同一组候选上对比，
# 输出选中轨迹的官方 PDMS、各子指标和 64 条最优。默认 navtest 随机 2000 个场景；--max-scenes 0 为全量。
# 每张可见 GPU 跑一个分片（CUDA_VISIBLE_DEVICES，默认 0,1），全部结束后合并、逐条打分并写汇总。
# 用法（任意目录执行）：
#   bash DrivoR/scripts/evaluation/run_cross_scorer_analysis.sh [--max-scenes 2000] [--generators front ...] [--scorers drivor ...]
# 已有分片时只重做合并和汇总：... --report --output-dir <目录>
# 输出：DrivoR/exp/analysis/cross_scorer_<时间戳>/summary.md、summary.json、pdm.npz、shard_*.npz/log。

DRIVOR_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd -- "${DRIVOR_EVAL_DIR}/../.." && pwd)"
DRIVOR_OUTPUT_DIR="${DRIVOR_ROOT}/exp/analysis/cross_scorer_$(date +%Y%m%d_%H%M%S)"
DRIVOR_ARGS=()
DRIVOR_REPORT_ONLY=false
while (( $# )); do
    case "$1" in
        --output-dir) DRIVOR_OUTPUT_DIR="$(realpath -m -- "$2")"; shift 2 ;;
        --report) DRIVOR_REPORT_ONLY=true; shift ;;
        *) DRIVOR_ARGS+=("$1"); shift ;;
    esac
done
run() { DRIVOR_ANALYSIS_SCRIPT=analyze_cross_scorer.py bash "${DRIVOR_EVAL_DIR}/run_scorer_oracle_gap_analysis.sh" \
            --output-dir "${DRIVOR_OUTPUT_DIR}" "${DRIVOR_ARGS[@]}" "$@"; }
mkdir -p -- "${DRIVOR_OUTPUT_DIR}"

if [[ "${DRIVOR_REPORT_ONLY}" == false ]]; then
    IFS=',' read -r -a DRIVOR_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
    DRIVOR_PIDS=()
    for i in "${!DRIVOR_GPU_IDS[@]}"; do
        CUDA_VISIBLE_DEVICES="${DRIVOR_GPU_IDS[$i]}" run --shard "$i" --num-shards "${#DRIVOR_GPU_IDS[@]}" \
            > "${DRIVOR_OUTPUT_DIR}/shard_${i}.log" 2>&1 &
        DRIVOR_PIDS+=($!)
    done
    printf '分片日志：%s/shard_*.log\n' "${DRIVOR_OUTPUT_DIR}" >&2
    for i in "${!DRIVOR_PIDS[@]}"; do
        if ! wait "${DRIVOR_PIDS[$i]}"; then
            printf '分片 %s 失败，见 %s/shard_%s.log\n' "$i" "${DRIVOR_OUTPUT_DIR}" "$i" >&2
            exit 1
        fi
    done
fi
CUDA_VISIBLE_DEVICES="" run --report
