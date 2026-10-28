"""Visualize the CLOVER pseudo experts used by run_drivor_action_dit_dinov3_vitl_pseudo_expert_training.sh.

That training (agent drivoR_action_dit_dinov3_vitl_pseudo_expert) adds a coverage loss to the multi-GT Action-DiT
objective: every pseudo expert pulls the nearest of the 64 one-step X-predictions (L1 over x, y, heading; weight 0.5).
The experts are picked from CLOVER's rule-generated, PDM-scored candidates of each navtrain scene
(dataset_decoupled_v2_clean.pkl, ~57 per scene, each with its generation family) by navsim/agents/drivoR_pseudo_expert:
  offline, selection.select_candidates (tools/build_clover_pseudo_experts.py -> exp/clover_data/pseudo_experts_k8_thr0p8):
    keep PDMS >= 0.8, sort by PDMS, take the best, then farthest-point sampling on d = mean |dx|, |dy| over the 8
    poses until 8 are chosen or every remaining candidate is within d < 0.05 of the chosen ones;
  online, targets.PseudoExpertTargetBuilder / selection.include_human_expert: the human trajectory is added when no
    expert is within d < 0.5 of it and a slot is free; without an eligible candidate it is the only target.

Stage 1 reads the 4-GB source once: statistics of all scenes (pool size, eligible count, the selected experts with
PDMS, family and FPS distance; every scene is re-selected and compared with the training pack) and the full candidate
pools of a random set of navtrain scenes from a few training logs. Stage 2 adds the human trajectory, the top4
multi-GT targets (the v-loss targets of the same training) and the official NAVSIM-v1 PDMS of every candidate, human
and top4 trajectory. Output: overview.png and summary.md (distributions) and one figure per drawn scene: the whole
pool on the BEV coloured by PDMS (grey: below 0.8, not eligible) next to the training targets it yields, PDMS against
distance to the human trajectory, x(t) and y(t). --reuse <dir> re-renders from the saved stage files.
"""

import argparse
import gc
import html
import json
import lzma
import multiprocessing
import os
import pickle
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import visualize_multi_gt as vmg
import visualize_proposals_bev as vpb
from navsim.agents.drivoR.proposal_analysis import (
    SPLITS, front_camera_config, log_of, make_scene_loader, sample_navtrain_tokens)
from navsim.agents.drivoR_pseudo_expert.selection import include_human_expert
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pairwise_pdm_score import PDM_FIELDS, pairwise_pdm_details, pairwise_pdm_scores

from visualize_proposals_bev import (  # palette shared with the other trajectory figures
    AGENT_FILL, AXIS, BLUE_STEPS, CRITICAL, DRIVABLE, GRID, INK, INK_2, LANE_EDGE, MUTED, SURFACE)

sys.path.insert(0, str(vpb.DRIVOR_ROOT / "tools"))
from build_world4scorer_bank import read_scenes  # noqa: E402  the CLOVER reader of both pack builders

EXP = vpb.DRIVOR_ROOT / "exp"
SCORE = PDM_FIELDS.index("score")
SUB = vmg.SUB
# Categorical slots 1, 2 and 7 of the reference palette (validated together, all pairs); the eligible candidates use
# the blue sequential ramp from step 250 when coloured by PDMS.
ELIGIBLE_COLOR, EXPERT_COLOR, TOP4_COLOR = "#2a78d6", "#eb6834", "#4a3aa7"
ELIGIBLE_CMAP = mcolors.LinearSegmentedColormap.from_list("eligible", BLUE_STEPS[3:])
GROUPS = {"dropped": "human not a target: 8 experts, none near it",
          "covered": "human covered by an expert",
          "appended": "human appended: fewer than 8 experts",
          "fallback": "fallback: no candidate reaches 0.8"}
FAMILY_SHORT = {"const_speed_portion": "const speed", "boundary_interp": "boundary interp", "stop_go": "stop-go",
                "accel_profile": "accel profile", "approach_brake": "approach-brake", "offroad": "offroad",
                "overshoot": "overshoot"}

_SIMULATOR = None
_SCORER = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=EXP / "world4scorer_data/source/dataset_decoupled_v2_clean.pkl")
    parser.add_argument("--pack", type=Path, default=EXP / "clover_data/pseudo_experts_k8_thr0p8",
                        help="pseudo-expert pack read by the training (manifest gives top_k, threshold, distances)")
    parser.add_argument("--top4-root", type=Path, default=vmg.DATA / "navtrain_multigt_top4_dense_0p1s_f64_8gpu")
    parser.add_argument("--metric-cache", type=Path, default=vpb.PROJECT_ROOT / "data/metric_cache_navtrain")
    parser.add_argument("--train-logs", type=int, default=12, help="random training logs of the scene pool")
    parser.add_argument("--pool", type=int, default=400, help="max pool scenes (human trajectory, official PDMS)")
    parser.add_argument("--per-group", type=int, default=2, help="drawn scenes per human-rule group (4 groups)")
    parser.add_argument("--tokens", nargs="*", help="draw these navtrain tokens instead of picking from the pool")
    parser.add_argument("--output-dir", type=Path, help="default: exp/viz/pseudo_experts_<time>, or the --reuse dir")
    parser.add_argument("--reuse", type=Path, help="re-render an earlier run from its stage files")
    parser.add_argument("--score-workers", type=int, default=min(20, len(os.sched_getaffinity(0))))
    parser.add_argument("--seed", type=int, default=0)
    args = vpb.resolve_caller_paths(parser.parse_args())
    if args.output_dir is None:
        args.output_dir = args.reuse or vpb.DRIVOR_ROOT / "exp/viz" / f"pseudo_experts_{datetime.now():%Y%m%d_%H%M%S}"
    return args


# ----------------------------------------------------------------------------- selection (as in training)

def mean_abs_xy(trajectories, reference):
    """d of both selection rules: mean |dx|, |dy| over the 8 poses, one value per trajectory."""
    trajectories = np.asarray(trajectories, dtype=np.float32).reshape(-1, 8, 3)
    return np.abs(trajectories[..., :2] - np.asarray(reference, dtype=np.float32)[:, :2]).mean((1, 2))


def select_indices(trajectories, scores, top_k, score_thr, min_distance):
    """selection.select_candidates, returning candidate indices in pick order and each pick's FPS distance to the
    experts chosen before it (the first pick, the best PDMS, gets inf)."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    scores = np.asarray(scores)
    keep = np.isfinite(trajectories).all((1, 2)) & np.isfinite(scores) & (scores >= score_thr)
    order = np.flatnonzero(keep)[np.argsort(-scores[keep], kind="stable")]
    if not len(order):
        return order, np.zeros(0)
    candidates = trajectories[order]
    selected, gaps = [0], [np.inf]
    distance = np.abs(candidates[:, :, :2] - candidates[0, :, :2]).mean((1, 2))
    distance[0] = -1
    for _ in range(min(top_k, len(candidates)) - 1):
        index = int(distance.argmax())
        if distance[index] < min_distance:
            break
        selected.append(index)
        gaps.append(float(distance[index]))
        distance = np.minimum(distance, np.abs(candidates[:, :, :2] - candidates[index, :, :2]).mean((1, 2)))
        distance[selected] = -1
    return order[selected], np.array(gaps)


def score_rows(scores):
    """CLOVER score dicts -> [N, 7] in PDM_FIELDS order (CLOVER calls the final score pdm_score)."""
    return np.array([[s["pdm_score" if field == "score" else field] for field in PDM_FIELDS] for s in scores],
                    dtype=np.float64)


def describe(generation):
    """Family and main parameters of a CLOVER candidate (speeds m/s, target lateral offset m, + = left)."""
    family = generation.get("family", "?")
    value = lambda key: generation.get(key, float("nan"))  # noqa: E731
    if family in ("const_speed_portion", "boundary_interp", "offroad", "overshoot"):
        return f"{FAMILY_SHORT[family]} {value('speed'):g} m/s, lat {value('target_lateral'):+.1f}"
    if family == "stop_go":
        return f"stop-go {value('v0'):g} m/s, -{value('decel'):g} m/s², stop {value('stop_time'):g} s"
    if family == "accel_profile":
        return f"accel {value('v0'):g} m/s {value('accel'):+g} m/s², lat {value('target_lateral'):+.1f}"
    if family == "approach_brake":
        return f"approach {value('v0'):g}→{value('v_peak'):g} m/s, brake {value('brake'):g}"
    return FAMILY_SHORT.get(family, family)


# ----------------------------------------------------------------------------- stage 1: the CLOVER source

def extract(args, manifest, pool_tokens):
    """Reads the source once; returns per-scene statistics, aggregate counts and the candidate pools of pool_tokens."""
    k, thr, min_distance = manifest["top_k"], manifest["score_thr"], manifest["diversity_min_distance"]
    tokens = np.load(args.pack / "tokens.npy", allow_pickle=False)
    index = {token.decode(): row for row, token in enumerate(tokens)}
    pack_experts = np.load(args.pack / "pseudo_experts.npy", mmap_mode="r", allow_pickle=False)
    pack_mask = np.load(args.pack / "pseudo_expert_mask.npy", mmap_mode="r", allow_pickle=False)
    start = time.time()
    print(f"loading {args.source} (about 4 GB) ...", flush=True)
    scenes = read_scenes(args.source)
    print(f"loaded {len(scenes)} CLOVER scenes in {time.time() - start:.0f} s", flush=True)
    n = len(index)
    stats = {"n_candidates": np.zeros(n, np.int16), "n_eligible": np.zeros(n, np.int16),
             "n_selected": np.full(n, -1, np.int8),
             "expert_scores": np.full((n, k, len(PDM_FIELDS)), np.nan, np.float32),
             "expert_gap": np.full((n, k), np.nan, np.float32), "expert_family": np.full((n, k), -1, np.int8)}
    families, counts = {}, {"all": Counter(), "eligible": Counter(), "selected": Counter()}
    all_scores, mismatched, missing, pools = [], [], 0, {}
    for number, scene in enumerate(scenes):
        trajectories, scores = scene.get("trajectories_relative", []), scene.get("scores", [])
        if not (scene.get("valid", False) and len(trajectories) and len(scores)):
            continue  # the builder's filter
        token = scene["token"]
        row = index.get(token)
        if row is None:
            missing += 1
            continue
        trajectories = np.asarray(trajectories, dtype=np.float32)
        pdms = np.asarray([s["pdm_score"] for s in scores])
        picks, gaps = select_indices(trajectories, pdms, k, thr, min_distance)
        count = int(pack_mask[row].sum())
        if count != len(picks) or not np.array_equal(pack_experts[row][:count], trajectories[picks]):
            mismatched.append(token)
        eligible = np.isfinite(trajectories).all((1, 2)) & np.isfinite(pdms) & (pdms >= thr)
        family = [g.get("family", "?") for g in scene["generation"]]
        stats["n_candidates"][row], stats["n_eligible"][row], stats["n_selected"][row] = len(pdms), eligible.sum(), len(picks)
        if len(picks):
            stats["expert_scores"][row, :len(picks)] = score_rows([scores[i] for i in picks])
            stats["expert_gap"][row, :len(picks)] = gaps
            stats["expert_family"][row, :len(picks)] = [families.setdefault(family[i], len(families)) for i in picks]
        counts["all"].update(family)
        counts["eligible"].update(f for f, e in zip(family, eligible) if e)
        counts["selected"].update(family[i] for i in picks)
        all_scores.append(pdms[np.isfinite(pdms)])
        if token in pool_tokens:
            pools[token] = {"trajectories": trajectories, "stored": score_rows(scores).astype(np.float32),
                            "family": family, "generation": [describe(g) for g in scene["generation"]],
                            "picks": picks, "gaps": gaps}
        if (number + 1) % 20000 == 0:
            print(f"re-selected {number + 1}/{len(scenes)} scenes", flush=True)
    del scenes
    gc.collect()
    edges = np.linspace(0.0, 1.0, 101)
    all_scores = np.concatenate(all_scores)
    expert_pdms = stats["expert_scores"][..., SCORE]
    info = {"scenes": int((stats["n_selected"] >= 0).sum()), "missing_from_pack": missing,
            "mismatched": len(mismatched), "mismatched_examples": mismatched[:10],
            "families": [name for name, _ in sorted(families.items(), key=lambda item: item[1])],
            "family_counts": {key: dict(value) for key, value in counts.items()},
            "histogram_edges": edges.tolist(),
            "histogram_all": np.histogram(all_scores, edges)[0].tolist(),
            "histogram_eligible": np.histogram(all_scores[all_scores >= thr], edges)[0].tolist(),
            "histogram_selected": np.histogram(expert_pdms[np.isfinite(expert_pdms)], edges)[0].tolist(),
            "seconds": round(time.time() - start)}
    print(f"re-selection vs pack: {info['scenes'] - len(mismatched)}/{info['scenes']} scenes identical, "
          f"{missing} source scenes not in the pack", flush=True)
    return stats, info, pools


# ----------------------------------------------------------------------------- stage 2: pool scenes

def _score_task(task):
    cache_path, trajectories = task
    with lzma.open(cache_path, "rb") as stream:
        metric_cache = pickle.load(stream)
    return pairwise_pdm_scores(metric_cache, trajectories, _SIMULATOR, _SCORER)


def build_pool(args, manifest, pools, cache_paths):
    """Human trajectory, training-target status, top4 multi-GT and official PDMS for every pool scene."""
    k, human_distance = manifest["top_k"], manifest["human_coverage_distance"]
    tokens = sorted(pools)
    loader = make_scene_loader("train", [log_of(cache_paths, t) for t in tokens], tokens, SensorConfig.build_no_sensors())
    records = {}
    for token in tokens:
        if token not in loader.scene_frames_dicts:
            continue
        pool = pools[token]
        human = loader.get_scene_from_token(token).get_future_trajectory(num_trajectory_frames=8).poses
        human = np.asarray(human, dtype=np.float32)
        picks = pool["picks"]
        experts, mask = np.zeros((k, 8, 3), np.float32), np.zeros(k, bool)
        experts[:len(picks)], mask[:len(picks)] = pool["trajectories"][picks], True
        _, final_mask = include_human_expert(experts, mask, human)
        appended = bool(final_mask.sum() > mask.sum())
        d_human = mean_abs_xy(pool["trajectories"][picks], human)
        covered = bool(len(picks) and d_human.min() < human_distance)
        status = "fallback" if not len(picks) else "covered" if covered else "appended" if appended else "dropped"
        if appended != (status in ("fallback", "appended")):
            raise RuntimeError(f"{token}: human-inclusion status disagrees with include_human_expert")
        top4 = vmg.load_set(args.top4_root, token)
        records[token] = dict(pool, token=token, human=human, status=status, d_human=d_human,
                              log_name=log_of(cache_paths, token), top4=top4,
                              command=top4["command"] if top4 else "?")
    empty = np.zeros((0, 8, 3), np.float32)
    tasks = [(cache_paths[t], np.concatenate([r["trajectories"], r["human"][None],
                                              r["top4"]["trajectories"] if r["top4"] else empty]))
             for t, r in records.items()]
    print(f"scoring {len(tasks)} pool scenes ({sum(len(t[1]) for t in tasks)} trajectories)", flush=True)
    with ProcessPoolExecutor(args.score_workers, mp_context=multiprocessing.get_context("fork")) as executor:
        for record, rows in zip(records.values(), executor.map(_score_task, tasks, chunksize=2)):
            n = len(record["trajectories"])
            record["rows"], record["human_row"] = rows[:n], rows[n]
            if record["top4"]:
                record["top4"]["rows"] = rows[n + 1:]
            record["stored_matches"] = bool(np.allclose(record["stored"], rows[:n], atol=1e-6, equal_nan=True))
    return records


# ----------------------------------------------------------------------------- summary

def write_summary(path, stats, info, records, manifest):
    k, thr = manifest["top_k"], manifest["score_thr"]
    valid = stats["n_selected"] >= 0
    n_sel, n_cand, n_elig = stats["n_selected"][valid], stats["n_candidates"][valid], stats["n_eligible"][valid]
    scores = stats["expert_scores"][valid]
    pdms = scores[..., SCORE]
    finite = np.isfinite(pdms)
    comfort = scores[..., PDM_FIELDS.index("comfort")]
    ep = scores[..., PDM_FIELDS.index("ego_progress")]
    early = np.mean((n_sel > 0) & (n_sel < np.minimum(k, n_elig)))
    no_comfort = (comfort == 0) & finite
    lines = ["# CLOVER pseudo expert：分布与训练时的选择", "",
             f"训练脚本 `run_drivor_action_dit_dinov3_vitl_pseudo_expert_training.sh` 使用 `{manifest['source']}` 的候选，"
             f"离线选择（`{manifest['selection']}`）：PDMS ≥ {thr:g} → 按 PDMS 排序取最好的一条 → 以 d = 8 个位姿 |dx|、|dy| "
             f"的平均为距离做最远点采样，最多 {k} 条，剩余候选 d < {manifest['diversity_min_distance']:g} 时停止；"
             f"训练时若没有 expert 离人类轨迹 d < {manifest['human_coverage_distance']:g} 且还有空位，再加入人类轨迹。"
             "覆盖损失把 64 条一步 X 预测中离每个 expert 最近的一条拉向它（权重 0.5）。", "",
             f"## 全部 {info['scenes']:,} 个场景", "",
             f"- 重新按同样的规则选择，与训练用的 pack 逐场景比对：{info['scenes'] - info['mismatched']:,}/{info['scenes']:,} 完全一致"
             + (f"（不一致示例 {info['mismatched_examples']}）" if info["mismatched"] else "") + "。",
             f"- 每个场景平均 {n_cand.mean():.1f} 条候选，其中 {n_elig.mean():.1f} 条 PDMS ≥ {thr:g}"
             f"（{n_elig.sum() / n_cand.sum():.1%}）。",
             "- 选出的 expert 条数分布（0 … 8）：" + " / ".join(f"{np.mean(n_sel == c):.1%}" for c in range(k + 1))
             + f"；0 条（只用人类轨迹）{np.mean(n_sel == 0):.1%}，满 {k} 条 {np.mean(n_sel == k):.1%}，"
             f"因距离 < {manifest['diversity_min_distance']:g} 提前停止 {early:.1%}。",
             f"- expert 的 PDMS：均值 {pdms[finite].mean():.3f}，中位数 {np.median(pdms[finite]):.3f}；"
             f"E1（最高分）均值 {np.nanmean(pdms[:, 0]):.3f}，E2…E{k}（最远点）均值 {np.nanmean(pdms[:, 1:]):.3f}；"
             f"PDMS < 0.9 的占 {np.mean(pdms[finite] < 0.9):.1%}，舒适度不达标（C = 0）的占 {np.mean(comfort[finite] == 0):.1%}，"
             f"EP 均值 {np.nanmean(ep):.3f}；C = 0 的比例按选择顺序：E1 {no_comfort[:, 0].sum() / finite[:, 0].sum():.1%}，"
             f"E2 {no_comfort[:, 1].sum() / finite[:, 1].sum():.1%}，E3…E{k} {no_comfort[:, 2:].sum() / finite[:, 2:].sum():.1%}。",
             "",
             "| 生成方式 | 全部候选 | PDMS ≥ 0.8 | 选出的 expert | 合格候选被选中的比例 |", "|---|---:|---:|---:|---:|"]
    totals = {key: sum(value.values()) for key, value in info["family_counts"].items()}
    for family, count in sorted(info["family_counts"]["all"].items(), key=lambda item: -item[1]):
        eligible = info["family_counts"]["eligible"].get(family, 0)
        selected = info["family_counts"]["selected"].get(family, 0)
        lines.append(f"| {FAMILY_SHORT.get(family, family)} | {count / totals['all']:.1%} | "
                     f"{eligible / max(totals['eligible'], 1):.1%} | {selected / max(totals['selected'], 1):.1%} | "
                     f"{selected / max(eligible, 1):.1%} |")
    rec = list(records.values())
    with_experts = [r for r in rec if len(r["picks"])]
    nearest = np.array([r["d_human"].min() for r in with_experts])
    status = Counter(r["status"] for r in rec)
    lines += ["", f"## 随机场景池（{len(rec)} 个 navtrain 场景，来自 {len({r['log_name'] for r in rec})} 个训练 log）", "",
              stored_check(rec),
              "- 人类轨迹：" + "，".join(f"{GROUPS[g]} {status[g] / len(rec):.1%}" for g in GROUPS) + "。",
              f"- 人类轨迹到最近 expert 的 d：中位数 {np.median(nearest):.2f} m，< {manifest['human_coverage_distance']:g} 的占 "
              f"{np.mean(nearest < manifest['human_coverage_distance']):.1%}；expert 到人类轨迹的平均 d "
              f"{np.mean(np.concatenate([r['d_human'] for r in with_experts])):.2f} m，每个场景最远的一条平均 "
              f"{np.mean([r['d_human'].max() for r in with_experts]):.2f} m。",
              f"- 人类轨迹的官方 PDMS 均值 {np.mean([r['human_row'][SCORE] for r in rec]):.3f}；"
              f"“{GROUPS['dropped']}”的场景中，人类 {np.mean([r['human_row'][SCORE] for r in rec if r['status'] == 'dropped'] or [np.nan]):.3f}"
              f" vs expert {np.mean([r['rows'][r['picks'], SCORE].mean() for r in rec if r['status'] == 'dropped'] or [np.nan]):.3f}。"]
    top4_d, spread_e, spread_t = [], [], []
    for r in with_experts:
        experts = r["trajectories"][r["picks"]]
        if len(experts) >= 2:
            spread_e.append(experts[:, -1, :2].std(0))
        if r["top4"]:
            real = r["top4"]["trajectories"][~r["top4"]["padded"]]
            top4_d += [mean_abs_xy(experts, gt).min() for gt in real]
            if len(real) >= 2:
                spread_t.append(real[:, -1, :2].std(0))
    spread_e, spread_t = np.array(spread_e), np.array(spread_t)
    lines += [f"- top4 多 GT（不含补位）中，离某个 expert d < 0.5 的占 {np.mean(np.array(top4_d) < 0.5):.1%}"
              f"（中位数 {np.median(top4_d):.2f} m）。",
              f"- 4 s 终点的离散程度（标准差，横向 / 纵向）：expert {spread_e[:, 1].mean():.2f} / {spread_e[:, 0].mean():.2f} m，"
              f"top4 多 GT {spread_t[:, 1].mean():.2f} / {spread_t[:, 0].mean():.2f} m。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def stored_check(records):
    """Summary line: CLOVER's stored scores against the official PDMS of every pool candidate."""
    stored = np.concatenate([r["stored"] for r in records])
    official = np.concatenate([r["rows"] for r in records])
    differ = ~np.isclose(stored, official, atol=1e-6, equal_nan=True).all(1)
    experts = sum(not np.isclose(r["stored"][r["picks"]], r["rows"][r["picks"]], atol=1e-6, equal_nan=True).all()
                  for r in records)
    line = f"- 候选的存储分数与官方 NAVSIM-v1 PDMS：{(~differ).sum():,}/{len(differ):,} 条完全一致"
    if differ.any():
        gap = np.abs(stored - official)[differ]
        line += (f"；其余 {differ.sum()} 条的官方 PDMS 都 ≤ {official[differ, SCORE].max():.3f}，差别只在 "
                 + "、".join(PDM_FIELDS[j] for j in np.flatnonzero((gap > 1e-6).any(0)))
                 + f" 的小数位（最大差 {np.nanmax(gap):.1e}）")
    return line + f"；受影响的 pseudo expert：{experts} 个场景。"


# ----------------------------------------------------------------------------- figures

def finish_bev(ax, view, title):
    x_min, x_max, y_min, y_max = view
    ax.set_xlim(y_max, y_min)  # left is +y
    ax.set_ylim(x_min, x_max)
    ax.set_aspect("equal", adjustable="box")
    vpb.style_axes(ax)
    ax.set_xlabel("lateral y [m], + = left", fontsize=9, color=INK_2)
    ax.set_ylabel("longitudinal x [m]", fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left")


def draw_human(ax, record, zorder=9):
    vpb.plot_path(ax, record["human"], ring=True, color=INK, linewidth=1.6, linestyle=(0, (1, 2)), zorder=zorder)
    vpb.mark_end(ax, record["human"], "X", INK, zorder=13)


def draw_pool(ax, ctx, view, record, thr):
    """Every candidate: PDMS >= thr on the blue ramp (higher on top), the others thin grey."""
    vmg.draw_context(ax, ctx)
    scores, trajectories = record["rows"][:, SCORE], record["trajectories"]
    eligible = scores >= thr
    norm = mcolors.Normalize(vmin=thr, vmax=1.0)
    for i in np.flatnonzero(~eligible):
        vpb.plot_path(ax, trajectories[i], color=MUTED, linewidth=0.8, alpha=0.55, zorder=4)
    order = np.flatnonzero(eligible)[np.argsort(scores[eligible], kind="stable")]
    for i in order:
        vpb.plot_path(ax, trajectories[i], color=ELIGIBLE_CMAP(norm(scores[i])), linewidth=1.5, zorder=5)
    if len(order):
        ax.scatter(trajectories[order, -1, 1], trajectories[order, -1, 0], s=14, zorder=6, edgecolors="none",
                   c=[ELIGIBLE_CMAP(norm(scores[i])) for i in order])
    draw_human(ax, record)
    finish_bev(ax, view, f"CLOVER candidate pool: {len(scores)} trajectories, {eligible.sum()} with PDMS ≥ {thr:g} "
               "(blue, darker = higher); grey = not eligible")


def draw_targets(ax, ctx, view, record, human_detail):
    """What the coverage loss trains on: the FPS experts (and the human when the rule adds it), top4 for reference."""
    vmg.draw_context(ax, ctx)
    for trajectory in record["trajectories"]:
        vpb.plot_path(ax, trajectory, color=AXIS, linewidth=0.7, alpha=0.8, zorder=3)
    top4 = record["top4"]
    if top4:
        for trajectory, padded in zip(top4["trajectories"], top4["padded"]):
            vpb.plot_path(ax, trajectory, color=TOP4_COLOR, linewidth=1.2 if padded else 1.8,
                          linestyle=(0, (5, 2.5)), zorder=6)
            vpb.mark_end(ax, trajectory, "D", TOP4_COLOR, size=6, zorder=11)
    for j, i in enumerate(record["picks"]):
        trajectory = record["trajectories"][i]
        vpb.plot_path(ax, trajectory, ring=True, color=EXPERT_COLOR, linewidth=2.8 if j == 0 else 2.2, zorder=8)
        vpb.mark_end(ax, trajectory, "*" if j == 0 else "o", EXPERT_COLOR, size=14 if j == 0 else 7, zorder=12)
        ax.annotate(f"E{j + 1}", (trajectory[-1, 1], trajectory[-1, 0]), xytext=(8, 3 - 10 * (j % 3)),
                    textcoords="offset points", fontsize=9, color=INK, fontweight="semibold", zorder=14,
                    bbox=dict(facecolor=SURFACE, alpha=0.8, edgecolor="none", pad=0.5))
    draw_human(ax, record, zorder=9.5)
    human = record["human"]
    target = record["status"] in ("appended", "fallback")
    ax.annotate("human" + (" = target" if target else ""), (human[-1, 1], human[-1, 0]), xytext=(-8, 9),
                textcoords="offset points", ha="right", fontsize=9, color=INK, fontweight="semibold", zorder=14,
                bbox=dict(facecolor=SURFACE, alpha=0.8, edgecolor="none", pad=0.5))
    vmg.draw_events(ax, ctx, human_detail, INK)
    count = len(record["picks"])
    finish_bev(ax, view, f"Training targets: {count} pseudo expert{'s' if count != 1 else ''} (orange; ★ E1 = best "
               f"PDMS, E2… farthest-point picks)" + (" + human" if target else "") + "; violet = top4 multi-GT")


def status_text(record, manifest):
    k, limit = manifest["top_k"], manifest["human_coverage_distance"]
    if record["status"] == "fallback":
        return f"no candidate reaches PDMS {manifest['score_thr']:g}, the human trajectory is the only target"
    j = int(np.argmin(record["d_human"]))
    d = record["d_human"][j]
    if record["status"] == "covered":
        return f"covered by E{j + 1} (d {d:.2f} < {limit:g}), not added"
    if record["status"] == "appended":
        return f"no expert within d {limit:g} (nearest E{j + 1}, {d:.2f}), appended as target {len(record['picks']) + 1}"
    return f"no expert within d {limit:g} (nearest E{j + 1}, {d:.2f}) and all {k} slots full: not a training target"


def draw_table(ax, record, expert_details, human_detail, top4_details):
    rows, picks = record["rows"], record["picks"]
    experts = record["trajectories"][picks]
    width = 40

    def values(row):
        return f"{row[SCORE]:>7.3f}" + "".join(f"{row[PDM_FIELDS.index(field)]:>6.2f}" for _, field in SUB)

    header = (f"{'':4}{'generation: family, speed, lateral target':{width}}{'stored':>7}{'PDMS':>7}"
              + "".join(f"{name:>6}" for name, _ in SUB) + f"{'prog m':>8}{'FPS d':>7}{'d→human':>9}{'d→expert':>10}")
    lines = [(header, INK, None)]
    for j, i in enumerate(picks):
        gap = "best" if j == 0 else f"{record['gaps'][j]:.2f}"
        lines.append((f"{'E%d' % (j + 1):4}{record['generation'][i][:width - 1]:{width}}{record['stored'][i, SCORE]:>7.3f}"
                      + values(rows[i]) + f"{expert_details[j]['progress_m']:>8.1f}{gap:>7}{record['d_human'][j]:>9.2f}",
                      INK if rows[i, SCORE] >= 0.5 else CRITICAL, ("expert", j)))
    nearest = f"{record['d_human'].min():.2f}" if len(picks) else "-"
    human_label = {"covered": "human log (covered)", "appended": "human log (appended)",
                   "dropped": "human log (not a target)", "fallback": "human log (only target)"}[record["status"]]
    lines.append((f"{'hum':4}{human_label:{width}}{'-':>7}" + values(record["human_row"])
                  + f"{human_detail['progress_m']:>8.1f}{'':>7}{0.0:>9.2f}{nearest:>10}",
                  INK if record["human_row"][SCORE] >= 0.5 else CRITICAL, ("human", 0)))
    top4 = record["top4"]
    if top4:
        for j, trajectory in enumerate(top4["trajectories"]):
            label = "top4 " + top4["labels"][j] + (" (padded)" if top4["padded"][j] else "")
            to_expert = f"{mean_abs_xy(experts, trajectory).min():.2f}" if len(picks) else "-"
            lines.append((f"{'T%d' % (j + 1):4}{label[:width - 1]:{width}}{top4['stored'][j]:>7.3f}"
                          + values(top4["rows"][j]) + f"{top4_details[j]['progress_m']:>8.1f}{'':>7}"
                          f"{mean_abs_xy(trajectory, record['human'])[0]:>9.2f}{to_expert:>10}",
                          INK if top4["rows"][j][SCORE] >= 0.5 else CRITICAL, ("top4", j)))
    ax.axis("off")
    step = min(1.0 / (len(lines) + 2.2), 0.068)
    for n, (text, color, marker) in enumerate(lines):
        y = 1.0 - n * step
        ax.text(0.03, y, text, family="monospace", fontsize=9.2, color=color, va="top", transform=ax.transAxes,
                fontweight="semibold" if n == 0 else "normal")
        if marker:
            kind, j = marker
            style = {"expert": dict(color=EXPERT_COLOR, linewidth=4 if j == 0 else 2.6),
                     "human": dict(color=INK, linewidth=1.8, linestyle=(0, (1, 1))),
                     "top4": dict(color=TOP4_COLOR, linewidth=2, linestyle=(0, (3, 1.5)))}[kind]
            ax.plot([0.0, 0.022], [y - 0.35 * step] * 2, transform=ax.transAxes, clip_on=False, **style)
    ax.text(0.0, 1.0 - len(lines) * step - 0.2 * step,
            "d = mean |dx|, |dy| over the 8 poses (the metric of both selection rules) · stored = CLOVER's score, "
            "PDMS = official NAVSIM-v1 · FPS d = distance to the experts picked before · d→expert = to the nearest expert",
            fontsize=9, color=INK_2, va="top", transform=ax.transAxes)


def draw_score_distance(ax, record, thr, human_distance):
    trajectories, scores = record["trajectories"], record["rows"][:, SCORE]
    d = mean_abs_xy(trajectories, record["human"])
    eligible = scores >= thr
    ax.scatter(d[~eligible], scores[~eligible], s=18, color=MUTED, alpha=0.6, edgecolors="none", zorder=3,
               label=f"not eligible (< {thr:g})")
    ax.scatter(d[eligible], scores[eligible], s=26, color=ELIGIBLE_COLOR, edgecolors=SURFACE, linewidths=0.8,
               zorder=4, label="eligible candidate")
    picks = record["picks"]
    ax.scatter(d[picks], scores[picks], s=110, facecolors="none", edgecolors=EXPERT_COLOR, linewidths=2.2, zorder=5,
               label="pseudo expert")
    for j, i in enumerate(picks):
        ax.annotate(f"E{j + 1}", (d[i], scores[i]), xytext=(6, -12 if j % 2 else 6), textcoords="offset points",
                    fontsize=8.5, color=INK, zorder=6)
    top4 = record["top4"]
    if top4:
        ax.scatter(mean_abs_xy(top4["trajectories"], record["human"]), top4["rows"][:, SCORE], marker="D", s=50,
                   color=TOP4_COLOR, edgecolors=SURFACE, linewidths=1, zorder=6, label="top4 multi-GT")
    ax.scatter([0.0], [record["human_row"][SCORE]], marker="X", s=110, color=INK, edgecolors=SURFACE, linewidths=1,
               zorder=7, label="human")
    ax.axhline(thr, color=AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(human_distance, color=AXIS, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
    ax.text(human_distance + 0.06, 0.03, f"human covered if an expert is left of this line (d = {human_distance:g})",
            rotation=90, fontsize=8.5, color=INK_2, va="bottom", ha="left")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(left=-0.1)
    vpb.style_axes(ax)
    ax.set_xlabel("d to the human trajectory [m]", fontsize=9, color=INK_2)
    ax.set_ylabel("official PDMS", fontsize=9, color=INK_2)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("All candidates: PDMS vs distance to the human trajectory", fontsize=11.5, color=INK, loc="left")


def draw_profile(ax, record, column, thr, title, ylabel):
    trajectories, scores = record["trajectories"], record["rows"][:, SCORE]
    times = vpb.POSE_TIMES
    for i in np.argsort(scores, kind="stable"):
        eligible = scores[i] >= thr
        ax.plot(times, np.r_[0.0, trajectories[i, :, column]], color=ELIGIBLE_COLOR if eligible else MUTED,
                linewidth=0.9, alpha=0.75 if eligible else 0.45, zorder=3 if eligible else 2)
    top4 = record["top4"]
    if top4:
        for trajectory in top4["trajectories"]:
            ax.plot(times, np.r_[0.0, trajectory[:, column]], color=TOP4_COLOR, linewidth=1.6,
                    linestyle=(0, (5, 2.5)), zorder=4)
    for j, i in enumerate(record["picks"]):
        ax.plot(times, np.r_[0.0, trajectories[i, :, column]], color=EXPERT_COLOR, linewidth=2.6 if j == 0 else 2.0,
                zorder=5)
    ax.plot(times, np.r_[0.0, record["human"][:, column]], color=INK, linewidth=1.6, linestyle=(0, (1, 2)), zorder=6)
    vpb.style_axes(ax)
    ax.set_xlim(0, 4)
    ax.set_xlabel("time [s]", fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left")


def render_scene(path, group, record, scene, metric_cache, manifest):
    thr, human_distance = manifest["score_thr"], manifest["human_coverage_distance"]
    picks = record["picks"]
    experts = record["trajectories"][picks]
    top4 = record["top4"]
    top4_trajectories = top4["trajectories"] if top4 else np.zeros((0, 8, 3), np.float32)
    check = np.concatenate([experts, record["human"][None], top4_trajectories])
    rows, details = pairwise_pdm_details(metric_cache, check, _SIMULATOR, _SCORER)
    expected = np.concatenate([record["rows"][picks], record["human_row"][None],
                               top4["rows"] if top4 else np.zeros((0, len(PDM_FIELDS)))])
    if not np.allclose(rows, expected, atol=1e-9, equal_nan=True):
        raise RuntimeError(f"{record['token']}: re-scoring differs from the pool scores")
    n = len(picks)
    expert_details, human_detail, top4_details = details[:n], details[n], details[n + 1:]

    fig = plt.figure(figsize=(24, 23.5), facecolor=SURFACE)
    eligible = record["rows"][:, SCORE] >= thr
    fig.text(0.04, 0.99, f"train · {record['token']} · command {record['command']} · {GROUPS[group]}", fontsize=17,
             fontweight="semibold", color=INK, va="top")
    fig.text(0.04, 0.972, f"log {record['log_name']}   ·   CLOVER pool {len(eligible)} candidates, {eligible.sum()} with "
             f"PDMS ≥ {thr:g} → {n} pseudo experts by farthest-point sampling   ·   human: {status_text(record, manifest)}",
             fontsize=11.5, color=INK_2, va="top")
    header = GridSpec(1, 2, figure=fig, left=0.04, right=0.985, top=0.95, bottom=0.765, width_ratios=[1, 2.4],
                      wspace=0.03)
    ax_cam = fig.add_subplot(header[0])
    camera = scene.frames[scene.scene_metadata.num_history_frames - 1].cameras.cam_f0
    if camera.image is not None:
        ax_cam.imshow(camera.image)
    ax_cam.axis("off")
    draw_table(fig.add_subplot(header[1]), record, expert_details, human_detail, top4_details)

    middle = GridSpec(1, 2, figure=fig, left=0.04, right=0.985, top=0.725, bottom=0.315, wspace=0.12)
    ax_pool, ax_targets = fig.add_subplot(middle[0]), fig.add_subplot(middle[1])
    points = np.concatenate([record["trajectories"][eligible, :, :2].reshape(-1, 2), experts[..., :2].reshape(-1, 2),
                             record["human"][:, :2], top4_trajectories[..., :2].reshape(-1, 2), vpb.EGO_BOX])
    box = (points[:, 0].min() - 3.0, points[:, 0].max() + 3.0, points[:, 1].min() - 1.5, points[:, 1].max() + 1.5)
    view, _ = vpb.fit_view(ax_pool, box, max_stretch=1.0)  # equal aspect: true vehicle shapes and headings
    ctx = vpb.scene_context(metric_cache, view)
    draw_pool(ax_pool, ctx, view, record, thr)
    draw_targets(ax_targets, ctx, view, record, human_detail)

    bottom = GridSpec(1, 3, figure=fig, left=0.04, right=0.985, top=0.265, bottom=0.055, wspace=0.2)
    draw_score_distance(fig.add_subplot(bottom[0]), record, thr, human_distance)
    draw_profile(fig.add_subplot(bottom[1]), record, 0, thr, "x(t): progress of every trajectory", "longitudinal x [m]")
    draw_profile(fig.add_subplot(bottom[2]), record, 1, thr, "y(t): lateral offset", "lateral y [m], + = left")

    handles = [Line2D([], [], color=ELIGIBLE_CMAP(0.6), linewidth=1.5, label=f"candidate, PDMS ≥ {thr:g}"),
               Line2D([], [], color=MUTED, linewidth=0.8, alpha=0.6, label=f"candidate, PDMS < {thr:g}"),
               Line2D([], [], color=EXPERT_COLOR, linewidth=2.2, marker="o", markersize=7, markeredgecolor=SURFACE,
                      label="pseudo expert (★ E1)"),
               Line2D([], [], color=TOP4_COLOR, linewidth=1.8, linestyle=(0, (5, 2.5)), marker="D", markersize=6,
                      label="top4 multi-GT"),
               Line2D([], [], color=INK, linewidth=1.6, linestyle=(0, (1, 2)), marker="X", markersize=8,
                      markeredgecolor=SURFACE, label="human"),
               Patch(facecolor=AGENT_FILL, edgecolor=INK_2, label="agents now"),
               Line2D([], [], color=INK_2, linewidth=0.8, alpha=0.5, label="agents at 1-4 s"),
               Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (6, 4)), label="route centreline"),
               Patch(facecolor=DRIVABLE, edgecolor=LANE_EDGE, label="drivable area"),
               Line2D([], [], color=CRITICAL, linewidth=2.0, label="object hit")]
    fig.legend(handles=handles, loc="lower left", ncol=10, frameon=False, fontsize=10, bbox_to_anchor=(0.04, 0.004))
    colorbar = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=thr, vmax=1.0), cmap=ELIGIBLE_CMAP),
                            cax=fig.add_axes([0.335, 0.297, 0.15, 0.006]), orientation="horizontal")
    colorbar.set_label("PDMS of eligible candidates (left BEV)", color=INK_2, fontsize=9)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8.5, colors=INK_2, length=0)
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def draw_overview(path, stats, info, records, manifest):
    k, thr = manifest["top_k"], manifest["score_thr"]
    fig, axes = plt.subplots(2, 3, figsize=(30, 14), facecolor=SURFACE, gridspec_kw=dict(wspace=0.26, hspace=0.42))
    valid = stats["n_selected"] >= 0

    ax = axes[0, 0]
    counts = np.bincount(stats["n_selected"][valid], minlength=k + 1)
    share = 100 * counts / counts.sum()
    bars = ax.bar(np.arange(k + 1), share, width=0.72, color=EXPERT_COLOR, edgecolor=SURFACE, linewidth=2)
    ax.bar_label(bars, labels=[f"{s:.1f}%" for s in share], fontsize=9, color=INK_2, padding=2)
    ax.set_xticks(range(k + 1))
    ax.set_xlabel(f"pseudo experts selected (PDMS ≥ {thr:g}, farthest-point sampling)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("% of scenes", fontsize=9.5, color=INK_2)
    ax.set_title(f"Experts per scene, all {counts.sum():,} navtrain scenes (0: human trajectory only)", fontsize=12,
                 color=INK, loc="left")

    ax = axes[0, 1]
    edges = np.array(info["histogram_edges"])
    lo = int(round(thr * 100))
    for key, color, label in (("histogram_eligible", ELIGIBLE_COLOR, f"eligible candidates (≥ {thr:g})"),
                              ("histogram_selected", EXPERT_COLOR, "selected experts")):
        values = np.array(info[key], dtype=float)[lo:]
        values = 100 * values / values.sum()
        ax.stairs(values, edges[lo:], color=color, linewidth=2, label=label)
        ax.stairs(values, edges[lo:], color=color, alpha=0.12, fill=True)
    ax.set_xlim(thr, 1.0)
    ax.set_xlabel("PDMS (CLOVER score = official NAVSIM-v1)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("% of the set per 0.01 bin", fontsize=9.5, color=INK_2)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    eligible_share = sum(info["histogram_eligible"]) / sum(info["histogram_all"])
    ax.set_title(f"PDMS of eligible candidates ({eligible_share:.0%} of all) vs the experts picked from them",
                 fontsize=12, color=INK, loc="left")

    ax = axes[0, 2]
    family_counts = info["family_counts"]
    families = [f for f, _ in sorted(family_counts["all"].items(), key=lambda item: -item[1])]
    y = np.arange(len(families))
    height = 0.27
    for j, (key, color, label) in enumerate((("all", MUTED, "all candidates"),
                                             ("eligible", ELIGIBLE_COLOR, f"eligible (≥ {thr:g})"),
                                             ("selected", EXPERT_COLOR, "selected experts"))):
        total = sum(family_counts[key].values())
        values = [100 * family_counts[key].get(f, 0) / total for f in families]
        bars = ax.barh(y + (j - 1) * height, values, height=height * 0.92, color=color, edgecolor=SURFACE,
                       linewidth=1, label=label)
        if key == "selected":
            ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], fontsize=8.5, color=INK_2, padding=3)
    ax.set_yticks(y, [FAMILY_SHORT.get(f, f) for f in families], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("% of the set", fontsize=9.5, color=INK_2)
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    ax.set_title("How each set splits over CLOVER's generation families", fontsize=12, color=INK, loc="left")

    ax = axes[1, 0]
    scores = stats["expert_scores"][valid][..., SCORE]
    x = np.arange(k)
    median = [np.nanmedian(scores[:, j]) for j in x]
    q1, q3 = ([np.nanpercentile(scores[:, j], q) for j in x] for q in (25, 75))
    ax.fill_between(x, q1, q3, color=EXPERT_COLOR, alpha=0.18, linewidth=0)
    ax.plot(x, median, color=EXPERT_COLOR, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE)
    for xi, m in zip(x, median):
        ax.annotate(f"{m:.3f}", (xi, m), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9,
                    color=INK)
    ax.axhline(thr, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(x, [f"E{j + 1}" for j in x])
    ax.set_ylim(thr - 0.02, 1.02)
    ax.set_xlabel("pick order (E1 = best PDMS, E2… = farthest from the experts picked before)", fontsize=9.5,
                  color=INK_2)
    ax.set_ylabel("expert PDMS", fontsize=9.5, color=INK_2)
    ax.set_title("Expert PDMS by pick order, all scenes: median and interquartile band", fontsize=12, color=INK,
                 loc="left")

    rec = list(records.values())
    ax = axes[1, 1]
    nearest = np.array([r["d_human"].min() for r in rec if len(r["picks"])])
    ax.hist(np.clip(nearest, 0, 4.0), bins=np.linspace(0, 4.0, 41), color=INK_2, edgecolor=SURFACE, linewidth=1)
    ax.axvline(manifest["human_coverage_distance"], color=EXPERT_COLOR, linewidth=2)
    status = Counter(r["status"] for r in rec)
    ax.text(0.98, 0.95, "\n".join(f"{GROUPS[g]}: {status[g] / len(rec):.1%}" for g in GROUPS), transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color=INK, linespacing=1.6)
    ax.text(manifest["human_coverage_distance"], ax.get_ylim()[1] * 0.97, f" covered if d < {manifest['human_coverage_distance']:g}",
            fontsize=9, color=INK_2, va="top")
    ax.set_xlabel("d from the human trajectory to the nearest expert [m] (last bin: ≥ 4)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("scenes", fontsize=9.5, color=INK_2)
    ax.set_title(f"Is the human trajectory among the targets? ({len(rec)} pool scenes)", fontsize=12, color=INK,
                 loc="left")

    ax = axes[1, 2]
    experts, top4 = [], []
    for r in rec:
        end = r["human"][-1, :2]
        experts += list(r["trajectories"][r["picks"], -1, :2] - end)
        if r["top4"]:
            top4 += list(r["top4"]["trajectories"][~r["top4"]["padded"], -1, :2] - end)
    experts, top4 = np.array(experts).reshape(-1, 2), np.array(top4).reshape(-1, 2)
    ax.scatter(experts[:, 1], experts[:, 0], s=12, color=EXPERT_COLOR, alpha=0.35, edgecolors="none",
               label=f"pseudo experts ({len(experts)})")
    ax.scatter(top4[:, 1], top4[:, 0], s=14, color=TOP4_COLOR, alpha=0.5, edgecolors="none",
               label=f"top4 multi-GT, not padded ({len(top4)})")
    ax.scatter([0], [0], marker="X", s=120, color=INK, edgecolors=SURFACE, linewidths=1, zorder=5,
               label="human endpoint")
    both = np.concatenate([experts, top4])
    lim_y = np.percentile(np.abs(both[:, 1]), 99) + 0.5
    lim_x = np.percentile(np.abs(both[:, 0]), 99) + 0.5
    ax.set_xlim(lim_y, -lim_y)
    ax.set_ylim(-lim_x, lim_x)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("lateral offset from the human endpoint [m], + = left", fontsize=9.5, color=INK_2)
    ax.set_ylabel("longitudinal offset [m], + = ahead", fontsize=9.5, color=INK_2)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
              markerscale=2)
    ax.set_title("4 s endpoints relative to the human endpoint (pool scenes)", fontsize=12, color=INK, loc="left")

    for ax in axes.ravel():
        vpb.style_axes(ax)
    fig.suptitle(f"CLOVER pseudo experts of the DINOv3 Action-DiT pseudo-expert training: PDMS ≥ {thr:g}, best first, "
                 f"then farthest-point sampling to {k}; human added online only if uncovered and a slot is free",
                 x=0.012, y=0.955, ha="left", fontsize=14, color=INK)
    fig.savefig(path, dpi=100, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def pick_scenes(records, per_group, rng):
    """per_group scenes per human-rule group, a turning one first when the group has one."""
    picked = []
    for group in GROUPS:
        tokens = sorted(t for t, r in records.items() if r["status"] == group)
        turns = [t for t in tokens if records[t]["command"] in ("LEFT", "RIGHT")]
        chosen = [rng.choice(turns)] if turns and per_group else []
        rest = [t for t in tokens if t not in chosen]
        chosen += rng.sample(rest, min(per_group - len(chosen), len(rest)))
        picked += [(group, t) for t in chosen]
    return picked


def write_index(output_dir, entries):
    sections = "\n".join(f'<section><p>{html.escape(e["caption"])}</p><a href="{html.escape(e["image"])}">'
                         f'<img src="{html.escape(e["image"])}" loading="lazy"></a></section>' for e in entries)
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>CLOVER pseudo experts</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#fcfcfb;color:#0b0b0b}"
        "section{margin:0 0 40px}p{color:#52514e}img{width:100%;max-width:2400px;border:1px solid #e1e0d9}</style>"
        "<h1>CLOVER pseudo experts (DINOv3 Action-DiT pseudo-expert training)</h1>\n"
        "<p><a href='summary.md'>summary.md</a></p>\n" + sections + "\n", encoding="utf-8")


def main():
    global _SIMULATOR, _SCORER
    from hydra.utils import instantiate

    from navsim.agents.drivoR.proposal_analysis import compose_eval_config

    args = parse_args()
    manifest = json.loads((args.pack / "manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = compose_eval_config()
    _SIMULATOR, _SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)
    cache_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    if args.reuse:
        stats = dict(np.load(args.reuse / "dataset_stats.npz", allow_pickle=False))
        info = json.loads((args.reuse / "dataset.json").read_text())
        with open(args.reuse / "pool.pkl", "rb") as stream:
            records = pickle.load(stream)
    else:
        pack_tokens = {token.decode() for token in np.load(args.pack / "tokens.npy", allow_pickle=False)}
        _, tokens = sample_navtrain_tokens(cache_paths, args.train_logs, args.pool, random.Random(f"{args.seed}-pool"))
        tokens = {t for t in tokens if t in pack_tokens} | set(args.tokens or [])
        stats, info, pools = extract(args, manifest, tokens)
        np.savez_compressed(args.output_dir / "dataset_stats.npz", **stats)
        (args.output_dir / "dataset.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        records = build_pool(args, manifest, pools, cache_paths)
        with open(args.output_dir / "pool.pkl", "wb") as stream:
            pickle.dump(records, stream)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {key: str(value) for key, value in vars(args).items()} | {"manifest": manifest}, indent=2), encoding="utf-8")
    write_summary(args.output_dir / "summary.md", stats, info, records, manifest)
    draw_overview(args.output_dir / "overview.png", stats, info, records, manifest)
    entries = [{"image": "overview.png", "caption": "overview: all navtrain scenes and the random pool"}]

    picked = [(records[t]["status"], t) for t in args.tokens if t in records] if args.tokens else \
        pick_scenes(records, args.per_group, random.Random(f"{args.seed}-pick"))
    tokens = [t for _, t in picked]
    os.environ["NAVSIM_OSS_SENSOR_BLOBS_PATH"] = SPLITS["train"]["oss"]  # read at image-load time
    loader = make_scene_loader("train", [records[t]["log_name"] for t in tokens], tokens, front_camera_config())
    for number, (group, token) in enumerate(picked):
        with lzma.open(cache_paths[token], "rb") as stream:
            metric_cache = pickle.load(stream)
        image = f"{number:02d}_{group}_{token}.png"
        record = records[token]
        render_scene(args.output_dir / image, group, record, loader.get_scene_from_token(token), metric_cache, manifest)
        caption = (f"{GROUPS[group]} · {token} · {record['command']} · {len(record['picks'])} experts, PDMS "
                   + " / ".join(f"{v:.3f}" for v in record["rows"][record["picks"], SCORE])
                   + f" · human {record['human_row'][SCORE]:.3f}")
        entries.append({"image": image, "caption": caption})
        print(f"[{number + 1}/{len(picked)}] {caption}", flush=True)
    write_index(args.output_dir, entries)
    print(f"Wrote {len(picked)} scenes to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
