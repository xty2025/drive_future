"""Visualize NAVSIM multi-GT trajectory sets (top4 and poor4) scene by scene, each GT with its official PDMS.

top4: up to four high-scoring trajectories per navtrain scene (drivezero_goal_gen; may include the human log, padded
with near-duplicates when fewer were found). poor4: up to four "poor negatives" that pass the hard gates but score at
least a margin below the scene's weakest top4 GT. Every GT, and in the drawn scenes the human trajectory, is scored with
the official NAVSIM-v1 PDMS (navsim.evaluate.pairwise_pdm_score) on the navtrain metric cache, next to the score
stored with the data.

One figure per scene: both sets on the same BEV view drawn from the metric cache (drivable area, centreline, agents
now and at 1-4 s, ego footprint where a GT collides / violates TTC / leaves the drivable area), x(t) and y(t) of every
GT, and a table of stored score, PDMS and sub-metrics per GT. A random pool of scenes is scored for summary.md (how
the sets compare, whether the stored scores match); the drawn scenes are picked from it by driving command.
"""

import argparse
import json
import lzma
import multiprocessing
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import visualize_proposals_bev as vpb
from navsim.agents.drivoR.proposal_analysis import SPLITS, front_camera_config, log_of, make_scene_loader
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pairwise_pdm_score import PDM_FIELDS, pairwise_pdm_details, pairwise_pdm_scores

from visualize_proposals_bev import (  # palette shared with the proposal figures
    AGENT_FILL, CRITICAL, DRIVABLE, INK, INK_2, LANE_EDGE, MUTED, SURFACE)

DATA = Path("/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/multi_traj_gt_data")
SETS = ("top4", "poor4")
# One hue per set, four lightness steps per set (dark = first GT).
SET_COLORS = {"top4": ["#0d366b", "#256abf", "#5598e7", "#86b6ef"],
              "poor4": ["#8a2e0a", "#c4471a", "#eb6834", "#f2a07b"]}
SET_TITLES = {"top4": "top4 multi-GT", "poor4": "poor4 negatives"}
SUB = (("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"), ("TTC", "time_to_collision_within_bound"),
       ("C", "comfort"), ("EP", "ego_progress"))
SCORE = PDM_FIELDS.index("score")
COMMANDS = ("LEFT", "STRAIGHT", "RIGHT")

_SIMULATOR = None
_SCORER = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top4-root", type=Path, default=DATA / "navtrain_multigt_top4_dense_0p1s_f64_8gpu")
    parser.add_argument("--poor4-root", type=Path, default=DATA / "navtrain_poor4_multigt_dense_0p1s_f64_16gpu")
    parser.add_argument("--metric-cache", type=Path, default=vpb.PROJECT_ROOT / "data/metric_cache_navtrain")
    parser.add_argument("--pool", type=int, default=300, help="random scenes scored for the summary and the picks")
    parser.add_argument("--scenes", type=int, default=9, help="scenes drawn, spread over LEFT / STRAIGHT / RIGHT")
    parser.add_argument("--tokens", nargs="*", help="draw these tokens instead of picking from the pool")
    parser.add_argument("--output-dir", type=Path, help="default: exp/viz/multi_gt_<time>")
    parser.add_argument("--score-workers", type=int, default=min(24, len(os.sched_getaffinity(0))))
    parser.add_argument("--seed", type=int, default=0)
    args = vpb.resolve_caller_paths(parser.parse_args())
    if args.output_dir is None:
        args.output_dir = vpb.DRIVOR_ROOT / "exp/viz" / f"multi_gt_{datetime.now():%Y%m%d_%H%M%S}"
    return args


# ----------------------------------------------------------------------------- data

def load_set(root, token):
    """GTs of one scene: trajectories [n, 8, 3] (ego frame, rear axle), labels, stored scores, flags; None if absent."""
    folder = Path(root) / token
    if not (folder / "gt_trajectories.npy").is_file():
        return None
    meta = json.loads((folder / "trajectory.json").read_text())
    debug = json.loads((folder / "debug.json").read_text())
    trajectories = np.load(folder / "gt_trajectories.npy").astype(np.float32).reshape(-1, 8, 3)
    keep = np.load(folder / "gt_valid_mask.npy").astype(bool).reshape(-1)
    n = len(trajectories)
    padded = np.array(meta.get("gt_padded", [False] * n), dtype=bool)
    behaviors = debug.get("selected_behaviors", [""] * n)
    return {"trajectories": trajectories[keep], "labels": [l for l, k in zip(meta.get("gt_labels", [""] * n), keep) if k],
            "stored": np.load(folder / "gt_scores.npy").astype(np.float64).reshape(-1)[keep],
            "padded": padded[keep], "behaviors": [b for b, k in zip(behaviors, keep) if k],
            "command": meta.get("command", "?"), "human_pdms_stored": debug.get("human_gt_pdms")}


def _score_task(task):
    cache_path, trajectories = task
    with lzma.open(cache_path, "rb") as stream:
        metric_cache = pickle.load(stream)
    return pairwise_pdm_scores(metric_cache, trajectories, _SIMULATOR, _SCORER)


def score_pool(args, cache_paths):
    """Scores every GT of a random pool of scenes that have both sets; returns {token: {set: data with 'rows'}}."""
    tokens = sorted(set(os.listdir(args.top4_root)) & set(os.listdir(args.poor4_root)) & set(cache_paths))
    tokens = sorted(random.Random(args.seed).sample(tokens, min(args.pool, len(tokens))))
    pool = {}
    for token in tokens:
        sets = {name: load_set(getattr(args, f"{name}_root"), token) for name in SETS}
        if all(s is not None for s in sets.values()) and len(sets["top4"]["trajectories"]):
            pool[token] = sets
    tasks = [(cache_paths[t], np.concatenate([pool[t][name]["trajectories"] for name in SETS])) for t in pool]
    with ProcessPoolExecutor(args.score_workers, mp_context=multiprocessing.get_context("fork")) as executor:
        for token, rows in zip(list(pool), executor.map(_score_task, tasks, chunksize=2)):
            n_top = len(pool[token]["top4"]["trajectories"])
            pool[token]["top4"]["rows"], pool[token]["poor4"]["rows"] = rows[:n_top], rows[n_top:]
    return pool


def pick_scenes(pool, count, rng):
    """Scenes with at least one poor GT, spread evenly over the driving commands; the first one has a poor GT that
    fails TTC (the only gate-passing way to fail besides low progress), if the pool has one."""
    by_command = {}
    for token, sets in pool.items():
        if len(sets["poor4"]["trajectories"]):
            by_command.setdefault(sets["top4"]["command"], []).append(token)
    ttc = PDM_FIELDS.index("time_to_collision_within_bound")
    failing = sorted(t for t in pool if len(pool[t]["poor4"]["trajectories"]) and (pool[t]["poor4"]["rows"][:, ttc] < 1).any())
    picked = [rng.choice(failing)] if failing and count else []
    commands = [c for c in COMMANDS if c in by_command] + [c for c in by_command if c not in COMMANDS]
    while len(picked) < count and any(by_command.get(c) for c in commands):
        for command in commands:
            candidates = [t for t in by_command.get(command, []) if t not in picked]
            if candidates and len(picked) < count:
                picked.append(rng.choice(candidates))
    return picked


# ----------------------------------------------------------------------------- summary

def write_pool(path, pool):
    """Official PDMS and sub-metrics of every pool GT next to its stored score."""
    with open(path, "w") as stream:
        stream.write("token,command,set,index,label,padded,stored," + ",".join(PDM_FIELDS) + "\n")
        for token, sets in pool.items():
            for name in SETS:
                data = sets[name]
                for i in range(len(data["trajectories"])):
                    stream.write(f"{token},{data['command']},{name},{i + 1},{data['labels'][i]},{bool(data['padded'][i])},"
                                 f"{data['stored'][i]:.6f}," + ",".join(f"{v:.6f}" for v in data["rows"][i]) + "\n")


def write_summary(path, pool):
    def mean(values):
        return float(np.mean(values)) if len(values) else float("nan")

    lines = ["# NAVSIM 多 GT：top4 与 poor4 的官方 PDMS", "",
             f"随机 {len(pool)} 个同时有两套 GT 的 navtrain 场景。官方 PDMS 为 NAVSIM-v1（与评测相同的打分）；"
             "“存储分数”为数据自带的 gt_scores。top4 的补位 GT（gt_padded，近似重复）单独计。", "",
             "| 集合 | 场景平均条数 | PDMS 均值 | PDMS 中位数 | NC 通过 | DAC 通过 | TTC 通过 | C 通过 | EP 均值 | 存储分数 vs 官方：平均绝对差 | 差 ≤ 0.01 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, subset in (("top4", lambda s: ~s["padded"]), ("top4 补位", lambda s: s["padded"]),
                         ("poor4", lambda s: np.ones(len(s["padded"]), bool))):
        key = "top4" if name.startswith("top4") else "poor4"
        rows = [pool[t][key]["rows"][subset(pool[t][key])] for t in pool if len(pool[t][key]["trajectories"])]
        stored = [pool[t][key]["stored"][subset(pool[t][key])] for t in pool if len(pool[t][key]["trajectories"])]
        rows, stored = np.concatenate(rows), np.concatenate(stored)
        if not len(rows):
            continue
        diff = np.abs(stored - rows[:, SCORE])
        lines.append(f"| {name} | {len(rows) / len(pool):.2f} | {rows[:, SCORE].mean():.3f} | "
                     f"{np.median(rows[:, SCORE]):.3f} | "
                     + " | ".join(f"{np.mean(rows[:, PDM_FIELDS.index(f)] == 1):.1%}" for _, f in SUB[:4])
                     + f" | {rows[:, PDM_FIELDS.index('ego_progress')].mean():.3f} | {diff.mean():.4f} | {np.mean(diff <= 0.01):.1%} |")
    gaps = [pool[t]["poor4"]["rows"][:, SCORE].max() - pool[t]["top4"]["rows"][~pool[t]["top4"]["padded"], SCORE].min()
            for t in pool if len(pool[t]["poor4"]["trajectories"]) and (~pool[t]["top4"]["padded"]).any()]
    human = [pool[t]["top4"]["human_pdms_stored"] for t in pool if pool[t]["top4"]["human_pdms_stored"] is not None]
    lines += ["", f"- 有 poor4 的场景：{mean([len(pool[t]['poor4']['trajectories']) > 0 for t in pool]):.1%}；"
              f"每个场景“最好的 poor4 − 最差的 top4”（官方 PDMS）：中位数 {np.median(gaps):+.3f}，"
              f"poor4 反而高于 top4 的场景 {mean(np.array(gaps) > 0):.1%}。",
              f"- 数据自带的人类轨迹 PDMS（debug.json 的 human_gt_pdms）均值 {mean(human):.3f}。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


# ----------------------------------------------------------------------------- figure

def draw_events(ax, ctx, detail, color):
    for name, text in vpb.EVENT_NAMES:
        event = detail[name]
        if event is None:
            continue
        rings = ctx["local"](event["ego"])
        for ring in rings:
            ax.plot(ring[:, 1], ring[:, 0], color=color, linewidth=1.6, linestyle=(0, (3, 2)), zorder=10)
        for obj in event.get("objects", []):
            for ring in ctx["local"](obj):
                ax.plot(ring[:, 1], ring[:, 0], color=CRITICAL, linewidth=2.0, zorder=10.5)
        if rings:
            centre = rings[0].mean(axis=0)
            ax.annotate(f"{text} {event['time_s']:.1f} s", (centre[1], centre[0]), xytext=(12, 10),
                        textcoords="offset points", fontsize=8.5, color=INK, zorder=14,
                        bbox=dict(facecolor=SURFACE, alpha=0.9, edgecolor=color, linewidth=1, pad=1.5),
                        arrowprops=dict(arrowstyle="-", color=color, linewidth=1))


def draw_context(ax, ctx):
    for ring in ctx["drivable"]:
        ax.fill(ring[:, 1], ring[:, 0], color=DRIVABLE, linewidth=0, zorder=0)
        ax.plot(ring[:, 1], ring[:, 0], color=LANE_EDGE, linewidth=0.6, zorder=0.5)
    ax.plot(ctx["centreline"][:, 1], ctx["centreline"][:, 0], color=MUTED, linewidth=1.0, linestyle=(0, (6, 4)), zorder=1)
    for seconds, alpha in zip(vpb.AGENT_TIMES[1:], (0.75, 0.55, 0.4, 0.28)):
        for red, ring in ctx["agents"][seconds]:
            if not red:
                ax.plot(ring[:, 1], ring[:, 0], color=INK_2, linewidth=0.8, alpha=alpha, zorder=2)
    for red, ring in ctx["agents"][0]:
        if red:
            ax.fill(ring[:, 1], ring[:, 0], color=CRITICAL, alpha=0.15, linewidth=0, zorder=1.5)
        else:
            ax.fill(ring[:, 1], ring[:, 0], facecolor=AGENT_FILL, edgecolor=INK_2, linewidth=0.9, zorder=3)
    ax.fill(vpb.EGO_BOX[:, 1], vpb.EGO_BOX[:, 0], facecolor=SURFACE, edgecolor=INK, linewidth=1.2, zorder=3.5)


def draw_bev(ax, ctx, view, sets, details, human, human_row, human_detail):
    """Every GT on one BEV: top4 solid blue (labels right of the end), poor4 dashed orange (labels left), human dotted."""
    draw_context(ax, ctx)
    for name, prefix, side in (("poor4", "P", -1), ("top4", "T", 1)):
        data = sets[name]
        for i, trajectory in enumerate(data["trajectories"]):
            color, padded = SET_COLORS[name][i % 4], bool(data["padded"][i])
            vpb.plot_path(ax, trajectory, ring=True, color=color, linewidth=1.2 if padded else 2.4 if side > 0 else 2.0,
                          linestyle="-" if side > 0 else (0, (5, 2.5)), zorder=(8 if side > 0 else 7) + 0.1 * i)
            vpb.mark_end(ax, trajectory, "o" if side > 0 else "s", color, size=7, zorder=12)
            ax.annotate(f"{prefix}{i + 1} {data['rows'][i, SCORE]:.2f}", (trajectory[-1, 1], trajectory[-1, 0]),
                        xytext=(12 * side, 10 - 13 * i), textcoords="offset points", ha="left" if side > 0 else "right",
                        fontsize=9, color=color, fontweight="semibold", zorder=13,
                        bbox=dict(facecolor=SURFACE, alpha=0.8, edgecolor="none", pad=0.6))
            draw_events(ax, ctx, details[name][i], color)
    # Human on top: where a top4 GT is the human log, the dotted line runs along it.
    vpb.plot_path(ax, human, ring=True, color=INK, linewidth=1.6, linestyle=(0, (1, 2)), zorder=9)
    vpb.mark_end(ax, human, "X", INK, zorder=13)
    ax.annotate(f"human {human_row[SCORE]:.2f}", (human[-1, 1], human[-1, 0]), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9, color=INK, fontweight="semibold", zorder=14,
                bbox=dict(facecolor=SURFACE, alpha=0.8, edgecolor="none", pad=0.6))
    draw_events(ax, ctx, human_detail, INK)
    x_min, x_max, y_min, y_max = view
    ax.set_xlim(y_max, y_min)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect("equal", adjustable="box")
    vpb.style_axes(ax)
    ax.set_xlabel("lateral y [m], + = left", fontsize=9, color=INK_2)
    ax.set_ylabel("longitudinal x [m]", fontsize=9, color=INK_2)
    counts = {name: len(sets[name]["trajectories"]) for name in SETS}
    ax.set_title(f"top4 multi-GT ({counts['top4']}, solid blue, T#)  ·  poor4 ({counts['poor4']}, dashed orange, P#)  ·  "
                 "human GT (dotted black)   —   labels: official PDMS", fontsize=11.5, color=INK, loc="left")


def draw_profiles(ax_x, ax_y, sets, human):
    for name in SETS:
        for i, trajectory in enumerate(sets[name]["trajectories"]):
            style = dict(color=SET_COLORS[name][i % 4], linewidth=1.8, linestyle="-" if name == "top4" else (0, (5, 2.5)))
            for ax, column in ((ax_x, 0), (ax_y, 1)):
                ax.plot(vpb.POSE_TIMES, np.r_[0.0, trajectory[:, column]], **style)
    for ax, column in ((ax_x, 0), (ax_y, 1)):
        ax.plot(vpb.POSE_TIMES, np.r_[0.0, human[:, column]], color=INK, linewidth=1.5, linestyle=(0, (1, 2)))
        vpb.style_axes(ax)
        ax.set_xlim(0, 4)
        ax.set_xlabel("time [s]", fontsize=9, color=INK_2)
    ax_x.set_ylabel("longitudinal x [m]", fontsize=9, color=INK_2)
    ax_y.set_ylabel("lateral y [m], + = left", fontsize=9, color=INK_2)
    ax_x.set_title("x(t): speed  (solid top4, dashed poor4, dotted human)", fontsize=11.5, color=INK, loc="left")
    ax_y.set_title("y(t): lane offset", fontsize=11.5, color=INK, loc="left")


def draw_table(ax, sets, details, human_row, human_detail):
    """One row per GT: set, label, behaviour, stored score, official PDMS and sub-metrics, progress, infraction."""
    header = f"{'':3}{'set':7}{'#':>2}  {'label':22}{'behaviour':10}{'stored':>7}{'PDMS':>7}" + \
        "".join(f"{name:>6}" for name, _ in SUB) + f"{'prog m':>8}  infraction"
    lines = [(header, INK, None)]

    def row(name, index, label, behavior, stored, values, detail, color):
        events = ", ".join(f"{text} {detail[key]['time_s']:.1f}s" for key, text in vpb.EVENT_NAMES if detail[key])
        text = (f"{'':3}{name:7}{index:>2}  {label[:21]:22}{behavior[:9]:10}{stored:>7}{values[SCORE]:>7.3f}"
                + "".join(f"{values[PDM_FIELDS.index(field)]:>6.2f}" for _, field in SUB)
                + f"{detail['progress_m']:>8.1f}  {events or '-'}")
        lines.append((text, INK if values[SCORE] >= 0.5 else CRITICAL, color))

    for name in SETS:
        data = sets[name]
        for i in range(len(data["trajectories"])):
            label = data["labels"][i] + (" (padded)" if data["padded"][i] else "")
            row(name, i + 1, label, data["behaviors"][i] if i < len(data["behaviors"]) else "",
                f"{data['stored'][i]:.3f}", data["rows"][i], details[name][i], SET_COLORS[name][i % 4])
    stored_human = sets["top4"]["human_pdms_stored"]
    row("human", "", "log", "", "-" if stored_human is None else f"{stored_human:.3f}", human_row, human_detail, INK)
    ax.axis("off")
    for k, (text, color, marker) in enumerate(lines):
        y = 1.0 - k * 0.075
        ax.text(0.0, y, text, family="monospace", fontsize=9.5, color=color, va="top", transform=ax.transAxes,
                fontweight="semibold" if k == 0 else "normal")
        if marker:
            ax.plot([0.002, 0.016], [y - 0.025, y - 0.025], color=marker, linewidth=3, transform=ax.transAxes,
                    clip_on=False, linestyle="-" if marker == INK or k <= len(sets["top4"]["trajectories"]) else (0, (2, 1)))
    ax.text(0.0, 1.0 - len(lines) * 0.075 - 0.02, "stored = gt_scores shipped with the data; PDMS and sub-metrics = "
            "official NAVSIM-v1 (red row: PDMS < 0.5); EP = own progress / max(own, PDM-Closed progress)",
            fontsize=9, color=INK_2, va="top", transform=ax.transAxes)


def render_scene(path, token, scene, sets, metric_cache, simulator, scorer):
    human = scene.get_future_trajectory(num_trajectory_frames=8).poses.astype(np.float32)
    trajectories = np.concatenate([sets[name]["trajectories"] for name in SETS] + [human[None]])
    rows, found = pairwise_pdm_details(metric_cache, trajectories, simulator, scorer)
    n_top, n_poor = (len(sets[name]["trajectories"]) for name in SETS)
    if not np.allclose(rows[:n_top + n_poor], np.concatenate([sets[name]["rows"] for name in SETS]), atol=1e-9,
                       equal_nan=True):
        raise RuntimeError(f"{token}: re-scoring differs from the pool scores")
    details = {"top4": found[:n_top], "poor4": found[n_top:n_top + n_poor]}
    human_row, human_detail = rows[-1], found[-1]

    fig = plt.figure(figsize=(24, 19.5), facecolor=SURFACE)
    outer = GridSpec(2, 2, figure=fig, height_ratios=[4.2, 13], width_ratios=[1.25, 1], left=0.045, right=0.985,
                     top=0.935, bottom=0.05, hspace=0.22, wspace=0.12)
    fig.text(0.045, 0.985, f"train · {token} · command {sets['top4']['command']}", fontsize=17, fontweight="semibold",
             color=INK, va="top")
    fig.text(0.045, 0.966, f"log {log_of_scene(scene)}   ·   NAVSIM multi-GT: top4 (high-scoring GTs) vs poor4 "
             "(gated negatives below the weakest top4)", fontsize=11.5, color=INK_2, va="top")
    header = GridSpec(1, 2, left=0.045, right=0.985, top=0.935, bottom=0.935 - 0.19, width_ratios=[1, 2.1], wspace=0.04,
                      figure=fig)
    ax_cam = fig.add_subplot(header[0])
    camera = scene.frames[scene.scene_metadata.num_history_frames - 1].cameras.cam_f0
    if camera.image is not None:
        ax_cam.imshow(camera.image)
    ax_cam.axis("off")
    draw_table(fig.add_subplot(header[1]), sets, details, human_row, human_detail)

    ax_bev = fig.add_subplot(outer[1, 0])
    points = np.concatenate([trajectories[..., :2].reshape(-1, 2), vpb.EGO_BOX])
    box = (points[:, 0].min() - 3.0, points[:, 0].max() + 3.0, points[:, 1].min() - 1.5, points[:, 1].max() + 1.5)
    # Equal aspect: a lateral stretch would widen every vehicle footprint and make cars look sideways.
    view, _ = vpb.fit_view(ax_bev, box, max_stretch=1.0)
    ctx = vpb.scene_context(metric_cache, view)
    draw_bev(ax_bev, ctx, view, sets, details, human, human_row, human_detail)
    profiles = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 1], hspace=0.28)
    draw_profiles(fig.add_subplot(profiles[0]), fig.add_subplot(profiles[1]), sets, human)
    handles = [Line2D([], [], color=SET_COLORS["top4"][1], linewidth=2.4, marker="o", label="top4 GT (thin = padded)"),
               Line2D([], [], color=SET_COLORS["poor4"][2], linewidth=2.0, linestyle=(0, (5, 2.5)), marker="s",
                      label="poor4 GT"),
               Line2D([], [], color=INK, linewidth=1.5, linestyle=(0, (1, 2)), marker="X", label="human GT"),
               Patch(facecolor=AGENT_FILL, edgecolor=INK_2, label="agents now"),
               Line2D([], [], color=INK_2, linewidth=0.8, alpha=0.5, label="agents at 1-4 s"),
               Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (6, 4)), label="route centreline"),
               Patch(facecolor=DRIVABLE, edgecolor=LANE_EDGE, label="drivable area"),
               Line2D([], [], color=INK_2, linewidth=1.6, linestyle=(0, (3, 2)), label="ego at infraction"),
               Line2D([], [], color=CRITICAL, linewidth=2.0, label="object hit")]
    fig.legend(handles=handles, loc="lower left", ncol=9, frameon=False, fontsize=10, bbox_to_anchor=(0.045, 0.002))
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    return rows, human_row


def log_of_scene(scene):
    return scene.scene_metadata.log_name


def main():
    global _SIMULATOR, _SCORER
    from hydra.utils import instantiate

    from navsim.agents.drivoR.proposal_analysis import compose_eval_config

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = compose_eval_config()
    _SIMULATOR, _SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)
    cache_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    pool = score_pool(args, cache_paths)
    print(f"scored {len(pool)} pool scenes", flush=True)
    write_summary(args.output_dir / "summary.md", pool)
    write_pool(args.output_dir / "pool.csv", pool)

    tokens = args.tokens or pick_scenes(pool, args.scenes, random.Random(args.seed))
    for token in tokens:
        if token not in pool:  # explicit token outside the pool
            pool[token] = {name: load_set(getattr(args, f"{name}_root"), token) for name in SETS}
            rows = _score_task((cache_paths[token], np.concatenate([pool[token][n]["trajectories"] for n in SETS])))
            n_top = len(pool[token]["top4"]["trajectories"])
            pool[token]["top4"]["rows"], pool[token]["poor4"]["rows"] = rows[:n_top], rows[n_top:]
    os.environ["NAVSIM_OSS_SENSOR_BLOBS_PATH"] = SPLITS["train"]["oss"]
    loader = make_scene_loader("train", [log_of(cache_paths, t) for t in tokens], tokens, front_camera_config())
    entries = []
    for k, token in enumerate(tokens):
        if token not in loader.scene_frames_dicts:
            print(f"skip {token}: not a navtrain scene with route", flush=True)
            continue
        with lzma.open(cache_paths[token], "rb") as stream:
            metric_cache = pickle.load(stream)
        sets = pool[token]
        image = f"{k:02d}_{sets['top4']['command'].lower()}_{token}.png"
        render_scene(args.output_dir / image, token, loader.get_scene_from_token(token), sets, metric_cache,
                     _SIMULATOR, _SCORER)
        caption = (f"{token} · {sets['top4']['command']} · top4 PDMS "
                   + " / ".join(f"{v:.3f}" for v in sets["top4"]["rows"][:, SCORE]) + " · poor4 PDMS "
                   + " / ".join(f"{v:.3f}" for v in sets["poor4"]["rows"][:, SCORE]))
        entries.append({"split": "train", "image": image, "caption": caption})
        print(f"[{k + 1}/{len(tokens)}] {caption}", flush=True)
    vpb.write_index(args.output_dir, entries, list(SET_TITLES.values()))
    print(f"Wrote {len(entries)} scenes to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
