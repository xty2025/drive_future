"""Compare two DrivoR checkpoints scene by scene: all 64 proposals, scorer predictions and official PDMS.

For every scene each model runs once with the README NAVSIM-v1 evaluation config. All 64 final
proposals, the six scorer heads and the model's ranking score are kept; every proposal and the
human GT are scored with the official NAVSIM-v1 PDMS (navsim.evaluate.pairwise_pdm_score,
identical to pdm_score()), i.e. the PDMS each proposal would get if the scorer picked it.

One figure per scene, one row per model. The BEV is drawn from the metric cache, i.e. what the
PDM scorer itself sees: drivable area (DAC), route centreline (EP), agents now and at 1-4 s (NC,
TTC). It is zoomed to the trajectories; when they are long and thin the lateral axis is stretched
(labelled) so the 64 proposals separate. Next to it: each proposal's x(t) and y(t), the output vs
the truly best proposal per sub-metric (predicted -> true, with where/when it collides, violates
TTC or leaves the drivable area), and predicted vs true PDMS of all 64.

Scenes come from navtest (grouped by both models' evaluation CSVs) and from the training logs (a
random pool of navtrain scenes, scored on the fly), in four groups: B better, A better, both low,
both high. --reuse <dir> re-renders an earlier run from its .npz files without models.
"""

import argparse
import csv
import html
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
from matplotlib import cm, colors as mcolors, patheffects
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from shapely.affinity import affine_transform
from shapely.geometry import box as shapely_box

from navsim.agents.drivoR.multi_gt import MultiTrajectoryTargetBuilder
from navsim.agents.drivoR.proposal_analysis import (
    HEADS, SPLITS, front_camera_config, make_scene_loader, sample_navtrain_tokens, scorer_order)
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pairwise_pdm_score import PDM_FIELDS, pairwise_pdm_details, pairwise_pdm_scores

DRIVOR_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DRIVOR_ROOT.parent
GROUPS = ("b_better", "a_better", "both_low", "both_high")
# Display order of the sub-metrics: (label, scorer head, official PDM field).
METRICS = (("NC", "no_at_fault_collisions", "no_at_fault_collisions"),
           ("DAC", "drivable_area_compliance", "drivable_area_compliance"),
           ("TTC", "time_to_collision_within_bound", "time_to_collision_within_bound"),
           ("EP", "ego_progress", "ego_progress"),
           ("C", "comfort", "comfort"))
SCORE = PDM_FIELDS.index("score")
AGENT_TIMES = (0, 1, 2, 3, 4)  # seconds; footprints from the metric cache's 10 Hz occupancy maps
POSE_TIMES = np.arange(9) * 0.5  # t = 0 and the 8 proposal poses

# Colours (dataviz reference palette; accents validated together, all pairs):
# PDMS is a magnitude -> one blue ramp; output / best true / multi-GT are accents; context stays gray.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OUTPUT_COLOR, BEST_COLOR, MULTI_GT_COLOR, CRITICAL = "#eb6834", "#1baf7a", "#4a3aa7", "#d03b3b"
DRIVABLE, LANE_EDGE, AGENT_FILL = "#ecebe6", "#d6d5cf", "#c3c2b7"
BLUE_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95", "#104281", "#0d366b"]  # steps 100..700
LINE_CMAP = mcolors.LinearSegmentedColormap.from_list("pdms_lines", BLUE_STEPS[3:])  # from step 250: visible lines
NORM = mcolors.Normalize(vmin=0.0, vmax=1.0)
EGO_BOX = np.array([[c.x, c.y] for c in CarFootprint.build_from_rear_axle(
    StateSE2(0, 0, 0), get_pacifica_parameters()).oriented_box.all_corners()])

# Set before the scoring pool forks; worker processes inherit them.
_SIMULATOR = None
_SCORER = None


def parse_args():
    exp = DRIVOR_ROOT / "exp" / "ke"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-a", default="DrivoR")
    parser.add_argument("--ckpt-a", type=Path, default=exp / "training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/"
                        "lightning_logs/version_2/checkpoints/epoch=24-step=33250.ckpt")
    parser.add_argument("--eval-dir-a", type=Path, default=exp / "eval_navtest_training_drivoR_Nav1_navtrain_traj_long_"
                        "25epochs_multinode_version_2_epoch_24-step_33250")
    parser.add_argument("--label-b", default="DrivoR multi-GT")
    parser.add_argument("--ckpt-b", type=Path, default=exp / "training_drivoR_Nav1_navtrain_multigt/drivor_multigt_seed2_"
                        "run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt")
    parser.add_argument("--eval-dir-b", type=Path, default=exp / "eval_navtest_training_drivoR_Nav1_navtrain_multigt_"
                        "drivor_multigt_seed2_run1_version_0_epoch_24-step_33250")
    parser.add_argument("--per-group", type=int, default=2, help="scenes per group and split (4 groups)")
    parser.add_argument("--margin", type=float, default=0.3, help="PDMS gap for the 'X better' groups")
    parser.add_argument("--test-tokens", nargs="*", help="explicit navtest tokens instead of automatic groups")
    parser.add_argument("--train-tokens", nargs="*", help="explicit navtrain tokens instead of automatic groups")
    parser.add_argument("--train-logs", type=int, default=6, help="random training logs for the train pool")
    parser.add_argument("--train-pool", type=int, default=240, help="max train-pool scenes scored to form groups")
    parser.add_argument("--splits", nargs="+", default=["test", "train"], choices=list(SPLITS))
    parser.add_argument("--test-metric-cache", type=Path, default=PROJECT_ROOT / "data/metric_cache_navtest")
    parser.add_argument("--train-metric-cache", type=Path, default=PROJECT_ROOT / "data/metric_cache_navtrain")
    parser.add_argument("--multi-gt-root", type=Path, default=Path(os.environ.get(
        "DRIVOR_MULTI_GT_ROOT", PROJECT_ROOT / "multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu")))
    parser.add_argument("--output-dir", type=Path, help="default: exp/viz/proposals_bev_<time>, or the --reuse dir")
    parser.add_argument("--reuse", type=Path, help="re-render the .npz files of an earlier run (no models, no scoring)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--score-workers", type=int, default=min(32, len(os.sched_getaffinity(0))))
    parser.add_argument("--seed", type=int, default=0)
    args = resolve_caller_paths(parser.parse_args())
    if args.output_dir is None:
        args.output_dir = args.reuse or DRIVOR_ROOT / "exp/viz" / f"proposals_bev_{datetime.now():%Y%m%d_%H%M%S}"
    return args


def resolve_caller_paths(args):
    """Relative path arguments are relative to where the launcher was run (it changes into DrivoR/)."""
    base = Path(os.environ.get("DRIVOR_CALLER_CWD", os.getcwd()))
    for key, value in vars(args).items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, key, base / value)
    return args


# ----------------------------------------------------------------------------- scene selection

def read_eval_scores(eval_dir):
    """Per-token PDMS from the newest CSV of an evaluation experiment (<exp>/<run>/*.csv) or of one run directory."""
    eval_dir = Path(eval_dir)
    csvs = sorted([*eval_dir.glob("*.csv"), *eval_dir.glob("*/*.csv")], key=lambda path: path.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No evaluation CSV under {eval_dir}")
    with open(csvs[-1]) as stream:
        return {row["token"]: float(row["score"]) for row in csv.DictReader(stream)
                if row["token"] != "average" and row["valid"] == "True"}


def pick_groups(scores_a, scores_b, per_group, margin, rng):
    """Random scenes per group; the 'X better' groups fall back to the largest gaps."""
    tokens = sorted(set(scores_a) & set(scores_b))
    gap = {token: scores_b[token] - scores_a[token] for token in tokens}
    rules = {
        "b_better": (lambda t: gap[t] > margin, lambda t: -gap[t]),
        "a_better": (lambda t: gap[t] < -margin, lambda t: gap[t]),
        "both_low": (lambda t: max(scores_a[t], scores_b[t]) < 0.5, lambda t: max(scores_a[t], scores_b[t])),
        "both_high": (lambda t: min(scores_a[t], scores_b[t]) >= 0.9, lambda t: -min(scores_a[t], scores_b[t])),
    }
    picked, used = [], set()
    for group in GROUPS:
        accept, order = rules[group]
        candidates = [token for token in tokens if token not in used and accept(token)]
        chosen = rng.sample(candidates, min(per_group, len(candidates)))
        if len(chosen) < per_group and group != "both_high":
            rest = sorted((token for token in tokens if token not in used and token not in chosen), key=order)
            chosen += rest[:per_group - len(chosen)]
        used.update(chosen)
        picked += [(group, token) for token in chosen]
    return picked


def log_of(cache_paths, token):
    # Metric cache layout: <cache>/<log_name>/unknown/<token>/metric_cache.pkl
    return Path(cache_paths[token]).parts[-4]


# ----------------------------------------------------------------------------- scoring

def _score_task(task):
    cache_path, trajectories = task
    with lzma.open(cache_path, "rb") as stream:
        metric_cache = pickle.load(stream)
    return pairwise_pdm_scores(metric_cache, trajectories, _SIMULATOR, _SCORER)


def compute_records(args, labels):
    """Selects scenes, runs both models and scores every proposal; returns {split: [record]}."""
    import torch
    from hydra.utils import instantiate

    from navsim.agents.drivoR.proposal_analysis import compose_eval_config, load_agent, run_proposals

    global _SIMULATOR, _SCORER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = compose_eval_config()
    _SIMULATOR, _SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)
    agents = {args.label_a: load_agent(cfg, args.ckpt_a, device), args.label_b: load_agent(cfg, args.ckpt_b, device)}
    sensor_config = agents[args.label_a].get_sensor_config()
    multi_gt_loader = MultiTrajectoryTargetBuilder(OmegaConf.create(
        {"root": str(args.multi_gt_root), "max_gt": 4, "missing_policy": "original"}))
    pool = ProcessPoolExecutor(max_workers=args.score_workers, mp_context=multiprocessing.get_context("fork"))

    records = {}
    for split in args.splits:
        rng = random.Random(f"{args.seed}-{split}")  # same picks whether or not the other split runs
        os.environ["NAVSIM_OSS_SENSOR_BLOBS_PATH"] = SPLITS[split]["oss"]  # read at image-load time
        cache_paths = MetricCacheLoader(args.test_metric_cache if split == "test" else args.train_metric_cache) \
            .metric_cache_paths
        explicit = args.test_tokens if split == "test" else args.train_tokens
        eval_scores = [read_eval_scores(args.eval_dir_a), read_eval_scores(args.eval_dir_b)] if split == "test" else None
        outputs = {}

        if explicit:
            picked = [("manual", token) for token in explicit]
        elif split == "test":
            picked = pick_groups(eval_scores[0], eval_scores[1], args.per_group, args.margin, rng)
        else:
            logs, pool_tokens = sample_navtrain_tokens(cache_paths, args.train_logs, args.train_pool, rng)
            print(f"[train] pool: {len(pool_tokens)} scenes from {len(logs)} training logs", flush=True)
            loader = make_scene_loader(split, logs, pool_tokens, sensor_config)
            pool_tokens = [token for token in pool_tokens if token in loader.scene_frames_dicts]
            outputs = run_proposals(agents, loader, pool_tokens, device, args.batch_size, args.loader_workers)
            selected_rows = list(pool.map(_score_task, [(cache_paths[t], np.stack([
                outputs[label][t]["proposals"][outputs[label][t]["selected"]] for label in labels])) for t in pool_tokens]))
            pool_scores = [{t: rows[i, SCORE] for t, rows in zip(pool_tokens, selected_rows)} for i in range(2)]
            for i, label in enumerate(labels):
                print(f"[train] pool PDMS of selected trajectories, {label}: "
                      f"{100 * np.mean(list(pool_scores[i].values())):.2f} over {len(pool_tokens)} scenes", flush=True)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output_dir / "train_pool.csv", "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["token", "log_name", f"pdms_{labels[0]}", f"pdms_{labels[1]}"])
                writer.writerows([t, log_of(cache_paths, t), rows[0, SCORE], rows[1, SCORE]]
                                 for t, rows in zip(pool_tokens, selected_rows))
            picked = pick_groups(pool_scores[0], pool_scores[1], args.per_group, args.margin, rng)

        tokens = [token for _, token in picked]
        missing = [t for t in tokens if t not in outputs.get(labels[0], {})]
        if missing:
            loader = make_scene_loader(split, [log_of(cache_paths, t) for t in missing], missing, sensor_config)
            new = run_proposals(agents, loader, missing, device, args.batch_size, min(4, args.loader_workers))
            for label in labels:
                outputs.setdefault(label, {}).update(new[label])
        # One task per scene: A's 64 proposals, B's 64 proposals, then the human GT.
        tasks = [(cache_paths[t], np.concatenate([outputs[labels[0]][t]["proposals"],
                                                  outputs[labels[1]][t]["proposals"],
                                                  outputs[labels[0]][t]["human"][None]])) for t in tokens]
        print(f"[{split}] scoring {len(tasks)} scenes x 129 trajectories", flush=True)
        records[split] = []
        for (group, token), rows in zip(picked, pool.map(_score_task, tasks)):
            multi_gt = np.zeros((0, 8, 3), dtype=np.float32)
            if split == "train":
                loaded = multi_gt_loader.load(token, outputs[labels[0]][token]["human"])
                if not bool(loaded["multi_gt_fallback"]):
                    multi_gt = loaded["trajectory_multi_gt"][loaded["trajectory_multi_gt_valid"]].numpy()
            models = {}
            for i, label in enumerate(labels):
                out = outputs[label][token]
                models[label] = {key: out[key] for key in ("proposals", "head_probs", "pdm_score", "pred_pdms")}
                models[label]["selected"] = int(out["selected"])
                models[label]["pdm_rows"] = rows[64 * i:64 * (i + 1)]
            records[split].append({
                "split": split, "group": group, "token": token, "log_name": log_of(cache_paths, token),
                "eval_csv": [np.nan, np.nan] if eval_scores is None else [eval_scores[0][token], eval_scores[1][token]],
                "human": outputs[labels[0]][token]["human"], "human_row": rows[128], "multi_gt": multi_gt,
                "models": models})
    pool.shutdown()
    return records


def save_record(path, record, labels):
    np.savez_compressed(
        path, split=record["split"], group=record["group"], token=record["token"], log_name=record["log_name"],
        labels=np.array(labels), heads=np.array(HEADS), pdm_fields=np.array(PDM_FIELDS),
        eval_csv=np.array(record["eval_csv"], dtype=np.float64), human=record["human"],
        human_row=record["human_row"], multi_gt=record["multi_gt"],
        **{f"{key}_{i}": np.asarray(record["models"][label][key]) for i, label in enumerate(labels)
           for key in ("proposals", "head_probs", "pdm_score", "pred_pdms", "selected", "pdm_rows")})


def load_records(directory):
    records, labels = {}, None
    for path in sorted(Path(directory).glob("*/*.npz")):
        data = np.load(path, allow_pickle=False)
        labels = data["labels"].tolist()
        record = {key: data[key].item() if data[key].ndim == 0 else data[key]
                  for key in ("split", "group", "token", "log_name", "eval_csv", "human", "human_row", "multi_gt")}
        record["models"] = {label: {key: data[f"{key}_{i}"] for key in ("proposals", "head_probs", "pdm_score",
                                                                        "pred_pdms", "pdm_rows")}
                            for i, label in enumerate(labels)}
        for i, label in enumerate(labels):
            record["models"][label]["selected"] = int(data[f"selected_{i}"])
        records.setdefault(record["split"], []).append(record)
    return records, labels


# ----------------------------------------------------------------------------- metrics

def model_metrics(model):
    order = scorer_order(model["pdm_score"])
    if int(order[0]) != int(model["selected"]):
        raise RuntimeError("Scorer ranking does not start with the model output")
    rank = np.empty(64, dtype=int)
    rank[order] = np.arange(1, 65)
    true = model["pdm_rows"][:, SCORE]
    best = int(min(np.flatnonzero(true == np.nanmax(true)), key=lambda i: rank[i]))  # ties: scorer's preference
    sel = int(model["selected"])
    varied = np.isfinite(true).all() and np.ptp(true) > 0
    return {
        "order": order, "rank": rank, "true": true, "selected": sel, "best": best,
        "selected_pred": float(model["pred_pdms"][sel]), "selected_true": float(true[sel]),
        "best_true": float(true[best]), "best_rank": int(rank[best]),
        "selected_true_rank": int(1 + np.sum(true > true[sel])), "num_good": int(np.sum(true >= 0.9)),
        "spearman": float(spearmanr(model["pred_pdms"], true).correlation) if varied else np.nan,
    }


def fmt_rho(value):
    return "n/a" if not np.isfinite(value) else f"{value:.2f}"


# ----------------------------------------------------------------------------- scene geometry (metric cache)

def ego_frame_matrix(origin):
    """shapely affine matrix taking global coordinates into the ego frame (x forward, y left, rear axle)."""
    c, s = np.cos(origin.heading), np.sin(origin.heading)
    return [c, s, -s, c, -(c * origin.x + s * origin.y), s * origin.x - c * origin.y]


def rings(geometry):
    """Exterior coordinates [N, 2] (x, y) of every polygon in a geometry."""
    if geometry is None or geometry.is_empty:
        return []
    parts = getattr(geometry, "geoms", [geometry])
    return [np.asarray(part.exterior.coords) for part in parts if part.geom_type == "Polygon" and not part.is_empty]


def scene_context(metric_cache, view):
    """What the PDM scorer sees, in the ego frame, clipped to the view (x_min, x_max, y_min, y_max)."""
    x_min, x_max, y_min, y_max = view
    matrix = ego_frame_matrix(metric_cache.ego_state.rear_axle)
    clip = shapely_box(x_min, y_min, x_max, y_max)

    def local(geometry):
        return rings(affine_transform(geometry, matrix).intersection(clip))

    centre = metric_cache.centerline._states_se2_array[:, :2]
    a, b, d, e, x_off, y_off = matrix
    agents, red_light = {}, metric_cache.observation.red_light_token
    for seconds in AGENT_TIMES:
        occupancy = metric_cache.observation[int(round(seconds * 10))]
        agents[seconds] = [(token.startswith(red_light), ring) for token, geometry
                           in zip(occupancy.tokens, occupancy._geometries) for ring in local(geometry)]
    return {
        "drivable": [ring for geometry in metric_cache.drivable_area_map._geometries for ring in local(geometry)],
        "centreline": np.column_stack([a * centre[:, 0] + b * centre[:, 1] + x_off,
                                       d * centre[:, 0] + e * centre[:, 1] + y_off]),
        "agents": agents,
        "local": local,
    }


def trajectory_box(record, labels):
    """Tight box (x_min, x_max, y_min, y_max) around every drawn trajectory and the ego footprint."""
    points = np.concatenate([record["models"][label]["proposals"][..., :2].reshape(-1, 2) for label in labels]
                            + [record["human"][:, :2], record["multi_gt"][..., :2].reshape(-1, 2), EGO_BOX])
    return (points[:, 0].min() - 3.0, points[:, 0].max() + 3.0, points[:, 1].min() - 1.5, points[:, 1].max() + 1.5)


def fit_view(ax, box, max_stretch=6.0):
    """Limits filling the axes. The lateral axis is stretched only when the trajectories are long and thin."""
    position = ax.get_position()
    width, height = position.width * ax.figure.get_figwidth(), position.height * ax.figure.get_figheight()
    x_min, x_max, y_min, y_max = box
    ratio = (width * (x_max - x_min)) / (height * (y_max - y_min))  # > 1: lateral could be magnified
    stretch = 1.0 if ratio < 1.25 else min(ratio, max_stretch)
    if ratio < 1.0:  # wide (e.g. turns): equal aspect, longer x range
        pad = (height * (y_max - y_min) / width - (x_max - x_min)) / 2.0
        x_min, x_max = x_min - pad, x_max + pad
    else:  # widen the lateral range to fill the axes at this stretch
        pad = (width * (x_max - x_min) / (height * stretch) - (y_max - y_min)) / 2.0
        y_min, y_max = y_min - pad, y_max + pad
    return (x_min, x_max, y_min, y_max), stretch


# ----------------------------------------------------------------------------- drawing

def plot_path(ax, poses, ring=False, **style):
    points = np.vstack([np.zeros((1, 2)), np.asarray(poses)[:, :2]])
    line, = ax.plot(points[:, 1], points[:, 0], solid_capstyle="round", solid_joinstyle="round", **style)
    if ring:  # 2px surface ring keeps the mark legible over the others
        line.set_path_effects([patheffects.Stroke(linewidth=style.get("linewidth", 2) + 2.5, foreground=SURFACE),
                               patheffects.Normal()])


def mark_end(ax, poses, marker, color, size=9, zorder=12):
    ax.plot(poses[-1, 1], poses[-1, 0], marker=marker, markersize=size, color=color, markeredgecolor=SURFACE,
            markeredgewidth=1.5, linestyle="none", zorder=zorder)


def style_axes(ax):
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)


EVENT_NAMES = (("collision", "collision"), ("ttc", "TTC violated"), ("off_road", "off drivable area"))


def draw_bev(ax, ctx, view, stretch, model, met, values, record, details, title):
    x_min, x_max, y_min, y_max = view
    for ring in ctx["drivable"]:
        ax.fill(ring[:, 1], ring[:, 0], color=DRIVABLE, linewidth=0, zorder=0)
        ax.plot(ring[:, 1], ring[:, 0], color=LANE_EDGE, linewidth=0.6, zorder=0.5)
    ax.plot(ctx["centreline"][:, 1], ctx["centreline"][:, 0], color=MUTED, linewidth=1.0, linestyle=(0, (6, 4)),
            zorder=1)
    for seconds, alpha in zip(AGENT_TIMES[1:], (0.75, 0.55, 0.4, 0.28)):  # future footprints, fading with time
        for red, ring in ctx["agents"][seconds]:
            if not red:
                ax.plot(ring[:, 1], ring[:, 0], color=INK_2, linewidth=0.8, alpha=alpha, zorder=2)
    for red, ring in ctx["agents"][0]:
        if red:
            ax.fill(ring[:, 1], ring[:, 0], color=CRITICAL, alpha=0.15, linewidth=0, zorder=1.5)
        else:
            ax.fill(ring[:, 1], ring[:, 0], facecolor=AGENT_FILL, edgecolor=INK_2, linewidth=0.9, zorder=3)
    ax.fill(EGO_BOX[:, 1], EGO_BOX[:, 0], facecolor=SURFACE, edgecolor=INK, linewidth=1.2, zorder=3.5)

    proposals = model["proposals"]
    order = np.argsort(np.nan_to_num(values, nan=-1.0), kind="stable")  # highest values drawn last, on top
    colors = [LINE_CMAP(NORM(values[i])) if np.isfinite(values[i]) else MUTED for i in order]
    for i, color in zip(order, colors):
        plot_path(ax, proposals[i], color=color, linewidth=1.3, alpha=0.95, zorder=4)
    ax.scatter(proposals[order, -1, 1], proposals[order, -1, 0], c=colors, s=14, edgecolors="none", zorder=5)
    for gt in record["multi_gt"]:
        plot_path(ax, gt, color=MULTI_GT_COLOR, linewidth=1.6, linestyle=(0, (5, 3)), zorder=6)
    plot_path(ax, record["human"], ring=True, color=INK, linewidth=1.4, linestyle=(0, (1, 2)), zorder=7)
    best, sel = proposals[met["best"]], proposals[met["selected"]]
    plot_path(ax, best, ring=True, color=BEST_COLOR, linewidth=2.6, zorder=8)
    plot_path(ax, sel, ring=True, color=OUTPUT_COLOR, linewidth=2.4, zorder=9)
    mark_end(ax, record["human"], "X", INK)
    mark_end(ax, best, "D", BEST_COLOR)
    mark_end(ax, sel, "o", OUTPUT_COLOR, size=8, zorder=13)

    # Where and when the output / best proposal loses points: simulated ego footprint at that moment.
    for row, (index, color) in enumerate(((met["selected"], OUTPUT_COLOR), (met["best"], BEST_COLOR))):
        if row == 1 and index == met["selected"]:
            break
        for name, label in EVENT_NAMES:
            event = details[index][name]
            if event is None:
                continue
            ego_rings = ctx["local"](event["ego"])
            for ring in ego_rings:
                ax.plot(ring[:, 1], ring[:, 0], color=color, linewidth=1.8, linestyle=(0, (3, 2)), zorder=10)
            for obj in event.get("objects", []):
                for ring in ctx["local"](obj):
                    ax.plot(ring[:, 1], ring[:, 0], color=CRITICAL, linewidth=2.0, zorder=10.5)
            if ego_rings:
                centre = ego_rings[0].mean(axis=0)
                ax.annotate(f"{label} {event['time_s']:.1f} s", (centre[1], centre[0]),
                            xytext=(-14 if row else 14, 12 if name == "collision" else -16),
                            textcoords="offset points", ha="right" if row else "left", fontsize=8.5, color=INK,
                            zorder=14, bbox=dict(facecolor=SURFACE, alpha=0.9, edgecolor=color, linewidth=1, pad=1.5),
                            arrowprops=dict(arrowstyle="-", color=color, linewidth=1))

    ax.set_xlim(y_max, y_min)  # left is +y
    ax.set_ylim(x_min, x_max)
    ax.set_aspect(1.0 / stretch, adjustable="box")
    style_axes(ax)
    ax.set_xlabel("lateral y [m], + = left" + (f"   (lateral stretched ×{stretch:.1f})" if stretch > 1 else ""),
                  fontsize=9, color=INK_2)
    ax.set_ylabel("longitudinal x [m]", fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left")


def draw_profiles(ax_x, ax_y, model, met, record, values):
    proposals = model["proposals"]
    for i in np.argsort(np.nan_to_num(values, nan=-1.0), kind="stable"):
        color = LINE_CMAP(NORM(values[i])) if np.isfinite(values[i]) else MUTED
        for ax, column in ((ax_x, 0), (ax_y, 1)):
            ax.plot(POSE_TIMES, np.r_[0.0, proposals[i, :, column]], color=color, linewidth=1.0, alpha=0.9, zorder=2)
    for poses, style in ((record["human"], dict(color=INK, linewidth=1.4, linestyle=(0, (1, 2)))),
                         (proposals[met["best"]], dict(color=BEST_COLOR, linewidth=2.4)),
                         (proposals[met["selected"]], dict(color=OUTPUT_COLOR, linewidth=2.2))):
        for ax, column in ((ax_x, 0), (ax_y, 1)):
            line, = ax.plot(POSE_TIMES, np.r_[0.0, poses[:, column]], zorder=4, solid_capstyle="round", **style)
            line.set_path_effects([patheffects.Stroke(linewidth=style["linewidth"] + 2.5, foreground=SURFACE),
                                   patheffects.Normal()])
    for ax in (ax_x, ax_y):
        style_axes(ax)
        ax.set_xlim(0, 4)
    ax_x.tick_params(labelbottom=False)
    ax_x.set_ylabel("longitudinal x [m]", fontsize=9, color=INK_2)
    ax_y.set_ylabel("lateral y [m], + = left", fontsize=9, color=INK_2)
    ax_y.set_xlabel("time [s]", fontsize=9, color=INK_2)
    ax_x.set_title("Position over time: speed (x) and lane offset (y) of all 64", fontsize=11.5, color=INK, loc="left")


def sub_scores(model, index):
    pred = [model["head_probs"][index, HEADS.index(head)] for _, head, _ in METRICS] + [model["pred_pdms"][index]]
    true = [model["pdm_rows"][index, PDM_FIELDS.index(field)] for _, _, field in METRICS] + \
        [model["pdm_rows"][index, SCORE]]
    return np.array(pred, dtype=float), np.array(true, dtype=float)


def describe(detail):
    events = [f"{label} at {detail[name]['time_s']:.1f} s" for name, label in EVENT_NAMES if detail[name] is not None]
    return (", ".join(events) if events else "no infraction") + f";  progress {detail['progress_m']:.1f} m"


def draw_comparison(ax, ax_numbers, model, met, details):
    names = [name for name, _, _ in METRICS] + ["PDMS"]
    entries = [(met["selected"], OUTPUT_COLOR, -0.17, f"output #{met['selected']} (rank 1)")]
    if met["best"] != met["selected"]:
        entries.append((met["best"], BEST_COLOR, 0.17, f"best true #{met['best']} (rank {met['best_rank']})"))
    rows = np.arange(len(names))
    numbers = blended_transform_factory(ax_numbers.transAxes, ax_numbers.transData)
    for column, (index, color, dy, label) in enumerate(entries):
        pred, true = sub_scores(model, index)
        ax.hlines(rows + dy, np.minimum(pred, true), np.maximum(pred, true), color=color, linewidth=2.2, alpha=0.5,
                  zorder=2)
        ax.scatter(pred, rows + dy, s=60, facecolors=SURFACE, edgecolors=color, linewidths=2, zorder=3)
        ax.scatter(true, rows + dy, s=60, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        x_text = 0.1 + 0.5 * column
        ax_numbers.text(x_text, -0.75, label.replace(" (", "\n("), transform=numbers, fontsize=9, color=INK,
                        va="bottom", fontweight="semibold")
        ax_numbers.plot([x_text - 0.07], [-1.0], marker="o", color=color, transform=numbers, clip_on=False,
                        markersize=7)
        for j in rows:
            mismatch = abs(pred[j] - true[j]) > 0.3
            ax_numbers.text(x_text, j, f"{pred[j]:.2f} → {true[j]:.2f}", transform=numbers, fontsize=9.5,
                            va="center", color=CRITICAL if mismatch else INK_2,
                            fontweight="semibold" if mismatch else "normal")
    ax_numbers.set_ylim(len(names) - 0.5, -0.5)
    ax_numbers.axis("off")
    ax.set_yticks(rows)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(-0.03, 1.03)
    ax.axhline(len(names) - 1.5, color=AXIS, linewidth=0.8)  # PDMS combines the five above
    style_axes(ax)
    ax.set_xlabel("○ scorer predicted   ●  true (official)", fontsize=9, color=INK_2)
    # Raised title: the number columns' headers sit just above the first row.
    ax.set_title("Output vs truly best proposal, per sub-metric\n(red numbers: predicted and true differ by > 0.3)",
                 fontsize=11.5, color=INK, loc="left", pad=34)
    closed = details[entries[0][0]]["pdm_closed_progress_m"]
    text = "\n".join([f"{label.split(' (')[0]}: {describe(details[index])}" for index, _, _, label in entries]
                     + [f"EP = own progress / max(own, PDM-Closed reference {closed:.1f} m)"])
    ax.text(0.0, -0.36, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="top", linespacing=1.6)


def draw_scatter(ax, model, met):
    pred, true = model["pred_pdms"], met["true"]
    others = np.ones(64, dtype=bool)
    others[[met["selected"], met["best"]]] = False
    ax.scatter(pred[others], true[others], s=26, color=BLUE_STEPS[6], edgecolors=SURFACE, linewidths=0.8, zorder=3)
    ax.scatter(pred[met["best"]], true[met["best"]], marker="D", s=90, color=BEST_COLOR, edgecolors=SURFACE,
               linewidths=1.5, zorder=5)
    ax.scatter(pred[met["selected"]], true[met["selected"]], marker="o", s=90, color=OUTPUT_COLOR,
               edgecolors=SURFACE, linewidths=1.5, zorder=6)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    style_axes(ax)
    ax.set_xlabel("scorer predicted PDMS", fontsize=9, color=INK_2)
    ax.set_ylabel("true PDMS", fontsize=9, color=INK_2)
    ax.set_title(f"All 64: predicted vs true PDMS (Spearman {fmt_rho(met['spearman'])})", fontsize=11.5, color=INK,
                 loc="left")


GROUP_NAMES = {"b_better": "{b} better", "a_better": "{a} better", "both_low": "both low", "both_high": "both high",
               "manual": "manual pick"}


def summary_text(record, labels, mets):
    names = [label[:16] for label in labels]
    lines = [f"{'':24}{names[0]:>17}{names[1]:>17}"]
    rows = [
        ("output true PDMS", lambda m: f"{m['selected_true']:.3f}"),
        ("output predicted PDMS", lambda m: f"{m['selected_pred']:.3f}"),
        ("best of 64 (true)", lambda m: f"{m['best_true']:.3f}"),
        ("best's scorer rank", lambda m: f"{m['best_rank']}/64"),
        ("output's true rank", lambda m: f"{m['selected_true_rank']}/64"),
        ("proposals PDMS >= 0.9", lambda m: f"{m['num_good']}/64"),
        ("Spearman(pred, true)", lambda m: fmt_rho(m["spearman"])),
    ]
    for name, fmt in rows:
        lines.append(f"{name:24}{fmt(mets[labels[0]]):>17}{fmt(mets[labels[1]]):>17}")
    if np.isfinite(record["eval_csv"]).all():
        lines.append(f"{'navtest eval CSV':24}{record['eval_csv'][0]:>17.3f}{record['eval_csv'][1]:>17.3f}")
    lines += ["", f"human GT PDMS {record['human_row'][SCORE]:.3f}"
              + (f"    ·    multi-GT training targets: {len(record['multi_gt'])}" if len(record["multi_gt"]) else "")]
    return "\n".join(lines)


def render_scene(path, scene, record, labels, metric_cache, simulator, scorer):
    mets = {label: model_metrics(record["models"][label]) for label in labels}
    details = {}
    for label in labels:
        model, met = record["models"][label], mets[label]
        indices = sorted({met["selected"], met["best"]})
        rows, found = pairwise_pdm_details(metric_cache, model["proposals"][indices], simulator, scorer)
        if not np.allclose(rows, model["pdm_rows"][indices], atol=1e-9, equal_nan=True):
            raise RuntimeError(f"{record['token']} {label}: re-scoring differs from the stored PDMS")
        details[label] = dict(zip(indices, found))

    fig = plt.figure(figsize=(24, 22), facecolor=SURFACE)
    outer = GridSpec(4, 4, figure=fig, height_ratios=[3.6, 7.6, 7.6, 0.3], width_ratios=[1, 1, 1.0, 1.05],
                     left=0.045, right=0.985, top=0.94, bottom=0.03, hspace=0.3, wspace=0.2)
    group = GROUP_NAMES[record["group"]].format(a=labels[0], b=labels[1])
    fig.text(0.045, 0.985, f"{record['split']} · {record['token']} · {group}", fontsize=17, fontweight="semibold",
             color=INK, va="top")
    fig.text(0.045, 0.966, "   ·   ".join(f"{label}: output {mets[label]['selected_true']:.3f} (predicted "
                                         f"{mets[label]['selected_pred']:.3f}), best of 64 {mets[label]['best_true']:.3f}"
                                         for label in labels) + f"   ·   log {record['log_name']}",
             fontsize=11.5, color=INK_2, va="top")
    ax_cam = fig.add_subplot(outer[0, :2])
    camera = scene.frames[scene.scene_metadata.num_history_frames - 1].cameras.cam_f0
    if camera.image is not None:
        ax_cam.imshow(camera.image)
    ax_cam.axis("off")
    ax_text = fig.add_subplot(outer[0, 2:])
    ax_text.axis("off")
    ax_text.text(0.0, 0.95, summary_text(record, labels, mets), family="monospace", fontsize=11, color=INK, va="top")

    bev_axes = {label: (fig.add_subplot(outer[row, 0]), fig.add_subplot(outer[row, 1]))
                for row, label in enumerate(labels, start=1)}
    view, stretch = fit_view(bev_axes[labels[0]][0], trajectory_box(record, labels))  # shared by all four BEVs
    ctx = scene_context(metric_cache, view)
    for row, label in enumerate(labels, start=1):
        model, met = record["models"][label], mets[label]
        ax_true, ax_pred = bev_axes[label]
        draw_bev(ax_true, ctx, view, stretch, model, met, met["true"], record, details[label],
                 f"{label}: 64 proposals, colour = true PDMS")
        draw_bev(ax_pred, ctx, view, stretch, model, met, model["pred_pdms"], record, details[label],
                 f"{label}: colour = scorer predicted PDMS")
        profiles = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, 2], hspace=0.1)
        draw_profiles(fig.add_subplot(profiles[0]), fig.add_subplot(profiles[1]), model, met, record, met["true"])
        right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, 3], height_ratios=[1.25, 1], hspace=0.75)
        compare = GridSpecFromSubplotSpec(1, 2, subplot_spec=right[0], width_ratios=[1, 0.72], wspace=0.04)
        draw_comparison(fig.add_subplot(compare[0]), fig.add_subplot(compare[1]), model, met, details[label])
        draw_scatter(fig.add_subplot(right[1]), model, met)

    handles = [
        Line2D([], [], color=LINE_CMAP(0.75), linewidth=1.3, marker="o", markersize=3.5, label="64 proposals"),
        Line2D([], [], color=OUTPUT_COLOR, linewidth=2.4, marker="o", markersize=7, markeredgecolor=SURFACE,
               label="output (scorer rank 1)"),
        Line2D([], [], color=BEST_COLOR, linewidth=2.6, marker="D", markersize=7, markeredgecolor=SURFACE,
               label="best true of 64"),
        Line2D([], [], color=INK, linewidth=1.4, linestyle=(0, (1, 2)), marker="X", markersize=8,
               markeredgecolor=SURFACE, label="human GT"),
        Patch(facecolor=AGENT_FILL, edgecolor=INK_2, label="agents now"),
        Line2D([], [], color=INK_2, linewidth=0.8, alpha=0.5, label="agents at 1-4 s"),
        Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (6, 4)), label="route centreline"),
        Patch(facecolor=DRIVABLE, edgecolor=LANE_EDGE, label="drivable area"),
        Line2D([], [], color=OUTPUT_COLOR, linewidth=1.8, linestyle=(0, (3, 2)), label="ego at infraction"),
        Line2D([], [], color=CRITICAL, linewidth=2.0, label="object hit"),
    ]
    if len(record["multi_gt"]):
        handles.insert(4, Line2D([], [], color=MULTI_GT_COLOR, linewidth=1.6, linestyle=(0, (5, 3)),
                                 label="multi-GT targets"))
    legend_ax = fig.add_subplot(outer[3, :3])
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center left", ncol=6, frameon=False, fontsize=10, handlelength=2.2,
                     columnspacing=1.3)
    colorbar = fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=LINE_CMAP), cax=fig.add_subplot(outer[3, 3]),
                            orientation="horizontal")
    colorbar.set_label("PDMS, true or predicted (0 = fail, 1 = best)", color=INK_2, fontsize=10)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=9, colors=INK_2, length=0)
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    return mets


def render_all(args, records, labels):
    from hydra.utils import instantiate

    from navsim.agents.drivoR.proposal_analysis import compose_eval_config

    cfg = compose_eval_config()
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    summary_rows, index_entries = [], []
    for split, split_records in records.items():
        cache_paths = MetricCacheLoader(args.test_metric_cache if split == "test" else args.train_metric_cache) \
            .metric_cache_paths
        scene_loader = make_scene_loader(split, [r["log_name"] for r in split_records],
                                         [r["token"] for r in split_records], front_camera_config())
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for record in split_records:
            group, token = record["group"], record["token"]
            name = f"{GROUPS.index(group) if group in GROUPS else 9}_{group}_{token}"
            save_record(split_dir / f"{name}.npz", record, labels)
            with lzma.open(cache_paths[token], "rb") as stream:
                metric_cache = pickle.load(stream)
            image = f"{split}/{name}.png"
            mets = render_scene(args.output_dir / image, scene_loader.get_scene_from_token(token), record, labels,
                                metric_cache, simulator, scorer)
            for i, label in enumerate(labels):
                summary_rows.append({
                    "split": split, "group": group, "token": token, "log_name": record["log_name"], "model": label,
                    **{key: mets[label][key] for key in ("selected", "selected_pred", "selected_true", "best",
                                                         "best_true", "best_rank", "selected_true_rank", "num_good",
                                                         "spearman")},
                    "human_pdms": float(record["human_row"][SCORE]), "eval_csv_pdms": float(record["eval_csv"][i]),
                    "num_multi_gt": len(record["multi_gt"]), "image": image})
            caption = (f"[{split}] {GROUP_NAMES[group].format(a=labels[0], b=labels[1])} · {token} · output true PDMS "
                       + " vs ".join(f"{label} {mets[label]['selected_true']:.3f}" for label in labels)
                       + " · best of 64 " + " vs ".join(f"{mets[label]['best_true']:.3f}" for label in labels))
            index_entries.append({"split": split, "image": image, "caption": caption})
            print(f"[{split}] {name}: {caption}", flush=True)
    with open(args.output_dir / "summary.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_index(args.output_dir, index_entries, labels)


def write_index(output_dir, entries, labels):
    sections = []
    for split in SPLITS:
        items = [entry for entry in entries if entry["split"] == split]
        if items:
            sections.append(f"<h2>{split}</h2>\n" + "\n".join(
                f'<section><p>{html.escape(e["caption"])}</p><a href="{html.escape(e["image"])}">'
                f'<img src="{html.escape(e["image"])}" loading="lazy"></a></section>' for e in items))
    (output_dir / "index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>DrivoR proposals: {html.escape(' vs '.join(labels))}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#fcfcfb;color:#0b0b0b}"
        "section{margin:0 0 40px}p{color:#52514e}img{width:100%;max-width:2400px;border:1px solid #e1e0d9}</style>"
        f"<h1>DrivoR proposals: {html.escape(' vs '.join(labels))}</h1>\n" + "\n".join(sections) + "\n",
        encoding="utf-8")


def main():
    args = parse_args()
    if args.reuse:
        records, labels = load_records(args.reuse)
    else:
        labels = [args.label_a, args.label_b]
        records = compute_records(args, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {key: str(value) for key, value in vars(args).items()}, indent=2), encoding="utf-8")
    render_all(args, records, labels)
    print(f"Wrote {sum(len(r) for r in records.values())} scenes to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
