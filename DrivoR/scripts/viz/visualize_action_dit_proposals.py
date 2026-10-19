"""Compare DrivoR and Action-DiT checkpoints scene by scene: the 64 proposals, the scorer's pick and the best one.

Every model runs with the overrides of its navtest evaluation: DrivoR in FP32 (deterministic, so its outputs match its
evaluation CSV), Action-DiT in BF16 with 10 Euler steps. Action-DiT samples its 64 proposals from random noise; here
the noise is fixed per scene and shared by all Action-DiT models (proposal_analysis.flow_noise), so differences between
their columns come from the models and a re-run gives the same figures, while a scene's output can differ from the one
in the evaluation CSV. Every proposal and the human GT are scored with the official NAVSIM-v1 PDMS
(navsim.evaluate.pairwise_pdm_score).

One figure per scene, one column per model, every BEV on the same view (drawn from the metric cache as in
visualize_proposals_bev.py): the 64 proposals coloured by true PDMS with the output and the best of 64, x(t)/y(t),
output vs best per sub-metric, and predicted vs true PDMS of all 64.

The scenes come from a pool per split (navtest: half random, half scenes that some model failed in its evaluation
CSV; train: random navtrain scenes of a few training logs). All models are run and scored on the whole pool and the
shown scenes are picked from these results: the first column's family (DrivoR) better than the other family
(Action-DiT) and vice versa, a scorer missing a good proposal, a model without any good proposal, all good.
overview_<split>.png and summary.md describe the random part of the pool. --reuse <dir> re-renders an earlier run
without models.
"""

import argparse
import csv
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
from matplotlib import cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from omegaconf import OmegaConf

import visualize_proposals_bev as vpb
from navsim.agents.drivoR.multi_gt import MultiTrajectoryTargetBuilder
from navsim.agents.drivoR.proposal_analysis import (
    ACTION_DIT_OVERRIDES, SPLITS, front_camera_config, log_of, make_scene_loader, sample_navtrain_tokens)
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pairwise_pdm_score import pairwise_pdm_details

from visualize_proposals_bev import (  # palette and constants shared with the two-model figures
    AGENT_FILL, AXIS, BEST_COLOR, CRITICAL, DRIVABLE, INK, INK_2, LANE_EDGE, LINE_CMAP, MULTI_GT_COLOR, MUTED, NORM,
    OUTPUT_COLOR, SCORE, SURFACE)

EXP = vpb.DRIVOR_ROOT / "exp" / "ke"
CKPT = "checkpoints/epoch=24-step=33250.ckpt"
FRONT_VIEW = ["agent=drivoR_action_dit"] + ACTION_DIT_OVERRIDES
MULTIVIEW = ["agent=drivoR_action_dit_multiview", "agent.config.num_scene_tokens=16"] + ACTION_DIT_OVERRIDES
DINOV3 = ["agent=drivoR_action_dit_dinov3_vitl", "agent.config.multi_gt.enabled=false"] + ACTION_DIT_OVERRIDES
# Columns. overrides: on top of the README NAVSIM-v1 config, as in each evaluation command. colour: overview only;
# gray DrivoR, blue front-view Action-DiT, violet multiview Action-DiT, lighter = multi-GT.
MODELS = {
    "drivor": dict(
        family="DrivoR", variant="single-GT", overrides=[], color="#52514e",
        checkpoint=EXP / "training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/lightning_logs/version_2" / CKPT,
        eval_dir=EXP / "eval_navtest_training_drivoR_Nav1_navtrain_traj_long_25epochs_multinode_version_2_epoch_24-"
        "step_33250/20261015_114507"),
    "drivor_multigt": dict(
        family="DrivoR", variant="multi-GT", overrides=[], color="#b4b3ac",
        checkpoint=EXP / "training_drivoR_Nav1_navtrain_multigt/drivor_multigt_seed2_run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_training_drivoR_Nav1_navtrain_multigt_drivor_multigt_seed2_run1_version_0_epoch_24-"
        "step_33250"),
    "front": dict(
        family="Action-DiT", variant="front-view", overrides=FRONT_VIEW, color="#256abf",
        checkpoint=EXP / "training_drivoR_action_dit_navtrain/multinode/lightning_logs/version_3" / CKPT,
        eval_dir=EXP / "eval_navtest_drivor_action_dit_navtrain_v3_epoch24"),
    "front_multigt": dict(
        family="Action-DiT", variant="front-view multi-GT", overrides=FRONT_VIEW, color="#86b6ef",
        checkpoint=EXP / "training_drivoR_action_dit_navtrain_multigt/action_dit_multigt_seed2_run1/lightning_logs/"
        "version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_drivor_action_dit_multigt_seed2_epoch24"),
    "multiview": dict(
        family="Action-DiT", variant="multiview", overrides=MULTIVIEW, color="#4a3aa7",
        checkpoint=EXP / "training_drivoR_action_dit_multiview_navtrain/action_dit_multiview_q16_seed2_run1/"
        "lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_drivor_action_dit_multiview_q16_seed2_epoch24"),
    "multiview_multigt": dict(
        family="Action-DiT", variant="multiview multi-GT", overrides=MULTIVIEW, color="#9d92d8",
        checkpoint=EXP / "training_drivoR_action_dit_multiview_navtrain_multigt/action_dit_multiview_multigt_seed2_run1/"
        "lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_drivor_action_dit_multiview_multigt_seed2_epoch24"),
    # V-JEPA 2 / 2.1 ViT-L (LoRA, front camera, native patches). Both evaluated bf16-mixed; the Action-DiT ones were
    # trained with multi-GT supervision (their agent configs inherit drivoR_action_dit_multigt).
    "drivor_vjepa2": dict(
        family="DrivoR", variant="V-JEPA 2", overrides=["agent=drivoR_vjepa2_vitl"], color="#52514e", bf16=True,
        checkpoint=EXP / "training_drivoR_vjepa2_vitl_navtrain/drivor_vjepa2_vitl_seed2_run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_drivoR_vjepa2_vitl/drivor_vjepa2_vitl_seed2_run1"),
    "drivor_vjepa2_1": dict(
        family="DrivoR", variant="V-JEPA 2.1", overrides=["agent=drivoR_vjepa2_1_vitl"], color="#b4b3ac", bf16=True,
        checkpoint=EXP / "training_drivoR_vjepa2_1_vitl_navtrain/drivor_vjepa2_1_vitl_seed2_run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_drivoR_vjepa2_1_vitl/drivor_vjepa2_1_vitl_seed2_run1"),
    "front_vjepa2": dict(
        family="Action-DiT", variant="V-JEPA 2 multi-GT", color="#256abf",
        overrides=["agent=drivoR_action_dit_vjepa2_vitl", "agent.config.multi_gt.enabled=false"] + ACTION_DIT_OVERRIDES,
        checkpoint=EXP / "training_drivoR_action_dit_vjepa2_vitl_navtrain_multigt/action_dit_vjepa2_vitl_seed2_train_eval_"
        "run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_action_dit_vjepa2_vitl"),
    "front_vjepa2_1": dict(
        family="Action-DiT", variant="V-JEPA 2.1 multi-GT", color="#86b6ef",
        overrides=["agent=drivoR_action_dit_vjepa2_1_vitl", "agent.config.multi_gt.enabled=false"] + ACTION_DIT_OVERRIDES,
        checkpoint=EXP / "training_drivoR_action_dit_vjepa2_1_vitl_navtrain_multigt/action_dit_vjepa2_1_vitl_seed2_train_"
        "eval_run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_action_dit_vjepa2_1_vitl"),
    # Action-DiT DINOv3 ViT-L/16 (LoRA, front camera, native patches), multi-GT top4; the variants add one training
    # signal: CLOVER pseudo-expert coverage, poor4 trajectories merged into the multi-GT targets, the CLOVER bank
    # scorer loss. The baseline was evaluated bf16-mixed and the variants in FP32; here all four run in FP32.
    "dinov3": dict(
        family="DINOv3 ViT-L", variant="baseline", color="#52514e", bf16=False, overrides=DINOV3,
        checkpoint=EXP / "training_drivoR_action_dit_dinov3_vitl_navtrain_multigt/action_dit_dinov3_vitl_seed2_train_eval_"
        "run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_action_dit_dinov3_vitl"),
    "dinov3_pseudo_expert": dict(
        family="DINOv3 ViT-L", variant="+ pseudo-expert", color="#256abf", bf16=False, overrides=DINOV3,
        checkpoint=EXP / "training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert/action_dit_dinov3_vitl_"
        "multigt_pseudo_expert_seed2_train_eval_run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_action_dit_dinov3_vitl_multigt_pseudo_expert"),
    "dinov3_poor": dict(
        family="DINOv3 ViT-L", variant="+ poor GT", color="#4a3aa7", bf16=False, overrides=DINOV3,
        checkpoint=EXP / "training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_poor/action_dit_dinov3_vitl_multigt_"
        "poor_seed2_train_eval_run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_action_dit_dinov3_vitl_multigt_poor"),
    "dinov3_bank": dict(
        family="DINOv3 ViT-L", variant="+ bank", color="#86b6ef", bf16=False, overrides=DINOV3,
        checkpoint=EXP / "training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_bank/action_dit_dinov3_vitl_multigt_"
        "bank_seed2_train_eval_run1/lightning_logs/version_0" / CKPT,
        eval_dir=EXP / "eval_navtest_action_dit_dinov3_vitl_multigt_bank"),
}
GROUP_ORDER = ("ref_better", "other_better", "disagree", "scorer_miss", "low_coverage", "all_good")
STAT_KEYS = ("selected_true", "selected_pred", "best_true", "num_good", "spread_lat_m", "spread_long_m", "min_ade_m")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS)[:6], choices=list(MODELS), help="columns, in order")
    parser.add_argument("--per-group", type=int, default=2, help="scenes per group and split")
    parser.add_argument("--test-pool", type=int, default=160, help="navtest scenes run and scored to pick from")
    parser.add_argument("--train-pool", type=int, default=160, help="navtrain scenes run and scored to pick from")
    parser.add_argument("--train-logs", type=int, default=6, help="random training logs of the train pool")
    parser.add_argument("--test-tokens", nargs="*", help="explicit navtest tokens instead of a pool")
    parser.add_argument("--train-tokens", nargs="*", help="explicit navtrain tokens instead of a pool")
    parser.add_argument("--splits", nargs="+", default=["test", "train"], choices=list(SPLITS))
    parser.add_argument("--test-metric-cache", type=Path, default=vpb.PROJECT_ROOT / "data/metric_cache_navtest")
    parser.add_argument("--train-metric-cache", type=Path, default=vpb.PROJECT_ROOT / "data/metric_cache_navtrain")
    parser.add_argument("--multi-gt-root", type=Path, default=Path(os.environ.get(
        "DRIVOR_MULTI_GT_ROOT", vpb.PROJECT_ROOT / "multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu")))
    parser.add_argument("--output-dir", type=Path, help="default: exp/viz/action_dit_proposals_<time>, or --reuse")
    parser.add_argument("--reuse", type=Path, help="re-render an earlier run (no models, no scoring)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--score-workers", type=int, default=min(32, len(os.sched_getaffinity(0))))
    parser.add_argument("--seed", type=int, default=0, help="pool sampling and the flow noise of every scene")
    args = vpb.resolve_caller_paths(parser.parse_args())
    if args.output_dir is None:
        args.output_dir = args.reuse or vpb.DRIVOR_ROOT / "exp/viz" / f"action_dit_proposals_{datetime.now():%Y%m%d_%H%M%S}"
    return args


# ----------------------------------------------------------------------------- pool and scene selection

def test_pool(eval_scores, cache_paths, size, rng):
    """Half random navtest scenes, half scenes some model failed (< 0.5) in its evaluation; (tokens, random ones)."""
    tokens = sorted(set(cache_paths).intersection(*eval_scores.values()))
    random_part = rng.sample(tokens, min(size // 2, len(tokens)))
    taken = set(random_part)
    failed = [t for t in tokens if t not in taken and min(scores[t] for scores in eval_scores.values()) < 0.5]
    return sorted(random_part + rng.sample(failed, min(size - len(random_part), len(failed)))), taken


def proposal_stats(model, rows, human):
    """The output, the best of 64 and the spread of the 64 proposals of one model in one scene."""
    true = rows[:, SCORE]
    selected = int(model["selected"])
    ends = model["proposals"][:, -1, :2]
    ade = np.linalg.norm(model["proposals"][:, :, :2] - human[None, :, :2], axis=-1).mean(axis=1)
    return {"selected_true": float(true[selected]), "selected_pred": float(model["pred_pdms"][selected]),
            "best_true": float(np.nanmax(true)), "num_good": int(np.sum(true >= 0.9)),
            "spread_lat_m": float(ends[:, 1].std()), "spread_long_m": float(ends[:, 0].std()),
            "min_ade_m": float(ade.min())}


def group_title(group, families):
    ref = families[0]
    other = next((family for family in families if family != ref), ref)
    base = group.removesuffix("_closest")
    title = {"ref_better": f"{ref} good, most {other} fail", "other_better": f"{other} good, {ref} fails",
             "disagree": "models disagree", "scorer_miss": "a scorer misses a good proposal",
             "low_coverage": "a model has no good proposal", "all_good": "all models good",
             "manual": "manual pick"}[base]
    return title if base == group else f"closest to '{title}' (no pool scene meets it)"


def file_stem(record):
    base = record["group"].removesuffix("_closest")
    return f"{GROUP_ORDER.index(base) if base in GROUP_ORDER else 9}_{record['group']}_{record['token']}"


def pick_groups(stats, families, per_group, rng):
    """Random scenes per group from the pool results; except 'all_good', missing ones are the closest scenes
    (group '<name>_closest').

    With two families (columns of the first family = reference, e.g. DrivoR): 'ref_better' = every reference output
    >= 0.8 while the other family's outputs average < 0.5, 'other_better' the reverse. One family: 'disagree'.
    """
    tokens = sorted(stats)
    sel = {t: np.array([s["selected_true"] for s in stats[t]]) for t in tokens}
    best = {t: np.array([s["best_true"] for s in stats[t]]) for t in tokens}
    ref = np.array([family == families[0] for family in families])
    rules = {}
    if not ref.all():
        rules["ref_better"] = (lambda t: sel[t][ref].min() >= 0.8 and sel[t][~ref].mean() < 0.5,
                               lambda t: sel[t][~ref].mean() - sel[t][ref].min())
        rules["other_better"] = (lambda t: sel[t][~ref].min() >= 0.8 and sel[t][ref].mean() < 0.5,
                                 lambda t: sel[t][ref].mean() - sel[t][~ref].min())
    else:
        rules["disagree"] = (lambda t: np.ptp(sel[t]) >= 0.3, lambda t: -np.ptp(sel[t]))
    rules["scorer_miss"] = (lambda t: np.max(best[t] - sel[t]) >= 0.5, lambda t: -np.max(best[t] - sel[t]))
    rules["low_coverage"] = (lambda t: np.min(best[t]) < 0.8, lambda t: np.min(best[t]))
    rules["all_good"] = (lambda t: np.min(sel[t]) >= 0.9, lambda t: -np.min(sel[t]))
    picked, used = [], set()
    for group, (accept, order) in rules.items():
        candidates = [t for t in tokens if t not in used and accept(t)]
        chosen = rng.sample(candidates, min(per_group, len(candidates)))
        picked += [(group, t) for t in chosen]
        used.update(chosen)
        if len(chosen) < per_group and group != "all_good":
            rest = sorted((t for t in tokens if t not in used), key=order)[:per_group - len(chosen)]
            picked += [(f"{group}_closest", t) for t in rest]
            used.update(rest)
    return picked


def compute_records(args, specs):
    """Runs every model on each split's pool, scores all proposals; returns ({split: [record]}, pool rows)."""
    import torch
    from hydra.utils import instantiate

    from navsim.agents.drivoR.proposal_analysis import (
        compose_eval_config, load_agent, run_proposals, union_sensor_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = compose_eval_config()
    vpb._SIMULATOR, vpb._SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)  # inherited by the pool
    labels = [spec["label"] for spec in specs]
    families = [spec["family"] for spec in specs]
    agents = {spec["label"]: load_agent(compose_eval_config(spec["overrides"]), spec["checkpoint"], device)
              for spec in specs}
    sensor_config = union_sensor_config(agents)
    multi_gt_loader = MultiTrajectoryTargetBuilder(OmegaConf.create(
        {"root": str(args.multi_gt_root), "max_gt": 4, "missing_policy": "original"}))
    pool = ProcessPoolExecutor(max_workers=args.score_workers, mp_context=multiprocessing.get_context("fork"))

    records, pool_rows = {}, []
    for split in args.splits:
        rng = random.Random(f"{args.seed}-{split}")  # same picks whether or not the other split runs
        os.environ["NAVSIM_OSS_SENSOR_BLOBS_PATH"] = SPLITS[split]["oss"]  # read at image-load time
        cache_paths = MetricCacheLoader(args.test_metric_cache if split == "test" else args.train_metric_cache) \
            .metric_cache_paths
        eval_scores = {spec["label"]: vpb.read_eval_scores(spec["eval_dir"]) for spec in specs} \
            if split == "test" else None
        explicit = args.test_tokens if split == "test" else args.train_tokens
        if explicit:
            tokens, randoms = list(explicit), set()
        elif split == "test":
            tokens, randoms = test_pool(eval_scores, cache_paths, args.test_pool, rng)
        else:
            _, tokens = sample_navtrain_tokens(cache_paths, args.train_logs, args.train_pool, rng)
            randoms = set(tokens)
        loader = make_scene_loader(split, [log_of(cache_paths, t) for t in tokens], tokens, sensor_config)
        tokens = [t for t in tokens if t in loader.scene_frames_dicts]
        print(f"[{split}] running {len(labels)} models on {len(tokens)} scenes", flush=True)
        outputs = run_proposals(agents, loader, tokens, device, args.batch_size, args.loader_workers, args.seed,
                                {spec["label"]: spec["bf16"] for spec in specs if "bf16" in spec})
        # One task per scene: 64 proposals of every model in column order, then the human GT.
        tasks = [(cache_paths[t], np.concatenate([outputs[label][t]["proposals"] for label in labels]
                                                 + [outputs[labels[0]][t]["human"][None]])) for t in tokens]
        print(f"[{split}] scoring {len(tasks)} scenes x {64 * len(labels) + 1} trajectories", flush=True)
        rows = dict(zip(tokens, pool.map(vpb._score_task, tasks)))
        stats = {}
        for t in tokens:
            human = outputs[labels[0]][t]["human"]
            stats[t] = [proposal_stats(outputs[label][t], rows[t][64 * i:64 * (i + 1)], human)
                        for i, label in enumerate(labels)]
            pool_rows += [{"split": split, "token": t, "log_name": log_of(cache_paths, t), "random": t in randoms,
                           "model": label, **stats[t][i], "human_pdms": float(rows[t][-1, SCORE]),
                           "eval_csv_pdms": eval_scores[label].get(t, np.nan) if eval_scores else np.nan}
                          for i, label in enumerate(labels)]

        picked = [("manual", t) for t in tokens] if explicit else pick_groups(stats, families, args.per_group, rng)
        records[split] = []
        for group, t in picked:
            multi_gt = np.zeros((0, 8, 3), dtype=np.float32)
            if split == "train":
                loaded = multi_gt_loader.load(t, outputs[labels[0]][t]["human"])
                if not bool(loaded["multi_gt_fallback"]):
                    multi_gt = loaded["trajectory_multi_gt"][loaded["trajectory_multi_gt_valid"]].numpy()
            models = {}
            for i, label in enumerate(labels):
                out = outputs[label][t]
                models[label] = {key: out[key] for key in ("proposals", "head_probs", "pdm_score", "pred_pdms")}
                models[label]["selected"] = int(out["selected"])
                models[label]["pdm_rows"] = rows[t][64 * i:64 * (i + 1)]
            records[split].append({
                "split": split, "group": group, "token": t, "log_name": log_of(cache_paths, t),
                "eval_csv": [eval_scores[label].get(t, np.nan) if eval_scores else np.nan for label in labels],
                "human": outputs[labels[0]][t]["human"], "human_row": rows[t][-1], "multi_gt": multi_gt,
                "models": models})
    pool.shutdown()
    return records, pool_rows


def write_pool(path, pool_rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pool_rows[0]))
        writer.writeheader()
        writer.writerows(pool_rows)


def read_pool(path):
    with open(path) as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["random"] = row["random"] == "True"
        for key in STAT_KEYS + ("human_pdms", "eval_csv_pdms"):
            row[key] = float(row[key])
    return rows


# ----------------------------------------------------------------------------- figures

def summary_text(record, labels, families, mets):
    width = 22
    stats = {label: proposal_stats(record["models"][label], record["models"][label]["pdm_rows"], record["human"])
             for label in labels}
    rows = [
        ("output true PDMS", lambda m, s: f"{m['selected_true']:.3f}"),
        ("output predicted PDMS", lambda m, s: f"{m['selected_pred']:.3f}"),
        ("best of 64 (true)", lambda m, s: f"{m['best_true']:.3f}"),
        ("best's scorer rank", lambda m, s: f"{m['best_rank']}/64"),
        ("output's true rank", lambda m, s: f"{m['selected_true_rank']}/64"),
        ("proposals PDMS >= 0.9", lambda m, s: f"{m['num_good']}/64"),
        ("Spearman(pred, true)", lambda m, s: vpb.fmt_rho(m["spearman"])),
        ("4 s endpoint spread", lambda m, s: f"lat {s['spread_lat_m']:.1f} long {s['spread_long_m']:.1f} m"),
        ("closest to human, ADE", lambda m, s: f"{s['min_ade_m']:.2f} m"),
    ]
    variants = [label[len(family):].strip() or label for label, family in zip(labels, families)]
    lines = [f"{'':24}" + "".join(f"{family:>{width}}" for family in families),
             f"{'':24}" + "".join(f"{variant:>{width}}" for variant in variants)]
    lines += [f"{name:24}" + "".join(f"{fmt(mets[label], stats[label]):>{width}}" for label in labels)
              for name, fmt in rows]
    if np.isfinite(record["eval_csv"]).all():
        lines.append(f"{'navtest eval CSV':24}" + "".join(f"{value:>{width}.3f}" for value in record["eval_csv"]))
    lines += ["", f"human GT PDMS {record['human_row'][SCORE]:.3f}"
              + (f"   ·   multi-GT training targets: {len(record['multi_gt'])}" if len(record["multi_gt"]) else ""),
              "Action-DiT columns start from the same noise in this scene (the evaluation used other noise); "
              "DrivoR is deterministic.",
              "Spread: std of the 64 endpoints at 4 s. ADE: mean distance of the closest proposal to the human GT."]
    return "\n".join(lines)


def legend_handles(record):
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
    return handles


def label_families(fig, bev_axes, families):
    """Family name over each run of adjacent columns of the same family."""
    spans = []
    for column, family in enumerate(families):
        if spans and spans[-1][0] == family:
            spans[-1][2] = column
        else:
            spans.append([family, column, column])
    if len(spans) < 2:
        return
    for family, first, last in spans:
        left, right = bev_axes[first].get_position().x0, bev_axes[last].get_position().x1
        top = bev_axes[first].get_position().y1 + 0.016
        fig.add_artist(Line2D([left, right], [top, top], transform=fig.transFigure, color=AXIS, linewidth=1.5))
        fig.text((left + right) / 2, top + 0.003, family, ha="center", va="bottom", fontsize=16, fontweight="semibold",
                 color=INK)


def render_scene(path, scene, record, labels, families, titles, metric_cache, simulator, scorer):
    mets = {label: vpb.model_metrics(record["models"][label]) for label in labels}
    details = {}
    for label in labels:
        model, met = record["models"][label], mets[label]
        indices = sorted({met["selected"], met["best"]})
        rows, found = pairwise_pdm_details(metric_cache, model["proposals"][indices], simulator, scorer)
        if not np.allclose(rows, model["pdm_rows"][indices], atol=1e-9, equal_nan=True):
            raise RuntimeError(f"{record['token']} {label}: re-scoring differs from the stored PDMS")
        details[label] = dict(zip(indices, found))

    n = len(labels)
    fig = plt.figure(figsize=(7.2 * n, 34.5), facecolor=SURFACE)
    # Row 4 is empty: room for the infraction / progress notes under the sub-metric comparison.
    outer = GridSpec(7, n, figure=fig, height_ratios=[4.6, 12, 5.4, 5.6, 0.3, 4.8, 0.45], left=0.035, right=0.99,
                     top=0.952, bottom=0.02, hspace=0.42, wspace=0.2)
    fig.text(0.035, 0.988, f"{record['split']} · {record['token']} · {group_title(record['group'], families)}",
             fontsize=19, fontweight="semibold", color=INK, va="top")
    fig.text(0.035, 0.975, "epoch 24  ·  output = proposal the scorer ranks first, best = highest official PDMS of "
             f"the 64  ·  log {record['log_name']}", fontsize=12.5, color=INK_2, va="top")
    ax_cam = fig.add_subplot(outer[0, :n // 2])
    camera = scene.frames[scene.scene_metadata.num_history_frames - 1].cameras.cam_f0
    if camera.image is not None:
        ax_cam.imshow(camera.image)
    ax_cam.axis("off")
    ax_text = fig.add_subplot(outer[0, n // 2:])
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, summary_text(record, labels, families, mets), family="monospace", fontsize=10.5,
                 color=INK, va="top")

    bev_axes = [fig.add_subplot(outer[1, column]) for column in range(n)]
    label_families(fig, bev_axes, families)
    view, stretch = vpb.fit_view(bev_axes[0], vpb.trajectory_box(record, labels))  # shared by every column
    ctx = vpb.scene_context(metric_cache, view)
    for column, label in enumerate(labels):
        model, met = record["models"][label], mets[label]
        vpb.draw_bev(bev_axes[column], ctx, view, stretch, model, met, met["true"], record, details[label],
                     titles[label])
        profiles = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2, column], hspace=0.1)
        ax_x = fig.add_subplot(profiles[0])
        vpb.draw_profiles(ax_x, fig.add_subplot(profiles[1]), model, met, record, met["true"])
        ax_x.set_title("x(t) and y(t) of all 64", fontsize=11.5, color=INK, loc="left")
        compare = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3, column], width_ratios=[1, 0.8], wspace=0.04)
        ax_compare = fig.add_subplot(compare[0])
        vpb.draw_comparison(ax_compare, fig.add_subplot(compare[1]), model, met, details[label])
        ax_compare.set_title("Output vs best of 64, per sub-metric\n(red: predicted and true differ by > 0.3)",
                             fontsize=11.5, color=INK, loc="left", pad=34)
        vpb.draw_scatter(fig.add_subplot(outer[5, column]), model, met)

    legend_ax = fig.add_subplot(outer[6, :n - 1])
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_handles(record), loc="center left", ncol=6, frameon=False, fontsize=10.5,
                     handlelength=2.2, columnspacing=1.3)
    colorbar = fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=LINE_CMAP), cax=fig.add_subplot(outer[6, n - 1]),
                            orientation="horizontal")
    colorbar.set_label("true PDMS of a proposal (0 = fail, 1 = best)", color=INK_2, fontsize=10.5)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=9, colors=INK_2, length=0)
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    return mets


def severe_miss(row):
    """The scorer picks a failing proposal although a good one exists."""
    return row["selected_true"] < 0.5 and row["best_true"] >= 0.9


def draw_overview(path, split, rows, labels, colors):
    data = {label: [row for row in rows if row["model"] == label] for label in labels}
    scenes = len(data[labels[0]])
    x = np.arange(len(labels))
    names = [label.replace(" ", "\n", 1) for label in labels]
    fig, axes = plt.subplots(2, 3, figsize=(31, 13), facecolor=SURFACE, gridspec_kw=dict(wspace=0.2, hspace=0.42))
    axes = axes.ravel()
    ax = axes[0]
    for offset, key, alpha in ((-0.2, "selected_true", 1.0), (0.2, "best_true", 0.4)):
        means = [100 * np.nanmean([row[key] for row in data[label]]) for label in labels]
        bars = ax.bar(x + offset, means, width=0.38, color=colors, alpha=alpha, edgecolor=INK_2, linewidth=0.6)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in means], fontsize=8.5, color=INK_2, padding=2)
    ax.legend(handles=[Patch(facecolor=MUTED, edgecolor=INK_2, label="output"),
                       Patch(facecolor=MUTED, alpha=0.4, edgecolor=INK_2, label="best of 64")],
              loc="upper left", ncol=2, fontsize=9, frameon=False)
    ax.set_ylim(0, 114)
    ax.set_title("Mean PDMS: output vs best of 64", fontsize=12, color=INK, loc="left")
    ax = axes[1]
    shares = [100 * np.mean([severe_miss(row) for row in data[label]]) for label in labels]
    bars = ax.bar(x, shares, width=0.6, color=colors, edgecolor=INK_2, linewidth=0.6)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in shares], fontsize=8.5, color=INK_2, padding=2)
    ax.set_ylim(0, max(shares + [1.0]) * 1.25)
    ax.set_ylabel("% of scenes", fontsize=9.5, color=INK_2)
    ax.set_title("Scorer picks a failing proposal (< 0.5) although one ≥ 0.9 exists", fontsize=12, color=INK,
                 loc="left")
    panels = (("num_good", "Proposals with PDMS ≥ 0.9", "count of 64"),
              ("spread_lat_m", "Lateral spread of the 64 endpoints", "std of y at 4 s [m]"),
              ("spread_long_m", "Longitudinal spread of the 64 endpoints", "std of x at 4 s [m]"),
              ("min_ade_m", "Closest proposal to the human GT", "min ADE over 64 [m]"))
    for ax, (key, title, ylabel) in zip(axes[2:], panels):
        values = [[row[key] for row in data[label]] for label in labels]
        boxes = ax.boxplot(values, positions=x, widths=0.55, patch_artist=True, showfliers=False,
                           medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK_2),
                           capprops=dict(color=INK_2))
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(INK_2)
        means = [np.mean(value) for value in values]
        ax.scatter(x, means, marker="D", s=36, color=SURFACE, edgecolors=INK, linewidths=1.2, zorder=4)
        for xi, mean in zip(x, means):
            ax.annotate(f"{mean:.2f}" if key != "num_good" else f"{mean:.1f}", (xi, mean), xytext=(8, 0),
                        textcoords="offset points", fontsize=8.5, color=INK, va="center")
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
    for ax in axes:
        vpb.style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9.5, color=INK)
    fig.suptitle(f"{split}: {scenes} random scenes (Action-DiT columns from the same noise per scene)   ·   "
                 "box = quartiles, whiskers = 1.5 IQR (outliers hidden), ◇ = mean", x=0.012, y=0.95, ha="left",
                 fontsize=13.5, color=INK)
    fig.savefig(path, dpi=100, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_summary(path, pool_rows, labels, titles):
    lines = ["# DrivoR 与 Action-DiT：64 条候选的分布", "",
             "统计只用池里的随机场景（测试池另一半按评测 CSV 挑的失败场景不计入）。PDMS ×100；"
             "严重错选 = 输出 < 0.5 而 64 条里有 ≥ 0.9 的；spread 为 64 条轨迹 4 s 终点的标准差；"
             "min ADE 为离人类轨迹最近的一条的平均位移误差。Action-DiT 每个场景用同一份初始噪声（与评测时不同），"
             "DrivoR 是确定性的，重跑应与评测 CSV 一致。", ""]
    for split in SPLITS:
        rows = [row for row in pool_rows if row["split"] == split and row["random"]]
        if not rows:
            continue
        test = split == "test"
        lines += [f"## {split}（{len(rows) // len(labels)} 个随机场景）", "",
                  "| 模型 | 输出 PDMS | 64 条最优 | 差距 | 严重错选 | PDMS≥0.9 条数 | 横向 spread | 纵向 spread | min ADE |"
                  + (" 同场景评测 CSV | 重跑与 CSV 差>0.3 |" if test else ""),
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|" if test else "")]
        for label in labels:
            r = [row for row in rows if row["model"] == label]
            sel, best = (100 * np.nanmean([row[key] for row in r]) for key in ("selected_true", "best_true"))
            line = (f"| {titles[label]} | {sel:.2f} | {best:.2f} | {best - sel:+.2f} | "
                    f"{np.mean([severe_miss(row) for row in r]):.1%} | {np.mean([row['num_good'] for row in r]):.1f} | "
                    f"{np.mean([row['spread_lat_m'] for row in r]):.2f} m | "
                    f"{np.mean([row['spread_long_m'] for row in r]):.2f} m | {np.mean([row['min_ade_m'] for row in r]):.2f} m |")
            if test:
                diff = np.array([row["selected_true"] - row["eval_csv_pdms"] for row in r])
                line += f" {100 * np.nanmean([row['eval_csv_pdms'] for row in r]):.2f} | {np.mean(np.abs(diff) > 0.3):.1%} |"
            lines.append(line)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_all(args, records, labels, meta, pool_rows):
    from hydra.utils import instantiate

    from navsim.agents.drivoR.proposal_analysis import compose_eval_config

    titles, families, colors = meta["titles"], meta["families"], meta["colors"]
    cfg = compose_eval_config()
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    names = {}
    for split, split_records in records.items():  # all .npz first: a failed render can be redone with --reuse
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)
        for record in split_records:
            names[split, record["token"]] = file_stem(record)
            vpb.save_record(args.output_dir / split / f"{names[split, record['token']]}.npz", record, labels)
    summary_rows, entries = [], []
    for split, split_records in records.items():
        random_rows = [row for row in pool_rows if row["split"] == split and row["random"]]
        if random_rows:
            draw_overview(args.output_dir / f"overview_{split}.png", split, random_rows, labels, colors)
            entries.append({"split": split, "image": f"overview_{split}.png",
                            "caption": f"[{split}] overview of {len(random_rows) // len(labels)} random pool scenes"})
        cache_paths = MetricCacheLoader(args.test_metric_cache if split == "test" else args.train_metric_cache) \
            .metric_cache_paths
        scene_loader = make_scene_loader(split, [r["log_name"] for r in split_records],
                                         [r["token"] for r in split_records], front_camera_config())
        for record in split_records:
            group, token = record["group"], record["token"]
            name = names[split, token]
            with lzma.open(cache_paths[token], "rb") as stream:
                metric_cache = pickle.load(stream)
            image = f"{split}/{name}.png"
            mets = render_scene(args.output_dir / image, scene_loader.get_scene_from_token(token), record, labels,
                                families, titles, metric_cache, simulator, scorer)
            for i, label in enumerate(labels):
                summary_rows.append({
                    "split": split, "group": group, "token": token, "log_name": record["log_name"], "model": label,
                    **{key: mets[label][key] for key in ("selected", "selected_pred", "selected_true", "best",
                                                         "best_true", "best_rank", "selected_true_rank", "num_good",
                                                         "spearman")},
                    "human_pdms": float(record["human_row"][SCORE]), "eval_csv_pdms": float(record["eval_csv"][i]),
                    "num_multi_gt": len(record["multi_gt"]), "image": image})
            caption = (f"[{split}] {group_title(group, families)} · {token} · output true PDMS "
                       + " / ".join(f"{mets[label]['selected_true']:.3f}" for label in labels)
                       + " · best of 64 " + " / ".join(f"{mets[label]['best_true']:.3f}" for label in labels)
                       + f" ({' / '.join(labels)})")
            entries.append({"split": split, "image": image, "caption": caption})
            print(f"[{split}] {name}: {caption}", flush=True)
    with open(args.output_dir / "summary.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_summary(args.output_dir / "summary.md", pool_rows, labels, titles)
    vpb.write_index(args.output_dir, entries, labels)


def main():
    args = parse_args()
    if args.reuse:
        records, labels = vpb.load_records(args.reuse)
        meta = json.loads((args.reuse / "models.json").read_text(encoding="utf-8"))
        # Runs before the DrivoR columns stored titles only: one family, default colours.
        meta.setdefault("families", [""] * len(labels))
        meta.setdefault("colors", [MODELS[key]["color"] for key in list(MODELS)[-len(labels):]])
        pool_rows = read_pool(args.reuse / "pool.csv")
    else:
        specs = [dict(MODELS[key], label=f"{MODELS[key]['family']} {MODELS[key]['variant']}") for key in args.models]
        labels = [spec["label"] for spec in specs]
        meta = {
            "titles": {spec["label"]: f"{spec['label']} · navtest "
                       f"{100 * np.mean(list(vpb.read_eval_scores(spec['eval_dir']).values())):.2f}" for spec in specs},
            "families": [spec["family"] for spec in specs],
            "colors": [spec["color"] for spec in specs],
            "models": {spec["label"]: {"checkpoint": str(spec["checkpoint"]), "eval_dir": str(spec["eval_dir"]),
                                       "overrides": spec["overrides"]} for spec in specs},
        }
        records, pool_rows = compute_records(args, specs)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_pool(args.output_dir / "pool.csv", pool_rows)
        (args.output_dir / "models.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {key: str(value) for key, value in vars(args).items()}, indent=2), encoding="utf-8")
    render_all(args, records, labels, meta, pool_rows)
    print(f"Wrote {sum(len(r) for r in records.values())} scenes to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
