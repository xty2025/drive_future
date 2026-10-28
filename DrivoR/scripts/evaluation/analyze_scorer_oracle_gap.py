"""Full-navtest gap between DrivoR's selected trajectory and the best of its own 64 proposals.

Each model runs once per navtest scene. All 64 proposals and the human GT are scored with the
official NAVSIM-v1 PDMS (navsim.evaluate.pairwise_pdm_score, identical to pdm_score()).
"Best of 64" is what a perfect scorer would reach with the same proposals, so the gap to the
selected trajectory bounds what any better scorer or re-ranker can gain.

--split train runs the same on navtrain scenes of random training logs (what the models train on),
e.g. to compare label balance and head quality between training and test data.

Stage 1 (compute) writes arrays.npz; stage 2 (report) writes scenes.csv, summary.json/.md and
oracle_gap_summary.png. --report-only <dir> reruns stage 2 from an existing arrays.npz.
"""

import argparse
import csv
import json
import lzma
import multiprocessing
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DRIVOR_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DRIVOR_ROOT.parent
TOPK = (1, 2, 3, 4, 5, 8, 16, 32, 64)
GAP_BUCKETS = ((0.0, 0.01, "<= 0.01"), (0.01, 0.1, "0.01-0.1"), (0.1, 0.5, "0.1-0.5"), (0.5, 1.01, "> 0.5"))

# Chart colors: models take categorical slots 1-2 (validated with the dataviz validator).
MODEL_COLORS = ("#2a78d6", "#eb6834")
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

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
    parser.add_argument("--split", choices=["test", "train"], default="test",
                        help="test: navtest; train: navtrain scenes of random training logs (what the models train on)")
    parser.add_argument("--metric-cache", type=Path, default=PROJECT_ROOT / "data/metric_cache_navtest")
    parser.add_argument("--train-metric-cache", type=Path, default=PROJECT_ROOT / "data/metric_cache_navtrain")
    parser.add_argument("--train-logs", type=int, default=40, help="random training logs sampled (--split train)")
    parser.add_argument("--max-scenes", type=int, default=None, help="random subset (test) or cap (train)")
    parser.add_argument("--output-dir", type=Path, help="default: exp/analysis/scorer_oracle_gap_<split>_<time>")
    parser.add_argument("--report-only", type=Path, help="existing output dir; rerun the report from arrays.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loader-workers", type=int, default=12)
    parser.add_argument("--score-workers", type=int, default=min(28, len(os.sched_getaffinity(0))))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DRIVOR_ROOT / "exp/analysis" / f"scorer_oracle_gap_{args.split}_{datetime.now():%Y%m%d_%H%M%S}"
    return args


def _score_task(task):
    from navsim.evaluate.pairwise_pdm_score import pairwise_pdm_scores

    cache_path, trajectories = task
    with lzma.open(cache_path, "rb") as stream:
        metric_cache = pickle.load(stream)
    return pairwise_pdm_scores(metric_cache, trajectories, _SIMULATOR, _SCORER)


def compute(args):
    import torch
    from hydra.utils import instantiate

    from navsim.agents.drivoR.proposal_analysis import (
        HEADS, SPLITS, compose_eval_config, iterate_proposals, load_agent, log_of, make_scene_loader,
        sample_navtrain_tokens)
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pairwise_pdm_score import PDM_FIELDS

    global _SIMULATOR, _SCORER
    labels = [args.label_a, args.label_b]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = compose_eval_config()
    _SIMULATOR, _SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)
    agents = {args.label_a: load_agent(cfg, args.ckpt_a, device), args.label_b: load_agent(cfg, args.ckpt_b, device)}

    if args.split == "test":
        cache_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
        tokens = sorted(cache_paths)
        if args.max_scenes:
            rng = np.random.default_rng(args.seed)
            tokens = sorted(rng.choice(tokens, size=min(args.max_scenes, len(tokens)), replace=False).tolist())
    else:
        cache_paths = MetricCacheLoader(args.train_metric_cache).metric_cache_paths
        _, tokens = sample_navtrain_tokens(cache_paths, args.train_logs, args.max_scenes or 10 ** 9,
                                           random.Random(args.seed))
    logs = [log_of(cache_paths, t) for t in tokens]
    os.environ["NAVSIM_OSS_SENSOR_BLOBS_PATH"] = SPLITS[args.split]["oss"]
    loader = make_scene_loader(args.split, logs, tokens, agents[args.label_a].get_sensor_config())
    tokens = [t for t in tokens if t in loader.scene_frames_dicts]
    logs = [log_of(cache_paths, t) for t in tokens]
    index = {token: i for i, token in enumerate(tokens)}
    n = len(tokens)
    print(f"{args.split} scenes: {n}", flush=True)

    proposals = np.zeros((n, 2, 64, 8, 3), dtype=np.float32)
    head_probs = np.zeros((n, 2, 64, len(HEADS)), dtype=np.float32)
    pdm_scores = np.zeros((n, 2, 64), dtype=np.float32)
    pred_pdms = np.zeros((n, 2, 64), dtype=np.float32)
    selected = np.zeros((n, 2), dtype=np.int16)
    human = np.zeros((n, 8, 3), dtype=np.float32)
    start, done = time.time(), 0
    for batch_tokens, batch_human, outputs in iterate_proposals(agents, loader, tokens, device, args.batch_size,
                                                                args.loader_workers):
        rows = [index[t] for t in batch_tokens]
        human[rows] = batch_human
        for m, label in enumerate(labels):
            proposals[rows, m] = outputs[label]["proposals"]
            head_probs[rows, m] = outputs[label]["head_probs"]
            pdm_scores[rows, m] = outputs[label]["pdm_score"]
            pred_pdms[rows, m] = outputs[label]["pred_pdms"]
            selected[rows, m] = outputs[label]["selected"]
        done += len(rows)
        if done % (args.batch_size * 40) < len(rows) or done == n:
            rate = done / (time.time() - start)
            print(f"[inference] {done}/{n} scenes, {rate:.1f} scenes/s, ETA {(n - done) / rate / 60:.1f} min", flush=True)

    # One task per scene: A's 64 proposals, B's 64 proposals, then the human GT (129 rows).
    tasks = ((cache_paths[t], np.concatenate([proposals[i, 0], proposals[i, 1], human[i][None]]))
             for i, t in enumerate(tokens))
    pdm = np.zeros((n, 129, len(PDM_FIELDS)), dtype=np.float32)
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for i, rows in enumerate(pool.map(_score_task, tasks, chunksize=4)):
            pdm[i] = rows
            if (i + 1) % 1000 == 0 or i + 1 == n:
                rate = (i + 1) / (time.time() - start)
                print(f"[scoring] {i + 1}/{n} scenes, {rate:.1f} scenes/s, ETA {(n - i - 1) / rate / 60:.1f} min",
                      flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "arrays.npz", split=args.split, tokens=np.array(tokens),
                        logs=np.array(logs),
                        labels=np.array(labels),
                        heads=np.array(HEADS), pdm_fields=np.array(PDM_FIELDS), proposals=proposals.astype(np.float16),
                        head_probs=head_probs, pdm_score=pdm_scores, pred_pdms=pred_pdms, selected=selected,
                        human=human, pdm=pdm)
    (args.output_dir / "run_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))


def read_eval_scores(eval_dir):
    csvs = sorted(Path(eval_dir).glob("*/*.csv"), key=lambda path: path.stat().st_mtime)
    if not csvs:
        return {}
    with open(csvs[-1]) as stream:
        return {row["token"]: float(row["score"]) for row in csv.DictReader(stream) if row["token"] != "average"}


def report(args, out_dir):
    data = np.load(out_dir / "arrays.npz", allow_pickle=False)
    tokens, logs, labels = data["tokens"].tolist(), data["logs"].tolist(), data["labels"].tolist()
    fields = data["pdm_fields"].tolist()
    pdm, pred_all, selected = data["pdm"].astype(np.float64), data["pred_pdms"].astype(np.float64), data["selected"]
    n = len(tokens)
    rows = np.arange(n)
    score = fields.index("score")
    human_pdms = pdm[:, 128, score]
    split = str(data["split"]) if "split" in data.files else "test"
    # Only navtest has evaluation CSVs to cross-check against.
    eval_scores = ([read_eval_scores(args.eval_dir_a), read_eval_scores(args.eval_dir_b)] if split == "test"
                   else [{}, {}])

    summary = {"split": split, "scenes": n, "human_gt_pdms": 100 * float(np.nanmean(human_pdms)), "models": {}}
    per_scene = {"token": tokens, "log_name": logs, "human_pdms": human_pdms}
    best, picks = [], []
    for m, label in enumerate(labels):
        block = pdm[:, 64 * m:64 * (m + 1)]
        pdms, pred, sel = block[..., score], pred_all[:, m], selected[:, m].astype(int)
        order = np.argsort(-data["pdm_score"][:, m], axis=1, kind="stable")  # the model's own ranking
        if not np.array_equal(order[:, 0], sel):
            raise RuntimeError(f"{label}: pdm_score ranking differs from the model output")
        sel_pdms, oracle = pdms[rows, sel], np.nanmax(pdms, axis=1)
        gap = oracle - sel_pdms
        # Several proposals often share the best PDMS (e.g. 1.0): take the one the scorer ranks highest.
        rank = np.empty_like(order)
        rank[rows[:, None], order] = np.arange(1, 65)
        rank_of_oracle = np.where(pdms >= oracle[:, None], rank, 65).min(axis=1)
        oracle_idx = order[rows, rank_of_oracle - 1]
        topk = {k: float(np.nanmean(np.nanmax(np.take_along_axis(pdms, order[:, :k], axis=1), axis=1))) for k in TOPK}
        csv_scores = np.array([eval_scores[m].get(t, np.nan) for t in tokens])
        csv_diff = np.abs(sel_pdms - csv_scores)

        severe = gap > 0.5
        sub = block[rows, sel]  # sub-scores of the selected trajectory
        fails = {"NC\n(at-fault collision)": sub[:, fields.index("no_at_fault_collisions")] < 1,
                 "DAC\n(off drivable area)": sub[:, fields.index("drivable_area_compliance")] < 1,
                 "TTC\n(time to collision)": sub[:, fields.index("time_to_collision_within_bound")] < 1,
                 "C\n(comfort)": sub[:, fields.index("comfort")] < 1}
        rhos = [spearmanr(pred[i], pdms[i]).correlation for i in range(n) if np.ptp(pdms[i]) > 0]
        buckets = []
        for low, high, name in GAP_BUCKETS:
            mask = (gap > low) & (gap <= high) if low > 0 else gap <= high
            buckets.append({"bucket": name, "scenes_pct": 100 * float(mask.mean()),
                            "loss_pct": 100 * float(gap[mask].sum() / max(gap.sum(), 1e-12))})
        summary["models"][label] = {
            "selected_pdms": 100 * float(np.nanmean(sel_pdms)),
            "best_of_64_pdms": 100 * float(np.nanmean(oracle)),
            "gap_points": 100 * float(np.nanmean(gap)),
            "best_of_topk_pdms": {str(k): 100 * v for k, v in topk.items()},
            "selected_predicted_pdms": 100 * float(np.mean(pred[rows, sel])),
            "scenes_selected_is_best_pct": 100 * float(np.mean(gap <= 1e-9)),
            "scenes_best_of_64_below_0.5_pct": 100 * float(np.mean(oracle < 0.5)),
            "scenes_all_64_zero_pct": 100 * float(np.mean(oracle == 0)),
            "mean_rank_of_best_proposal": float(np.mean(rank_of_oracle)),
            "median_rank_of_best_proposal": float(np.median(rank_of_oracle)),
            "mean_spearman_pred_vs_true": float(np.nanmean(rhos)),
            "gap_buckets": buckets,
            "severe_scorer_failures": {"scenes_pct": 100 * float(severe.mean()),
                                       "failed_metric_of_selected_pct": {k: 100 * float(v[severe].mean())
                                                                        for k, v in fails.items()}},
            "eval_csv_check": {"compared": int(np.isfinite(csv_diff).sum()),
                               "max_abs_diff": float(np.nanmax(csv_diff)) if np.isfinite(csv_diff).any() else None,
                               "mismatches_gt_1e-6": int(np.sum(csv_diff > 1e-6))},
        }
        summary["models"][label]["label_balance"] = head_balance(data, m, fields)
        best.append(oracle)
        picks.append(sel_pdms)
        per_scene.update({f"{label}|selected_idx": sel, f"{label}|selected_pdms": sel_pdms,
                          f"{label}|selected_pred": pred[rows, sel], f"{label}|best_of_64": oracle,
                          f"{label}|best_idx": oracle_idx, f"{label}|gap": gap,
                          f"{label}|rank_of_best": rank_of_oracle,
                          f"{label}|best_of_top4": np.nanmax(np.take_along_axis(pdms, order[:, :4], axis=1), axis=1),
                          f"{label}|num_ge_0.9": np.sum(pdms >= 0.9, axis=1), f"{label}|eval_csv_pdms": csv_scores})
    summary["best_of_128_both_models_pdms"] = 100 * float(np.nanmean(np.maximum(best[0], best[1])))
    summary["better_of_two_outputs_pdms"] = 100 * float(np.nanmean(np.maximum(picks[0], picks[1])))

    with open(out_dir / "scenes.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(list(per_scene))
        for i in range(n):
            writer.writerow([per_scene[key][i] for key in per_scene])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", summary, labels)
    plot_summary(out_dir / "oracle_gap_summary.png", summary, labels)
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


BALANCE_HEADS = (("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"),
                 ("TTC", "time_to_collision_within_bound"), ("C", "comfort"), ("DDC", "driving_direction_compliance"))
CALIBRATION_BINS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0001)


def head_balance(data, m, fields):
    """Label balance and quality of each binary head: are labels mostly positive, does the head lean positive?"""
    from sklearn.metrics import roc_auc_score

    heads = data["heads"].tolist()
    probs = data["head_probs"][:, m].astype(np.float64)
    truth = data["pdm"][:, 64 * m:64 * (m + 1)].astype(np.float64)
    result = {}
    for name, key in BALANCE_HEADS:
        passed = truth[..., fields.index(key)] >= 1.0  # NC / DDC 0.5 count as failing, as in training
        failing = ~passed
        sigma = probs[..., heads.index(key)]
        mixed = np.flatnonzero(passed.any(axis=1) & failing.any(axis=1))
        calibration = []
        for low, high in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
            inside = (sigma >= low) & (sigma < high)
            if inside.any():
                calibration.append({"sigma_bin": f"[{low:.2f}, {min(high, 1.0):.2f})", "proposals": int(inside.sum()),
                                    "pass_pct": 100 * float(passed[inside].mean())})
        has_both = bool(failing.any() and passed.any())
        result[name] = {
            "pass_pct": 100 * float(passed.mean()),
            "mean_sigma_pct": 100 * float(sigma.mean()),
            "scenes_with_failing_pct": 100 * float(failing.any(axis=1).mean()),
            "failing_per_scene_if_any": float(failing.sum(axis=1)[failing.any(axis=1)].mean()) if failing.any() else 0.0,
            "auroc_pooled": float(roc_auc_score(passed.ravel(), sigma.ravel())) if has_both else None,
            "auroc_within_scene": float(np.mean([roc_auc_score(passed[i], sigma[i]) for i in mixed]))
            if len(mixed) else None,
            "failing_sigma_median": float(np.median(sigma[failing])) if failing.any() else None,
            "failing_sigma_above_0.9_pct": 100 * float(np.mean(sigma[failing] > 0.9)) if failing.any() else None,
            "calibration": calibration,
        }
    return result


def fmt_optional(value, spec):
    return "n/a" if value is None else format(value, spec)


def write_markdown(path, summary, labels):
    models = [summary["models"][label] for label in labels]
    where = "navtest" if summary.get("split", "test") == "test" else "a navtrain sample (training logs)"
    lines = [f"# Scorer oracle gap on {where} ({summary['scenes']} scenes)", "",
             f"| | {' | '.join(labels)} |", "|---|" + "---:|" * len(labels)]
    rows = [
        ("PDMS of the selected trajectory (model output)", lambda m: f"{m['selected_pdms']:.2f}"),
        ("PDMS with a perfect scorer (best of the same 64)", lambda m: f"{m['best_of_64_pdms']:.2f}"),
        ("**Max gain from a better scorer**", lambda m: f"**+{m['gap_points']:.2f}**"),
        ("Best within the scorer's top-2 / top-4 / top-8",
         lambda m: " / ".join(f"{m['best_of_topk_pdms'][k]:.2f}" for k in ("2", "4", "8"))),
        ("Scorer's predicted PDMS of its pick (calibration)", lambda m: f"{m['selected_predicted_pdms']:.2f}"),
        ("Scenes where the pick is already the best", lambda m: f"{m['scenes_selected_is_best_pct']:.1f}%"),
        ("Scenes where even the best of 64 is < 0.5", lambda m: f"{m['scenes_best_of_64_below_0.5_pct']:.2f}%"),
        ("Scenes where all 64 score 0", lambda m: f"{m['scenes_all_64_zero_pct']:.2f}%"),
        ("Scorer rank of the best proposal (mean / median)",
         lambda m: f"{m['mean_rank_of_best_proposal']:.1f} / {m['median_rank_of_best_proposal']:.0f}"),
        ("Spearman(predicted, true) per scene, mean", lambda m: f"{m['mean_spearman_pred_vs_true']:.2f}"),
        ("Severe picks (gap > 0.5), share of scenes", lambda m: f"{m['severe_scorer_failures']['scenes_pct']:.2f}%"),
        ("Selected PDMS vs eval CSV, max abs diff", lambda m: fmt_optional(m['eval_csv_check']['max_abs_diff'], ".1e")),
    ]
    for name, fmt in rows:
        lines.append(f"| {name} | {' | '.join(fmt(m) for m in models)} |")
    lines += ["", f"Human GT trajectory PDMS: {summary['human_gt_pdms']:.2f}. "
              f"Best of both models' 128 proposals: {summary['best_of_128_both_models_pdms']:.2f}. "
              f"Better of the two outputs per scene: {summary['better_of_two_outputs_pdms']:.2f}.", "",
              "## Where the scorer loses points (gap = best of 64 - selected)", "",
              "| gap bucket | " + " | ".join(f"{label}: scenes / share of lost points" for label in labels) + " |",
              "|---|" + "---:|" * len(labels)]
    for b in range(len(GAP_BUCKETS)):
        cells = [f"{m['gap_buckets'][b]['scenes_pct']:.1f}% / {m['gap_buckets'][b]['loss_pct']:.1f}%" for m in models]
        lines.append(f"| {models[0]['gap_buckets'][b]['bucket']} | {' | '.join(cells)} |")
    lines += ["", "## Failed metric of the selected trajectory in severe picks (gap > 0.5)", "",
              f"| metric | {' | '.join(labels)} |", "|---|" + "---:|" * len(labels)]
    for name in models[0]["severe_scorer_failures"]["failed_metric_of_selected_pct"]:
        lines.append(f"| {name.replace(chr(10), ' ')} | " + " | ".join(
            f"{m['severe_scorer_failures']['failed_metric_of_selected_pct'][name]:.1f}%" for m in models) + " |")
    lines += ["", "## Label balance and head quality (NC / DDC 0.5 counted as failing, as in training)", "",
              "| head | model | pass rate | mean σ | scenes with ≥1 failing | failing per such scene | "
              "AUROC pooled | AUROC within scene | failing proposals with σ > 0.9 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, _ in BALANCE_HEADS:
        for label, m in zip(labels, models):
            b = m["label_balance"][name]
            lines.append(f"| {name} | {label} | {b['pass_pct']:.1f}% | {b['mean_sigma_pct']:.1f}% | "
                         f"{b['scenes_with_failing_pct']:.1f}% | {b['failing_per_scene_if_any']:.1f} | "
                         f"{fmt_optional(b['auroc_pooled'], '.3f')} | {fmt_optional(b['auroc_within_scene'], '.3f')} | "
                         f"{fmt_optional(b['failing_sigma_above_0.9_pct'], '.1f')}% |")
    lines += ["", "Calibration (share of proposals that really pass, per σ bin):", "",
              "| σ bin | " + " | ".join(f"{name} {label}" for name in ("NC", "DAC") for label in labels) + " |",
              "|---|" + "---:|" * (2 * len(labels))]
    bins = [c["sigma_bin"] for c in models[0]["label_balance"]["NC"]["calibration"]]
    for b in bins:
        cells = []
        for name in ("NC", "DAC"):
            for m in models:
                hit = [c for c in m["label_balance"][name]["calibration"] if c["sigma_bin"] == b]
                cells.append(f"{hit[0]['pass_pct']:.1f}% (n={hit[0]['proposals']:,})" if hit else "-")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(path, summary, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK})
    models = [summary["models"][label] for label in labels]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # 1) Best true PDMS within the scorer's top-k: k=1 is the model output, k=64 a perfect scorer.
    ax = axes[0]
    for m, (label, model) in enumerate(zip(labels, models)):
        values = [model["best_of_topk_pdms"][str(k)] for k in TOPK]
        ax.plot(TOPK, values, color=MODEL_COLORS[m], linewidth=2, marker="o", markersize=8, markeredgecolor="white",
                markeredgewidth=2, label=label, solid_capstyle="round", zorder=3)
        ax.annotate(f"{values[0]:.2f}", (TOPK[0], values[0]), xytext=(9, -14 if m else 6),
                    textcoords="offset points", ha="left", color=INK_2, fontsize=10)
        ax.annotate(f"{values[-1]:.2f}  (+{values[-1] - values[0]:.2f})", (TOPK[-1], values[-1]),
                    xytext=(9, 3 if m == 0 else -13), textcoords="offset points", ha="left", color=INK_2,
                    fontsize=10)
    ax.axhline(summary["human_gt_pdms"], color=INK, linewidth=1, linestyle="--", zorder=2)
    ax.annotate(f"human GT {summary['human_gt_pdms']:.2f}", (TOPK[0], summary["human_gt_pdms"]), xytext=(0, 5),
                textcoords="offset points", color=INK_2, fontsize=9.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TOPK)
    ax.set_xticklabels([str(k) for k in TOPK])
    ax.set_xlim(0.75, 300)  # room right of k = 64 for the end labels
    ax.set_xlabel("k = number of top-ranked proposals (by the scorer) the choice may come from")
    ax.set_ylabel("navtest PDMS")
    ax.set_title("Best true PDMS within the scorer's top-k\n(k = 1: model output, k = 64: perfect scorer)",
                 fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.legend(frameon=False, loc="lower right")

    # 2) Share of the scorer's lost points per gap bucket (bars) with the share of scenes (labels).
    ax = axes[1]
    width = 0.36
    x = np.arange(len(GAP_BUCKETS))
    for m, (label, model) in enumerate(zip(labels, models)):
        loss = [b["loss_pct"] for b in model["gap_buckets"]]
        scenes = [b["scenes_pct"] for b in model["gap_buckets"]]
        bars = ax.bar(x + (m - 0.5) * (width + 0.02), loss, width=width, color=MODEL_COLORS[m], label=label, zorder=3)
        for bar, share in zip(bars, scenes):
            ax.annotate(f"{share:.1f}%\nscenes", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                        color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in GAP_BUCKETS])
    ax.set_xlabel("per-scene gap = best of 64 - selected PDMS")
    ax.set_ylabel("share of all points the scorer loses (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Where the scorer loses points\n(bar: share of lost points; label: share of scenes)",
                 fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.legend(frameon=False, loc="upper left")

    # 3) Which metric the selected trajectory fails when a much better proposal existed.
    ax = axes[2]
    names = list(models[0]["severe_scorer_failures"]["failed_metric_of_selected_pct"])
    x = np.arange(len(names))
    for m, (label, model) in enumerate(zip(labels, models)):
        values = [model["severe_scorer_failures"]["failed_metric_of_selected_pct"][k] for k in names]
        bars = ax.bar(x + (m - 0.5) * (width + 0.02), values, width=width, color=MODEL_COLORS[m], label=label,
                      zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.0f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("share of severe picks (%)")
    ax.set_ylim(0, 105)
    severe = " / ".join(f"{m['severe_scorer_failures']['scenes_pct']:.1f}%" for m in models)
    ax.set_title(f"Failed metric of the selected trajectory\nin severe picks (gap > 0.5: {severe} of scenes)",
                 fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    args = parse_args()
    if args.report_only:
        report(args, args.report_only)
        return
    compute(args)
    report(args, args.output_dir)


if __name__ == "__main__":
    main()
