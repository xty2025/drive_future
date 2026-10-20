"""64-scene BEV visualization + scorer-selection statistics (joint e2e lineage).

Companion to ``eval_navtest_official_pdms_e2e.py``: it reuses that script's model,
oracle and candidate construction VERBATIM (imported), so what is drawn and counted
here is produced by the exact same deployed forward + official ``pdm_score`` path as
the full held-out eval -- only on a seeded sample of scenes instead of all of them.

For each sampled scene it:
  * renders the ego-BEV (``e2e.draw_bev_eval``): all 64 candidates colored by their
    official PDMS, the psi-SELECTED candidate, the ORACLE-BEST candidate, every valid
    official GT trajectory and the ego history;
  * records the PRE-SELECTION pool (all 64 candidate PDMS) versus the SELECTED one.

It then reports selection statistics -- "what the scorer picked" versus "the pool
before it picked" -- and writes a montage of the BEVs plus aggregate distribution
figures.  Read-only w.r.t. every package file, trainer script and checkpoint.
"""
from __future__ import annotations

import os

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/data/navsim/maps")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_navtrain_wan_e2e_pdms_bev as e2e  # noqa: E402
import eval_navtest_official_pdms_e2e as off  # noqa: E402
import run_navtrain_wan_e2e_ddp as trainer  # noqa: E402
from official_navtest_oracle import OfficialNavsimOracle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scenes", type=int, default=64, help="how many navtest scenes to sample + render")
    parser.add_argument("--seed", type=int, default=17, help="seed for the scene sample (reproducible)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--target-dim", type=int, default=0)
    parser.add_argument("--wan-backbone", default=trainer.DEFAULT_WAN_BACKBONE)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--dino-checkpoint", default="")
    parser.add_argument("--drivor-cache", default=e2e.NAVTEST_CACHE,
                        help="navtest DrivoR cache; used ONLY to resolve the scene index, never for scoring")
    parser.add_argument("--official-cache", default=off.OFFICIAL_NAVTEST_CACHE)
    parser.add_argument("--log-root", default=e2e.NAVTEST_LOG_ROOT)
    parser.add_argument("--sensor-root", default=e2e.NAVTEST_SENSOR_ROOT)
    parser.add_argument("--index-dir", default=str(REPO_ROOT / "artifacts/navtest_wan_joint_eval_index"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--montage-cols", type=int, default=8)
    return parser.parse_args()


def make_montage(png_paths: list[Path], out_path: Path, cols: int, title: str) -> None:
    """Tile the per-scene BEVs into one contact sheet (downsampled to stay light)."""
    n = len(png_paths)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.5), dpi=90)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, png_paths):
        img = plt.imread(str(path))
        # downsample ~3x to keep the montage memory + file size small
        img = img[::3, ::3]
        ax.imshow(img)
        ax.set_title(path.stem.replace("_bev", ""), fontsize=5)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_stats_figure(recs: list[dict], out_path: Path, title: str) -> dict:
    """Distribution + per-scene contrast figures; returns the aggregate stat dict."""
    sel = np.array([r["selected_pdms"] for r in recs], dtype=float)
    best = np.array([r["oracle_best_pdms"] for r in recs], dtype=float)
    pmean = np.array([r["pool_mean"] for r in recs], dtype=float)
    cand0 = np.array([r["cand0_pdms"] for r in recs], dtype=float)
    gain = sel - pmean           # how much better the pick is than the average candidate
    regret = best - sel          # how far below the best the pick lands
    match = np.array([r["oracle_match"] for r in recs], dtype=float)
    rel_rank = np.array([r["selected_rank"] / max(r["pool_size"], 1) for r in recs], dtype=float)
    pool_all = np.concatenate([np.asarray(r["pool"], dtype=float) for r in recs])
    pool_all = pool_all[np.isfinite(pool_all)]

    def _m(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")

    stats = {
        "n_scenes": len(recs),
        "preselection_pool_mean_pdms": round(_m(pmean), 4),      # "没选之前的": avg candidate
        "preselection_cand0_pdms": round(_m(cand0), 4),          # prior anchor #0 (no scorer)
        "selected_mean_pdms": round(_m(sel), 4),                 # "score选的": scorer's pick
        "oracle_best_mean_pdms": round(_m(best), 4),             # upper bound
        "selection_gain_over_pool_mean": round(_m(gain), 4),     # selected - pool mean
        "selection_regret_vs_oracle": round(_m(regret), 4),      # oracle - selected
        "oracle_match_rate": round(_m(match), 4),
        "selected_beats_cand0_rate": round(_m((sel > cand0).astype(float)), 4),
        "mean_selected_rank_in_pool": round(_m(np.array([r["selected_rank"] for r in recs], float)), 3),
        "mean_selected_relative_rank": round(_m(rel_rank), 4),   # 0=best .. 1=worst
        "pool_pdms_min": round(float(pool_all.min()), 4) if pool_all.size else None,
        "pool_pdms_max": round(float(pool_all.max()), 4) if pool_all.size else None,
        "pool_pdms_std": round(float(pool_all.std()), 4) if pool_all.size else None,
    }

    fig = plt.figure(figsize=(16, 9), dpi=110)
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # (0,0) distribution of the whole pre-selection pool + key means
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(pool_all, bins=60, color="#9ecae1", edgecolor="none")
    ax0.axvline(_m(pmean), color="#6a51a3", ls="--", lw=1.6, label=f"pool mean {_m(pmean):.3f}")
    ax0.axvline(_m(sel), color="#00a651", ls="-", lw=2.0, label=f"selected {_m(sel):.3f}")
    ax0.axvline(_m(best), color="#ff9f00", ls=":", lw=2.0, label=f"oracle-best {_m(best):.3f}")
    ax0.set(title="pre-selection pool: all 64 candidate PDMS", xlabel="official PDMS", ylabel="count")
    ax0.legend(fontsize=7)

    # (0,1) boxplot: pool vs selected vs oracle
    ax1 = fig.add_subplot(gs[0, 1])
    bp = ax1.boxplot([pool_all, sel, best], tick_labels=["pool (all 64)", "selected", "oracle-best"],
                     showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#9ecae1", "#00a651", "#ff9f00"]):
        patch.set_facecolor(c)
    ax1.set(title="PDMS by selection stage", ylabel="official PDMS")

    # (0,2) per-scene: pool mean vs selected
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(pmean, sel, s=16, c="#00a651", alpha=0.7, label="selected")
    ax2.scatter(pmean, best, s=12, c="#ff9f00", alpha=0.5, label="oracle-best")
    lo = float(np.nanmin([pmean.min(), sel.min()])) if pmean.size else 0.0
    hi = float(np.nanmax([pmean.max(), best.max()])) if pmean.size else 1.0
    ax2.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1.0)
    ax2.set(title="per scene: pool mean (x) vs pick (y)", xlabel="pool mean PDMS", ylabel="PDMS")
    ax2.legend(fontsize=7)

    # (1,0) selection gain histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(gain[np.isfinite(gain)], bins=40, color="#00a651", edgecolor="none")
    ax3.axvline(0, color="gray", ls="--", lw=1.0)
    ax3.axvline(_m(gain), color="black", ls="-", lw=1.6, label=f"mean {_m(gain):+.3f}")
    ax3.set(title="selection gain (selected - pool mean)", xlabel="PDMS delta", ylabel="count")
    ax3.legend(fontsize=7)

    # (1,1) regret histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(regret[np.isfinite(regret)], bins=40, color="#ff9f00", edgecolor="none")
    ax4.axvline(_m(regret), color="black", ls="-", lw=1.6, label=f"mean {_m(regret):.3f}")
    ax4.set(title="regret (oracle-best - selected)", xlabel="PDMS delta", ylabel="count")
    ax4.legend(fontsize=7)

    # (1,2) selected rank within pool
    ax5 = fig.add_subplot(gs[1, 2])
    ranks = np.array([r["selected_rank"] for r in recs], dtype=int)
    ax5.hist(ranks, bins=np.arange(0.5, min(int(ranks.max()) + 2, 66), 1), color="#6a51a3", edgecolor="none")
    ax5.axvline(_m(ranks.astype(float)), color="black", ls="-", lw=1.6, label=f"mean rank {_m(ranks.astype(float)):.1f}")
    ax5.set(title="rank of selected within pool (1=best)", xlabel="rank", ylabel="count")
    ax5.legend(fontsize=7)

    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return stats


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir or str(REPO_ROOT / "artifacts/navtest_selection_viz" / args.tag))
    bev_dir = output_dir / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)

    net, model, scorer, meta = trainer.load_joint_checkpoint(
        args.checkpoint, device=device, depth=args.depth, heads=args.heads,
        target_dim=args.target_dim, dino_checkpoint=args.dino_checkpoint,
        wan_backbone=args.wan_backbone, grad_checkpointing=args.grad_checkpointing,
    )
    opt_step, epoch = meta["opt_step"], meta["epoch"]
    print(json.dumps({"event": "ckpt_loaded", **meta}), flush=True)

    # SAME universe as the full eval: navtest index intersected with the official cache.
    candidates = e2e.build_navtest_index(args.drivor_cache, args.log_root, Path(args.index_dir), 0)
    kept = []
    official_paths: dict[str, str] = {}
    for c in candidates:
        p = Path(args.official_cache) / c.log_name / "unknown" / c.token / "metric_cache.pkl"
        if p.exists():
            official_paths[c.token] = str(p)
            kept.append(c)
    candidates = kept
    if not candidates:
        raise RuntimeError("no navtest scenes with an official metric cache")

    # seeded sample of --scenes from the filtered universe (reproducible)
    n_take = min(args.scenes, len(candidates))
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(candidates), size=n_take, replace=False))
    sampled = [candidates[i] for i in pick]
    print(json.dumps({"event": "sampled", "n_take": n_take, "universe": len(candidates),
                      "seed": args.seed}), flush=True)

    oracle = OfficialNavsimOracle(official_paths, evaluator_version=off.ORACLE_VERSION)
    dataset = e2e.NavtestSceneDataset(sampled, args.sensor_root, args.log_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=trainer.collate, drop_last=False,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=2 if args.workers > 0 else None)

    modes = trainer.ImmutableModeBuffer().subset(64).to(device)
    initializer = trainer.StructuredSourceInitializer(jitter_std=0.0)

    recs: list[dict] = []
    png_paths: list[Path] = []
    oracle_failures = 0
    batch_i = 0
    run_start = time.time()

    for batch in loader:
        observation, targets, scene_ids = trainer.batch_to_device(batch, device)
        size = observation.rgb_history.shape[0]
        mode_codes = modes.expand_for_batch(size, device)
        source = initializer.clean(observation.ego_history, mode_codes)
        with torch.no_grad():
            endpoint, trajectory, logits = net.inference(observation, source, mode_codes)
            del endpoint
            try:
                labels = oracle.evaluate(scene_ids, trajectory, batch_id=f"viz{batch_i}")
                labels_ok = (not labels.failure_diagnostics) and bool(labels.final_valid.all())
            except Exception as error:
                labels, labels_ok = None, False
                print(json.dumps({"event": "oracle_error", "batch": batch_i, "error": repr(error)}), flush=True)
            assignment = trainer.hungarian_xy(trajectory, targets["gt_trajectory"], targets["gt_valid"])
            selected, _utility = scorer.select(logits)

        if not labels_ok:
            oracle_failures += 1

        traj_np = trajectory.detach().cpu().numpy()
        gt_np = targets["gt_trajectory"].cpu().numpy()
        gtvalid_np = targets["gt_valid"].cpu().numpy()
        ego_hist_np = observation.relative_ego_poses.cpu().numpy()
        sel_np = selected.cpu().numpy()
        fs_np = labels.final_score.cpu().numpy() if labels_ok else None

        for row in range(size):
            if fs_np is None:
                continue
            token = scene_ids[row]
            pool = fs_np[row]
            finite = pool[np.isfinite(pool)]
            if finite.size == 0:
                continue
            sel_idx = int(sel_np[row])
            oracle_idx = int(np.argmax(pool))
            sel_pdms = float(pool[sel_idx])
            best_pdms = float(pool[oracle_idx])
            pool_mean = float(finite.mean())
            # 1-based rank of the selected candidate within the finite pool
            sel_rank = int((finite > sel_pdms).sum()) + 1
            ade_pairs = [(traj_np[row, c, :, :2] - gt_np[row, g, :, :2]) for g, c in assignment[row].as_pairs()]
            ade_row = float(np.mean([np.linalg.norm(d, axis=-1).mean() for d in ade_pairs])) if ade_pairs else float("nan")

            recs.append({
                "scene_id": token,
                "pool": [round(float(v), 4) if np.isfinite(v) else None for v in pool],
                "pool_size": int(finite.size),
                "pool_mean": round(pool_mean, 4),
                "pool_median": round(float(np.median(finite)), 4),
                "pool_min": round(float(finite.min()), 4),
                "pool_max": round(float(finite.max()), 4),
                "pool_std": round(float(finite.std()), 4),
                "cand0_pdms": round(float(pool[0]), 4) if np.isfinite(pool[0]) else None,
                "selected_index": sel_idx,
                "selected_pdms": round(sel_pdms, 4),
                "selected_rank": sel_rank,
                "oracle_best_index": oracle_idx,
                "oracle_best_pdms": round(best_pdms, 4),
                "oracle_match": bool(sel_idx == oracle_idx),
                "selection_gain": round(sel_pdms - pool_mean, 4),
                "selection_regret": round(best_pdms - sel_pdms, 4),
                "matched_gt_ade_m": round(ade_row, 4),
                "n_valid_gt": int(gtvalid_np[row].sum()),
            })

            out_png = bev_dir / f"{token}_bev.png"
            e2e.draw_bev_eval(
                scene_id=token, candidates=traj_np[row], final_score=pool,
                selected_idx=sel_idx, oracle_idx=oracle_idx,
                gt_paths=gt_np[row], gt_valid=gtvalid_np[row], ego_history=ego_hist_np[row],
                out_path=out_png, sel_pdms=sel_pdms, best_pdms=best_pdms,
                ade_row=ade_row, opt_step=opt_step,
            )
            png_paths.append(out_png)
        batch_i += 1
        print(json.dumps({"event": "batch", "batch": batch_i, "rendered": len(png_paths),
                          "elapsed_s": round(time.time() - run_start, 1)}), flush=True)

    if not recs:
        raise RuntimeError("no scenes produced a valid official score; nothing to visualize")

    title = (f"navtest selection viz | {args.tag} | epoch={epoch} opt_step={opt_step} | "
             f"{len(recs)} scenes | scorer-selected vs pre-selection pool")
    stats = make_stats_figure(recs, output_dir / "selection_stats.png", title)
    make_montage(png_paths, output_dir / "montage.png", args.montage_cols, title)

    payload = {
        "checkpoint": str(args.checkpoint),
        "tag": args.tag,
        "epoch": epoch,
        "opt_step": opt_step,
        "lineage": "nvs_v6_wan_e2e",
        "split": "navtest (held-out)",
        "scoring": "official navsim pdm_score path (official MetricCache + official PDMScorer)",
        "oracle_version": off.ORACLE_VERSION,
        "seed": args.seed,
        "scenes_sampled": len(recs),
        "scenes_requested": args.scenes,
        "oracle_failures": oracle_failures,
        "selection_stats": stats,
        "elapsed_s": round(time.time() - run_start, 1),
        "per_scene": recs,
    }
    (output_dir / "selection_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "summary", "output_dir": str(output_dir), **stats}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
