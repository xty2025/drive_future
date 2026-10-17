"""navtest PDMS scored the OFFICIAL navsim way, for the joint e2e lineage.

This is the held-out counterpart of ``eval_navtrain_wan_e2e_pdms_bev.py``: it
reuses that script's model/scorer construction and candidate generation, but the
candidates are scored by ``official_navtest_oracle.OfficialNavsimOracle`` --
official ``MetricCache`` + official ``PDMScorer`` + ``pdm_score`` aggregation --
rather than DrivoR's ``get_sub_score`` on the float32-quantized train cache.

  * identical navtest scene index (same DrivoR navtest cache + same log root +
    the same sidecar in ``--index-dir``), so the scene set is bit-identical to
    the baseline run and the numbers are directly comparable;
  * candidates from ``JointStage2Module.inference``: the deployed single forward,
    the 64-anchor prior denoised once at ``sigma = 1`` and inverted algebraically
    to a metric trajectory.  No rollout, so what is scored is what ships;
  * identical scorer selection (``scorer.select``).

Per-scene records carry the keys ``merge_navtest_shards.py`` recomputes its summary
from, so sharding and reporting reuse the existing tooling unchanged.
Read-only w.r.t. every package file, trainer script and checkpoint.
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

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_navtrain_wan_e2e_pdms_bev as e2e  # noqa: E402
import run_navtrain_wan_e2e_ddp as trainer  # noqa: E402
from official_navtest_oracle import OfficialNavsimOracle, verify_against_pdm_score  # noqa: E402

OFFICIAL_NAVTEST_CACHE = (
    "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/"
    "drivor_official_navtest_full_cache_20261007/metric_cache"
)
ORACLE_VERSION = "navsim_official_pdm_score_navtest_20261007"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 = every navtest scene with an official cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--depth", type=int, default=0, help="0 = read depth from the checkpoint config")
    parser.add_argument("--heads", type=int, default=0,
                        help="0 = read from the checkpoint config (the Wan payload fixes num_heads=4 at hidden_dim=256)")
    parser.add_argument("--target-dim", type=int, default=0, help="0 = read from the checkpoint config")
    parser.add_argument("--wan-backbone", default=trainer.DEFAULT_WAN_BACKBONE,
                        help="only needed to build theta before the checkpoint overwrites it")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dino-checkpoint", default="")
    parser.add_argument("--drivor-cache", default=e2e.NAVTEST_CACHE,
                        help="navtest DrivoR cache; used ONLY to resolve the scene index, never for scoring")
    parser.add_argument("--official-cache", default=OFFICIAL_NAVTEST_CACHE)
    parser.add_argument("--log-root", default=e2e.NAVTEST_LOG_ROOT)
    parser.add_argument("--sensor-root", default=e2e.NAVTEST_SENSOR_ROOT)
    parser.add_argument("--index-dir", default=str(REPO_ROOT / "artifacts/navtest_wan_joint_eval_index"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--verify-scenes", type=int, default=3,
                        help="cross-check the batch oracle against official pdm_score on this many scenes")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="evaluate every shard_count-th scene starting here (0-based)")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="split the scene list across this many independent processes (1 = no split)")
    parser.add_argument("--device", default="",
                        help="torch device for this process (default cuda:0)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError(f"--shard-index {args.shard_index} outside [0, {args.shard_count})")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir or str(REPO_ROOT / "artifacts/navtest_official_pdms_e2e" / args.tag))
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("result.json" if args.shard_count == 1 else f"result.shard{args.shard_index}.json")

    net, model, scorer, meta = trainer.load_joint_checkpoint(
        args.checkpoint,
        device=device,
        depth=args.depth,
        heads=args.heads,
        target_dim=args.target_dim,
        dino_checkpoint=args.dino_checkpoint,
        wan_backbone=args.wan_backbone,
        grad_checkpointing=args.grad_checkpointing,
    )
    opt_step, epoch = meta["opt_step"], meta["epoch"]
    print(json.dumps({"event": "ckpt_loaded", **meta}), flush=True)

    index_start = time.time()
    candidates = e2e.build_navtest_index(args.drivor_cache, args.log_root, Path(args.index_dir), args.max_scenes)
    official_paths: dict[str, str] = {}
    missing_official = 0
    kept = []
    for candidate in candidates:
        official = Path(args.official_cache) / candidate.log_name / "unknown" / candidate.token / "metric_cache.pkl"
        if official.exists():
            official_paths[candidate.token] = str(official)
            kept.append(candidate)
        else:
            missing_official += 1
    candidates = kept
    total_indexed = len(candidates)
    if args.shard_count > 1:
        # Strided slice over the sorted, official-cache-filtered list: every shard
        # sees a disjoint subset and the union is exactly the unsharded run.
        candidates = candidates[args.shard_index::args.shard_count]
    if not candidates:
        raise RuntimeError("no navtest scenes with an official metric cache")
    print(json.dumps({"event": "index_ready", "scenes": len(candidates),
                      "scenes_total": total_indexed,
                      "shard_index": args.shard_index, "shard_count": args.shard_count,
                      "missing_official_cache": missing_official,
                      "elapsed_s": round(time.time() - index_start, 2)}), flush=True)

    oracle = OfficialNavsimOracle(official_paths, evaluator_version=ORACLE_VERSION)
    dataset = e2e.NavtestSceneDataset(candidates, args.sensor_root, args.log_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=trainer.collate, drop_last=False,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=2 if args.workers > 0 else None)

    modes = trainer.ImmutableModeBuffer().subset(64).to(device)
    initializer = trainer.StructuredSourceInitializer(jitter_std=0.0)

    acc = torch.zeros(6)
    per_scene: list[dict] = []
    oracle_failures = 0
    ns_mean_sum = 0.0
    ns_cand0_sum = 0.0
    ns_count = 0
    verify_worst = 0.0
    verified = 0
    # Only shard 0 cross-checks against pdm_score; every shard re-verifying the
    # same oracle code path would just cost minutes.
    verify_budget = args.verify_scenes if args.shard_index == 0 else 0
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
                labels = oracle.evaluate(scene_ids, trajectory, batch_id=f"eval{batch_i}")
                labels_ok = (not labels.failure_diagnostics) and bool(labels.final_valid.all())
            except Exception as error:
                labels, labels_ok = None, False
                print(json.dumps({"event": "oracle_error", "batch": batch_i, "error": repr(error)}), flush=True)
            assignment = trainer.hungarian_xy(trajectory, targets["gt_trajectory"], targets["gt_valid"])
            metrics = trainer.batch_eval_metrics(scorer, logits, trajectory, targets, assignment, labels, labels_ok, device)
            nearest_ade, supervised = trainer.nearest_gt_ade(
                trajectory, targets["gt_trajectory"], targets["gt_valid"])
            selected, _utility = scorer.select(logits)

        acc += metrics.cpu()
        if not labels_ok:
            oracle_failures += 1

        traj_np = trajectory.detach().cpu().numpy()
        gt_np = targets["gt_trajectory"].cpu().numpy()
        gtvalid_np = targets["gt_valid"].cpu().numpy()
        sel_np = selected.cpu().numpy()
        fs_np = labels.final_score.cpu().numpy() if labels_ok else None
        near_np = nearest_ade.cpu().numpy()
        sup_np = supervised.cpu().numpy()

        for row in range(size):
            token = scene_ids[row]
            if fs_np is not None and verified < verify_budget:
                verify_worst = max(verify_worst, verify_against_pdm_score(oracle, token, traj_np[row]))
                verified += 1
            ade_pairs = [(traj_np[row, c, :, :2] - gt_np[row, g, :, :2]) for g, c in assignment[row].as_pairs()]
            ade_row = float(np.mean([np.linalg.norm(d, axis=-1).mean() for d in ade_pairs])) if ade_pairs else float("nan")
            row_near = near_np[row][sup_np[row]]
            rec: dict = {"scene_id": token, "n_valid_gt": int(gtvalid_np[row].sum()),
                         "matched_gt_ade_m": round(ade_row, 4),
                         "all64_nearest_gt_ade_m": (round(float(row_near.mean()), 4) if row_near.size else None)}
            if fs_np is not None:
                sel_idx = int(sel_np[row])
                rec.update({
                    "selected_index": sel_idx,
                    "selected_pdms_official": round(float(fs_np[row, sel_idx]), 4),
                    "oracle_best_pdms_official": round(float(fs_np[row].max()), 4),
                    "oracle_match": bool(sel_idx == int(np.argmax(fs_np[row]))),
                    "noscorer_mean64_official": round(float(np.nanmean(fs_np[row])), 4),
                    "noscorer_cand0_official": round(float(fs_np[row, 0]), 4) if np.isfinite(fs_np[row, 0]) else None,
                })
                fin = fs_np[row][np.isfinite(fs_np[row])]
                if fin.size:
                    ns_mean_sum += float(fin.mean())
                    ns_cand0_sum += float(fs_np[row, 0]) if np.isfinite(fs_np[row, 0]) else 0.0
                    ns_count += 1
            per_scene.append(rec)
        batch_i += 1
        print(json.dumps({"event": "batch", "batch": batch_i, "scenes": int(acc[5].item()),
                          "elapsed_s": round(time.time() - run_start, 1)}), flush=True)

    denom_scene = max(float(acc[5].item()), 1.0)
    denom_ade = max(float(acc[4].item()), 1.0)
    denom_near = max(float(acc[7].item()), 1.0)
    summary = {
        "checkpoint": str(args.checkpoint),
        "opt_step": opt_step,
        "epoch": epoch,
        "lineage": "nvs_v6_wan_e2e",
        "depth": meta["depth"],
        "heads": meta["heads"],
        "target_dim": meta["target_dim"],
        # Recorded separately because it is no longer ``target_dim``: the future head
        # emits the Teacher's feature width, and a checkpoint loaded at the wrong one
        # is a different model, not a differently-named same one.
        "register_target_dim": meta["register_target_dim"],
        "scene_tokens": int(model.scene_tokens),
        "split": "navtest (held-out)",
        "scoring": "official navsim pdm_score path (official MetricCache + official PDMScorer)",
        "oracle_version": ORACLE_VERSION,
        "scenes_evaluated": int(denom_scene),
        "scenes_indexed": len(candidates),
        "scenes_indexed_total": total_indexed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_pdms": round(float(acc[0].item()) / denom_scene, 4),
        "oracle_best_pdms": round(float(acc[1].item()) / denom_scene, 4),
        "oracle_match_rate": round(float(acc[2].item()) / denom_scene, 4),
        "matched_gt_ade_m": round(float(acc[3].item()) / denom_ade, 4),
        "all64_nearest_gt_ade_m": round(float(acc[6].item()) / denom_near, 4),
        "noscorer_mean64_pdms": round(ns_mean_sum / max(ns_count, 1), 4),
        "noscorer_cand0_pdms": round(ns_cand0_sum / max(ns_count, 1), 4),
        "oracle_failures": oracle_failures,
        "verify_scenes": verified,
        "verify_max_abs_diff_vs_pdm_score": round(verify_worst, 9),
        "elapsed_s": round(time.time() - run_start, 1),
    }
    result_path.write_text(json.dumps({"summary": summary, "per_scene": per_scene}, indent=2) + "\n")
    print(json.dumps({"event": "summary", "result_path": str(result_path), **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
