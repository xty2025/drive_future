"""Attribute the PDMS gap on navtrain and emit the hard-scene training lists.

Reads the frozen-generator cache produced by ``build_scorer_cache.py`` and answers
the question the experiment hinges on: for the low-PDMS scenes, is the candidate
POOL too weak (nothing good to pick) or is the SELECTION wrong (a good candidate
exists and was not picked)?  The two have different remedies, so they are counted
separately and trained separately.

The breakdown is byte-for-byte the same shape as the navtest diagnosis so the two
splits can be read side by side:

* per-``oracle_best`` bucket: scenes, share, mean selected PDMS, contribution to
  the total (oracle_best - selected) gap;
* hard scenes (selected < ``--hard-threshold``) split into
  selection-limited / middle / pool-limited with each one's recoverable headroom;
* the "perfect selection" ceilings.

It also flags a known navtrain cache artifact: scenes where ALL 64 candidates get
drivable-area-compliance == 0, the signature of the float32-quantized drivable
area in the DrivoR train cache.  Every hard list is emitted twice -- with and
without those scenes -- because a scorer cannot learn anything useful from them.

Single process, streams shard by shard, memory O(scenes).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_CACHE = "artifacts/scorer_hard_mining/cache"
DEFAULT_OUTPUT = "artifacts/scorer_hard_mining"

BUCKETS = [(0.99, 1.01), (0.95, 0.99), (0.90, 0.95), (0.80, 0.90), (0.50, 0.80), (0.00, 0.50)]
PDMS = 6
DAC = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--hard-threshold", type=float, default=0.80, help="selected PDMS below this = hard scene")
    parser.add_argument("--selection-threshold", type=float, default=0.95, help="oracle_best at/above this = selection-limited")
    parser.add_argument("--pool-threshold", type=float, default=0.90, help="oracle_best below this = pool-limited")
    parser.add_argument("--heldout-scenes", type=int, default=2000, help="held-out scenes for in-training model selection")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_shard(path: Path) -> dict[str, Any]:
    """mmap the shard so the big scene/trajectory tensors never hit RSS."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def collect(cache_dir: Path) -> dict[str, dict[str, float]]:
    """scene_id -> per-scene scalars, deduplicated across shards (first wins)."""
    rows: dict[str, dict[str, float]] = {}
    duplicates = 0
    for shard_path in sorted(cache_dir.glob("shard_*.pt")):
        shard = load_shard(shard_path)
        raw7 = shard["raw7"]
        valid = shard["final_valid"].to(torch.bool)
        selected = shard["selected_index"].to(torch.long)
        scene_ids = list(shard["scene_ids"])
        for row, scene_id in enumerate(scene_ids):
            if scene_id in rows:
                duplicates += 1
                continue
            score = raw7[row]
            if not bool(valid[row].all()) or not bool(torch.isfinite(score).all()):
                continue
            pdms = score[:, PDMS]
            chosen = int(selected[row].clamp(0, pdms.shape[0] - 1))
            order = torch.argsort(pdms, descending=True, stable=True)
            rank_of_selected = int((order == chosen).nonzero(as_tuple=True)[0][0])
            dac = score[:, DAC]
            rows[scene_id] = {
                "selected_pdms": float(pdms[chosen]),
                "oracle_best": float(pdms.max()),
                "mean64": float(pdms.mean()),
                "selected_index": float(chosen),
                "oracle_index": float(int(order[0])),
                "rank_of_selected": float(rank_of_selected),
                "dac_all_zero": float(bool((dac == 0).all())),
                "nc_all_zero": float(bool((score[:, 0] == 0).all())),
                "n_valid": float(int(valid[row].sum())),
            }
    if duplicates:
        print(json.dumps({"event": "duplicate_scene_ids_dropped", "count": duplicates}), flush=True)
    return rows


def diagnose(rows: dict[str, dict[str, float]], args) -> dict[str, Any]:
    total = len(rows)
    selected = np.array([r["selected_pdms"] for r in rows.values()], dtype=np.float64)
    best = np.array([r["oracle_best"] for r in rows.values()], dtype=np.float64)
    mean64 = np.array([r["mean64"] for r in rows.values()], dtype=np.float64)
    rank_sel = np.array([r["rank_of_selected"] for r in rows.values()], dtype=np.float64)
    artifact = np.array([r["dac_all_zero"] for r in rows.values()], dtype=np.bool_)

    gap = best - selected
    buckets = []
    for low, high in BUCKETS:
        mask = (best >= low) & (best < high)
        count = int(mask.sum())
        buckets.append({
            "bucket": f"[{low:.2f}, {high:.2f})",
            "scenes": count,
            "share": round(count / max(total, 1), 4),
            "mean_selected_pdms": round(float(selected[mask].mean()), 4) if count else None,
            "gap_contribution": round(float(gap[mask].sum()) / max(total, 1), 4),
        })

    hard = selected < args.hard_threshold
    selection_limited = hard & (best >= args.selection_threshold)
    pool_limited = hard & (best < args.pool_threshold)
    middle = hard & ~selection_limited & ~pool_limited

    def group(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        return {
            "scenes": count,
            "share_of_all": round(count / max(total, 1), 4),
            "share_of_hard": round(count / max(int(hard.sum()), 1), 4),
            "recoverable_gap": round(float(gap[mask].sum()) / max(total, 1), 4),
            "mean_selected_pdms": round(float(selected[mask].mean()), 4) if count else None,
            "mean_oracle_best": round(float(best[mask].mean()), 4) if count else None,
            "suspect_map_artifact": int((mask & artifact).sum()),
        }

    perfect_where_selectable = np.where(best >= args.selection_threshold, best, selected)
    return {
        "scenes": total,
        "mean_selected_pdms": round(float(selected.mean()), 4),
        "mean_oracle_best": round(float(best.mean()), 4),
        "mean_noscorer_mean64": round(float(mean64.mean()), 4),
        "total_gap": round(float(gap.mean()), 4),
        "oracle_match_rate": round(float((rank_sel == 0).mean()), 4),
        "median_rank_of_selected": float(np.median(rank_sel)) if total else None,
        "buckets": buckets,
        "hard_threshold": args.hard_threshold,
        "selection_threshold": args.selection_threshold,
        "pool_threshold": args.pool_threshold,
        "hard": group(hard),
        "hard_selection_limited": group(selection_limited),
        "hard_middle": group(middle),
        "hard_pool_limited": group(pool_limited),
        "non_hard_gap": round(float(gap[~hard].sum()) / max(total, 1), 4),
        "hard_share_of_total_gap": round(float(gap[hard].sum()) / max(float(gap.sum()), 1e-12), 4),
        "ceiling_perfect_where_selectable": round(float(perfect_where_selectable.mean()), 4),
        "ceiling_perfect_everywhere": round(float(best.mean()), 4),
        "suspect_map_artifact_scenes": int(artifact.sum()),
        "suspect_map_artifact_share": round(float(artifact.mean()), 4) if total else None,
        "suspect_map_artifact_within_hard": int((hard & artifact).sum()),
    }


def write_list(output_dir: Path, name: str, scenes: list[str], meta: dict[str, Any]) -> None:
    payload = {"_meta": meta, "count": len(scenes), "scenes": sorted(scenes)}
    (output_dir / f"scenes_{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(cache_dir.glob("shard_*.pt"))
    if not shards:
        raise RuntimeError(f"no shard_*.pt under {cache_dir}; run build_scorer_cache.py first")

    rows = collect(cache_dir)
    if not rows:
        raise RuntimeError("cache produced zero usable scenes")
    diagnosis = diagnose(rows, args)
    print(json.dumps({"event": "diagnosis", **{k: v for k, v in diagnosis.items() if k != "buckets"}}), flush=True)
    print(json.dumps({"event": "buckets", "buckets": diagnosis["buckets"]}), flush=True)

    rng = random.Random(args.seed)
    all_scenes = sorted(rows)
    heldout = set(rng.sample(all_scenes, min(args.heldout_scenes, len(all_scenes))))
    pool = [s for s in all_scenes if s not in heldout]

    def is_hard(sid: str) -> bool:
        return rows[sid]["selected_pdms"] < args.hard_threshold

    def is_selection(sid: str) -> bool:
        return rows[sid]["oracle_best"] >= args.selection_threshold

    def is_pool_limited(sid: str) -> bool:
        return rows[sid]["oracle_best"] < args.pool_threshold

    def is_artifact(sid: str) -> bool:
        return bool(rows[sid]["dac_all_zero"])

    hard_all = [s for s in pool if is_hard(s)]
    hard_selection = [s for s in hard_all if is_selection(s)]
    hard_pool = [s for s in hard_all if is_pool_limited(s)]
    hard_middle = [s for s in hard_all if s not in set(hard_selection) | set(hard_pool)]
    hard_all_clean = [s for s in hard_all if not is_artifact(s)]
    hard_selection_clean = [s for s in hard_selection if not is_artifact(s)]
    non_hard_clean = [s for s in pool if not is_hard(s) and not is_artifact(s)]
    non_hard = [s for s in pool if not is_hard(s)]

    random_matched = rng.sample(non_hard, min(len(hard_all), len(non_hard)))
    random_matched_clean = rng.sample(non_hard_clean, min(len(hard_all_clean), len(non_hard_clean)))
    all_train = list(pool)

    meta = {
        "cache_dir": str(Path(cache_dir).resolve()),
        "seed": args.seed,
        "hard_threshold": args.hard_threshold,
        "selection_threshold": args.selection_threshold,
        "pool_threshold": args.pool_threshold,
        "heldout_excluded": True,
        "oracle": "DrivoR get_sub_score on the navtrain float32 metric cache",
        "split": "navtrain (navtest is never used for mining)",
    }
    write_list(output_dir, "all", all_train, meta)
    write_list(output_dir, "hard_all", hard_all, meta)
    write_list(output_dir, "hard_all_clean", hard_all_clean, meta)
    write_list(output_dir, "hard_selection", hard_selection, meta)
    write_list(output_dir, "hard_selection_clean", hard_selection_clean, meta)
    write_list(output_dir, "hard_pool", hard_pool, meta)
    write_list(output_dir, "hard_middle", hard_middle, meta)
    write_list(output_dir, "random_matched", random_matched, meta)
    write_list(output_dir, "random_matched_clean", random_matched_clean, meta)
    write_list(output_dir, "heldout", sorted(heldout), meta)

    overlap = {
        "heldout_and_hard_all": len(heldout & set(hard_all)),
        "heldout_and_random_matched": len(heldout & set(random_matched)),
        "hard_selection_and_hard_pool": len(set(hard_selection) & set(hard_pool)),
        "hard_split_sum_equals_hard_all": (len(hard_selection) + len(hard_pool) + len(hard_middle)) == len(hard_all),
    }
    summary = {
        "diagnosis": diagnosis,
        "lists": {
            "all": len(all_train), "hard_all": len(hard_all), "hard_all_clean": len(hard_all_clean),
            "hard_selection": len(hard_selection), "hard_selection_clean": len(hard_selection_clean),
            "hard_pool": len(hard_pool), "hard_middle": len(hard_middle),
            "random_matched": len(random_matched), "random_matched_clean": len(random_matched_clean),
            "heldout": len(heldout),
        },
        "overlap_checks": overlap,
    }
    (output_dir / "diagnosis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "lists_written", **summary["lists"], **overlap}), flush=True)
    if not overlap["hard_split_sum_equals_hard_all"] or overlap["heldout_and_hard_all"] or overlap["heldout_and_random_matched"]:
        raise AssertionError(f"scene-list invariants violated: {overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
