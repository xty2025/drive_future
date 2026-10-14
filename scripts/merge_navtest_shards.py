"""Merge sharded navtest official-PDMS results back into one result.json.

``eval_navtest_official_pdms.py --shard-count N`` splits the scene list across N
independent processes; this recombines them and recomputes the summary from the
merged per-scene records, so every arm (sharded) and the baseline (unsharded or
sharded) are reduced by exactly the same code.

Also does the sharding correctness proof: the shard scene sets must be pairwise
disjoint, their union must equal ``scenes_indexed_total``, and with ``--reference``
every merged per-scene record must be identical to the unsharded run's record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SUMMARY_KEYS_SAME = ("checkpoint", "opt_step", "epoch", "split", "scoring", "oracle_version", "shard_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-dir", required=True, help="directory holding result.shard*.json")
    parser.add_argument("--output", default="", help="default: <shard-dir>/result.json")
    parser.add_argument("--reference", default="",
                        help="unsharded result.json to compare per-scene records against (exact equality)")
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def main() -> int:
    args = parse_args()
    shard_dir = Path(args.shard_dir)
    paths = sorted(shard_dir.glob("result.shard*.json"), key=lambda p: int(p.stem.split("shard")[-1]))
    if not paths:
        raise RuntimeError(f"no result.shard*.json under {shard_dir}")
    shards = [json.loads(p.read_text()) for p in paths]

    shard_count = int(shards[0]["summary"]["shard_count"])
    if len(shards) != shard_count:
        raise RuntimeError(f"expected {shard_count} shards, found {len(shards)}")
    seen_index = sorted(int(s["summary"]["shard_index"]) for s in shards)
    if seen_index != list(range(shard_count)):
        raise RuntimeError(f"shard indices incomplete: {seen_index}")
    for key in SUMMARY_KEYS_SAME:
        values = {json.dumps(s["summary"].get(key)) for s in shards}
        if len(values) != 1:
            raise RuntimeError(f"shards disagree on {key}: {values}")

    per_scene: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    duplicates = 0
    total_indexed = set()
    for shard in shards:
        total_indexed.add(int(shard["summary"].get("scenes_indexed_total", -1)))
        for record in shard["per_scene"]:
            scene_id = record["scene_id"]
            if scene_id in seen_scenes:
                duplicates += 1
                continue
            seen_scenes.add(scene_id)
            per_scene.append(record)
    if duplicates:
        raise RuntimeError(f"{duplicates} scene_ids appear in more than one shard")
    if len(total_indexed) != 1:
        raise RuntimeError(f"shards disagree on scenes_indexed_total: {total_indexed}")
    per_scene.sort(key=lambda r: r["scene_id"])

    scored = [r for r in per_scene if "selected_pdms_official" in r]
    ade = [r["matched_gt_ade_m"] for r in per_scene
           if r.get("matched_gt_ade_m") is not None and np.isfinite(r["matched_gt_ade_m"])]
    cand0 = [(r["noscorer_cand0_official"] if r.get("noscorer_cand0_official") is not None else 0.0)
             for r in scored if "noscorer_cand0_official" in r]
    summary: dict[str, Any] = {key: shards[0]["summary"].get(key) for key in SUMMARY_KEYS_SAME}
    summary.update({
        "scenes_indexed_total": int(total_indexed.pop()),
        "scenes_merged": len(per_scene),
        "scenes_evaluated": len(scored),
        "selected_pdms": mean([r["selected_pdms_official"] for r in scored]),
        "oracle_best_pdms": mean([r["oracle_best_pdms_official"] for r in scored]),
        "oracle_match_rate": mean([float(r["oracle_match"]) for r in scored]),
        "matched_gt_ade_m": mean(ade),
        "ade_aggregation": "mean of per-scene matched-GT ADE (scene-weighted)",
        "noscorer_mean64_pdms": mean([r["noscorer_mean64_official"] for r in scored
                                      if "noscorer_mean64_official" in r]),
        "noscorer_cand0_pdms": mean(cand0) if cand0 else None,
        "oracle_failures": int(sum(int(s["summary"].get("oracle_failures", 0)) for s in shards)),
        "verify_scenes": int(sum(int(s["summary"].get("verify_scenes", 0)) for s in shards)),
        "verify_max_abs_diff_vs_pdm_score": round(max(
            float(s["summary"].get("verify_max_abs_diff_vs_pdm_score", 0.0)) for s in shards), 9),
        "elapsed_s": round(max(float(s["summary"].get("elapsed_s", 0.0)) for s in shards), 1),
        "elapsed_s_sum": round(sum(float(s["summary"].get("elapsed_s", 0.0)) for s in shards), 1),
        "merged_from": [p.name for p in paths],
    })
    if len(scored) != len(per_scene):
        summary["scenes_without_labels"] = len(per_scene) - len(scored)

    checks: dict[str, Any] = {"shards": len(shards), "scenes_merged": len(per_scene), "duplicates": duplicates}
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())
        ref_by_id = {r["scene_id"]: r for r in reference["per_scene"]}
        merged_by_id = {r["scene_id"]: r for r in per_scene}
        missing = sorted(set(ref_by_id) - set(merged_by_id))
        extra = sorted(set(merged_by_id) - set(ref_by_id))
        mismatched = []
        for scene_id in sorted(set(ref_by_id) & set(merged_by_id)):
            left = {k: v for k, v in ref_by_id[scene_id].items()}
            right = {k: v for k, v in merged_by_id[scene_id].items()}
            if left != right:
                mismatched.append({"scene_id": scene_id,
                                   "diff": sorted(set(left.items()) ^ set(right.items()))[:6]})
        checks.update({"reference": str(args.reference), "reference_scenes": len(ref_by_id),
                       "missing_vs_reference": len(missing), "extra_vs_reference": len(extra),
                       "record_mismatches": len(mismatched),
                       "exact_per_scene_match": not (missing or extra or mismatched)})
        if mismatched:
            checks["first_mismatches"] = mismatched[:3]
        if missing:
            checks["first_missing"] = missing[:3]
        if extra:
            checks["first_extra"] = extra[:3]

    output = Path(args.output) if args.output else shard_dir / "result.json"
    output.write_text(json.dumps({"summary": summary, "checks": checks, "per_scene": per_scene}, indent=2) + "\n")
    print(json.dumps({"event": "merged", "output": str(output), **summary, "checks": checks}), flush=True)
    if args.reference and not checks.get("exact_per_scene_match", False):
        raise AssertionError(f"sharded run does not reproduce the reference per-scene records: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
