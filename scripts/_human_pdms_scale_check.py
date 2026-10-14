"""Cache-scale sanity check: score the LOGGED HUMAN future through each split's
official metric_cache with the official ``get_sub_score``.

If the human-driven trajectory scores ~high on BOTH navtrain and navtest
caches, the two caches share semantics/scale and any model navtrain-vs-navtest
gap is generalization, not a cache adaptation bug.  A systematically lower
human score on navtest would indicate split difficulty or a cache-build
semantics difference instead.

Read-only: touches no package file; caches and logs are trusted local data.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np

from navsim.agents.drivoR.score_module.compute_navsim_score import get_sub_score
from nvs_v6_wan.real_data import _relative_pose

NAVTRAIN_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache"
NAVTEST_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtest_20261006/metric_cache"
NAVTRAIN_LOG = "/mnt/data/navsim/trainval_navsim_logs/trainval"
NAVTEST_LOG = "/mnt/data/navsim/test_navsim_logs/test"


def human_future(log_root: str, log_name: str, token: str) -> np.ndarray:
    with (Path(log_root) / f"{log_name}.pkl").open("rb") as handle:
        frames = pickle.load(handle)
    idx = next(i for i, fr in enumerate(frames) if str(fr["token"]) == token)
    if idx + 8 >= len(frames):
        raise ValueError("no future context")
    reference = np.asarray(frames[idx]["ego2global"], dtype=np.float64)
    return np.stack([_relative_pose(reference, np.asarray(frames[i]["ego2global"], dtype=np.float64))
                     for i in range(idx + 1, idx + 9)]).astype(np.float32)  # [8,3]


def score_split(tag: str, cache_root: str, log_root: str, scenes: list[dict], n: int, seed: int) -> dict:
    picked = random.Random(seed).sample(scenes, min(n, len(scenes)))
    finals = []
    for rec in picked:
        traj = human_future(log_root, rec["log_name"], rec["token"])[None]  # [1,8,3]
        scores, *_ = get_sub_score(rec["cache_path"], traj, True)
        finals.append(float(np.asarray(scores, dtype=np.float32)[0, 6]))
    finals = np.asarray(finals, dtype=np.float64)
    out = {"tag": tag, "n": int(finals.size), "human_pdms_mean": round(float(finals.mean()), 4),
           "human_pdms_median": round(float(np.median(finals)), 4),
           "human_pdms_p10": round(float(np.percentile(finals, 10)), 4),
           "human_pdms_min": round(float(finals.min()), 4)}
    print(json.dumps({"event": "human_scale", **out}), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenes-json", default="artifacts/bev_map_random64/scenes_random64.json")
    args = parser.parse_args()

    with open(args.scenes_json) as handle:
        navtrain = json.load(handle)

    navtest = []
    for path in sorted(Path(NAVTEST_CACHE).glob("*/*/*/metric_cache.pkl")):
        navtest.append({"token": path.parent.name, "log_name": path.parent.parent.parent.name,
                        "cache_path": str(path)})
    print(json.dumps({"event": "navtest_pool", "scenes": len(navtest)}), flush=True)

    score_split("navtrain", NAVTRAIN_CACHE, NAVTRAIN_LOG, navtrain, args.n, args.seed)
    score_split("navtest", NAVTEST_CACHE, NAVTEST_LOG, navtest, args.n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
