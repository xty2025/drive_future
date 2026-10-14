"""Compare the OFFICIAL navsim PDMS path with DrivoR's ``get_sub_score`` on the
same navtest scenes, scoring the logged human future in both.

Official path : official MetricCache (has ``trajectory`` = PDM-Closed proposal)
                + ``navsim.evaluate.pdm_score.pdm_score`` + official ``PDMScorer``
                -> proposals are [PDM-Closed, prediction], progress is normalized
                   by ``np.max(raw_progress)`` over BOTH.
DrivoR path   : DrivoR train MetricCache (has scalar ``pdm_progress``)
                + ``compute_navsim_score.get_sub_score`` + adapted ``PDMScorer``
                -> only the prediction is scored, progress is normalized by
                   ``np.maximum(raw_progress, pdm_progress)``.

Read-only; caches and logs are trusted local data.
"""
from __future__ import annotations

import os

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/data/navsim/maps")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import argparse
import json
import lzma
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.agents.drivoR.score_module.compute_navsim_score import get_sub_score

from nvs_v6_wan.real_data import _relative_pose

OFFICIAL_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/drivor_official_navtest_full_cache_20261007/metric_cache"
DRIVOR_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtest_20261006/metric_cache"
NAVTEST_LOG = "/mnt/data/navsim/test_navsim_logs/test"

PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
FUTURE_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)
SUBSCORE_NAMES = ["nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms"]

_SIMULATOR = None
_SCORER = None


def _official_objects():
    global _SIMULATOR, _SCORER
    if _SIMULATOR is None:
        _SIMULATOR = PDMSimulator(PROPOSAL_SAMPLING)
        _SCORER = PDMScorer(PROPOSAL_SAMPLING, PDMScorerConfig())
    return _SIMULATOR, _SCORER


def _load_cache(path: str):
    with lzma.open(path, "rb") as handle:
        return pickle.load(handle)


def human_future(log_root: str, log_name: str, token: str) -> np.ndarray:
    with (Path(log_root) / f"{log_name}.pkl").open("rb") as handle:
        frames = pickle.load(handle)
    idx = next(i for i, fr in enumerate(frames) if str(fr["token"]) == token)
    if idx + 8 >= len(frames):
        raise ValueError("no future context")
    reference = np.asarray(frames[idx]["ego2global"], dtype=np.float64)
    return np.stack(
        [
            _relative_pose(reference, np.asarray(frames[i]["ego2global"], dtype=np.float64))
            for i in range(idx + 1, idx + 9)
        ]
    ).astype(np.float32)


def score_scene(rec: dict) -> dict:
    simulator, scorer = _official_objects()
    out = {"token": rec["token"], "log_name": rec["log_name"]}
    traj = human_future(NAVTEST_LOG, rec["log_name"], rec["token"])
    out["human_traj"] = traj.tolist()

    official_cache = _load_cache(rec["official_cache_path"])
    result = pdm_score(
        metric_cache=official_cache,
        model_trajectory=Trajectory(traj),
        future_sampling=PROPOSAL_SAMPLING,
        simulator=simulator,
        scorer=scorer,
    )
    out["official"] = [
        float(result.no_at_fault_collisions),
        float(result.drivable_area_compliance),
        float(result.ego_progress),
        float(result.time_to_collision_within_bound),
        float(result.comfort),
        float(result.driving_direction_compliance),
        float(result.score),
    ]

    drivor_cache = _load_cache(rec["drivor_cache_path"])
    pdm_progress = getattr(drivor_cache, "pdm_progress", None)
    out["drivor_pdm_progress"] = None if pdm_progress is None else np.asarray(pdm_progress).ravel().tolist()
    out["official_has_trajectory"] = hasattr(official_cache, "trajectory")

    raw, *_ = get_sub_score(rec["drivor_cache_path"], traj[None], True)
    out["drivor"] = np.asarray(raw, dtype=np.float64)[0].tolist()

    # Same DrivoR scorer, but progress normalized the official way (reference =
    # the batch max, i.e. the prediction itself) -> isolates the reference effect.
    scores_none, *_ = _score_with_progress_none(rec["drivor_cache_path"], traj)
    out["drivor_no_pdm_ref"] = scores_none
    return out


def _score_with_progress_none(cache_path: str, traj: np.ndarray) -> tuple[list[float], None]:
    from navsim.agents.drivoR.score_module import compute_navsim_score as cns

    cache = _load_cache(cache_path)
    initial_ego_state = cache.ego_state
    pred = cns.transform_trajectory(Trajectory(traj), initial_ego_state)
    states = cns.get_trajectory_as_array(pred, cns.simulator.proposal_sampling, initial_ego_state.time_point)[None]
    simulated = cns.simulator.simulate_proposals(states, initial_ego_state)
    scorer = cns.PDMScorer(cns.proposal_sampling, cns.PDMScorerConfig())
    final = scorer.score_proposals(
        simulated,
        cache.observation,
        cache.centerline,
        cache.route_lane_ids,
        cache.drivable_area_map,
        None,
    )
    cols = [
        scorer._multi_metrics[cns.MultiMetricIndex.NO_COLLISION, 0],
        scorer._multi_metrics[cns.MultiMetricIndex.DRIVABLE_AREA, 0],
        scorer._weighted_metrics[cns.WeightedMetricIndex.PROGRESS, 0],
        scorer._weighted_metrics[cns.WeightedMetricIndex.TTC, 0],
        scorer._weighted_metrics[cns.WeightedMetricIndex.COMFORTABLE, 0],
        scorer._weighted_metrics[cns.WeightedMetricIndex.DRIVING_DIRECTION, 0],
        final[0],
    ]
    return [float(v) for v in cols], None


def aggregate(rows: list[dict], key: str) -> dict:
    arr = np.asarray([r[key] for r in rows], dtype=np.float64)
    return {name: round(float(arr[:, i].mean()), 4) for i, name in enumerate(SUBSCORE_NAMES)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="artifacts/human_pdms_official_vs_drivor/result.json")
    args = parser.parse_args()

    pool = []
    for path in sorted(Path(DRIVOR_CACHE).glob("*/*/*/metric_cache.pkl")):
        token = path.parent.name
        log_name = path.parent.parent.parent.name
        official = Path(OFFICIAL_CACHE) / log_name / "unknown" / token / "metric_cache.pkl"
        if official.exists():
            pool.append(
                {
                    "token": token,
                    "log_name": log_name,
                    "drivor_cache_path": str(path),
                    "official_cache_path": str(official),
                }
            )
    print(json.dumps({"event": "pool", "drivor_scenes_with_official_cache": len(pool)}), flush=True)

    picked = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, row in enumerate(executor.map(score_scene, picked)):
            rows.append(row)
            if (i + 1) % 25 == 0:
                print(json.dumps({"event": "progress", "done": i + 1, "total": len(picked)}), flush=True)

    summary = {
        "n": len(rows),
        "official": aggregate(rows, "official"),
        "drivor": aggregate(rows, "drivor"),
        "drivor_no_pdm_ref": aggregate(rows, "drivor_no_pdm_ref"),
    }
    official = np.asarray([r["official"][6] for r in rows])
    drivor = np.asarray([r["drivor"][6] for r in rows])
    summary["official_pdms_median"] = round(float(np.median(official)), 4)
    summary["drivor_pdms_median"] = round(float(np.median(drivor)), 4)
    summary["mean_delta_official_minus_drivor"] = round(float((official - drivor).mean()), 4)
    summary["n_delta_gt_0p02"] = int((official - drivor > 0.02).sum())
    summary["n_official_zero"] = int((official == 0.0).sum())
    summary["n_drivor_zero"] = int((drivor == 0.0).sum())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_scene": rows}, indent=2))
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
