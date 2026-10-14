"""Diagnose why 15/1000 navtest scenes are DAC=0 in the DrivoR train metric cache
but DAC=1 in the official navsim metric cache for the same token.

Compares the two caches field by field and re-runs the off-road test with each
cache's own ``drivable_area_map``.  Read-only; caches are trusted local data.
"""
from __future__ import annotations

import os

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/data/navsim/maps")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import argparse
import copy
import lzma
import pickle
from pathlib import Path

import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex

OFFICIAL_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/drivor_official_navtest_full_cache_20261007/metric_cache"
DRIVOR_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtest_20261006/metric_cache"
NAVTEST_LOG = "/mnt/data/navsim/test_navsim_logs/test"
PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)


def load(path: str):
    with lzma.open(path, "rb") as handle:
        return pickle.load(handle)


def describe(tag: str, cache) -> dict:
    dam = cache.drivable_area_map
    layers = {str(layer): int(len(getattr(dam, "layers", {}).get(layer, []))) for layer in getattr(dam, "layers", {})}
    info = {
        "tag": tag,
        "dam_type": type(dam).__name__,
        "dam_layers": layers,
        "centerline_len_m": round(float(cache.centerline.length), 3),
        "route_lane_ids": len(cache.route_lane_ids),
        "observation_unique_objects": int(len(cache.observation.unique_objects)),
        "observation_shape": list(np.asarray(cache.observation.state_se2_array).shape)
        if hasattr(cache.observation, "state_se2_array")
        else None,
        "ego_x": round(float(cache.ego_state.rear_axle.x), 4),
        "ego_y": round(float(cache.ego_state.rear_axle.y), 4),
        "ego_h": round(float(cache.ego_state.rear_axle.heading), 6),
        "ego_time_us": int(cache.ego_state.time_us),
    }
    if hasattr(cache, "trajectory"):
        traj = cache.trajectory
        states = getattr(traj, "states", None) or getattr(traj, "interpolated_states", None)
        info["pdm_traj_states"] = int(len(states)) if states is not None else -1
    if hasattr(cache, "pdm_progress"):
        info["pdm_progress"] = np.asarray(cache.pdm_progress).ravel().tolist()
    return info


def offroad_mask(cache, traj: np.ndarray) -> np.ndarray:
    simulator = PDMSimulator(PROPOSAL_SAMPLING)
    scorer = PDMScorer(PROPOSAL_SAMPLING, PDMScorerConfig())
    pred = transform_trajectory(Trajectory(traj.astype(np.float32)), cache.ego_state)
    states = get_trajectory_as_array(pred, PROPOSAL_SAMPLING, cache.ego_state.time_point)[None]
    simulated = simulator.simulate_proposals(states, cache.ego_state)
    scorer._reset(simulated, cache.observation, cache.centerline, cache.route_lane_ids, cache.drivable_area_map)
    scorer._calculate_ego_area()
    return scorer._ego_areas[0, :, EgoAreaIndex.NON_DRIVABLE_AREA].astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", required=True)
    parser.add_argument("--log-name", required=True)
    args = parser.parse_args()

    with (Path(NAVTEST_LOG) / f"{args.log_name}.pkl").open("rb") as handle:
        frames = pickle.load(handle)

    from nvs_v6_wan.real_data import _relative_pose

    for token in args.tokens:
        idx = next(i for i, fr in enumerate(frames) if str(fr["token"]) == token)
        reference = np.asarray(frames[idx]["ego2global"], dtype=np.float64)
        traj = np.stack(
            [_relative_pose(reference, np.asarray(frames[i]["ego2global"], dtype=np.float64)) for i in range(idx + 1, idx + 9)]
        ).astype(np.float32)

        official = load(f"{OFFICIAL_CACHE}/{args.log_name}/unknown/{token}/metric_cache.pkl")
        drivor = load(f"{DRIVOR_CACHE}/{args.log_name}/unknown/{token}/metric_cache.pkl")
        print("== token", token)
        for info in (describe("official", official), describe("drivor", drivor)):
            print("  ", info)
        mask_o = offroad_mask(official, traj)
        mask_d = offroad_mask(drivor, traj)
        print("   offroad_steps official:", mask_o.sum(), "drivor:", mask_d.sum(), "first_drivor:", int(np.argmax(mask_d)) if mask_d.any() else -1)

        # Cross test: DrivoR observation + official drivable map and vice versa.
        cross_a = offroad_mask(_swap(drivor, official.drivable_area_map), traj)
        cross_b = offroad_mask(_swap(official, drivor.drivable_area_map), traj)
        print("   cross drivor-cache+official-map offroad:", int(cross_a.sum()), " official-cache+drivor-map offroad:", int(cross_b.sum()))


def _swap(cache, dam):
    clone = copy.deepcopy(cache)
    clone.drivable_area_map = dam
    return clone


if __name__ == "__main__":
    raise SystemExit(main())
