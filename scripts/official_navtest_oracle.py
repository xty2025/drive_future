"""Official navsim PDMS oracle for batch candidate scoring.

Scores with the OFFICIAL navsim evaluation path:
  * official ``MetricCache`` (float64 drivable-area polygons + the PDM-Closed
    ``trajectory`` proposal),
  * official ``navsim...scoring.pdm_scorer.PDMScorer`` and ``PDMSimulator``,
  * official aggregation from ``navsim/evaluate/pdm_score.py``.

``pdm_score`` scores the pair [PDM-Closed, prediction].  The only cross-proposal
coupling in the official aggregation is ``max_raw_progress = np.max(raw_progress)``
over that pair, so we simulate/score [PDM, cand_0 ... cand_63] once and re-apply
the two-way max per candidate -- numerically identical to 64 separate
``pdm_score`` calls (asserted by ``verify_against_pdm_score``).

Returned columns match ``get_sub_score``: [NC, DAC, EP, TTC, comfort, DDC, PDMS].
Read-only; metric caches are trusted local data produced by our own caching runs.
"""
from __future__ import annotations

import lzma
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

from nvs_v6_wan_e2e.oracle import HEAD_INDICES, OnlineOracle, OracleLabels, _hash_object, _hash_tensor

PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)


def load_official_metric_cache(path: str | Path) -> Any:
    with lzma.open(path, "rb") as handle:
        return pickle.load(handle)


class OfficialNavsimOracle(OnlineOracle):
    """Official navsim PDM-Score for [B,64,8,3] candidate batches."""

    def __init__(self, cache_paths: dict[str, str | Path], evaluator_version: str) -> None:
        self.cache_paths = {key: str(value) for key, value in cache_paths.items()}
        self.evaluator_version = evaluator_version
        self.simulator = PDMSimulator(PROPOSAL_SAMPLING)
        self.config = PDMScorerConfig()
        self.scorer = PDMScorer(PROPOSAL_SAMPLING, self.config)
        self._cache: dict[str, Any] = {}

    def metric_cache(self, scene_id: str) -> Any:
        path = self.cache_paths[scene_id]
        if path not in self._cache:
            self._cache = {path: load_official_metric_cache(path)}  # keep only one cache resident
        return self._cache[path]

    def subscores(self, scene_id: str, candidates: np.ndarray) -> np.ndarray:
        """[N,7] official subscores for N candidate trajectories [N,8,3]."""
        cache = self.metric_cache(scene_id)
        initial_ego_state = cache.ego_state

        pdm_states = get_trajectory_as_array(cache.trajectory, PROPOSAL_SAMPLING, initial_ego_state.time_point)
        cand_states = np.stack(
            [
                get_trajectory_as_array(
                    transform_trajectory(Trajectory(row), initial_ego_state),
                    PROPOSAL_SAMPLING,
                    initial_ego_state.time_point,
                )
                for row in candidates
            ]
        )
        states = np.concatenate([pdm_states[None], cand_states], axis=0)
        simulated = self.simulator.simulate_proposals(states, initial_ego_state)

        scorer = self.scorer
        scorer._reset(simulated, cache.observation, cache.centerline, cache.route_lane_ids, cache.drivable_area_map)
        scorer._calculate_ego_area()
        scorer._calculate_no_at_fault_collision()
        scorer._calculate_drivable_area_compliance()
        scorer._calculate_driving_direction_compliance()
        scorer._calculate_progress()
        scorer._calculate_ttc()
        scorer._calculate_is_comfortable()

        multiplicative = scorer._multi_metrics.prod(axis=0)
        raw_progress = scorer._progress_raw * multiplicative
        # Official reference: the PDM-Closed proposal of the same scene only.
        max_raw_progress = np.maximum(raw_progress, raw_progress[0])
        ego_progress = np.ones_like(raw_progress)
        fast = max_raw_progress > self.config.progress_distance_threshold
        ego_progress[fast] = raw_progress[fast] / max_raw_progress[fast]
        ego_progress[(~fast) & (multiplicative == 0.0)] = 0.0

        weighted = scorer._weighted_metrics.copy()
        weighted[WeightedMetricIndex.PROGRESS] = ego_progress
        weights = self.config.weighted_metrics_array
        weighted_score = (weighted * weights[..., None]).sum(axis=0) / weights.sum()
        final = multiplicative * weighted_score

        columns = np.stack(
            [
                scorer._multi_metrics[MultiMetricIndex.NO_COLLISION],
                scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA],
                ego_progress,
                weighted[WeightedMetricIndex.TTC],
                weighted[WeightedMetricIndex.COMFORTABLE],
                weighted[WeightedMetricIndex.DRIVING_DIRECTION],
                final,
            ],
            axis=-1,
        )
        return columns[1:].astype(np.float32)  # drop the PDM-Closed proposal row

    @torch.no_grad()
    def evaluate(self, scene_ids: list[str], trajectories: torch.Tensor, batch_id: str = "") -> OracleLabels:
        del batch_id
        if trajectories.ndim != 4 or trajectories.shape[1:] != (64, 8, 3):
            raise ValueError("official oracle requires [B,64,8,3] trajectories")
        raw_rows: list[torch.Tensor] = []
        scene_hashes: list[str] = []
        trajectory_hashes: list[list[str]] = []
        failures: list[dict[str, Any]] = []
        for batch, scene_id in enumerate(scene_ids):
            if scene_id not in self.cache_paths:
                raise ValueError(f"unknown oracle scene: {scene_id}")
            scene_hashes.append(_hash_object(self.cache_paths[scene_id]))
            trajectory_hashes.append([_hash_tensor(item) for item in trajectories[batch].detach().cpu()])
            try:
                candidates = trajectories[batch].detach().cpu().numpy().astype("float32")
                raw_rows.append(torch.as_tensor(self.subscores(scene_id, candidates), dtype=torch.float32))
            except Exception as error:
                failures.append({"scene_id": scene_id, "error": repr(error)})
                raw_rows.append(torch.full((64, 7), float("nan"), dtype=torch.float32))
        raw = torch.stack(raw_rows)
        valid = torch.isfinite(raw)
        named = raw[..., list(HEAD_INDICES)]
        named_valid = valid[..., list(HEAD_INDICES)]
        return OracleLabels(
            raw,
            named,
            named_valid,
            raw[..., 6],
            valid[..., 6],
            scene_hashes,
            trajectory_hashes,
            self.evaluator_version,
            failures,
        )


def verify_against_pdm_score(oracle: OfficialNavsimOracle, scene_id: str, candidates: np.ndarray, limit: int = 4) -> float:
    """Max |batch oracle - official pdm_score| over a few candidates."""
    from navsim.evaluate.pdm_score import pdm_score

    cache = load_official_metric_cache(oracle.cache_paths[scene_id])
    batch = oracle.subscores(scene_id, candidates[:limit])
    worst = 0.0
    for idx in range(limit):
        result = pdm_score(
            metric_cache=cache,
            model_trajectory=Trajectory(candidates[idx].astype(np.float32)),
            future_sampling=PROPOSAL_SAMPLING,
            simulator=oracle.simulator,
            scorer=PDMScorer(PROPOSAL_SAMPLING, PDMScorerConfig()),
        )
        reference = np.array(
            [
                result.no_at_fault_collisions,
                result.drivable_area_compliance,
                result.ego_progress,
                result.time_to_collision_within_bound,
                result.comfort,
                result.driving_direction_compliance,
                result.score,
            ],
            dtype=np.float64,
        )
        worst = max(worst, float(np.abs(reference - batch[idx].astype(np.float64)).max()))
    return worst
