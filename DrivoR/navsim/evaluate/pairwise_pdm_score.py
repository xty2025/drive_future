"""Official NAVSIM-v1 PDMS of many candidate trajectories of one scene, scored in one batch."""

import numpy as np

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    EgoAreaIndex,
    MultiMetricIndex,
    WeightedMetricIndex,
)

# Column order of the returned rows; same fields and order as navsim.common.dataclasses.PDMResults.
PDM_FIELDS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def pairwise_pdm_scores(
    metric_cache: MetricCache,
    trajectories: np.ndarray,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> np.ndarray:
    """
    Returns [K, 7] rows; row k equals pdm_score(metric_cache, Trajectory(trajectories[k]), ...).
    pdm_score() simulates [PDM-Closed, candidate] and normalises ego progress by the larger of
    the two. Simulation and every other metric are computed per proposal, so all candidates are
    simulated together with PDM-Closed and only the progress normalisation is redone per pair.
    :param metric_cache: metric cache of the scene
    :param trajectories: [K, 8, 3] candidate poses in the ego frame (default 4s / 0.5s sampling)
    :param simulator: PDM simulator of the evaluation config
    :param scorer: PDM scorer of the evaluation config
    :return: array of PDM sub-scores and score per candidate, columns as PDM_FIELDS
    """
    sampling = simulator.proposal_sampling
    ego_state = metric_cache.ego_state
    states = [get_trajectory_as_array(metric_cache.trajectory, sampling, ego_state.time_point)]
    for poses in trajectories:
        candidate = transform_trajectory(Trajectory(np.asarray(poses, dtype=np.float32)), ego_state)
        states.append(get_trajectory_as_array(candidate, sampling, ego_state.time_point))

    simulated = simulator.simulate_proposals(np.stack(states), ego_state)
    scorer.score_proposals(
        simulated,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    # Same aggregation as PDMScorer._aggregate_scores, restricted to (PDM-Closed, candidate).
    multiplicative = scorer._multi_metrics.prod(axis=0)
    raw_progress = scorer._progress_raw * multiplicative
    weights = scorer._config.weighted_metrics_array
    rows = np.empty((len(trajectories), len(PDM_FIELDS)), dtype=np.float64)
    for k in range(len(trajectories)):
        i = k + 1
        pair_max = max(raw_progress[0], raw_progress[i])
        if pair_max > scorer._config.progress_distance_threshold:
            progress = raw_progress[i] / pair_max
        else:
            progress = 0.0 if multiplicative[i] == 0.0 else 1.0
        weighted = scorer._weighted_metrics[:, i].copy()
        weighted[WeightedMetricIndex.PROGRESS] = progress
        rows[k] = (
            scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, i],
            scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, i],
            progress,
            weighted[WeightedMetricIndex.TTC],
            weighted[WeightedMetricIndex.COMFORTABLE],
            weighted[WeightedMetricIndex.DRIVING_DIRECTION],
            multiplicative[i] * float(weighted @ weights) / float(weights.sum()),
        )
    return rows


def pairwise_pdm_details(
    metric_cache: MetricCache,
    trajectories: np.ndarray,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> tuple:
    """
    Scores like pairwise_pdm_scores and also reports why each candidate lost points.
    :return: (rows [K, 7] as pairwise_pdm_scores, list of K dicts) where each dict holds the first
        at-fault collision / TTC violation / off-road time index (None if none), the simulated ego
        polygon (global frame) at each of those indices, the colliding objects' polygons, the progress
        along the route in metres, and PDM-Closed's progress (EP normalisation). Progress is reported
        before the multiplicative metrics zero it, so a colliding candidate still shows how far it went.
    """
    rows = pairwise_pdm_scores(metric_cache, trajectories, simulator, scorer)
    raw_progress = scorer._progress_raw
    details = []
    for k in range(len(trajectories)):
        i = k + 1
        off_road = np.flatnonzero(scorer._ego_areas[i, :, EgoAreaIndex.NON_DRIVABLE_AREA])
        events = {
            "collision": scorer._collision_time_idcs[i],
            "ttc": scorer._ttc_time_idcs[i],
            "off_road": off_road[0] if len(off_road) else np.inf,
        }
        detail = {"progress_m": float(raw_progress[i]), "pdm_closed_progress_m": float(raw_progress[0])}
        for name, index in events.items():
            if not np.isfinite(index):
                detail[name] = None
                continue
            index = int(index)
            ego = scorer._ego_polygons[i, index]
            event = {"index": index, "time_s": index * simulator.proposal_sampling.interval_length, "ego": ego}
            if name == "collision":
                occupancy = metric_cache.observation[index]
                event["objects"] = [occupancy._geometries[occupancy.token_to_idx[token]]
                                    for token in occupancy.intersects(ego)]
            detail[name] = event
        details.append(detail)
    return rows, details
