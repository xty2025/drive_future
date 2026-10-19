import torch

from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import lzma
import pickle
import os
import atexit
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
import numpy as np
from .train_pdm_scorer import PDMScorerConfig, PDMScorer
from .proposal_array import get_candidate_states, get_candidate_states_reference
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer as ReferencePDMScorer
# from .train_pdm_scorer_v2_dev import PDMScorerConfig, PDMScorer

# metric_cache_loader = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/metric_cache"))
proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig( )
scorer = PDMScorer(proposal_sampling, config)


def get_reference_progress(metric_cache):
    """Return the PDM reference's collision/drivable-area weighted progress.

    Each candidate must be normalized against this reference, as in official
    ``pdm_score(reference, candidate)``, never against other candidates.
    """
    cached_progress = getattr(metric_cache, "pdm_progress", None)
    if cached_progress is not None:
        return cached_progress

    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory, proposal_sampling, initial_ego_state.time_point)
    simulated_states = simulator.simulate_proposals(pdm_states[None], initial_ego_state)
    reference_scorer = ReferencePDMScorer(proposal_sampling)
    reference_scorer._reset(
        simulated_states, metric_cache.observation, metric_cache.centerline,
        metric_cache.route_lane_ids, metric_cache.drivable_area_map)
    # Area flags are required by both collision attribution and off-road checks.
    reference_scorer._calculate_ego_area()
    reference_scorer._calculate_no_at_fault_collision()
    reference_scorer._calculate_drivable_area_compliance()
    reference_scorer._calculate_progress()
    return reference_scorer._progress_raw * reference_scorer._multi_metrics.prod(axis=0)


# Cache immutable decompressed bytes, not the mutable metric-cache objects used
# by scorers. A fresh unpickle on every call prevents cross-call contamination.
# Eight typical navtrain scenes use about 12 MiB per process; 0 disables reuse.
_METRIC_CACHE_ENTRIES = max(0, int(os.getenv("NAVSIM_ORACLE_CACHE_ENTRIES", "8")))
_metric_cache_records = OrderedDict()
_FAST_PATH = os.getenv("NAVSIM_ORACLE_FAST_PATH", "1") != "0"
# Map microbenchmarks improve, but complete-oracle gains vary; keep opt-in.
_PRUNED_AREAS = os.getenv("NAVSIM_ORACLE_PRUNED_AREAS", "0") != "0"
# Candidate parallelism is opt-in: spawning processes and repeating per-scene
# setup can outweigh the gain, especially when CPU cores are shared by ranks.
_ORACLE_WORKERS = int(os.getenv("NAVSIM_ORACLE_WORKERS", "1"))
if _ORACLE_WORKERS < 1:
    raise ValueError("NAVSIM_ORACLE_WORKERS must be at least 1")
_oracle_executor = None
_oracle_executor_pid = None
_oracle_executor_workers = None
_worker_thread_limit = None


def _initialize_oracle_worker():
    """Each spawned process owns its scorer, simulator and bounded byte cache."""
    global _worker_thread_limit
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    _worker_thread_limit = threadpool_limits(limits=1)


def shutdown_oracle_executor():
    """Release this process's optional pool (also useful for bounded profiling)."""
    global _oracle_executor, _oracle_executor_pid, _oracle_executor_workers
    if _oracle_executor is not None and _oracle_executor_pid == os.getpid():
        _oracle_executor.shutdown(wait=True, cancel_futures=True)
    _oracle_executor = _oracle_executor_pid = _oracle_executor_workers = None


atexit.register(shutdown_oracle_executor)


def _get_oracle_executor(workers):
    global _oracle_executor, _oracle_executor_pid, _oracle_executor_workers
    if (_oracle_executor is None or _oracle_executor_pid != os.getpid()
            or _oracle_executor_workers != workers):
        shutdown_oracle_executor()
        # Never fork a rank after CUDA initialization. Spawn also prevents
        # sharing the simulator/scorer's mutable arrays with another process.
        _oracle_executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_oracle_worker)
        _oracle_executor_pid = os.getpid()
        _oracle_executor_workers = workers
    return _oracle_executor


def load_metric_cache_and_reference(path):
    """Load a fresh cache object and its candidate-independent PDM progress.

    Reuse is bounded per process and invalidated by path, inode, size, mtime and
    ctime. File replacement and in-place rewrites therefore miss the cache.
    """
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size,
           stat.st_mtime_ns, stat.st_ctime_ns)
    record = _metric_cache_records.get(key)
    if record is not None:
        _metric_cache_records.move_to_end(key)
        payload, reference = record
        return pickle.loads(payload), np.asarray(reference, dtype=np.float64)

    with lzma.open(path, "rb") as handle:
        payload = handle.read()
    metric_cache = pickle.loads(payload)
    reference = np.asarray(get_reference_progress(metric_cache), dtype=np.float64)
    if _METRIC_CACHE_ENTRIES:
        _metric_cache_records[key] = (payload, tuple(reference.reshape(-1)))
        while len(_metric_cache_records) > _METRIC_CACHE_ENTRIES:
            _metric_cache_records.popitem(last=False)
    return metric_cache, reference


def get_scores(args):

    return [get_sub_score(a["token"],a["poses"],a["test"]) for a in args]


def get_sub_score(metric_cache, poses, test, fast_path=None, use_pruned_areas=None, workers=None):
    """Return candidate-ordered labels, optionally using persistent CPU workers.

    Each chunk uses the same scene's PDM reference, never a chunk maximum for
    progress normalization. Only a path, NumPy poses and flags cross processes;
    no CUDA tensors, model or mutable metric-cache object are submitted.
    """
    workers = _ORACLE_WORKERS if workers is None else int(workers)
    if workers < 1:
        raise ValueError("Oracle workers must be at least 1")
    if workers == 1 or len(poses) < 2:
        return _get_sub_score_serial(metric_cache, poses, test, fast_path, use_pruned_areas)
    if not isinstance(poses, np.ndarray):
        raise TypeError("Parallel Oracle expects CPU NumPy poses")
    # Resolve optional settings in the caller so all workers use the same mode.
    fast_path = _FAST_PATH if fast_path is None else fast_path
    use_pruned_areas = (fast_path and _PRUNED_AREAS) if use_pruned_areas is None else use_pruned_areas
    workers = min(workers, len(poses))
    executor = _get_oracle_executor(workers)
    futures = [executor.submit(_get_sub_score_serial, str(Path(metric_cache).resolve()),
                               chunk, test, fast_path, use_pruned_areas)
               for chunk in np.array_split(poses, workers)]
    # Future order is chunk order, regardless of completion order.
    results = [future.result() for future in futures]
    if any(np.any(result[0][:, 3] == 2.0) for result in results):
        # The legacy TTC exception handler assigns its sentinel to an entire
        # batch. Preserve that batch-level behavior if any chunk hits it.
        return _get_sub_score_serial(metric_cache, poses, test, fast_path, use_pruned_areas)
    return tuple(np.concatenate([result[index] for result in results], axis=0) for index in range(4))


def _get_sub_score_serial(metric_cache, poses, test, fast_path=None, use_pruned_areas=None):

    metric_cache, pdm_progress = load_metric_cache_and_reference(metric_cache)
    initial_ego_state = metric_cache.ego_state

    fast_path = _FAST_PATH if fast_path is None else fast_path
    use_pruned_areas = (fast_path and _PRUNED_AREAS) if use_pruned_areas is None else use_pruned_areas
    if fast_path:
        trajectory_states = get_candidate_states(poses, initial_ego_state, simulator.proposal_sampling)
    else:
        trajectory_states = get_candidate_states_reference(poses, initial_ego_state, simulator.proposal_sampling)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)#32,41,11

    final_scores=scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        pdm_progress,
        use_pruned_areas=use_pruned_areas,
    )

    num_col=2

    key_agent_corners = np.zeros([len(final_scores), scorer.proposal_sampling.num_poses ,num_col, 4, 2])
    key_agent_labels = np.zeros([len(final_scores), scorer.proposal_sampling.num_poses ,num_col],dtype=bool)
    ego_areas = scorer._ego_areas[:,1:,1:]

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
    driving_direction_compliance = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, :  ]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, :]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :]


    scores=np.stack([no_at_fault_collisions,drivable_area_compliance,
                     ego_progress,time_to_collision_within_bound,comfort, driving_direction_compliance, final_scores
                     ],axis=-1)#[:,None]

    if not test:
        for i in range(len(scores)):
            # proposal_collided_track_ids=scorer.proposal_collided_track_ids[i]
            proposal_fault_collided_track_ids = scorer.proposal_fault_collided_track_ids[i]
            # temp_collided_track_ids=scorer.temp_collided_track_ids[i]

            if len(proposal_fault_collided_track_ids):
                col_token=proposal_fault_collided_track_ids[0]
                collision_time_idcs = int(scorer._collision_time_idcs[i])+1

                for time_idx in range(1,collision_time_idcs):
                    if  col_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx-1,0] = True
                        key_agent_corners[i][time_idx-1,0]=np.array(scorer._observation[time_idx][col_token].boundary.xy).T[:4]

            ttc_collided_track_ids = scorer.ttc_collided_track_ids[i]

            if len(ttc_collided_track_ids):
                ttc_token=ttc_collided_track_ids[0]
                ttc_time_idcs = int(scorer._ttc_time_idcs[i])+1

                for time_idx in range(1,ttc_time_idcs):
                    if  ttc_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx-1,1] = True
                        key_agent_corners[i][time_idx-1,1]=np.array(scorer._observation[time_idx][ttc_token].boundary.xy).T[:4]

        theta = initial_ego_state.rear_axle.heading
        origin_x = initial_ego_state.rear_axle.x
        origin_y = initial_ego_state.rear_axle.y

        c, s = np.cos(theta), np.sin(theta)
        mat = np.array([[c, -s],
                        [s, c]])

        key_agent_corners[...,0]-=origin_x
        key_agent_corners[...,1]-=origin_y

        key_agent_corners=key_agent_corners.dot(mat)

    return scores,key_agent_corners,key_agent_labels,ego_areas
