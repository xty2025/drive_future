"""Batch equivalent of NAVSIM's relative-trajectory interpolation path."""

import numpy as np
from scipy.interpolate import interp1d

from nuplan.common.geometry.compute import principal_value
from nuplan.common.geometry.convert import matrix_from_pose
from nuplan.planning.simulation.planner.ml_planner.transform_utils import _get_fixed_timesteps
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


def get_candidate_states_reference(poses, initial_ego_state, proposal_sampling):
    """Original NAVSIM object path, retained for parity checks and rollback."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory

    return np.stack([
        get_trajectory_as_array(transform_trajectory(Trajectory(pose), initial_ego_state),
                                proposal_sampling, initial_ego_state.time_point)
        for pose in poses
    ])


def _rear_axle_round_trip(xy, heading, distance):
    # EgoState.build_from_rear_axle stores a center footprint, and reading its
    # rear axle subtracts the same offset. Keep those floating-point operations
    # so coordinates match the object-based implementation, including at knots.
    offset = distance * np.stack((np.cos(heading), np.sin(heading)), axis=-1)
    return (xy + offset) - offset


def get_candidate_states(
    poses,
    initial_ego_state,
    proposal_sampling,
    trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=0.5),
):
    """Return [M, T+1, 11] without creating M*(T+1) EgoState objects.

    This preserves ``transform_trajectory`` + ``get_trajectory_as_array``:
    SE(2) matrix composition, integer-microsecond timestamps, piecewise linear
    interpolation and unwrapped angular interpolation. It does not change the
    subsequent vehicle simulation or any PDM metric.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (trajectory_sampling.num_poses, 3):
        raise ValueError("Candidate poses must match trajectory_sampling as [M, num_poses, 3]")

    relative_matrices = np.zeros((*poses.shape[:2], 3, 3), dtype=np.float64)
    c, s = np.cos(poses[..., 2]), np.sin(poses[..., 2])
    relative_matrices[..., 0, 0] = relative_matrices[..., 1, 1] = c
    relative_matrices[..., 0, 1] = -s
    relative_matrices[..., 1, 0] = s
    relative_matrices[..., :2, 2] = poses[..., :2]
    relative_matrices[..., 2, 2] = 1
    absolute_matrices = matrix_from_pose(initial_ego_state.rear_axle) @ relative_matrices
    headings = np.arctan2(absolute_matrices[..., 1, 0], absolute_matrices[..., 0, 0])
    distance = initial_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center

    knots = np.zeros((len(poses), poses.shape[1] + 1, StateIndex.size()), dtype=np.float64)
    knots[:, 0] = ego_state_to_state_array(initial_ego_state)
    knots[:, 1:, :2] = _rear_axle_round_trip(absolute_matrices[..., :2, 2], headings, distance)
    knots[:, 1:, StateIndex.HEADING] = headings
    # EgoState's SplitState interpolation carries velocity, acceleration and
    # steering angle, but reconstructs these three rate fields with zero defaults.
    knots[..., StateIndex.STEERING_RATE:] = 0
    knots[..., StateIndex.HEADING] = np.unwrap(knots[..., StateIndex.HEADING], axis=1)

    knot_times = [initial_ego_state.time_us] + [
        int(t * 1e6) for t in _get_fixed_timesteps(
            initial_ego_state, trajectory_sampling.time_horizon, trajectory_sampling.interval_length)
    ]
    sample_seconds = np.arange(
        0.0, proposal_sampling.time_horizon + proposal_sampling.interval_length,
        proposal_sampling.interval_length)
    sample_seconds += initial_ego_state.time_point.time_s
    sample_times = np.clip([int(t * 1e6) for t in sample_seconds], knot_times[0], knot_times[-1])
    states = interp1d(knot_times, knots, axis=1)(sample_times)
    states[..., StateIndex.HEADING] = principal_value(states[..., StateIndex.HEADING])
    states[..., :2] = _rear_axle_round_trip(states[..., :2], states[..., StateIndex.HEADING], distance)
    return states
