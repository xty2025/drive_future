"""The fast candidate array path must match NAVSIM's object interpolation."""

import numpy as np
import pytest

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.agents.drivoR.score_module.proposal_array import get_candidate_states


@pytest.mark.parametrize("time_us,heading", [(0, 0.0), (1600000000123457, np.pi - 1e-10)])
@pytest.mark.parametrize("num_candidates,num_poses,interval", [(1, 8, 0.5), (64, 12, 0.25)])
def test_batch_array_matches_official_objects(time_us, heading, num_candidates, num_poses, interval):
    ego = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(123456.125, -45678.25, heading),
        rear_axle_velocity_2d=StateVector2D(12.0, 0.25),
        rear_axle_acceleration_2d=StateVector2D(0.4, -0.2),
        tire_steering_angle=0.15, time_point=TimePoint(time_us),
        vehicle_parameters=get_pacifica_parameters(),
        angular_vel=0.3, angular_accel=0.1, tire_steering_rate=0.2,
    )
    random = np.random.default_rng(19)
    poses = random.normal(size=(num_candidates, num_poses, 3))
    poses[..., :2] *= 15
    # Repeated wrap crossings exercise the AngularInterpolator semantics.
    poses[..., 2] = np.linspace(-4 * np.pi, 4 * np.pi, num_poses)
    trajectory_sampling = TrajectorySampling(num_poses=num_poses, interval_length=interval)
    # Include clipping beyond the candidate horizon, as get_trajectory_as_array does.
    proposal_sampling = TrajectorySampling(num_poses=55, interval_length=0.1)
    expected = np.stack([
        get_trajectory_as_array(transform_trajectory(Trajectory(pose, trajectory_sampling), ego),
                                proposal_sampling, ego.time_point)
        for pose in poses
    ])
    actual = get_candidate_states(poses, ego, proposal_sampling, trajectory_sampling)
    # Scalar and SIMD trig can differ by a few float64 ulps at wrap boundaries.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_rejects_inconsistent_pose_count():
    with pytest.raises(ValueError, match="trajectory_sampling"):
        get_candidate_states(np.zeros((64, 7, 3)), None, None)


@pytest.mark.parametrize("empty", [False, True])
def test_indexed_containment_preserves_boundaries_holes_and_overlap(empty):
    from shapely.geometry import Polygon
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap

    polygons = [] if empty else [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]]),
        Polygon([(2, 0), (5, 0), (5, 3), (2, 3)]),
    ]
    area = PDMDrivableMap([str(i) for i in range(len(polygons))],
                          [SemanticMapLayer.ROADBLOCK] * len(polygons), polygons)
    points = np.array([[[0, 0], [1, 1.5], [1.5, 1.5], [3, 2]],
                       [[4, 2], [4.5, 2], [-1, 2], [0.5, 0.5]]], dtype=np.float64)
    np.testing.assert_array_equal(area.points_in_polygons_pruned(points), area.points_in_polygons(points))
    no_points = np.zeros((2, 0, 2))
    np.testing.assert_array_equal(area.points_in_polygons_pruned(no_points), area.points_in_polygons(no_points))
