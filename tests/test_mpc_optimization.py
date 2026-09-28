"""Numerical and action equivalence against the original Python MPC math."""
from types import SimpleNamespace

import numpy as np
import pytest

from agents.mpc import base, cbf, costs, defensive, kinematic, mpcc, obstacles
from agents.mpc.costs import MPCCWeights, mpcc_geometry_cost, trajectory_cost
from agents.mpc.rollout import (
    kinematic_bicycle_step, normalize_actions, normalize_pose, rollout_kinematic_bicycle,
)
from agents.mpc.track_geometry import (
    heading_error, prepare_centerline_geometry, progress_along_centerline, project_to_centerline,
)


def reference_geometry_cost(trajectory, geometry, *, weights=MPCCWeights()):
    # Keep the original independent contour, heading, and endpoint searches.
    contour, lag, heading = [], [], []
    last_index = None
    for pose in trajectory:
        projection = project_to_centerline(geometry, pose[:2], last_index=last_index)
        last_index = projection.index
        contour.append(projection.contouring_error)
        lag.append(projection.lag_error)
        heading.append(heading_error(geometry, pose, last_index=projection.index))
    contour, lag, heading = [np.asarray(values, dtype=np.float32)
                             for values in (contour, lag, heading)]
    return (weights.contouring * float(np.mean(contour * contour))
            + weights.lag * float(np.mean(lag * lag))
            + weights.heading * float(np.mean(heading * heading))
            - weights.progress * progress_along_centerline(geometry, trajectory))


def reference_rollout(pose, actions, *, dt=.1, wheelbase=.3302, horizon=None):
    if horizon is None:
        horizon = 0 if actions is None else (1 if np.asarray(actions).ndim == 1 else len(actions))
    horizon = max(0, int(horizon))
    actions = normalize_actions(actions, horizon)
    trajectory = np.zeros((horizon + 1, 3), dtype=np.float32)
    trajectory[0] = normalize_pose(pose)
    for step in range(horizon):
        trajectory[step + 1] = kinematic_bicycle_step(
            trajectory[step], actions[step], dt=dt, wheelbase=wheelbase,
        )
    return trajectory


def reference_trajectory_cost(trajectory, actions, *, centerline=None, target_speed=None,
                              weights=costs.CostWeights()):
    if len(costs._normalize_trajectory(trajectory)) == 0:
        return 0.0
    return (
        weights.path_tracking * costs.path_tracking_cost(trajectory, centerline)
        + weights.heading_error * costs.heading_error_cost(trajectory, centerline)
        + weights.target_speed * costs.target_speed_cost(actions, target_speed)
        + weights.control_effort * costs.control_effort_cost(actions)
        + weights.steering_smoothness * costs.steering_smoothness_cost(actions)
        - weights.progress * costs.progress_reward(trajectory, centerline)
    )


def reference_smooth_action(self, pose, centerline, target_speed):
    best_cost, best_action = float("inf"), self._fallback_action
    for sequence in self._candidate_sequences:
        trajectory = reference_rollout(pose, sequence, dt=self.dt, wheelbase=self.wheelbase)
        cost = reference_trajectory_cost(trajectory, sequence, centerline=centerline,
                                         target_speed=target_speed, weights=self._weights)
        steer_delta = float(sequence[0, 0] - self._previous_action[0])
        speed_delta = float(sequence[0, 1] - self._previous_action[1])
        cost += self._speed_smoothness_weight * (speed_delta * speed_delta + steer_delta * steer_delta)
        if cost < best_cost:
            best_cost, best_action = cost, sequence[0]
    return best_action.copy()


@pytest.mark.parametrize("path_size", [0, 1, 2, 80])
@pytest.mark.parametrize("horizon", [0, 1, 10, 30])
@pytest.mark.parametrize("target_speed", [None, 2.5, np.nan])
def test_fused_trajectory_cost_matches_independent_terms(path_size, horizon, target_speed):
    rng = np.random.default_rng(912)
    path = rng.normal(size=(path_size, 2)).astype(np.float32)
    trajectory = rng.normal(size=(horizon, 3)).astype(np.float32)
    actions = rng.normal(size=(max(0, horizon - 1), 2)).astype(np.float32)
    if horizon > 1:
        trajectory[0, 0] = np.nan
    if len(actions):
        actions[0, 1] = np.inf
    weights = costs.CostWeights(path_tracking=2., heading_error=.5, progress=1.)
    kwargs = dict(centerline=path, target_speed=target_speed, weights=weights)
    np.testing.assert_allclose(trajectory_cost(trajectory, actions, **kwargs),
                               reference_trajectory_cost(trajectory, actions, **kwargs),
                               atol=2e-6, rtol=2e-6)


def test_fused_trajectory_cost_keeps_ties_endpoint_heading_and_input_normalization():
    path = np.array([[0, 0], [0, 0], [1, 0], [np.nan, 1], [1, 1]], dtype=np.float32)
    poses = np.array([[.5, 0, -np.pi], [0, 0, 0], [1, 1, np.pi]], dtype=np.float32)
    for trajectory, actions in [(poses, None), (poses[:, :2], [0., 1.]), (None, None)]:
        kwargs = dict(centerline=path, target_speed=1., weights=costs.CostWeights(progress=1.))
        np.testing.assert_allclose(trajectory_cost(trajectory, actions, **kwargs),
                                   reference_trajectory_cost(trajectory, actions, **kwargs),
                                   atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("kind", ["circle", "hairpin", "duplicates"])
def test_geometry_cost_preserves_windows_seams_and_degenerate_segments(closed, kind):
    angle = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    path = np.column_stack([5 * np.cos(angle), 5 * np.sin(angle)])
    if kind == "hairpin":
        path[:, 1] *= .02
    elif kind == "duplicates":
        path[70:90] = path[70]
    geometry = prepare_centerline_geometry(path, closed=closed)
    weights = MPCCWeights(contouring=4.0, lag=.5, heading=.5, progress=2.0)
    rng = np.random.default_rng(42)
    for start in (0, 75, 150, 295):
        indices = (start + np.arange(13)) % len(path)
        trajectory = np.column_stack([path[indices], angle[indices] + np.pi / 2]).astype(np.float32)
        trajectory[:, :2] += rng.normal(0, .1, (13, 2)).astype(np.float32)
        for sample in (trajectory, trajectory[::6], trajectory[[0]], trajectory[::-1]):
            expected = reference_geometry_cost(sample, geometry, weights=weights)
            actual = mpcc_geometry_cost(sample, geometry, weights=weights)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_geometry_cost_normalizes_missing_and_nonfinite_inputs():
    path = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    assert mpcc_geometry_cost(None, path) == 0
    assert mpcc_geometry_cost(np.zeros((2, 3)), None) == 0
    assert mpcc_geometry_cost(np.zeros((2, 3)), np.zeros((3, 2))) == 0
    trajectory = np.array([[np.nan, .2], [1, np.inf]], dtype=np.float32)
    expected = np.array([[0, .2, 0], [1, 0, 0]], dtype=np.float32)
    assert mpcc_geometry_cost(trajectory, path) == mpcc_geometry_cost(expected, path)


@pytest.mark.parametrize("horizon", [0, 1, 12, 30])
@pytest.mark.parametrize("dt,wheelbase", [(.05, .3302), (.1, .4), (-1.0, .3302), (.01, 0.0)])
def test_compiled_rollout_matches_public_step(horizon, dt, wheelbase):
    rng = np.random.default_rng(731)
    pose = np.array([2.5, -1.2, 3.13], dtype=np.float32)
    actions = rng.uniform([-.42, -.5], [.42, 3.0], (20, 2)).astype(np.float32)
    actions[5] = [np.nan, np.inf]
    kwargs = dict(horizon=horizon, dt=dt, wheelbase=wheelbase)
    np.testing.assert_allclose(rollout_kinematic_bicycle(pose, actions, **kwargs),
                               reference_rollout(pose, actions, **kwargs), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(rollout_kinematic_bicycle(None, None, **kwargs),
                                  reference_rollout(None, None, **kwargs))


@pytest.mark.parametrize("agent_class", [
    kinematic.KinematicMPCAgent, obstacles.ObstacleAwareMPCAgent,
    defensive.DefensiveMPCAgent, cbf.CBFMPCAgent, mpcc.MPCCAgent,
])
@pytest.mark.parametrize("closed", [False, True])
def test_traffic_actions_match_reference_with_obstacles_and_previous_action(monkeypatch, agent_class, closed):
    angle = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    path = np.column_stack([5 * np.cos(angle), 5 * np.sin(angle)]).astype(np.float32)
    env = SimpleNamespace(centerline_points=path)
    observations = []
    for index, speed, distance in [(0, 0., 10.), (1, 1., .9), (239, 2., .3)]:
        pose = np.array([*path[index], angle[index] + np.pi / 2], dtype=np.float32)
        heading = np.array([np.cos(pose[2]), np.sin(pose[2])])
        observations.append({
            "pose": pose, "velocity": np.array([speed, 0], dtype=np.float32),
            "scan": np.full(64, distance, dtype=np.float32),
            "target_pose": np.array([*(pose[:2] - .8 * heading), pose[2]], dtype=np.float32),
        })
    config = {"params": {"dt": .05, "closed_centerline": closed}}
    fast, reference = agent_class(config), agent_class(config)
    fast.set_env(env)
    reference.set_env(env)
    actual = [fast.act(obs) for obs in observations]
    monkeypatch.setattr(mpcc, "mpcc_geometry_cost", reference_geometry_cost)
    for module in (base, kinematic, obstacles, defensive):
        monkeypatch.setattr(module, "trajectory_cost", reference_trajectory_cost)
    for module in (base, kinematic, obstacles, defensive, cbf, mpcc):
        monkeypatch.setattr(module, "rollout_kinematic_bicycle", reference_rollout)
    monkeypatch.setattr(kinematic.KinematicMPCAgent, "_select_with_previous_action_smoothness",
                        reference_smooth_action)
    expected = [reference.act(obs) for obs in observations]
    np.testing.assert_array_equal(actual, expected)


def test_kinematic_scores_each_candidate_once_after_first_action(monkeypatch):
    agent = kinematic.KinematicMPCAgent({"params": {"max_candidate_sequences": 35}})
    agent.set_env(SimpleNamespace(centerline_points=np.array([[0, 0], [10, 0]], dtype=np.float32)))
    obs = {"pose": np.zeros(3), "velocity": np.zeros(2)}
    agent.act(obs)
    calls = 0
    original_cost = kinematic.trajectory_cost

    def count_cost(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_cost(*args, **kwargs)

    def redundant_search(*args, **kwargs):
        pytest.fail("Repeated full search before applying previous-action smoothness")

    monkeypatch.setattr(kinematic, "trajectory_cost", count_cost)
    monkeypatch.setattr(kinematic, "evaluate_action_sequences", redundant_search)
    agent.act(obs)
    assert calls == len(agent._candidate_sequences)
