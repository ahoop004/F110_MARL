from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.env.spawn import (
    SpawnRequest,
    extract_spawn_points,
    load_spawn_points_from_map,
    resolve_reset_spawn,
    sample_centerline_relative_spawn,
    sample_random_spawn,
)


def test_extract_spawn_points_from_metadata_list():
    metadata = {
        "annotations": {
            "spawn_points": [
                {"name": "a", "pose": [1.0, 2.0]},
                {"name": "b", "pose": [3.0, 4.0, 0.5]},
            ]
        }
    }

    points = extract_spawn_points(None, metadata)

    assert sorted(points) == ["a", "b"]
    assert points["a"].dtype == np.float32
    np.testing.assert_allclose(points["a"], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(points["b"], np.array([3.0, 4.0, 0.5], dtype=np.float32))


def test_sample_random_spawn_outputs_pose3_per_agent():
    spawn_points = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([3.0, 4.0, 0.5], dtype=np.float32),
    }
    rng = np.random.default_rng(1)

    sampled = sample_random_spawn(
        spawn_points=spawn_points,
        spawn_point_names=["a", "b"],
        agent_ids=["car_0", "car_1"],
        rng=rng,
        allow_reuse=False,
    )

    assert sampled is not None
    mapping, poses = sampled
    assert set(mapping) == {"car_0", "car_1"}
    assert set(mapping.values()) == {"a", "b"}
    assert poses.shape == (2, 3)
    assert poses.dtype == np.float32


def test_load_spawn_points_from_map_uses_yaml_annotations():
    poses = load_spawn_points_from_map(
        "maps/line2/line2.yaml",
        ["spawn_1", "spawn_2"],
    )

    assert poses.shape == (2, 3)
    assert poses.dtype == np.float64


def test_resolve_reset_spawn_prefers_explicit_options():
    request = SpawnRequest(
        agent_ids=["car_0"],
        n_agents=1,
        agent_index={"car_0": 0},
        options={
            "poses": [1.0, 2.0, 0.25],
            "velocities": {"car_0": 0.7},
            "lock_speed_steps": 5,
        },
        current_start_poses=np.array([[9.0, 9.0, 0.0]], dtype=np.float32),
        random_spawn_enabled=True,
        spawn_points={"a": np.array([3.0, 4.0, 0.0], dtype=np.float32)},
        spawn_point_names=["a"],
        rng=np.random.default_rng(0),
    )

    result = resolve_reset_spawn(request)

    assert result.update_start_poses is True
    np.testing.assert_allclose(result.poses, np.array([[1.0, 2.0, 0.25]], dtype=np.float32))
    np.testing.assert_allclose(result.velocities, np.array([0.7], dtype=np.float32))
    assert result.locked_velocities == {"car_0": 0.7}
    assert result.lock_speed_steps == 5
    assert result.spawn_mapping == {}


def test_resolve_reset_spawn_falls_back_to_start_poses():
    start_poses = np.array([[9.0, 9.0, 0.0]], dtype=np.float32)
    request = SpawnRequest(
        agent_ids=["car_0"],
        n_agents=1,
        agent_index={"car_0": 0},
        current_start_poses=start_poses,
    )

    result = resolve_reset_spawn(request)

    assert result.update_start_poses is False
    np.testing.assert_allclose(result.poses, start_poses)


def test_resolve_reset_spawn_uses_centerline_before_random_named():
    request = SpawnRequest(
        agent_ids=["car_0"],
        n_agents=1,
        agent_index={"car_0": 0},
        random_spawn_enabled=True,
        spawn_points={"a": np.array([3.0, 4.0, 0.0], dtype=np.float32)},
        spawn_point_names=["a"],
        rng=np.random.default_rng(0),
    )

    def centerline_spawn():
        return (
            np.array([[5.0, 6.0, 0.75]], dtype=np.float32),
            {"car_0": 0.4},
            {"spawn_source": "centerline_relative", "spawn_s": 0.5},
        )

    result = resolve_reset_spawn(request, centerline_spawn_fn=centerline_spawn)

    assert result.update_start_poses is True
    np.testing.assert_allclose(result.poses, np.array([[5.0, 6.0, 0.75]], dtype=np.float32))
    np.testing.assert_allclose(result.velocities, np.array([0.4], dtype=np.float32))
    assert result.metadata["spawn_source"] == "centerline_relative"
    assert result.spawn_mapping == {}


def test_resolve_reset_spawn_uses_random_named_before_start_poses():
    request = SpawnRequest(
        agent_ids=["car_0"],
        n_agents=1,
        agent_index={"car_0": 0},
        current_start_poses=np.array([[9.0, 9.0, 0.0]], dtype=np.float32),
        random_spawn_enabled=True,
        spawn_points={"a": np.array([3.0, 4.0, 0.0], dtype=np.float32)},
        spawn_point_names=["a"],
        rng=np.random.default_rng(0),
    )

    result = resolve_reset_spawn(request)

    assert result.update_start_poses is True
    np.testing.assert_allclose(result.poses, np.array([[3.0, 4.0, 0.0]], dtype=np.float32))
    assert result.spawn_mapping == {"car_0": "a"}


def test_sample_centerline_relative_spawn_round_robin():
    centerline = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = sample_centerline_relative_spawn(
        spawn_policy="centerline_relative",
        centerline=centerline,
        map_split_mode="eval",
        spawn_centerline_cfg={
            "mode": "random",
            "min_progress": 0.0,
            "max_progress": 1.0,
            "avoid_finish": False,
        },
        spawn_offsets_cfg={"s_offset": 1.0, "d_offset": 0.25},
        spawn_target_cfg={"enabled": True, "speed": 0.2},
        spawn_ego_cfg={"speed": 0.6},
        agent_ids=["car_0", "car_1"],
        rng=np.random.default_rng(0),
        current_index=0,
    )

    assert result is not None
    assert result.poses.shape == (2, 3)
    np.testing.assert_allclose(result.poses[1, :2], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result.poses[0, :2], np.array([1.0, 0.25], dtype=np.float32))
    assert result.velocities == {"car_0": 0.6, "car_1": 0.2}
    assert result.metadata == {"spawn_s": 0.0, "spawn_d": 0.25}
    assert result.next_index == 1
