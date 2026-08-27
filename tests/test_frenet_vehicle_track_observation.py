from __future__ import annotations

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario
from env.f110ParallelEnv import F110ParallelEnv
from utils.track_preview import TrackPreviewGeometry
from wrappers.observations.composer import ObservationComposer
from wrappers.observations.frenet_vehicle_track import FrenetVehicleTrackComponent


def _circle(radius: float, count: int = 240) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle))).astype(
        np.float32
    )


def test_track_preview_is_uniform_ahead_and_uses_track_geometry_maxima() -> None:
    geometry = TrackPreviewGeometry.build(
        _circle(10.0),
        {0: _circle(9.0), 1: _circle(11.0)},
        spacing=0.3,
    )

    assert geometry is not None
    preview = geometry.preview(np.array([10.0, 0.0], dtype=np.float32), 8)
    assert np.asarray(preview["curvature"]).shape == (8,)
    assert np.asarray(preview["width"]).shape == (8,)
    assert np.mean(np.abs(preview["curvature"])) == pytest.approx(0.1, abs=0.025)
    assert np.mean(preview["width"]) == pytest.approx(2.0, abs=0.12)
    assert preview["curvature_max"] >= np.max(np.abs(preview["curvature"]))
    assert preview["width_max"] >= np.max(preview["width"])

    near_seam = geometry.nearest_index(
        np.array([10.0, -0.05], dtype=np.float32)
    )
    wrapped = geometry.nearest_index(
        np.array([10.0, 0.05], dtype=np.float32),
        last_index=near_seam,
        search_window=3,
    )
    assert wrapped < 3


def test_frenet_vehicle_track_observation_order_and_normalization() -> None:
    component = FrenetVehicleTrackComponent(
        points=2,
        wheel_radius=0.5,
        maxima={key: 2.0 for key in (
            "vx", "vy", "u", "n", "r", "delta", "delta_ref",
            "omega_ref_dot", "omega_ref", "omega",
        )},
    )
    raw = {
        "velocity": np.array([1.0, -1.0], dtype=np.float32),
        "angular_velocity": 1.0,
        "steering_angle": 1.0,
        "steering_reference": -1.0,
        "speed_reference_rate": 0.5,
        "speed_reference": 0.5,
    }
    info = {
        "centerline": {"heading_error": 1.0, "d": -1.0},
        "track_preview": {
            "curvature": np.array([1.0, -2.0], dtype=np.float32),
            "width": np.array([2.0, 4.0], dtype=np.float32),
            "curvature_max": 2.0,
            "width_max": 4.0,
        },
    }

    observation = component.compute(raw, info)

    assert observation == pytest.approx(
        [
            0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5,
            0.5, 0.5, 1.0, 0.5, -1.0, 0.5, 1.0,
        ]
    )


def test_lidar_and_frenet_track_composer_dimension() -> None:
    composer = ObservationComposer.from_config(
        {
            "observation": {
                "lidar": {"enabled": True, "normalize": True},
                "frenet_vehicle_track": {
                    "enabled": True,
                    "points": 3,
                    "wheel_radius": 0.05,
                },
            }
        },
        {"lidar_beams": 4, "lidar_range": 10.0},
    )
    assert composer.obs_dim == 4 + 10 + 2 * 3


def test_latest_speed_reference_rate_survives_repeated_physics_actions() -> None:
    env = F110ParallelEnv.__new__(F110ParallelEnv)
    env._agent_id_to_index = {"car_0": 0}
    env._control_timestep = 0.02
    env._last_control_commands = np.zeros((1, 2), dtype=np.float32)
    env._last_speed_reference_rates = np.zeros(1, dtype=np.float32)
    env._control_command_initialized = np.ones(1, dtype=bool)

    command = np.array([[0.1, 1.0]], dtype=np.float32)
    env._record_control_commands(command, ("car_0",))
    assert env._last_speed_reference_rates[0] == pytest.approx(50.0)

    # action_repeat sends the same command again; preserve the latest policy
    # change rather than replacing it with a misleading zero derivative.
    env._record_control_commands(command, ("car_0",))
    assert env._last_speed_reference_rates[0] == pytest.approx(50.0)


def test_complete_4_frenet_scenario_is_opt_in() -> None:
    scenario = load_and_expand_scenario(
        "scenarios/complete_4_vehicle_track_frenet.yaml"
    )
    assert scenario["experiment"]["name"] == "complete_4_vehicle_track_frenet"
    assert scenario["environment"]["track_preview"] == {
        "points": 20,
        "spacing": 0.3,
    }
    for config in scenario["agents"].values():
        assert config["observation"].endswith(
            "configs/observations/rl_racer_vehicle_track_frenet.yaml"
        )
