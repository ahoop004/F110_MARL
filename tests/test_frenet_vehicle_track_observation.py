from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario
from core.env_builder import build_env_kwargs
from core.feature_requirements import derive_environment_feature_requirements
from core.setup import create_training_setup
from env.centerline_state import (
    CenterlineProgressTracker,
    build_relative_frenet_facts,
)
from env.f110ParallelEnv import F110ParallelEnv
from physics.simulaton import Simulator
from utils.track_preview import TrackPreviewGeometry
from wrappers.observations.composer import ObservationComposer
from wrappers.observations.frenet_neighbors import FrenetNeighborsComponent
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


def test_frenet_velocity_uses_body_frame_and_is_rotation_invariant() -> None:
    tracker = CenterlineProgressTracker(["car_0"])
    agent_index = {"car_0": 0}
    v_long = np.array([4.0], dtype=np.float32)
    v_lat = np.array([1.0], dtype=np.float32)

    horizontal = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    facts_horizontal = tracker.update(
        horizontal,
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        v_long,
        v_lat,
        agent_index,
    )["car_0"]

    tracker.reset()
    vertical = np.array(
        [
            [0.0, 0.0, np.pi / 2.0],
            [0.0, 1.0, np.pi / 2.0],
            [0.0, 2.0, np.pi / 2.0],
        ],
        dtype=np.float32,
    )
    facts_vertical = tracker.update(
        vertical,
        np.array([0.0]),
        np.array([1.0]),
        np.array([np.pi / 2.0]),
        v_long,
        v_lat,
        agent_index,
    )["car_0"]

    assert facts_horizontal["vs"] == pytest.approx(4.0)
    assert facts_horizontal["vd"] == pytest.approx(1.0)
    assert facts_vertical["vs"] == pytest.approx(facts_horizontal["vs"])
    assert facts_vertical["vd"] == pytest.approx(facts_horizontal["vd"])


def test_simulator_slip_velocity_flows_into_frenet_components() -> None:
    speed = 5.0
    slip_angle = 0.4
    state = np.zeros(7, dtype=np.float64)
    state[6] = slip_angle
    simulator = Simulator.__new__(Simulator)
    simulator.num_agents = 1
    simulator.agents = [SimpleNamespace(state=state)]
    simulator._linear_vels_x = np.zeros(1, dtype=np.float32)
    simulator._linear_vels_y = np.zeros(1, dtype=np.float32)
    simulator._ang_vels_z = np.zeros(1, dtype=np.float32)
    simulator.set_agent_speed(0, speed)
    v_long = float(simulator._linear_vels_x[0])
    v_lat = float(simulator._linear_vels_y[0])

    tracker = CenterlineProgressTracker(["car_0"])
    facts = tracker.update(
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([v_long], dtype=np.float32),
        np.array([v_lat], dtype=np.float32),
        {"car_0": 0},
    )["car_0"]

    assert v_long == pytest.approx(speed * np.cos(slip_angle))
    assert v_lat == pytest.approx(speed * np.sin(slip_angle))
    assert facts["vs"] == pytest.approx(speed * np.cos(slip_angle))
    assert facts["vd"] == pytest.approx(speed * np.sin(slip_angle))

    # The dynamics use a no-slip kinematic model below 0.5 m/s, even if the
    # slip-angle state still contains its previous dynamic-model value.
    simulator.set_agent_speed(0, 0.25)
    assert simulator._linear_vels_x[0] == pytest.approx(0.25)
    assert simulator._linear_vels_y[0] == pytest.approx(0.0)


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


def test_relative_frenet_neighbors_wrap_sort_and_normalize() -> None:
    relative = build_relative_frenet_facts(
        {
            "car_0": {"s": 98.0, "d": 0.5, "vs": 4.0, "vd": 0.5},
            "car_1": {"s": 2.0, "d": -0.5, "vs": 6.0, "vd": 1.5},
            "car_2": {"s": 80.0, "d": 1.5, "vs": 3.0, "vd": -0.5},
        },
        track_length=100.0,
        closed=True,
    )

    assert [item["agent_id"] for item in relative["car_0"]] == ["car_1", "car_2"]
    assert relative["car_0"][0] == {
        "agent_id": "car_1",
        "delta_s": pytest.approx(4.0),
        "delta_d": pytest.approx(-1.0),
        "delta_vs": pytest.approx(2.0),
        "delta_vd": pytest.approx(1.0),
    }

    component = FrenetNeighborsComponent(
        max_neighbors=3,
        maxima={
            "delta_s": 20.0,
            "delta_d": 5.0,
            "delta_vs": 10.0,
            "delta_vd": 5.0,
        },
    )
    observation = component.compute({}, {"frenet_neighbors": relative["car_0"]})
    assert observation == pytest.approx(
        [
            0.2, -0.2, 0.2, 0.2, 1.0,
            -0.9, 0.2, -0.1, -0.2, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    )


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
    scenario = load_and_expand_scenario("scenarios/complete_4_frenet.yaml")
    assert scenario["experiment"]["name"] == "complete_4_frenet"
    assert scenario["environment"]["track_preview"] == {
        "points": 20,
        "spacing": 0.3,
    }
    for config in scenario["agents"].values():
        assert config["observation"].endswith(
            "configs/observations/rl_racer_vehicle_track_frenet.yaml"
        )
    composer = ObservationComposer.from_file(
        str(
            (Path("scenarios") / scenario["agents"]["car_0"]["observation"])
            .resolve()
        ),
        scenario["environment"],
    )
    assert composer.obs_dim == 108 + 10 + 2 * 20

    env_kwargs = build_env_kwargs(
        {**scenario["environment"], "map": "dummy.yaml"},
        scenario["agents"],
        seed=42,
    )
    assert env_kwargs["track_preview"] == {"points": 20, "spacing": 0.3}
    assert env_kwargs["action_repeat"] == 2


def test_complete_4_frenet_neighbors_is_a_separate_privileged_arm() -> None:
    scenario = load_and_expand_scenario(
        "scenarios/complete_4_frenet_neighbors.yaml"
    )
    assert scenario["experiment"]["name"] == "complete_4_frenet_neighbors"
    assert "privileged-neighbors" in scenario["wandb"]["tags"]
    observation_path = str(
        (Path("scenarios") / scenario["agents"]["car_0"]["observation"]).resolve()
    )
    composer = ObservationComposer.from_file(
        observation_path,
        scenario["environment"],
    )
    assert composer.obs_dim == 108 + 10 + 2 * 20 + 5 * 3


@pytest.mark.parametrize(
    ("scenario_path", "preview", "neighbors"),
    [
        ("scenarios/complete_4.yaml", (), ()),
        (
            "scenarios/complete_4_frenet.yaml",
            ("car_0", "car_1", "car_2", "car_3"),
            (),
        ),
        (
            "scenarios/complete_4_frenet_neighbors.yaml",
            ("car_0", "car_1", "car_2", "car_3"),
            ("car_0", "car_1", "car_2", "car_3"),
        ),
    ],
)
def test_complete_4_feature_requirements_match_observation_arms(
    scenario_path: str,
    preview: tuple[str, ...],
    neighbors: tuple[str, ...],
) -> None:
    scenario = load_and_expand_scenario(scenario_path)
    requirements = derive_environment_feature_requirements(
        scenario["agents"],
        scenario_dir=Path(scenario_path).resolve().parent,
        centerline_render=scenario["environment"].get("centerline_render", False),
    )

    assert requirements.centerline_progress_agents == (
        "car_0", "car_1", "car_2", "car_3"
    )
    assert requirements.track_preview_agents == preview
    assert requirements.frenet_neighbor_agents == neighbors


def test_feature_requirements_aggregate_heterogeneous_agents(tmp_path: Path) -> None:
    requirements = derive_environment_feature_requirements(
        {
            "car_0": {
                "observation": {
                    "observation": {"progress": {"enabled": True}}
                }
            },
            "car_1": {
                "observation": {
                    "observation": {
                        "frenet_vehicle_track": {"enabled": True}
                    }
                }
            },
            "car_2": {
                "observation": {
                    "observation": {"frenet_neighbors": {"enabled": True}}
                }
            },
            "car_3": {
                "reward": {
                    "reward": {"wrong_way_penalty": {"enabled": True}}
                }
            },
        },
        scenario_dir=tmp_path,
    )

    assert requirements.centerline_progress_agents == (
        "car_0", "car_1", "car_2", "car_3"
    )
    assert requirements.frenet_vehicle_state_agents == ("car_1",)
    assert requirements.track_preview_agents == ("car_1",)
    assert requirements.frenet_neighbor_agents == ("car_2",)


def test_gated_frenet_payloads_match_direct_geometry_computation() -> None:
    geometry = TrackPreviewGeometry.build(
        _circle(10.0),
        {0: _circle(9.0), 1: _circle(11.0)},
        spacing=0.3,
    )
    assert geometry is not None

    env = F110ParallelEnv.__new__(F110ParallelEnv)
    env._track_preview_geometry = geometry
    env._track_preview_agents = frozenset({"car_0"})
    env._frenet_neighbor_agents = frozenset({"car_0"})
    env._track_preview_points = 8
    env._track_preview_last_indices = {"car_0": -1, "car_1": -1}
    env.possible_agents = ["car_0", "car_1"]
    env._agent_id_to_index = {"car_0": 0, "car_1": 1}
    env.poses_x = np.array([10.0, 0.0], dtype=np.float32)
    env.poses_y = np.array([0.0, 10.0], dtype=np.float32)
    env._last_centerline_facts = {
        "car_0": {"s": 0.0, "d": 0.0, "vs": 4.0, "vd": 0.0},
        "car_1": {"s": 3.0, "d": 0.5, "vs": 5.0, "vd": -0.5},
    }
    env._centerline_progress_tracker = SimpleNamespace(
        track_length=geometry.projection_geometry.total_length,
        closed=geometry.closed,
    )

    infos = {"car_0": {}, "car_1": {}}
    env._inject_track_previews(infos)
    env._inject_frenet_neighbors(infos)

    direct_index = geometry.nearest_index(np.array([10.0, 0.0], dtype=np.float32))
    direct_preview = geometry.preview(
        np.array([10.0, 0.0], dtype=np.float32),
        8,
        start_index=direct_index,
    )
    assert infos["car_0"]["track_preview"]["curvature"] == pytest.approx(
        direct_preview["curvature"]
    )
    assert infos["car_0"]["track_preview"]["width"] == pytest.approx(
        direct_preview["width"]
    )
    direct_neighbors = build_relative_frenet_facts(
        env._last_centerline_facts,
        track_length=geometry.projection_geometry.total_length,
        closed=geometry.closed,
    )
    assert infos["car_0"]["frenet_neighbors"] == direct_neighbors["car_0"]
    assert "track_preview" not in infos["car_1"]
    assert "frenet_neighbors" not in infos["car_1"]


@pytest.mark.parametrize(
    ("scenario_path", "has_preview", "has_neighbors"),
    [
        ("scenarios/complete_4.yaml", False, False),
        ("scenarios/complete_4_frenet.yaml", True, False),
        ("scenarios/complete_4_frenet_neighbors.yaml", True, True),
    ],
)
def test_complete_4_reset_emits_only_required_frenet_payloads(
    scenario_path: str,
    has_preview: bool,
    has_neighbors: bool,
) -> None:
    scenario = load_and_expand_scenario(scenario_path)
    scenario_dir = Path(scenario_path).resolve().parent
    env, _, _ = create_training_setup(
        scenario,
        mode="train",
        scenario_dir=scenario_dir,
    )
    try:
        _, infos = env.reset(seed=42)
        for agent_id in env.possible_agents:
            assert "centerline" in infos[agent_id]
            assert ("track_preview" in infos[agent_id]) is has_preview
            assert ("frenet_neighbors" in infos[agent_id]) is has_neighbors
        assert (env._track_preview_geometry is not None) is has_preview
    finally:
        env.close()
