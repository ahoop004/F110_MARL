"""Build raw environment action and observation space specs."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from src.env.spaces import DictSpaceSpec, SpaceSpec


def build_action_spaces(
    possible_agents: Sequence[str],
    vehicle_params: Mapping[str, float],
) -> Tuple[SpaceSpec, Dict[str, SpaceSpec]]:
    single_action_space = SpaceSpec(
        shape=(2,),
        low=np.array([vehicle_params["s_min"], vehicle_params["v_min"]], dtype=np.float32),
        high=np.array([vehicle_params["s_max"], vehicle_params["v_max"]], dtype=np.float32),
    )
    return single_action_space, {aid: single_action_space for aid in possible_agents}


def build_observation_spaces(
    *,
    possible_agents: Sequence[str],
    agent_sensor_spec: Mapping[str, Tuple[str, ...]],
    default_sensors: Tuple[str, ...],
    central_state_dim: int,
    lidar_beam_count: int,
    lidar_range: float,
    vehicle_params: Mapping[str, float],
    target_laps: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Dict[str, DictSpaceSpec]:
    pose_low = np.array([x_min, y_min, -np.pi], dtype=np.float32)
    pose_high = np.array([x_max, y_max, np.pi], dtype=np.float32)
    v_min = float(vehicle_params.get("v_min", -5.0))
    v_max = float(vehicle_params.get("v_max", 20.0))
    vel_low = np.array([v_min, v_min], dtype=np.float32)
    vel_high = np.array([v_max, v_max], dtype=np.float32)

    accel_cap = float(vehicle_params.get("a_max", 10.0))
    accel_low = np.array([-accel_cap, -accel_cap], dtype=np.float32)
    accel_high = np.array([accel_cap, accel_cap], dtype=np.float32)

    ang_cap = float(vehicle_params.get("ang_vel_max", 10.0))

    lap_cap = float(target_laps)
    lap_low = np.array([0.0, 0.0], dtype=np.float32)
    lap_high = np.array([lap_cap, 1e5], dtype=np.float32)

    obs_spaces: Dict[str, DictSpaceSpec] = {}
    for aid in possible_agents:
        sensors = agent_sensor_spec.get(aid, default_sensors)
        components: Dict[str, SpaceSpec] = {}

        if "lidar" in sensors:
            components["lidar"] = SpaceSpec(
                shape=(lidar_beam_count,),
                low=np.zeros(lidar_beam_count, dtype=np.float32),
                high=np.full(lidar_beam_count, lidar_range, dtype=np.float32),
            )
        if "pose" in sensors:
            components["pose"] = SpaceSpec(shape=(3,), low=pose_low, high=pose_high)
        if "velocity" in sensors:
            components["velocity"] = SpaceSpec(shape=(2,), low=vel_low, high=vel_high)
        if "acceleration" in sensors:
            components["acceleration"] = SpaceSpec(shape=(2,), low=accel_low, high=accel_high)
        if "angular_velocity" in sensors:
            components["angular_velocity"] = SpaceSpec(
                shape=(1,),
                low=np.array([-ang_cap], dtype=np.float32),
                high=np.array([ang_cap], dtype=np.float32),
            )
        if "target_pose" in sensors:
            components["target_pose"] = SpaceSpec(shape=(3,), low=pose_low, high=pose_high)
        if "target_collision" in sensors:
            components["target_collision"] = SpaceSpec(
                shape=(1,),
                low=np.zeros(1, dtype=np.float32),
                high=np.ones(1, dtype=np.float32),
            )
        if "lap" in sensors:
            components["lap"] = SpaceSpec(shape=(2,), low=lap_low, high=lap_high)
        if "collision" in sensors:
            components["collision"] = SpaceSpec(
                shape=(1,),
                low=np.zeros(1, dtype=np.float32),
                high=np.ones(1, dtype=np.float32),
            )

        components["state"] = SpaceSpec(
            shape=(central_state_dim,),
            low=np.full(central_state_dim, -np.inf, dtype=np.float32),
            high=np.full(central_state_dim, np.inf, dtype=np.float32),
        )
        obs_spaces[aid] = DictSpaceSpec(spaces=components)

    return obs_spaces
