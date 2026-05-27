"""Per-agent observation assembly from simulator joint output.

``split_joint_obs`` converts the flat ``{key: (n_agents, ...)}`` dicts that
the physics simulator emits into the per-agent observation dicts expected by
the env contract.  It is a pure function — all dependencies are passed as
explicit arguments so the function can be tested without constructing an env.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def split_joint_obs(
    joint: Mapping[str, np.ndarray],
    *,
    possible_agents: Sequence[str],
    agent_sensor_spec: Mapping[str, Tuple[str, ...]],
    agent_target_index: Mapping[str, Optional[int]],
    default_sensors: Tuple[str, ...],
    lidar_beam_count: int,
    timestep: float,
    lap_counts: np.ndarray,
    lap_times: np.ndarray,
    prev_vels_x: Optional[np.ndarray] = None,
    prev_vels_y: Optional[np.ndarray] = None,
    velocity_initialized: bool = False,
    fallback_collisions: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Split a simulator joint observation into per-agent dicts.

    Parameters
    ----------
    joint:
        Raw ``{key: (n_agents, ...)}`` observation dict from the simulator.
    possible_agents:
        Ordered sequence of all agent IDs.
    agent_sensor_spec:
        Per-agent tuple of enabled sensor names.  Falls back to
        *default_sensors* when an agent has no explicit spec.
    agent_target_index:
        Mapping from agent_id to the integer index of its target agent, or
        ``None`` when no target is configured.
    default_sensors:
        Sensor names used when an agent has no entry in *agent_sensor_spec*.
    lidar_beam_count:
        Fallback LiDAR beam count when the joint obs is missing scans.
    timestep:
        Simulation timestep used for acceleration finite-difference.
    lap_counts:
        Per-agent lap counts (shape ``(n_agents,)``).
    lap_times:
        Per-agent lap times (shape ``(n_agents,)``).
    prev_vels_x, prev_vels_y:
        Previous-step velocity arrays used to compute acceleration.  When
        *velocity_initialized* is False these are ignored and acceleration
        is reported as zero.
    velocity_initialized:
        True once the state buffer has at least one valid velocity reading.
    fallback_collisions:
        Collision array used when ``joint["collisions"]`` is not available.

    Returns
    -------
    Dict mapping each agent_id to its per-sensor observation dict.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}

    scans = joint.get("scans")
    poses_x = joint.get("poses_x")
    poses_y = joint.get("poses_y")
    poses_theta = joint.get("poses_theta")
    linear_vels_x = joint.get("linear_vels_x")
    linear_vels_y = joint.get("linear_vels_y")
    ang_vels_z = joint.get("ang_vels_z")
    collisions = joint.get("collisions")

    dt = max(float(timestep), 1e-6)

    for i, aid in enumerate(possible_agents):
        sensors = agent_sensor_spec.get(aid, default_sensors)
        agent_obs: Dict[str, np.ndarray] = {}
        target_idx = agent_target_index.get(aid)

        has_scan = isinstance(scans, np.ndarray) and i < scans.shape[0]

        if "lidar" in sensors:
            if has_scan:
                lidar_reading = np.asarray(scans[i], dtype=np.float32)
            else:
                n_beams = (
                    scans.shape[1]
                    if isinstance(scans, np.ndarray) and scans.ndim == 2
                    else lidar_beam_count
                )
                lidar_reading = np.zeros((n_beams,), dtype=np.float32)
            agent_obs["lidar"] = lidar_reading
            agent_obs["scans"] = lidar_reading

        if "pose" in sensors:
            if isinstance(poses_x, np.ndarray) and i < poses_x.shape[0]:
                agent_obs["pose"] = np.array(
                    [np.float32(poses_x[i]), np.float32(poses_y[i]), np.float32(poses_theta[i])],
                    dtype=np.float32,
                )
            else:
                agent_obs["pose"] = np.zeros(3, dtype=np.float32)

        curr_vx = (
            float(linear_vels_x[i])
            if isinstance(linear_vels_x, np.ndarray) and i < linear_vels_x.shape[0]
            else 0.0
        )
        curr_vy = (
            float(linear_vels_y[i])
            if isinstance(linear_vels_y, np.ndarray) and i < linear_vels_y.shape[0]
            else 0.0
        )

        if "velocity" in sensors:
            agent_obs["velocity"] = np.array([curr_vx, curr_vy], dtype=np.float32)

        if "acceleration" in sensors:
            if (
                velocity_initialized
                and prev_vels_x is not None
                and prev_vels_y is not None
                and i < prev_vels_x.shape[0]
            ):
                ax = (curr_vx - float(prev_vels_x[i])) / dt
                ay = (curr_vy - float(prev_vels_y[i])) / dt
            else:
                ax = 0.0
                ay = 0.0
            agent_obs["acceleration"] = np.array([ax, ay], dtype=np.float32)

        if "angular_velocity" in sensors:
            agent_obs["angular_velocity"] = (
                np.float32(ang_vels_z[i])
                if isinstance(ang_vels_z, np.ndarray) and i < ang_vels_z.shape[0]
                else np.float32(0.0)
            )

        if "target_pose" in sensors:
            if target_idx is not None and isinstance(poses_x, np.ndarray) and target_idx < poses_x.shape[0]:
                agent_obs["target_pose"] = np.array(
                    [
                        np.float32(poses_x[target_idx]),
                        np.float32(poses_y[target_idx]),
                        np.float32(poses_theta[target_idx]),
                    ],
                    dtype=np.float32,
                )
            else:
                agent_obs["target_pose"] = np.zeros(3, dtype=np.float32)

        if "target_collision" in sensors:
            if target_idx is not None and isinstance(collisions, np.ndarray) and target_idx < collisions.shape[0]:
                agent_obs["target_collision"] = np.float32(float(collisions[target_idx]))
            else:
                agent_obs["target_collision"] = np.float32(0.0)

        if "lap" in sensors:
            lap_count_val = float(lap_counts[i]) if i < len(lap_counts) else 0.0
            lap_time_val = float(lap_times[i]) if i < len(lap_times) else 0.0
            agent_obs["lap"] = np.array([lap_count_val, lap_time_val], dtype=np.float32)

        if "collision" in sensors:
            if isinstance(collisions, np.ndarray) and i < collisions.shape[0]:
                col_val = float(collisions[i])
            elif fallback_collisions is not None and i < fallback_collisions.shape[0]:
                col_val = float(fallback_collisions[i])
            else:
                col_val = 0.0
            agent_obs["collision"] = np.float32(col_val)

        out[aid] = agent_obs

    return out
