"""Public state views for centralized critics, datasets, and diagnostics."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.env.types import AgentState, GlobalState, ProgressState


def central_state_tensor(
    joint: Mapping[str, np.ndarray],
    *,
    n_agents: int,
    central_state_keys: Sequence[str],
) -> np.ndarray:
    central = np.zeros((n_agents * len(central_state_keys),), dtype=np.float32)
    if n_agents == 0:
        return central

    span = n_agents
    offset = 0
    for key in central_state_keys:
        arr = joint.get(key)
        if arr is None:
            offset += span
            continue
        view = np.asarray(arr, dtype=np.float32).reshape(-1)
        if view.size >= span:
            central[offset:offset + span] = view[:span]
        else:
            central[offset:offset + view.size] = view
        offset += span
    return central


def build_agent_state(
    agent_id: str,
    *,
    agent_index: Mapping[str, int],
    poses_x: np.ndarray,
    poses_y: np.ndarray,
    poses_theta: np.ndarray,
    linear_vels_x: np.ndarray,
    linear_vels_y: np.ndarray,
    angular_vels: np.ndarray,
    collision_flags: np.ndarray,
    lap_counts: Optional[np.ndarray] = None,
    lap_times: Optional[np.ndarray] = None,
    finish_crossed: Optional[np.ndarray] = None,
    centerline_facts: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentState:
    idx = agent_index[agent_id]
    pose = np.array(
        [poses_x[idx], poses_y[idx], poses_theta[idx]],
        dtype=np.float32,
    )
    velocity = np.array(
        [linear_vels_x[idx], linear_vels_y[idx]],
        dtype=np.float32,
    )
    lap_count = int(lap_counts[idx]) if lap_counts is not None and idx < len(lap_counts) else 0
    lap_time = float(lap_times[idx]) if lap_times is not None and idx < len(lap_times) else 0.0
    finished = bool(finish_crossed[idx]) if finish_crossed is not None and idx < len(finish_crossed) else False

    cl = centerline_facts or {}
    progress = ProgressState(
        lap_count=lap_count,
        lap_time=lap_time,
        finished=finished,
        progress=float(cl.get("progress", 0.0)),
        progress_delta=float(cl.get("progress_delta", 0.0)),
        wrong_way=bool(cl.get("wrong_way", False)),
    )
    return AgentState(
        agent_id=agent_id,
        pose=pose,
        velocity=velocity,
        angular_velocity=float(angular_vels[idx]) if idx < len(angular_vels) else 0.0,
        collision=bool(collision_flags[idx]) if idx < len(collision_flags) else False,
        progress=progress,
        metadata=dict(metadata or {}),
    )


def build_masks(
    possible_agents: Sequence[str],
    active_agents: Sequence[str],
    *,
    controlled_agents: Optional[Sequence[str]] = None,
    trainable_agents: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    active_set = set(active_agents)
    controlled_set = set(controlled_agents) if controlled_agents is not None else set(possible_agents)
    trainable_set = set(trainable_agents or ())
    active_mask = np.array([agent_id in active_set for agent_id in possible_agents], dtype=bool)
    controlled_mask = np.array([agent_id in controlled_set for agent_id in possible_agents], dtype=bool)
    trainable_mask = np.array([agent_id in trainable_set for agent_id in possible_agents], dtype=bool)
    return {
        "active_mask": active_mask,
        "terminated_mask": np.logical_not(active_mask),
        "controlled_mask": controlled_mask,
        "trainable_mask": trainable_mask,
    }


def build_global_state(
    *,
    possible_agents: Sequence[str],
    active_agents: Sequence[str],
    central_vector: np.ndarray,
    controlled_agents: Optional[Sequence[str]] = None,
    trainable_agents: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> GlobalState:
    return GlobalState(
        agent_ids=tuple(possible_agents),
        vector=np.asarray(central_vector, dtype=np.float32).copy(),
        masks=build_masks(
            possible_agents,
            active_agents,
            controlled_agents=controlled_agents,
            trainable_agents=trainable_agents,
        ),
        metadata=dict(metadata or {}),
    )
