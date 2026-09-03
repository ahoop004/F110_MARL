"""Public state views for centralized critics, datasets, and diagnostics."""
from __future__ import annotations

import copy
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.env.types import AgentLifecycleRecord, AgentRaceStatus, AgentState, GlobalState, ProgressState


def _freeze_snapshot_value(value: Any) -> Any:
    """Detach and recursively freeze metadata stored in a cached state view."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_snapshot_value(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        array = value.copy()
        array.setflags(write=False)
        return array
    if isinstance(value, list):
        return tuple(_freeze_snapshot_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_snapshot_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_snapshot_value(item) for item in value)
    return copy.deepcopy(value)


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
    lifecycle: Optional[AgentLifecycleRecord] = None,
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
    finished = (
        lifecycle.status == AgentRaceStatus.FINISHED
        if lifecycle is not None
        else bool(finish_crossed[idx]) if finish_crossed is not None and idx < len(finish_crossed) else False
    )

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
        metadata={
            **dict(metadata or {}),
            **(
                {
                    "status": lifecycle.status.value,
                    "terminal_reason": lifecycle.terminal_reason.value if lifecycle.terminal_reason else None,
                    "terminal_step": lifecycle.terminal_step,
                    "finish_position": lifecycle.finish_position,
                    "target_laps": lifecycle.target_laps,
                    "race_completed": lifecycle.race_completed,
                }
                if lifecycle is not None
                else {}
            ),
        },
    )


def build_masks(
    possible_agents: Sequence[str],
    active_agents: Sequence[str],
    *,
    controlled_agents: Optional[Sequence[str]] = None,
    trainable_agents: Optional[Sequence[str]] = None,
    lifecycle_records: Optional[Mapping[str, AgentLifecycleRecord]] = None,
) -> Dict[str, np.ndarray]:
    active_set = set(active_agents)
    controlled_set = set(controlled_agents) if controlled_agents is not None else set(possible_agents)
    trainable_set = set(trainable_agents or ())
    active_mask = np.array([agent_id in active_set for agent_id in possible_agents], dtype=bool)
    controlled_mask = np.array([agent_id in controlled_set for agent_id in possible_agents], dtype=bool)
    trainable_mask = np.array([agent_id in trainable_set for agent_id in possible_agents], dtype=bool)
    masks = {
        "active_mask": active_mask,
        "terminated_mask": np.logical_not(active_mask),
        "controlled_mask": controlled_mask,
        "trainable_mask": trainable_mask,
    }
    if lifecycle_records is not None:
        for status in (
            AgentRaceStatus.FINISHED,
            AgentRaceStatus.CRASHED,
            AgentRaceStatus.TRUNCATED,
        ):
            masks[f"{status.value}_mask"] = np.array(
                [lifecycle_records[aid].status == status for aid in possible_agents],
                dtype=bool,
            )
    return masks


def build_global_state(
    *,
    possible_agents: Sequence[str],
    active_agents: Sequence[str],
    central_vector: np.ndarray,
    controlled_agents: Optional[Sequence[str]] = None,
    trainable_agents: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    lifecycle_records: Optional[Mapping[str, AgentLifecycleRecord]] = None,
) -> GlobalState:
    vector = np.asarray(central_vector, dtype=np.float32).copy()
    vector.setflags(write=False)
    masks = build_masks(
        possible_agents,
        active_agents,
        controlled_agents=controlled_agents,
        trainable_agents=trainable_agents,
        lifecycle_records=lifecycle_records,
    )
    for mask in masks.values():
        mask.setflags(write=False)
    return GlobalState(
        agent_ids=tuple(possible_agents),
        vector=vector,
        masks=MappingProxyType(masks),
        metadata=_freeze_snapshot_value(dict(metadata or {})),
    )
