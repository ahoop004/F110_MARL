"""Step/reset info payload assembly helpers."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.env.centerline_state import inject_finish_line_info
from src.env.types import AgentLifecycleRecord, AgentState, GlobalState, StepFacts


MINIMAL_INFO_KEYS = {
    "collision",
    "target_collision",
    "target_finished",
    "finish_line",
    "time_limit",
    "lap_crossed",
    "lap_count",
    "target_laps",
    "race_completed",
    "terminal_reason",
    "terminal_step",
    "finish_position",
    "status",
    "centerline",
    "track_preview",
    "frenet_neighbors",
    "target_lap_count",
    "target_race_completed",
}

# Keys guaranteed stable across env changes.  Online trainers, offline
# dataset writers, and heuristic policies may depend on these names.
# New keys should be added here only when their semantics are settled.
STABLE_STEP_INFO_KEYS: frozenset = frozenset(
    {
        "collision",         # bool — this agent collided this step
        "target_collision",  # bool — target agent collided this step
        "target_finished",   # bool — target agent crossed finish this step
        "time_limit",        # bool — episode was truncated by max_steps
        "lap_crossed",       # bool — accepted forward crossing on this step
        "lap_count",         # int  — completed laps after this step
        "target_laps",       # int  — configured race distance
        "race_completed",    # bool — lap_count reached target_laps
        "terminal_reason",   # str | None — immutable terminal cause
        "terminal_step",     # int | None — first terminal simulator step
        "finish_position",   # int | None — immutable one-based order
        "status",            # str — active/finished/crashed/truncated
        "target_lap_count",
        "target_race_completed",
        "target_terminal_reason",
        "target_finish_position",
        "centerline",        # dict — CenterlineProgressTracker facts (when enabled)
        "track_preview",     # dict — forward curvature/width samples and maxima
        "frenet_neighbors",  # list — ego-relative Frenet state of other agents
    }
)

STABLE_RESET_INFO_KEYS: frozenset = frozenset(
    {
        "map_bundle",   # str  — active map bundle name
        "spawn_point",  # str  — named spawn point selected for this agent
        "spawn_s",      # float — centerline s-position used for spawn
        "spawn_d",      # float — lateral offset used for spawn
    }
)


def add_step_info_fields(
    infos: Mapping[str, Dict[str, Any]],
    *,
    possible_agents: Sequence[str],
    agent_target_index: Mapping[str, Optional[int]],
    collision_flags: np.ndarray,
    collision_events: Optional[np.ndarray] = None,
    finish_crossed: Optional[np.ndarray],
    lifecycle_records: Optional[Mapping[str, AgentLifecycleRecord]] = None,
    locked_velocities: Mapping[str, float],
    lock_speed_steps: int,
    episode_step_count: int,
) -> None:
    for idx, agent_id in enumerate(possible_agents):
        if agent_id not in infos:
            continue
        payload = infos[agent_id]
        event_array = collision_events if collision_events is not None else collision_flags
        payload["collision"] = bool(event_array[idx])
        payload["collision_event"] = bool(event_array[idx])

        target_idx = agent_target_index.get(agent_id)
        if target_idx is not None and target_idx < len(collision_flags):
            payload["target_collision"] = bool(event_array[target_idx])
        else:
            payload["target_collision"] = False

        if (
            target_idx is not None
            and finish_crossed is not None
            and target_idx < len(finish_crossed)
        ):
            payload["target_finished"] = bool(finish_crossed[target_idx])
        else:
            payload["target_finished"] = False

        if lifecycle_records is not None:
            record = lifecycle_records[agent_id]
            payload.update(
                {
                    "lap_crossed": bool(record.lap_crossed),
                    "lap_count": int(record.lap_count),
                    "target_laps": int(record.target_laps),
                    "race_completed": bool(record.race_completed),
                    "terminal_reason": (
                        record.terminal_reason.value if record.terminal_reason else None
                    ),
                    "terminal_step": record.terminal_step,
                    "finish_position": record.finish_position,
                    "status": record.status.value,
                }
            )
            if target_idx is not None and target_idx < len(possible_agents):
                target = lifecycle_records[possible_agents[target_idx]]
                payload["target_lap_count"] = int(target.lap_count)
                payload["target_race_completed"] = bool(target.race_completed)
                payload["target_finished"] = bool(target.race_completed)
                payload["target_terminal_reason"] = (
                    target.terminal_reason.value if target.terminal_reason else None
                )
                payload["target_finish_position"] = target.finish_position
            else:
                payload["target_lap_count"] = 0
                payload["target_race_completed"] = False
                payload["target_terminal_reason"] = None
                payload["target_finish_position"] = None

        if agent_id in locked_velocities:
            payload["locked_velocity"] = float(locked_velocities[agent_id])
            payload["lock_speed_active"] = bool(
                lock_speed_steps > 0 and episode_step_count <= lock_speed_steps
            )
        else:
            payload["locked_velocity"] = None
            payload["lock_speed_active"] = False


def add_episode_metadata(
    infos: Mapping[str, Dict[str, Any]],
    *,
    map_bundle: Optional[Any] = None,
    spawn_metadata: Optional[Mapping[str, Any]] = None,
    protocol_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    if map_bundle:
        for payload in infos.values():
            payload["map_bundle"] = str(map_bundle)
    if spawn_metadata:
        metadata = dict(spawn_metadata)
        for payload in infos.values():
            payload.update(metadata)
    if protocol_metadata:
        metadata = dict(protocol_metadata)
        for payload in infos.values():
            payload.update(metadata)


def add_spawn_mapping(
    infos: Mapping[str, Dict[str, Any]],
    spawn_mapping: Mapping[str, Any],
) -> None:
    for agent_id, name in spawn_mapping.items():
        payload = infos.get(agent_id)
        if payload is not None:
            payload["spawn_point"] = str(name)


def build_reset_info_payloads(
    *,
    agent_ids: Sequence[str],
    map_bundle: Optional[Any] = None,
    spawn_mapping: Optional[Mapping[str, Any]] = None,
    spawn_metadata: Optional[Mapping[str, Any]] = None,
    protocol_metadata: Optional[Mapping[str, Any]] = None,
    finish_line_data: Optional[Mapping[str, Any]] = None,
    finish_crossed: Optional[np.ndarray] = None,
    agent_id_to_index: Optional[Mapping[str, int]] = None,
    info_level: str = "training",
) -> Dict[str, Dict[str, Any]]:
    infos = {str(agent_id): {} for agent_id in agent_ids}
    add_episode_metadata(
        infos,
        map_bundle=map_bundle,
        spawn_metadata=spawn_metadata,
        protocol_metadata=protocol_metadata,
    )
    if spawn_mapping:
        add_spawn_mapping(infos, spawn_mapping)
    if agent_id_to_index is not None:
        inject_finish_line_info(
            finish_line_data,
            finish_crossed,
            agent_id_to_index,
            infos,
        )
    return filter_info_payloads(infos, info_level=info_level)


def add_time_limit_info(
    infos: Mapping[str, Dict[str, Any]],
    *,
    truncated: bool = False,
    truncations: Optional[Mapping[str, bool]] = None,
) -> None:
    for agent_id, payload in infos.items():
        value = bool(truncations.get(agent_id, False)) if truncations is not None else bool(truncated)
        payload["time_limit"] = value


def build_step_facts(
    *,
    agent_ids: Sequence[str],
    agent_states: Mapping[str, AgentState],
    global_state: GlobalState,
    collision_flags: np.ndarray,
    terminations: Mapping[str, bool],
    truncations: Mapping[str, bool],
    infos: Mapping[str, Mapping[str, Any]],
) -> StepFacts:
    collisions = {}
    for idx, agent_id in enumerate(agent_ids):
        collisions[str(agent_id)] = bool(collision_flags[idx]) if idx < len(collision_flags) else False
    return StepFacts(
        agent_states={str(agent_id): agent_states[str(agent_id)] for agent_id in agent_ids if str(agent_id) in agent_states},
        global_state=global_state,
        collisions=collisions,
        terminations={str(agent_id): bool(value) for agent_id, value in terminations.items()},
        truncations={str(agent_id): bool(value) for agent_id, value in truncations.items()},
        info={
            str(agent_id): dict(payload)
            for agent_id, payload in infos.items()
        },
    )


def filter_info_payloads(
    infos: Mapping[str, Dict[str, Any]],
    *,
    info_level: str,
) -> Dict[str, Dict[str, Any]]:
    level = str(info_level or "training").strip().lower()
    if level in {"training", "debug"}:
        return {agent_id: dict(payload) for agent_id, payload in infos.items()}
    if level != "minimal":
        return {agent_id: dict(payload) for agent_id, payload in infos.items()}
    return {
        agent_id: {
            key: value
            for key, value in payload.items()
            if key in MINIMAL_INFO_KEYS
        }
        for agent_id, payload in infos.items()
    }
