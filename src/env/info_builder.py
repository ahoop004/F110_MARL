"""Step/reset info payload assembly helpers."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.env.centerline_state import inject_finish_line_info
from src.env.types import AgentState, GlobalState, StepFacts


MINIMAL_INFO_KEYS = {
    "collision",
    "target_collision",
    "target_finished",
    "finish_line",
    "time_limit",
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
        "centerline",        # dict — CenterlineProgressTracker facts (when enabled)
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
    finish_crossed: Optional[np.ndarray],
    locked_velocities: Mapping[str, float],
    lock_speed_steps: int,
    episode_step_count: int,
) -> None:
    for idx, agent_id in enumerate(possible_agents):
        if agent_id not in infos:
            continue
        payload = infos[agent_id]
        payload["collision"] = bool(collision_flags[idx])

        target_idx = agent_target_index.get(agent_id)
        if target_idx is not None and target_idx < len(collision_flags):
            payload["target_collision"] = bool(collision_flags[target_idx])
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
) -> None:
    if map_bundle:
        for payload in infos.values():
            payload["map_bundle"] = str(map_bundle)
    if spawn_metadata:
        metadata = dict(spawn_metadata)
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
    truncated: bool,
) -> None:
    if not truncated:
        return
    for payload in infos.values():
        payload["time_limit"] = True


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
