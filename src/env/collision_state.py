"""Collision state and termination helpers."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from src.env.collision import build_terminations


def update_collision_flags(
    possible_agents: Sequence[str],
    collisions: np.ndarray,
    collision_flags: np.ndarray,
    collision_steps: np.ndarray,
    elapsed_steps: int,
) -> None:
    collision_array = np.asarray(collisions)
    for idx, _agent_id in enumerate(possible_agents):
        collided = idx < collision_array.shape[0] and bool(collision_array[idx])
        if collided and not collision_flags[idx]:
            collision_flags[idx] = True
            collision_steps[idx] = int(elapsed_steps)


def build_step_terminations(
    possible_agents: Sequence[str],
    collision_flags: np.ndarray,
    lap_completion: Mapping[str, bool],
    terminate_on_collision: Mapping[str, bool],
) -> Dict[str, bool]:
    terminations = build_terminations(
        possible_agents,
        collision_flags,
        lap_completion,
        terminate_on_collision,
    )
    any_collision_termination = any(
        collision_flags[idx] and terminate_on_collision.get(agent_id, True)
        for idx, agent_id in enumerate(possible_agents)
    )
    if any_collision_termination:
        for agent_id in possible_agents:
            terminations[agent_id] = True
    return terminations


def build_truncations(
    possible_agents: Sequence[str],
    *,
    max_steps: int,
    elapsed_steps: int,
) -> tuple[Dict[str, bool], bool]:
    trunc_flag = (max_steps > 0) and (elapsed_steps + 1 >= max_steps)
    return {agent_id: bool(trunc_flag) for agent_id in possible_agents}, bool(trunc_flag)
