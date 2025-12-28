"""Collision termination helpers for the parallel F110 environment."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def build_terminations(
    agent_ids: Sequence[str],
    collisions: np.ndarray,
    lap_completion: Mapping[str, bool],
    terminate_on_collision: Mapping[str, bool],
) -> Dict[str, bool]:
    terminations: Dict[str, bool] = {}
    collision_array = np.asarray(collisions)
    for idx, agent_id in enumerate(agent_ids):
        collided = bool(collision_array[idx]) if idx < collision_array.shape[0] else False
        collision_done = collided and terminate_on_collision.get(agent_id, True)
        lap_done = bool(lap_completion.get(agent_id, False))
        terminations[agent_id] = collision_done or lap_done
    return terminations

