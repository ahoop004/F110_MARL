"""Step/reset info payload assembly helpers."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


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
