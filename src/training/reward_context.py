"""Shared reward-context assembly for training loops."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def build_reward_context(
    *,
    env: Any,
    agent_id: str,
    info_dict: Dict[str, Any],
    obs_dict: Dict[str, Any],
    actions: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    try:
        global_state = env.get_global_state().vector
    except Exception:
        global_state = np.zeros(0, dtype=np.float32)

    trainable_agent_ids = _string_list(getattr(env, "trainable_agents", []) or [])
    fixed_policy_agent_ids = _string_list(getattr(env, "fixed_policy_agents", []) or [])
    team_set = set(trainable_agent_ids)

    get_target_id = getattr(env, "get_target_id", None)
    target_id = get_target_id(agent_id) if callable(get_target_id) else None

    opponent_agent_ids = [aid for aid in fixed_policy_agent_ids if aid != agent_id]
    if not opponent_agent_ids:
        opponent_agent_ids = [
            str(aid)
            for aid in (info_dict or {})
            if str(aid) != str(agent_id) and str(aid) not in team_set
        ]

    return {
        "agent_id": agent_id,
        "target_id": target_id,
        "trainable_agent_ids": trainable_agent_ids,
        "teammate_ids": [aid for aid in trainable_agent_ids if aid != agent_id],
        "opponent_agent_ids": opponent_agent_ids,
        "all_infos": info_dict or {},
        "all_obs": obs_dict or {},
        "all_actions": actions or {},
        "global_state": global_state,
        "last_step_facts": getattr(env, "last_step_facts", None),
    }


def _string_list(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values]
