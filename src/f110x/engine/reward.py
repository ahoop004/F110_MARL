"""Reward curriculum helpers shared across training and evaluation."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from f110x.envs import F110ParallelEnv
from f110x.tasks.reward import reward_task_registry
from f110x.utils.map_loader import MapData
from f110x.wrappers.reward import RewardRuntimeContext, RewardWrapper


CurriculumSchedule = List[Tuple[Optional[int], str]]


def build_curriculum_schedule(raw_curriculum: Iterable[Dict[str, Any]]) -> CurriculumSchedule:
    """Normalise curriculum definitions into an ordered schedule."""

    schedule: CurriculumSchedule = []
    for stage in raw_curriculum:
        if not isinstance(stage, dict):
            continue
        mode = stage.get("task") or stage.get("mode")
        if mode is None:
            continue
        upper = stage.get("until", stage.get("episodes"))
        if upper is not None:
            try:
                upper = int(upper)
            except (TypeError, ValueError):
                upper = None
        schedule.append((upper, str(mode)))
    schedule.sort(key=lambda item: float("inf") if item[0] is None else item[0])
    return schedule


def resolve_reward_mode(
    curriculum: CurriculumSchedule,
    episode_idx: int,
    *,
    default_sequence: Optional[List[Tuple[int, str]]] = None,
) -> str:
    """Resolve the canonical reward task for the given episode index."""

    if curriculum:
        for threshold, mode in curriculum:
            if threshold is None or episode_idx < threshold:
                resolved = mode
                break
        else:
            resolved = curriculum[-1][1]
    else:
        defaults = default_sequence or [
            (1000, "basic"),
            (2000, "pursuit"),
        ]
        resolved = defaults[-1][1] if defaults else "gaplock"
        for threshold, mode in defaults:
            if episode_idx < threshold:
                resolved = mode
                break

    try:
        return reward_task_registry.normalize(resolved, default="gaplock")
    except KeyError as exc:
        raise ValueError(f"Unknown reward task '{resolved}' in curriculum") from exc


def build_reward_wrapper(
    reward_cfg: Dict[str, Any],
    env: F110ParallelEnv,
    map_data: MapData,
    episode_idx: int,
    *,
    curriculum: CurriculumSchedule,
    roster: Optional[Any] = None,
) -> RewardWrapper:
    """Construct a reward wrapper instance for the given episode."""

    wrapper_cfg = dict(reward_cfg)
    task_id = resolve_reward_mode(curriculum, episode_idx)
    wrapper_cfg["task"] = task_id
    wrapper_cfg["mode"] = task_id  # Retain legacy field for migration tooling

    runtime_context = RewardRuntimeContext(env=env, map_data=map_data, roster=roster)
    wrapper = RewardWrapper(config=wrapper_cfg, context=runtime_context)
    wrapper.reset(episode_idx)
    return wrapper


__all__ = [
    "CurriculumSchedule",
    "build_curriculum_schedule",
    "build_reward_wrapper",
    "resolve_reward_mode",
]
