"""Track-completion reward components."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from wrappers.rewards.base import RewardComponent


def _centerline_info(step_info: dict) -> dict:
    info = step_info.get("info") or {}
    centerline = info.get("centerline", {}) if isinstance(info, dict) else {}
    return centerline if isinstance(centerline, dict) else {}


def _centerline_from_info(info: object) -> dict:
    if not isinstance(info, dict):
        return {}
    centerline = info.get("centerline", {})
    return centerline if isinstance(centerline, dict) else {}


def _float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_ids(values: Optional[Iterable[Any]]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _agent_progress_delta(step_info: dict, agent_id: str) -> Optional[float]:
    agent_id = str(agent_id)
    if str(step_info.get("agent_id")) == agent_id:
        info = step_info.get("info") or {}
        if (
            info.get("status") not in {None, "active"}
            and not info.get("lap_crossed", False)
            and not info.get("collision_event", False)
        ):
            return None
        return _float_or_none(_centerline_info(step_info).get("progress_delta"))

    all_infos = step_info.get("all_infos") or {}
    if not isinstance(all_infos, dict):
        return None
    agent_info = all_infos.get(agent_id)
    if isinstance(agent_info, dict) and (
        agent_info.get("status") not in {None, "active"}
        and not agent_info.get("lap_crossed", False)
        and not agent_info.get("collision_event", False)
    ):
        return None
    return _float_or_none(_centerline_from_info(agent_info).get("progress_delta"))


def _aggregate_progress_delta(
    step_info: dict,
    agent_ids: Iterable[str],
    *,
    positive_only: bool,
    aggregation: str,
) -> Optional[float]:
    deltas = []
    for agent_id in agent_ids:
        delta = _agent_progress_delta(step_info, agent_id)
        if delta is None:
            continue
        if positive_only:
            delta = max(delta, 0.0)
        deltas.append(delta)

    if not deltas:
        return None
    if aggregation == "sum":
        return sum(deltas)
    return sum(deltas) / len(deltas)


class ProgressDeltaBonusComponent(RewardComponent):
    """Reward positive lap-fraction progress from the centerline tracker."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 100.0))
        self.positive_only = bool(config.get("positive_only", True))
        max_delta = config.get("max_delta")
        self.max_delta = _float_or_none(max_delta) if max_delta is not None else None

    def compute(self, step_info: dict) -> Dict[str, float]:
        centerline = _centerline_info(step_info)
        progress_delta = _float_or_none(centerline.get("progress_delta"))
        if progress_delta is None:
            return {}

        if self.positive_only:
            progress_delta = max(progress_delta, 0.0)
        if self.max_delta is not None:
            progress_delta = min(progress_delta, self.max_delta)

        return {"progress_delta/bonus": self.weight * progress_delta}


class RelativeProgressBonusComponent(RewardComponent):
    """Reward lap-fraction progress relative to a configured target agent."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 100.0))
        self.positive_only = bool(config.get("positive_only", False))
        self.target_id = config.get("target_id")
        max_abs_delta = config.get("max_abs_delta")
        self.max_abs_delta = (
            _float_or_none(max_abs_delta) if max_abs_delta is not None else None
        )

    def compute(self, step_info: dict) -> Dict[str, float]:
        ego_delta = _float_or_none(_centerline_info(step_info).get("progress_delta"))
        target_delta = self._target_progress_delta(step_info)
        if ego_delta is None or target_delta is None:
            return {}

        relative_delta = ego_delta - target_delta
        if self.positive_only:
            relative_delta = max(relative_delta, 0.0)
        if self.max_abs_delta is not None:
            bound = abs(self.max_abs_delta)
            relative_delta = max(-bound, min(bound, relative_delta))

        return {"relative_progress/bonus": self.weight * relative_delta}

    def _target_progress_delta(self, step_info: dict) -> Optional[float]:
        target_id = self._target_id(step_info)
        if not target_id:
            return None

        all_infos = step_info.get("all_infos") or {}
        if not isinstance(all_infos, dict):
            return None
        target_info = all_infos.get(target_id)
        return _float_or_none(_centerline_from_info(target_info).get("progress_delta"))

    def _target_id(self, step_info: dict) -> Optional[str]:
        target_id = step_info.get("target_id") or self.target_id
        if target_id:
            return str(target_id)

        agent_id = step_info.get("agent_id")
        all_infos = step_info.get("all_infos") or {}
        if not agent_id or not isinstance(all_infos, dict):
            return None

        other_ids = [str(aid) for aid in all_infos if str(aid) != str(agent_id)]
        if len(other_ids) == 1:
            return other_ids[0]
        return None


class FinishAheadBonusComponent(RewardComponent):
    """Sparse bonus when ego finishes before its configured target."""

    def __init__(self, config: dict) -> None:
        self.bonus = float(config.get("bonus", 100.0))
        self.require_clean = bool(config.get("require_clean", True))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        if not bool(info.get("race_completed", False)):
            return {}
        if self.require_clean and bool(info.get("collision", False)):
            return {}

        ego_position = info.get("finish_position")
        target_position = info.get("target_finish_position")
        if target_position is not None and (
            ego_position is None or int(ego_position) >= int(target_position)
        ):
            return {}

        return {"finish_ahead/bonus": self.bonus}


class TeamProgressBonusComponent(RewardComponent):
    """Reward aggregate lap-fraction progress for a configured team."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 100.0))
        self.positive_only = bool(config.get("positive_only", True))
        self.aggregation = str(config.get("aggregation", "mean")).lower()
        self.team_agent_ids = _string_ids(config.get("team_agent_ids"))

    def compute(self, step_info: dict) -> Dict[str, float]:
        team_ids = self.team_agent_ids or _string_ids(step_info.get("trainable_agent_ids"))
        if not team_ids:
            return {}

        progress = _aggregate_progress_delta(
            step_info,
            team_ids,
            positive_only=self.positive_only,
            aggregation=self.aggregation,
        )
        if progress is None:
            return {}
        return {"team_progress/bonus": self.weight * progress}


class TeamRelativeProgressBonusComponent(RewardComponent):
    """Reward aggregate team progress relative to aggregate opponent progress."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 100.0))
        self.positive_only = bool(config.get("positive_only", False))
        self.aggregation = str(config.get("aggregation", "mean")).lower()
        self.team_agent_ids = _string_ids(config.get("team_agent_ids"))
        self.opponent_agent_ids = _string_ids(config.get("opponent_agent_ids"))

    def compute(self, step_info: dict) -> Dict[str, float]:
        team_ids = self.team_agent_ids or _string_ids(step_info.get("trainable_agent_ids"))
        opponent_ids = self.opponent_agent_ids or _string_ids(
            step_info.get("opponent_agent_ids")
        )
        if not team_ids or not opponent_ids:
            return {}

        team_progress = _aggregate_progress_delta(
            step_info,
            team_ids,
            positive_only=False,
            aggregation=self.aggregation,
        )
        opponent_progress = _aggregate_progress_delta(
            step_info,
            opponent_ids,
            positive_only=False,
            aggregation=self.aggregation,
        )
        if team_progress is None or opponent_progress is None:
            return {}

        relative_progress = team_progress - opponent_progress
        if self.positive_only:
            relative_progress = max(relative_progress, 0.0)
        return {"team_relative_progress/bonus": self.weight * relative_progress}


class StepTimePenaltyComponent(RewardComponent):
    """Small per-decision penalty to prefer shorter completion times."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -0.01))
        self.apply_on_terminal = bool(config.get("apply_on_terminal", True))

    def compute(self, step_info: dict) -> Dict[str, float]:
        if not self.apply_on_terminal and (
            step_info.get("done") or step_info.get("terminated") or step_info.get("truncated")
        ):
            return {}
        return {"step_time/penalty": self.penalty}
