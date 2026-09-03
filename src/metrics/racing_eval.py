"""Fact-based racing evaluation metrics.

These helpers intentionally consume environment facts such as finish-line
crossings, centerline progress, truncation flags, and collision flags.  They do
not infer racing outcomes from reward values.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


DEFERRED_COLLISION_PAIR_METRICS = (
    "teammate_collision_rate",
    "opponent_collision_rate",
    "wall_collision_rate",
)


@dataclass
class AgentEpisodeFacts:
    agent_id: str
    team: str
    reward_total: float = 0.0
    individual_reward_total: float = 0.0
    reward_components: Dict[str, float] = field(default_factory=dict)
    active_steps: int = 0
    done_step: Optional[int] = None
    finish_step: Optional[int] = None
    collision_step: Optional[int] = None
    timed_out: bool = False
    final_progress: Optional[float] = None
    speed_samples: list[float] = field(default_factory=list)
    outcome: str = "unknown"
    terminal_reason: Optional[str] = None
    finish_position: Optional[int] = None
    final_lap_count: int = 0

    @property
    def completed(self) -> bool:
        return self.finish_step is not None

    @property
    def collided(self) -> bool:
        return self.collision_step is not None

    @property
    def clean_finish(self) -> bool:
        if self.finish_step is None:
            return False
        return self.collision_step is None or self.collision_step > self.finish_step

    @property
    def progress_score(self) -> float:
        if self.completed:
            return 1.0
        if self.final_progress is None or not np.isfinite(self.final_progress):
            return 0.0
        return float(np.clip(self.final_progress, 0.0, 1.0))


@dataclass
class EvalEpisodeFacts:
    episode: int
    steps: int
    agents: Dict[str, AgentEpisodeFacts]
    trainable_team: tuple[str, ...]
    opponent_team: tuple[str, ...]
    deferred: set[str] = field(default_factory=set)


def create_episode_facts(
    *,
    episode: int,
    agent_ids: Sequence[str],
    trainable_ids: Sequence[str],
    opponent_ids: Sequence[str],
) -> EvalEpisodeFacts:
    trainable_set = set(trainable_ids)
    agents = {
        aid: AgentEpisodeFacts(
            agent_id=aid,
            team="trainable" if aid in trainable_set else "opponent",
        )
        for aid in agent_ids
    }
    return EvalEpisodeFacts(
        episode=episode,
        steps=0,
        agents=agents,
        trainable_team=tuple(trainable_ids),
        opponent_team=tuple(opponent_ids),
        deferred=set(DEFERRED_COLLISION_PAIR_METRICS),
    )


def update_agent_step_facts(
    episode: EvalEpisodeFacts,
    *,
    step_idx: int,
    infos: Mapping[str, Mapping[str, Any]],
    terminations: Optional[Mapping[str, bool]] = None,
    truncations: Optional[Mapping[str, bool]] = None,
    agent_states: Optional[Mapping[str, Any]] = None,
) -> None:
    episode.steps = max(episode.steps, int(step_idx))
    terminations = terminations or {}
    truncations = truncations or {}
    agent_states = agent_states or {}

    for agent_id, facts in episode.agents.items():
        info = infos.get(agent_id, {}) if isinstance(infos, Mapping) else {}
        if not isinstance(info, Mapping):
            info = {}

        if facts.done_step is not None:
            continue
        facts.active_steps += 1
        facts.final_lap_count = int(info.get("lap_count", facts.final_lap_count))
        terminal_reason = info.get("terminal_reason")
        if terminal_reason:
            facts.terminal_reason = str(terminal_reason)
        if bool(info.get("race_completed", False)) and facts.finish_step is None:
            facts.finish_step = int(info.get("terminal_step", step_idx))
            position = info.get("finish_position")
            facts.finish_position = int(position) if position is not None else None
        if terminal_reason == "collision" and facts.collision_step is None:
            facts.collision_step = int(info.get("terminal_step", step_idx))
        if bool(info.get("time_limit", False)) or bool(truncations.get(agent_id, False)):
            facts.timed_out = True

        done = bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
        if done and facts.done_step is None:
            facts.done_step = int(step_idx)

        centerline = info.get("centerline")
        if isinstance(centerline, Mapping):
            progress = _float_or_none(centerline.get("progress"))
            if progress is not None:
                facts.final_progress = float(np.clip(progress, 0.0, 1.0))
            speed = _float_or_none(centerline.get("vs"))
            if speed is not None:
                facts.speed_samples.append(abs(speed))

        state = agent_states.get(agent_id)
        if state is not None:
            progress = getattr(state, "progress", None)
            if facts.final_progress is None and progress is not None:
                state_progress = _float_or_none(getattr(progress, "progress", None))
                if state_progress is not None:
                    facts.final_progress = float(np.clip(state_progress, 0.0, 1.0))
            if not facts.speed_samples:
                velocity = getattr(state, "velocity", None)
                if velocity is not None:
                    speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float32)))
                    if np.isfinite(speed):
                        facts.speed_samples.append(speed)


def finalize_episode_facts(episode: EvalEpisodeFacts) -> EvalEpisodeFacts:
    for facts in episode.agents.values():
        if facts.terminal_reason == "race_complete" or facts.clean_finish:
            facts.outcome = "finished"
        elif facts.terminal_reason == "collision" or facts.collided:
            facts.outcome = "crashed"
        elif facts.terminal_reason == "time_limit" or facts.timed_out:
            facts.outcome = "truncated"
        else:
            facts.outcome = "incomplete"
    return episode


def aggregate_eval_episodes(
    episodes: Sequence[EvalEpisodeFacts],
    *,
    focal_agent_id: Optional[str] = None,
    opponent_agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    total = len(episodes)
    if total == 0:
        return {"episodes": 0}

    trainable_ids = list(episodes[0].trainable_team)
    opponent_ids = list(episodes[0].opponent_team)
    focal_agent_id = focal_agent_id or (trainable_ids[0] if trainable_ids else None)
    opponent_agent_id = opponent_agent_id or (opponent_ids[0] if opponent_ids else None)

    summary: Dict[str, Any] = {
        "episodes": total,
        "mean_episode_length": _mean(ep.steps for ep in episodes),
        "deferred_metrics": sorted(set().union(*(ep.deferred for ep in episodes))),
    }

    all_agent_ids = list(episodes[0].agents)
    summary["per_agent_rewards_mean"] = {
        aid: _mean(ep.agents[aid].reward_total for ep in episodes if aid in ep.agents)
        for aid in all_agent_ids
    }
    summary["per_agent_individual_rewards_mean"] = {
        aid: _mean(
            ep.agents[aid].individual_reward_total
            for ep in episodes
            if aid in ep.agents
        )
        for aid in all_agent_ids
    }
    summary["per_agent_reward_components_mean"] = _aggregate_reward_components(
        episodes, all_agent_ids
    )
    summary["per_agent_outcomes"] = {
        aid: dict(Counter(ep.agents[aid].outcome for ep in episodes if aid in ep.agents))
        for aid in all_agent_ids
    }
    summary["per_agent_completion_rate"] = {
        aid: _rate(ep.agents[aid].completed for ep in episodes if aid in ep.agents)
        for aid in all_agent_ids
    }
    summary["per_agent_timeout_rate"] = {
        aid: _rate(ep.agents[aid].timed_out for ep in episodes if aid in ep.agents)
        for aid in all_agent_ids
    }

    trainable_rewards = [
        sum(ep.agents[aid].reward_total for aid in trainable_ids if aid in ep.agents)
        for ep in episodes
    ]
    summary["mean_episode_reward"] = _mean(trainable_rewards)
    summary["completion_rate"] = _rate(
        any(ep.agents[aid].completed for aid in trainable_ids if aid in ep.agents)
        for ep in episodes
    )
    summary["timeout_rate"] = _rate(
        any(ep.agents[aid].timed_out for aid in trainable_ids if aid in ep.agents)
        for ep in episodes
    )
    summary["collision_rate"] = _rate(
        any(ep.agents[aid].collided for aid in trainable_ids if aid in ep.agents)
        for ep in episodes
    )
    finish_steps = [
        float(facts.finish_step)
        for ep in episodes
        for aid in trainable_ids
        if (facts := ep.agents.get(aid)) is not None and facts.finish_step is not None
    ]
    # ``None`` distinguishes "no finish" from a genuinely immediate finish and
    # lets checkpoint selection rank incomplete policies without a false speed
    # advantage.
    summary["mean_finish_steps"] = (
        float(np.mean(finish_steps)) if finish_steps else None
    )
    summary["self_crash_rate"] = summary["collision_rate"]
    summary["mean_progress"] = _mean(
        _team_mean_progress(ep, trainable_ids) for ep in episodes
    )
    summary["mean_speed"] = _mean(
        speed
        for ep in episodes
        for aid in trainable_ids
        for speed in ep.agents.get(aid, AgentEpisodeFacts(aid, "trainable")).speed_samples
    )

    if len(trainable_ids) == 1 and len(opponent_ids) == 1 and focal_agent_id and opponent_agent_id:
        summary.update(
            _aggregate_1v1(episodes, focal_agent_id=focal_agent_id, opponent_agent_id=opponent_agent_id)
        )

    if len(trainable_ids) > 1 or len(opponent_ids) > 1:
        summary.update(_aggregate_team(episodes, trainable_ids, opponent_ids))

    return summary


def _aggregate_1v1(
    episodes: Sequence[EvalEpisodeFacts],
    *,
    focal_agent_id: str,
    opponent_agent_id: str,
) -> Dict[str, Any]:
    wins = 0
    finish_ahead = 0
    finish_behind = 0
    opponent_finished = 0

    for ep in episodes:
        focal = ep.agents.get(focal_agent_id)
        opponent = ep.agents.get(opponent_agent_id)
        if focal is None or opponent is None:
            continue

        wins += int(_agent_beats(focal, opponent))
        opponent_finished += int(opponent.completed)
        finish_ahead += int(_finished_ahead(focal, opponent))
        finish_behind += int(_finished_ahead(opponent, focal))

    total = len(episodes)
    return {
        "win_rate": wins / total,
        "target_finish_rate": opponent_finished / total,
        "opponent_finish_rate": opponent_finished / total,
        "finish_ahead_rate": finish_ahead / total,
        "finish_behind_rate": finish_behind / total,
    }


def _aggregate_team(
    episodes: Sequence[EvalEpisodeFacts],
    trainable_ids: Sequence[str],
    opponent_ids: Sequence[str],
) -> Dict[str, Any]:
    team_wins = 0
    strict_wins = 0
    team_collisions = 0

    for ep in episodes:
        trainable = [ep.agents[aid] for aid in trainable_ids if aid in ep.agents]
        opponents = [ep.agents[aid] for aid in opponent_ids if aid in ep.agents]
        if not trainable:
            continue

        team_collisions += int(any(t.collided for t in trainable))
        if not opponents:
            continue

        team_wins += int(any(_agent_beats(t, o) for t in trainable for o in opponents))
        strict_wins += int(all(_agent_beats(t, o) for t in trainable for o in opponents))

    total = len(episodes)
    return {
        "team_win_rate": team_wins / total,
        "strict_team_win_rate": strict_wins / total,
        "best_teammate_progress": _mean(
            max((ep.agents[aid].progress_score for aid in trainable_ids if aid in ep.agents), default=0.0)
            for ep in episodes
        ),
        "mean_team_progress": _mean(_team_mean_progress(ep, trainable_ids) for ep in episodes),
        "opponent_team_progress": _mean(_team_mean_progress(ep, opponent_ids) for ep in episodes),
        "team_collision_rate": team_collisions / total,
        "team_completion_rate": _rate(
            any(ep.agents[aid].completed for aid in trainable_ids if aid in ep.agents)
            for ep in episodes
        ),
        "team_both_finished_rate": _rate(
            all(ep.agents[aid].completed for aid in trainable_ids if aid in ep.agents)
            for ep in episodes
        ),
        "team_dnf_rate": _rate(
            any(not ep.agents[aid].completed for aid in trainable_ids if aid in ep.agents)
            for ep in episodes
        ),
        "team_mean_finish_position": _mean(
            facts.finish_position
            for ep in episodes
            for aid in trainable_ids
            if (facts := ep.agents.get(aid)) is not None and facts.finish_position is not None
        ),
        "team_best_finish_position": _mean(
            min(
                (
                    ep.agents[aid].finish_position
                    for aid in trainable_ids
                    if aid in ep.agents and ep.agents[aid].finish_position is not None
                ),
                default=0,
            )
            for ep in episodes
        ),
    }


def _agent_beats(left: AgentEpisodeFacts, right: AgentEpisodeFacts) -> bool:
    left_rank = _race_rank(left)
    right_rank = _race_rank(right)
    return left_rank > right_rank


def _race_rank(facts: AgentEpisodeFacts) -> tuple[float, float, float]:
    if facts.clean_finish:
        return (2.0, -float(facts.finish_step or 0), facts.progress_score)
    if facts.collided:
        return (0.0, facts.progress_score, 0.0)
    return (1.0, facts.progress_score, 0.0)


def _finished_ahead(left: AgentEpisodeFacts, right: AgentEpisodeFacts) -> bool:
    if not left.completed:
        return False
    if not right.completed:
        return True
    return int(left.finish_step or 0) < int(right.finish_step or 0)


def _team_mean_progress(ep: EvalEpisodeFacts, agent_ids: Sequence[str]) -> float:
    values = [ep.agents[aid].progress_score for aid in agent_ids if aid in ep.agents]
    return _mean(values)


def _aggregate_reward_components(
    episodes: Sequence[EvalEpisodeFacts],
    agent_ids: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for aid in agent_ids:
        by_component: Dict[str, list[float]] = defaultdict(list)
        for ep in episodes:
            facts = ep.agents.get(aid)
            if facts is None:
                continue
            for name, value in facts.reward_components.items():
                by_component[name].append(float(value))
        result[aid] = {
            name: _mean(values)
            for name, values in sorted(by_component.items())
        }
    return result


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _rate(values: Iterable[bool]) -> float:
    vals = list(values)
    return float(sum(bool(v) for v in vals) / len(vals)) if vals else 0.0


def _float_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


__all__ = [
    "AgentEpisodeFacts",
    "EvalEpisodeFacts",
    "DEFERRED_COLLISION_PAIR_METRICS",
    "aggregate_eval_episodes",
    "create_episode_facts",
    "finalize_episode_facts",
    "update_agent_step_facts",
]
