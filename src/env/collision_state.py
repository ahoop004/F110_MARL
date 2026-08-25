"""Collision state and termination helpers."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.env.collision import build_terminations
from src.env.types import AgentLifecycleRecord, AgentRaceStatus, TerminalReason


def validate_target_laps(value: object) -> int:
    """Validate and return the configured positive integer race distance."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("environment.target_laps must be a positive integer")
    target_laps = int(value)
    if target_laps <= 0:
        raise ValueError("environment.target_laps must be a positive integer")
    return target_laps


class RaceLifecycle:
    """Framework-independent owner of monotonic per-agent race results."""

    def __init__(self, agent_ids: Sequence[str], target_laps: int) -> None:
        self.agent_ids = tuple(str(agent_id) for agent_id in agent_ids)
        self.target_laps = validate_target_laps(target_laps)
        self.records: Dict[str, AgentLifecycleRecord] = {}
        self._next_finish_position = 1
        self.reset()

    def reset(self) -> None:
        self.records = {
            agent_id: AgentLifecycleRecord(
                agent_id=agent_id,
                target_laps=self.target_laps,
            )
            for agent_id in self.agent_ids
        }
        self._next_finish_position = 1

    @property
    def active_agents(self) -> Tuple[str, ...]:
        return tuple(
            agent_id
            for agent_id in self.agent_ids
            if self.records[agent_id].is_active
        )

    @property
    def episode_done(self) -> bool:
        return all(not record.is_active for record in self.records.values())

    def begin_step(self) -> None:
        for record in self.records.values():
            record.lap_crossed = False

    def record_lap_crossing(self, agent_id: str, *, step: int) -> bool:
        """Record one accepted crossing and finish at the configured distance."""
        record = self._record(agent_id)
        if not record.is_active:
            return False
        record.lap_crossed = True
        record.lap_count += 1
        if record.race_completed:
            return self._transition(
                agent_id,
                AgentRaceStatus.FINISHED,
                TerminalReason.RACE_COMPLETE,
                step=step,
                finish_position=self._next_finish_position,
            )
        return False

    def record_collision(self, agent_id: str, *, step: int) -> bool:
        return self._transition(
            agent_id,
            AgentRaceStatus.CRASHED,
            TerminalReason.COLLISION,
            step=step,
        )

    def truncate_active(self, *, step: int) -> Tuple[str, ...]:
        transitioned = []
        for agent_id in self.active_agents:
            if self._transition(
                agent_id,
                AgentRaceStatus.TRUNCATED,
                TerminalReason.TIME_LIMIT,
                step=step,
            ):
                transitioned.append(agent_id)
        return tuple(transitioned)

    def _transition(
        self,
        agent_id: str,
        status: AgentRaceStatus,
        reason: TerminalReason,
        *,
        step: int,
        finish_position: Optional[int] = None,
    ) -> bool:
        record = self._record(agent_id)
        if not record.is_active:
            return False
        record.status = status
        record.terminal_reason = reason
        record.terminal_step = int(step)
        record.finish_position = finish_position
        if status is AgentRaceStatus.FINISHED:
            self._next_finish_position += 1
        return True

    def _record(self, agent_id: str) -> AgentLifecycleRecord:
        try:
            return self.records[str(agent_id)]
        except KeyError as exc:
            raise KeyError(f"unknown agent_id: {agent_id}") from exc


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
    return build_terminations(
        possible_agents,
        collision_flags,
        lap_completion,
        terminate_on_collision,
    )


def normalize_episode_termination_mode(value: object) -> str:
    """Return the canonical joint-episode termination policy name."""
    aliases = {
        "any": "any_agent",
        "any_agent": "any_agent",
        "any_relevant_agent": "any_agent",
        "all": "all_agents",
        "all_agents": "all_agents",
        "individual": "all_agents",
        "all_trainable": "all_trainable",
    }
    mode = aliases.get(str(value or "any_agent").strip().lower())
    if mode is None:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported episode_termination.mode {value!r}; expected one of: {supported}"
        )
    return mode


def apply_episode_termination_policy(
    terminations: Mapping[str, bool],
    truncations: Mapping[str, bool],
    *,
    active_agents: Sequence[str],
    possible_agents: Sequence[str],
    trainable_agents: Sequence[str],
    mode: str,
) -> Tuple[Dict[str, bool], bool]:
    """Apply a scenario-level policy to per-agent terminal events.

    A time limit always ends the joint episode. For ``any_agent``, a terminal
    event from one active agent ends all active agents. The other policies keep
    individual terminal flags until their configured group has finished.
    """
    resolved = normalize_episode_termination_mode(mode)
    result = {aid: bool(terminations.get(aid, False)) for aid in possible_agents}
    active = set(active_agents)
    remaining = {aid for aid in active if not result.get(aid, False)}

    time_limit = any(bool(truncations.get(aid, False)) for aid in active)
    if time_limit:
        return result, True

    if resolved == "any_agent":
        episode_done = any(result.get(aid, False) for aid in active)
        if episode_done:
            for aid in active:
                result[aid] = True
        return result, episode_done

    if resolved == "all_trainable":
        relevant = set(trainable_agents) & active
        episode_done = bool(relevant) and not any(aid in remaining for aid in relevant)
        return result, episode_done

    return result, not remaining


def build_truncations(
    possible_agents: Sequence[str],
    *,
    max_steps: int,
    elapsed_steps: int,
) -> tuple[Dict[str, bool], bool]:
    trunc_flag = (max_steps > 0) and (elapsed_steps + 1 >= max_steps)
    return {agent_id: bool(trunc_flag) for agent_id in possible_agents}, bool(trunc_flag)
