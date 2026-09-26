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

from metrics.race_penalties import POLICY, RacePenaltyEvent, penalty_totals, terminal_penalty_event


DEFERRED_COLLISION_PAIR_METRICS = (
    "teammate_collision_rate",
    "opponent_collision_rate",
    "wall_collision_rate",
)


def team_finish_result(infos: Mapping[str, Any], team_ids: Sequence[str],
                       opponent_ids: Sequence[str]) -> Dict[str, float]:
    """2v2 result from immutable finish facts; unfinished/DNF cars earn no points.

    Rank points are (4-position)/3, averaged over the configured two teammates.
    Thus 1st+4th and 2nd+3rd tie. Clean finishers rank ahead of DNFs; no finish
    produces neither a first-place win nor a sweep. Later contact with a parked
    finisher does not undo its authoritative race-complete terminal cause.
    """
    ids = [*team_ids, *opponent_ids]
    if len(team_ids) != 2 or len(opponent_ids) != 2 or len(set(ids)) != 4:
        raise ValueError("Team race results require two teammates and two opponents")
    if any(aid not in infos for aid in ids):
        raise ValueError("Team race results require lifecycle info for all four cars")
    positions = []
    for aid in team_ids:
        info = infos[aid]
        if info.get("terminal_reason") != "race_complete":
            continue
        position = info.get("finish_position")
        if position not in (1, 2, 3, 4):
            raise ValueError("Completed racers require a finish position from 1 to 4")
        positions.append(position)
    return {
        "both_finished": float(len(positions) == 2),
        "rank_score": sum((4 - p) / 3 for p in positions) / 2,
        "first_place": float(1 in positions),
        "sweep": float(sorted(positions) == [1, 2]),
    }


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
    finish_elapsed_steps: Optional[int] = None
    collision_step: Optional[int] = None
    timed_out: bool = False
    final_progress: Optional[float] = None
    speed_samples: list[float] = field(default_factory=list)
    outcome: str = "unknown"
    terminal_reason: Optional[str] = None
    finish_position: Optional[int] = None
    penalty_events: list[RacePenaltyEvent] = field(default_factory=list)
    final_lap_count: int = 0
    net_progress: float = 0.0
    progress_delta_samples: int = 0
    lap_times_steps: list[int] = field(default_factory=list)
    valid_lap_times_steps: list[int] = field(default_factory=list)
    lap_offtrack_sums: list[float] = field(default_factory=list)
    lap_boundary_violations: list[bool] = field(default_factory=list)
    _lap_started: bool = False
    _lap_offtrack_sum: float = 0.0
    _lap_exceeded: bool = False

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
    collect_speed: bool = True,
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
        limits = info.get("track_limits")
        if isinstance(limits, Mapping):
            if facts._lap_started:
                facts._lap_offtrack_sum += float(limits["offtrack_distance"])
                facts._lap_exceeded |= bool(limits["exceeded"])
            if info.get("lap_crossed") and info.get("lap_time_steps") is not None:
                duration = int(info["lap_time_steps"])
                facts.lap_times_steps.append(duration)
                facts.lap_offtrack_sums.append(facts._lap_offtrack_sum)
                facts.lap_boundary_violations.append(facts._lap_exceeded)
                if not facts._lap_exceeded:
                    facts.valid_lap_times_steps.append(duration)
                facts._lap_offtrack_sum = 0.0
                facts._lap_exceeded = False
            if info.get("lap_start_step") is not None:
                facts._lap_started = True
        facts.final_lap_count = int(info.get("lap_count", facts.final_lap_count))
        penalty = terminal_penalty_event(agent_id, info)
        if penalty is not None and penalty not in facts.penalty_events:
            facts.penalty_events.append(penalty)
        terminal_reason = info.get("terminal_reason")
        if terminal_reason:
            facts.terminal_reason = str(terminal_reason)
        if bool(info.get("race_completed", False)) and facts.finish_step is None:
            facts.finish_step = int(info.get("terminal_step", step_idx))
            # Callers count completed physics steps; terminal_step may be zero-based.
            facts.finish_elapsed_steps = int(step_idx)
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
            delta = _float_or_none(centerline.get("progress_delta"))
            if delta is not None and np.isfinite(delta):
                # Sum signed, seam-corrected lap fractions. Absolute position
                # rewards spawn placement; positive-only sums reward oscillation.
                facts.net_progress += delta
                facts.progress_delta_samples += 1
            progress = _float_or_none(centerline.get("progress"))
            if progress is not None:
                facts.final_progress = float(np.clip(progress, 0.0, 1.0))
            speed = _float_or_none(centerline.get("vs"))
            if speed is not None and collect_speed:
                facts.speed_samples.append(abs(speed))

        state = agent_states.get(agent_id)
        if state is not None:
            progress = getattr(state, "progress", None)
            if facts.final_progress is None and progress is not None:
                state_progress = _float_or_none(getattr(progress, "progress", None))
                if state_progress is not None:
                    facts.final_progress = float(np.clip(state_progress, 0.0, 1.0))
            if not facts.speed_samples and collect_speed:
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


def capture_spawn_context(env, agent_ids) -> Dict[str, Any]:
    """Capture actual reset poses even when the spawn mode has no named metadata."""
    manager = getattr(env, "_spawn_manager", None)
    context = dict(getattr(manager, "last_spawn_metadata", {}) or {})
    context["spawn_ids"] = dict(getattr(manager, "last_spawn_mapping", {}) or {})
    context["initial_states"] = {}
    if hasattr(env, "get_agent_state"):
        for aid in agent_ids:
            try:
                state = env.get_agent_state(aid)
            except KeyError:
                context["initial_states"][aid] = None
                continue
            context["initial_states"][aid] = {
                field: np.asarray(getattr(state, field)).tolist()
                for field in ("pose", "velocity") if getattr(state, field, None) is not None}
    return context


def episode_race_record(episode: EvalEpisodeFacts, *, timestep: float,
                        finite_race: bool = True, include_rewards: bool = True) -> Dict[str, Any]:
    """Small shared training/evaluation record; no trajectory arrays or inferred blame.

    Progress is signed earned lap fractions. Continuous training has no finish
    objective; its finish-derived fields are unavailable, rather than failures.
    """
    agents = {}
    for aid, f in episode.agents.items():
        valid = [steps * timestep for steps in f.valid_lap_times_steps]
        agents[aid] = {
            "team": f.team, "finished": f.completed if finite_race else None,
            "finish_position": f.finish_position if finite_race else None,
            "terminal_reason": f.terminal_reason, "outcome": f.outcome,
            "collision_dnf": f.terminal_reason == "collision",
            "boundary_dnf": f.terminal_reason == "track_boundary",
            "timeout": f.timed_out, "laps": f.final_lap_count,
            "net_progress_laps": f.net_progress if f.progress_delta_samples else None,
            "active_time_s": f.active_steps * timestep,
            "clean_finish_time_s": (f.finish_elapsed_steps * timestep
                if finite_race and f.clean_finish and f.finish_elapsed_steps is not None else None),
            "valid_lap_count": len(valid), "measured_lap_count": len(f.lap_times_steps),
            "mean_valid_lap_time_s": float(np.mean(valid)) if valid else None,
        }
        if include_rewards and aid in episode.trainable_team:
            agents[aid].update(reward=f.reward_total, individual_reward=f.individual_reward_total,
                               reward_components=dict(f.reward_components))
    own = [agents[aid] for aid in episode.trainable_team]
    others = [agents[aid] for aid in episode.opponent_team]
    finishes = [a["clean_finish_time_s"] for a in own if a["clean_finish_time_s"] is not None]
    progress = [a["net_progress_laps"] for a in own]
    record = {
        "metric_contract": "race_facts_v1", "phase": "training",
        "race_mode": "finite" if finite_race else "continuous",
        "physics_steps": episode.steps, "duration_s": episode.steps * timestep,
        "timestep_s": timestep,
        "at_least_one_finished": any(a["finished"] for a in own) if finite_race else None,
        "both_finished": all(a["finished"] for a in own) if finite_race else None,
        "first_place": None, "sweep": None, "rank_score": None,
        "any_learner_collision_dnf": any(a["collision_dnf"] for a in own),
        "mean_net_progress_laps": float(np.mean(progress)) if progress and all(p is not None for p in progress) else None,
        "mean_learner_laps": float(np.mean([a["laps"] for a in own])) if own else None,
        "clean_finish_count": len(finishes),
        "mean_clean_finish_time_s": float(np.mean(finishes)) if finishes else None,
        "agents": agents,
    }
    for prefix, rows in (("own", own), ("opponent", others)):
        for kind in ("collision_dnf", "boundary_dnf", "timeout"):
            record[f"{prefix}_{kind}_count"] = sum(a[kind] for a in rows)
    if len(own) == len(others) == 2:
        if finite_race:
            record.update(team_finish_result(
                {aid: {"terminal_reason": f.terminal_reason, "finish_position": f.finish_position}
                 for aid, f in episode.agents.items()}, episode.trainable_team, episode.opponent_team))
        totals = penalty_totals([e for f in episode.agents.values() for e in f.penalty_events],
                                episode.trainable_team, episode.opponent_team)
        record.update({f"penalty_{key}": value for key, value in totals.items()})
    return record


def aggregate_eval_episodes(
    episodes: Sequence[EvalEpisodeFacts],
    *,
    focal_agent_id: Optional[str] = None,
    opponent_agent_id: Optional[str] = None,
    timestep: Optional[float] = None,
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
    # The first learner is the progressing car in asymmetric support races.
    if focal_agent_id:
        focal = [ep.agents[focal_agent_id] for ep in episodes]
        summary["focal_agent_id"] = focal_agent_id
        summary["focal_completion_rate"] = _rate(f.completed for f in focal)
        summary["focal_opponent_win_rate"] = _rate(
            ep.agents[focal_agent_id].completed and all(
                _agent_beats(ep.agents[focal_agent_id], ep.agents[aid]) for aid in opponent_ids)
            for ep in episodes)
        summary["focal_mean_net_progress"] = _mean(f.net_progress for f in focal)
        times = [f.finish_elapsed_steps * timestep for f in focal
                 if timestep is not None and f.clean_finish and f.finish_elapsed_steps is not None]
        summary["focal_mean_clean_finish_time_s"] = _mean(times) if times else None
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
    clean_finishes = [
        facts for ep in episodes for aid in trainable_ids
        if (facts := ep.agents.get(aid)) is not None and facts.clean_finish
    ]
    summary["clean_finish_count"] = len(clean_finishes)
    if timestep is not None:
        finish_times = [
            facts.finish_elapsed_steps * timestep for facts in clean_finishes
            if facts.finish_elapsed_steps is not None
        ]
        summary["finish_time_sample_count"] = len(finish_times)
        summary["mean_clean_finish_time_s"] = float(np.mean(finish_times)) if finish_times else None
    summary["self_crash_rate"] = summary["collision_rate"]
    progress_facts = [ep.agents[aid] for ep in episodes for aid in trainable_ids if aid in ep.agents]
    if timestep is not None:
        laps = [v * timestep for f in progress_facts for v in f.lap_times_steps]
        valid = [v * timestep for f in progress_facts for v in f.valid_lap_times_steps]
        offtrack = [v * timestep for f in progress_facts for v in f.lap_offtrack_sums]
        violations = [v for f in progress_facts for v in f.lap_boundary_violations]
        summary.update({
            "measured_laps": len(laps),
            "valid_laps": len(valid),
            "valid_lap_time_sample_count": len(valid),
            "mean_valid_lap_time_s": float(np.mean(valid)) if valid else None,
            "fastest_valid_lap_s": min(valid) if valid else None,
            "mean_lap_time_s": float(np.mean(laps)) if laps else None,
            "std_lap_time_s": float(np.std(laps)) if laps else None,
            "offtrack_error_m_s_per_lap": float(np.mean(offtrack)) if offtrack else None,
            "boundary_violation_lap_rate": float(np.mean(violations)) if violations else None,
        })
    summary["mean_net_progress"] = (
        _mean(facts.net_progress for facts in progress_facts)
        if progress_facts and all(facts.progress_delta_samples for facts in progress_facts)
        else None
    )
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
    if len(trainable_ids) == 2 and len(opponent_ids) == 2:
        results = [team_finish_result(
            {aid: {"terminal_reason": facts.terminal_reason,
                   "finish_position": facts.finish_position}
             for aid, facts in ep.agents.items()}, trainable_ids, opponent_ids,
        ) for ep in episodes]
        for key in ("rank_score", "first_place", "sweep"):
            summary[f"team_{key}"] = _mean(result[key] for result in results)
        episode_events = [[event for facts in ep.agents.values() for event in facts.penalty_events]
                          for ep in episodes]
        totals = [penalty_totals(events, trainable_ids, opponent_ids) for events in episode_events]
        summary["race_penalty_policy"] = POLICY
        summary["mean_own_penalty_points"] = _mean(row["own_points"] for row in totals)
        summary["mean_opponent_penalty_points"] = _mean(row["opponent_points"] for row in totals)
        summary["mean_team_penalty_score"] = _mean(row["own_score"] + row["opponent_score"] for row in totals)
        summary["team_rank_penalty_score"] = summary["team_rank_score"] + summary["mean_team_penalty_score"]
        summary["race_penalty_events"] = [
            {"episode": ep.episode, **event.to_dict()}
            for ep, events in zip(episodes, episode_events) for event in events
        ]

    # Explicit denominators supplement the legacy keys used by selection.
    summary["race_count"] = total
    summary["at_least_one_finished_count"] = sum(
        any(ep.agents[aid].completed for aid in trainable_ids) for ep in episodes)
    summary["both_finished_count"] = sum(
        all(ep.agents[aid].completed for aid in trainable_ids) for ep in episodes)
    summary["any_learner_collision_dnf_count"] = sum(
        any(ep.agents[aid].terminal_reason == "collision" for aid in trainable_ids) for ep in episodes)
    if len(trainable_ids) == 2 and len(opponent_ids) == 2:
        for key in ("first_place", "sweep"):
            summary[f"{key}_count"] = sum(int(row[key]) for row in results)
    summary["per_car"] = {}
    for aid in all_agent_ids:
        facts = [ep.agents[aid] for ep in episodes if aid in ep.agents]
        times = [f.finish_elapsed_steps * timestep for f in facts
                 if timestep is not None and f.clean_finish and f.finish_elapsed_steps is not None]
        summary["per_car"][aid] = {
            "race_count": len(facts), "finished_count": sum(f.completed for f in facts),
            "collision_dnf_count": sum(f.terminal_reason == "collision" for f in facts),
            "boundary_dnf_count": sum(f.terminal_reason == "track_boundary" for f in facts),
            "timeout_count": sum(f.timed_out for f in facts),
            "clean_finish_count": sum(f.clean_finish for f in facts),
            "finish_time_sample_count": len(times),
            "mean_clean_finish_time_s": float(np.mean(times)) if times else None,
        }
    for prefix, ids in (("own", trainable_ids), ("opponent", opponent_ids)):
        for kind in ("collision_dnf", "boundary_dnf", "timeout"):
            summary[f"{prefix}_{kind}_count"] = sum(summary["per_car"][aid][f"{kind}_count"] for aid in ids)

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
