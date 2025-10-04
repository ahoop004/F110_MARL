"""Rollout utilities shared by training and evaluation runners."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from f110x.envs import F110ParallelEnv
from f110x.trainer.base import Trainer, Transition
from f110x.utils.builders import AgentTeam
from f110x.wrappers.reward import RewardWrapper


class TrajectoryBuffer:
    """Accumulates transitions before dispatching them to a trainer."""

    __slots__ = ("trainer", "_items", "_capacity", "off_policy", "updates_per_step")

    def __init__(
        self,
        trainer: Trainer,
        *,
        capacity: int = 64,
        off_policy: bool = False,
        updates_per_step: int = 1,
    ) -> None:
        self.trainer = trainer
        self._items: List[Transition] = []
        self._capacity = max(int(capacity), 1)
        self.off_policy = bool(off_policy)
        self.updates_per_step = max(int(updates_per_step), 1)

    def append(self, item: Transition) -> bool:
        self._items.append(item)
        if len(self._items) >= self._capacity:
            return self.flush()
        return False

    def flush(self) -> bool:
        if not self._items:
            return False
        observe_batch = getattr(self.trainer, "observe_batch", None)
        if callable(observe_batch):
            observe_batch(self._items)
        else:
            for item in self._items:
                self.trainer.observe(item)
        self._items.clear()
        return True


class IdleTerminationTracker:
    """Tracks per-step speeds to decide when to truncate idle episodes."""

    __slots__ = ("threshold", "patience", "counter", "triggered")

    def __init__(self, speed_threshold: float, patience_steps: int) -> None:
        self.threshold = float(speed_threshold)
        self.patience = max(int(patience_steps), 0)
        self.counter = 0
        self.triggered = False

    def reset(self) -> None:
        self.counter = 0
        self.triggered = False

    def observe(self, speed_values: np.ndarray, present_mask: np.ndarray) -> bool:
        if self.patience <= 0:
            self.triggered = False
            return False

        if present_mask.any():
            active = speed_values[present_mask]
            if active.size and np.all(active < self.threshold):
                self.counter += 1
            else:
                self.counter = 0
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self.triggered = True
        else:
            self.triggered = False
        return self.triggered


class BestReturnTracker:
    """Maintains a rolling average of returns and triggers on improvements."""

    __slots__ = ("_values", "best", "save_callback")

    def __init__(self, window: int, save_callback: Optional[Callable[[float], None]] = None) -> None:
        if window <= 0:
            window = 1
        self._values = deque(maxlen=int(window))
        self.best = float("-inf")
        self.save_callback = save_callback

    def observe(self, value: float) -> Optional[float]:
        self._values.append(float(value))
        if len(self._values) < self._values.maxlen:
            return None

        avg_return = float(sum(self._values) / len(self._values))
        if avg_return > self.best:
            self.best = avg_return
            if self.save_callback is not None:
                self.save_callback(avg_return)
            return avg_return
        return None


@dataclass
class EpisodeTraceStep:
    """Single step record captured during an episode rollout."""

    step: int
    observations: Dict[str, Any]
    actions: Dict[str, Any]
    rewards: Dict[str, float]
    next_observations: Dict[str, Any]
    done: Dict[str, bool]
    collisions: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "done": self.done,
            "collisions": list(self.collisions),
        }


@dataclass
class EpisodeRollout:
    """Aggregated metrics for a single environment episode."""

    episode_index: int
    reward_mode: str
    steps: int
    returns: Dict[str, float]
    reward_breakdown: Dict[str, Dict[str, float]]
    terms: Dict[str, bool]
    truncations: Dict[str, bool]
    done_flags: Dict[str, bool]
    collisions: Dict[str, int]
    collision_steps: Dict[str, int]
    idle_triggered: bool
    spawn_points: Dict[str, str]
    spawn_option: Optional[str]
    average_speeds: Dict[str, float]
    causes: List[str] = field(default_factory=list)
    trace: Optional[List[EpisodeTraceStep]] = None

    @property
    def cause(self) -> str:
        return ",".join(self.causes) if self.causes else "unknown"


def build_trajectory_buffers(
    team: AgentTeam,
    trainer_map: Mapping[str, Trainer],
    *,
    off_policy_algorithms: Iterable[str] = ("td3", "sac", "dqn"),
    on_policy_capacity: int = 64,
) -> Tuple[Dict[str, TrajectoryBuffer], Set[str]]:
    """Create trajectory buffers for all trainers in the roster."""

    buffers: Dict[str, TrajectoryBuffer] = {}
    off_policy_ids: Set[str] = set()
    off_policy_lookup = {algo.lower() for algo in off_policy_algorithms}

    for agent_id, trainer in trainer_map.items():
        bundle = team.by_id.get(agent_id)
        algo_name = bundle.algo.lower() if bundle else ""
        is_off_policy = algo_name in off_policy_lookup
        if is_off_policy:
            off_policy_ids.add(agent_id)

        metadata = dict(bundle.metadata.get("config", {})) if bundle else {}
        updates_per_step = int(metadata.get("updates_per_step", metadata.get("gradient_steps", 1) or 1))
        capacity = 1 if is_off_policy else int(metadata.get("on_policy_buffer_size", on_policy_capacity) or on_policy_capacity)

        buffers[agent_id] = TrajectoryBuffer(
            trainer,
            capacity=max(capacity, 1),
            off_policy=is_off_policy,
            updates_per_step=max(updates_per_step, 1),
        )

    return buffers, off_policy_ids


def collect_trajectory(
    trace_steps: Optional[List[EpisodeTraceStep]],
    *,
    transform: Optional[Callable[[EpisodeTraceStep], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return a serialisable view of the captured trajectory."""

    if not trace_steps:
        return []
    if transform is None:
        return [step.as_dict() for step in trace_steps]
    return [transform(step) for step in trace_steps]


def run_episode(
    *,
    env: F110ParallelEnv,
    team: AgentTeam,
    trainer_map: Mapping[str, Trainer],
    trajectory_buffers: Mapping[str, TrajectoryBuffer],
    reward_wrapper_factory: Callable[[int], RewardWrapper],
    compute_actions: Callable[[Dict[str, Any], Dict[str, bool]], Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]]],
    prepare_next_observation: Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    idle_tracker: IdleTerminationTracker,
    episode_index: int,
    reset_fn: Callable[[], Tuple[Dict[str, Any], Dict[str, Any]]],
    agent_ids: Optional[Sequence[str]] = None,
    render_condition: Optional[Callable[[int, int], bool]] = None,
    on_offpolicy_flush: Optional[Callable[[str, Trainer, TrajectoryBuffer], None]] = None,
    trace_buffer: Optional[List[EpisodeTraceStep]] = None,
) -> EpisodeRollout:
    """Execute a single environment episode and gather rollout statistics."""

    obs, infos = reset_fn()
    team.reset_actions()

    agent_order = list(agent_ids) if agent_ids is not None else list(env.possible_agents)
    agent_count = len(agent_order)
    id_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_order)}

    spawn_selection: Dict[str, str] = {}
    spawn_option_id: Optional[str] = None
    for aid, info in infos.items():
        if not isinstance(info, dict):
            continue
        spawn_name = info.get("spawn_point")
        if spawn_name:
            spawn_selection[aid] = str(spawn_name)
        if spawn_option_id is None:
            option_name = info.get("spawn_option")
            if option_name is not None:
                spawn_option_id = str(option_name)

    totals_array = np.zeros(agent_count, dtype=np.float32)
    collision_counts_array = np.zeros(agent_count, dtype=np.int32)
    collision_step_array = np.full(agent_count, -1, dtype=np.int32)
    collision_flags = np.zeros(agent_count, dtype=bool)
    speed_sums_array = np.zeros(agent_count, dtype=np.float32)
    speed_counts_array = np.zeros(agent_count, dtype=np.int32)
    done_flags = np.zeros(agent_count, dtype=bool)

    step_collision_mask = np.zeros(agent_count, dtype=bool)
    step_speed_values = np.zeros(agent_count, dtype=np.float32)
    step_speed_present = np.zeros(agent_count, dtype=bool)
    shaped_rewards = np.zeros(agent_count, dtype=np.float32)

    done_map = {agent_id: False for agent_id in agent_order}
    reward_wrapper = reward_wrapper_factory(episode_index)
    components_extractor = getattr(reward_wrapper, "get_last_components", None)
    reward_breakdown: Dict[str, Dict[str, float]] = {agent_id: {} for agent_id in agent_order}

    idle_tracker.reset()
    steps = 0
    terms: Dict[str, bool] = {}
    truncs: Dict[str, bool] = {}
    causes: List[str] = []

    while True:
        step_collision_mask.fill(False)
        step_speed_present.fill(False)
        step_speed_values.fill(0.0)
        shaped_rewards.fill(0.0)

        actions, processed_obs, controller_infos = compute_actions(obs, done_map)
        if not actions:
            break

        obs_snapshot = {aid: obs.get(aid) for aid in agent_order if aid in obs}
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        steps += 1

        for idx, agent_id in enumerate(agent_order):
            base_reward = rewards.get(agent_id, 0.0)
            if next_obs.get(agent_id) is not None:
                base_reward = reward_wrapper(
                    next_obs,
                    agent_id,
                    rewards.get(agent_id, 0.0),
                    done=terms.get(agent_id, False) or truncs.get(agent_id, False),
                    info=infos.get(agent_id, {}),
                    all_obs=next_obs,
                    step_index=steps,
                )
            totals_array[idx] += float(base_reward)
            shaped_rewards[idx] = float(base_reward)

            if callable(components_extractor):
                try:
                    components = components_extractor(agent_id)
                except TypeError:
                    components = None
                if components:
                    agent_breakdown = reward_breakdown.setdefault(agent_id, {})
                    for name, value in components.items():
                        agent_breakdown[name] = agent_breakdown.get(name, 0.0) + float(value)

        for idx, agent_id in enumerate(agent_order):
            collided = bool(next_obs.get(agent_id, {}).get("collision", False))
            if collided:
                step_collision_mask[idx] = True
                collision_counts_array[idx] += 1
                if collision_step_array[idx] < 0:
                    collision_step_array[idx] = steps
                collision_flags[idx] = True

        for agent_id, trainer in trainer_map.items():
            buffer = trajectory_buffers.get(agent_id)
            if buffer is None:
                continue
            if agent_id not in processed_obs:
                continue

            agent_idx = id_to_index.get(agent_id)
            if agent_idx is None:
                continue

            terminated = terms.get(agent_id, False) or bool(step_collision_mask[agent_idx])
            truncated = truncs.get(agent_id, False)

            patched_next = prepare_next_observation(agent_id, next_obs, infos)
            try:
                next_wrapped = team.observation(agent_id, patched_next)
            except Exception:
                next_wrapped = processed_obs[agent_id]

            shaped = float(shaped_rewards[agent_idx]) if agent_idx is not None else float(rewards.get(agent_id, 0.0))
            info_payload = infos.get(agent_id)
            extra_info = controller_infos.get(agent_id)
            if extra_info:
                merged_info: Dict[str, Any]
                if info_payload is None:
                    merged_info = dict(extra_info)
                else:
                    merged_info = dict(info_payload)
                    merged_info.update(extra_info)
                info_payload = merged_info

            transition = Transition(
                agent_id=agent_id,
                obs=processed_obs[agent_id],
                action=actions.get(agent_id),
                reward=shaped,
                next_obs=next_wrapped,
                terminated=terminated,
                truncated=truncated,
                info=info_payload,
                raw_obs=obs,
            )
            flushed = buffer.append(transition)
            if buffer.off_policy and flushed and on_offpolicy_flush is not None:
                on_offpolicy_flush(agent_id, trainer, buffer)

        for idx, agent_id in enumerate(agent_order):
            done_flag = bool(terms.get(agent_id, False) or truncs.get(agent_id, False) or collision_flags[idx])
            done_flags[idx] = done_flag
            done_map[agent_id] = done_flag

        obs = next_obs

        for idx, agent_id in enumerate(agent_order):
            velocity = next_obs.get(agent_id, {}).get("velocity")
            if velocity is None:
                continue
            speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float32)))
            if np.isnan(speed):
                speed = 0.0
            step_speed_values[idx] = speed
            step_speed_present[idx] = True
            speed_sums_array[idx] += speed
            speed_counts_array[idx] += 1

        if trace_buffer is not None:
            step_collisions = [agent_id for agent_id in agent_order if step_collision_mask[id_to_index[agent_id]]]
            trace_buffer.append(
                EpisodeTraceStep(
                    step=steps,
                    observations=obs_snapshot,
                    actions=dict(actions),
                    rewards={agent_id: float(rewards.get(agent_id, 0.0)) for agent_id in agent_order},
                    next_observations=next_obs,
                    done={agent_id: bool(done_map.get(agent_id, False)) for agent_id in agent_order},
                    collisions=step_collisions,
                )
            )

        if idle_tracker.observe(step_speed_values, step_speed_present):
            for agent_id in env.possible_agents:
                truncs[agent_id] = True
                done_map[agent_id] = True
            causes.append("idle")
            break

        if collision_flags.any():
            if "collision" not in causes:
                causes.append("collision")
        if done_flags.all():
            break

        if render_condition and render_condition(episode_index, steps):
            try:
                env.render()
            except Exception:
                # Disable further renders for this episode by clearing condition
                render_condition = None

    if "collision" not in causes and collision_flags.any():
        causes.append("collision")

    term_flags = {agent_id: bool(terms.get(agent_id, False)) for agent_id in agent_order}
    trunc_flags = {agent_id: bool(truncs.get(agent_id, False)) for agent_id in agent_order}
    done_map = {agent_id: bool(done_flags[id_to_index[agent_id]]) for agent_id in agent_order}

    for agent_id, flagged in term_flags.items():
        if flagged:
            causes.append(f"term:{agent_id}")
    for agent_id, flagged in trunc_flags.items():
        if flagged:
            causes.append(f"trunc:{agent_id}")

    returns = {agent_id: float(totals_array[idx]) for idx, agent_id in enumerate(agent_order)}
    collisions = {agent_id: int(collision_counts_array[idx]) for idx, agent_id in enumerate(agent_order)}
    collision_steps = {agent_id: int(collision_step_array[idx]) for idx, agent_id in enumerate(agent_order)}
    avg_speeds: Dict[str, float] = {}
    for idx, agent_id in enumerate(agent_order):
        count = speed_counts_array[idx]
        avg_speeds[agent_id] = float(speed_sums_array[idx] / count) if count > 0 else 0.0

    reward_breakdown = {
        agent_id: dict(components) if components else {}
        for agent_id, components in reward_breakdown.items()
    }

    rollout = EpisodeRollout(
        episode_index=episode_index,
        reward_mode=str(getattr(reward_wrapper, "mode", "unknown")),
        steps=steps,
        returns=returns,
        reward_breakdown=reward_breakdown,
        terms=term_flags,
        truncations=trunc_flags,
        done_flags=done_map,
        collisions=collisions,
        collision_steps=collision_steps,
        idle_triggered="idle" in causes,
        spawn_points=spawn_selection,
        spawn_option=spawn_option_id,
        average_speeds=avg_speeds,
        causes=list(dict.fromkeys(causes)),
        trace=trace_buffer,
    )
    return rollout


__all__ = [
    "BestReturnTracker",
    "EpisodeRollout",
    "EpisodeTraceStep",
    "IdleTerminationTracker",
    "TrajectoryBuffer",
    "build_trajectory_buffers",
    "collect_trajectory",
    "run_episode",
]
