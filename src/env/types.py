"""Shared environment data structures.

These types are intentionally lightweight and framework-agnostic so they can be
used by online trainers, offline dataset tooling, and heuristic policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


class AgentRaceStatus(str, Enum):
    """Monotonic per-agent race lifecycle states."""

    ACTIVE = "active"
    FINISHED = "finished"
    CRASHED = "crashed"
    TRUNCATED = "truncated"


class TerminalReason(str, Enum):
    """Immutable cause for an agent leaving the active decision set."""

    RACE_COMPLETE = "race_complete"
    COLLISION = "collision"
    TIME_LIMIT = "time_limit"


@dataclass
class AgentLifecycleRecord:
    """Authoritative mutable race result for one physical agent.

    ``status``, ``terminal_reason``, ``terminal_step``, and ``finish_position``
    become immutable once the agent leaves :attr:`AgentRaceStatus.ACTIVE`.
    ``lap_crossed`` is a one-step event and is cleared at the beginning of the
    next lifecycle update.
    """

    agent_id: str
    target_laps: int
    status: AgentRaceStatus = AgentRaceStatus.ACTIVE
    terminal_reason: Optional[TerminalReason] = None
    terminal_step: Optional[int] = None
    finish_position: Optional[int] = None
    lap_crossed: bool = False
    lap_count: int = 0

    @property
    def race_completed(self) -> bool:
        return self.lap_count >= self.target_laps

    @property
    def is_active(self) -> bool:
        return self.status is AgentRaceStatus.ACTIVE


@dataclass(frozen=True)
class MapRuntimeConfig:
    """Resolved map paths and metadata for one active map."""

    map_dir: Path
    map_ext: str
    map_name: Optional[str]
    map_yaml: Optional[str]
    map_path: Path
    yaml_path: Path
    metadata: Dict[str, Any]
    image_path: Path
    image_size: Tuple[int, int]


@dataclass(frozen=True)
class SpawnState:
    """Resolved start pose metadata for one agent."""

    agent_id: str
    pose: np.ndarray
    spawn_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpawnPlan:
    """Deterministic collection of spawn states for a reset.

    Extended with dataset-level metadata so that recorded rollouts can be
    replayed exactly: same map, same vehicle physics, same spawn poses.

    Parameters
    ----------
    states:
        Per-agent :class:`SpawnState` tuples with resolved poses and optional
        named spawn IDs.
    plan_id:
        Optional human-readable identifier for this plan (e.g. ``"eval_0"``).
    map_id:
        Map name/bundle active when this plan was created.  Stored in dataset
        metadata so offline consumers know which map was used.
    vehicle_params:
        Snapshot of vehicle physics parameters (``v_max``, ``a_max``, …) at
        the time of recording.  Allows exact physical replay.
    scenario_hash:
        SHA-256 (hex, first 16 chars) of the scenario YAML used for the run.
        Links dataset files back to the configuration that produced them.
    """

    states: Tuple[SpawnState, ...]
    plan_id: Optional[str] = None
    map_id: Optional[str] = None
    vehicle_params: Optional[Dict[str, float]] = None
    scenario_hash: Optional[str] = None


@dataclass
class ProgressState:
    """Factual centerline/lap progress for one agent."""

    lap_count: int = 0
    lap_time: float = 0.0
    progress: float = 0.0
    progress_delta: float = 0.0
    wrong_way: bool = False
    finished: bool = False


@dataclass(frozen=True)
class AgentState:
    """Public per-agent state view for diagnostics and datasets."""

    agent_id: str
    pose: np.ndarray
    velocity: np.ndarray
    angular_velocity: float
    collision: bool
    progress: Optional[ProgressState] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalState:
    """Ordered multi-agent state view for centralized critics and datasets."""

    agent_ids: Tuple[str, ...]
    vector: np.ndarray
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepFacts:
    """Factual step outputs before reward or observation shaping."""

    agent_states: Mapping[str, AgentState]
    global_state: GlobalState
    collisions: Mapping[str, bool]
    terminations: Mapping[str, bool]
    truncations: Mapping[str, bool]
    info: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class TransitionRecord:
    """One agent-decision record for offline RL and dataset logging.

    Fields are aligned with standard offline RL dataset conventions (d3rlpy,
    D4RL, etc.) while carrying F110-specific metadata for deterministic replay.

    All array fields are **float32** numpy arrays.  String fields use plain
    Python ``str``; ``reward_components`` may be an empty dict if the trainer
    does not track component breakdowns.

    Parameters
    ----------
    obs:
        Composed, flattened observation seen by the agent (output of
        :class:`~wrappers.observations.composer.ObservationComposer`).
    action_norm:
        Normalized action in ``[-1, 1]`` produced by the agent.
    action_phys:
        Physical (denormalized + constrained) action sent to the env.
    reward:
        Total scalar reward for this decision (sum across action_repeat
        sub-steps if action_repeat > 1).
    reward_components:
        Per-component breakdown ``{name: value}`` from
        :class:`~wrappers.rewards.composer.RewardComposer`.  Empty dict when
        not tracked.
    next_obs:
        Composed observation after the action (after all action_repeat steps).
    terminated:
        True when the env signals a terminal event (collision, task success).
    truncated:
        True when the episode ends due to a time limit.
    info:
        Raw step info dict from the env (contains facts: collision, lap count,
        progress delta, etc.).
    global_state:
        Global state vector from ``env.get_global_state().vector`` at the time
        of the decision.  Zero-length array if global state is unavailable.
    map_id:
        Map name/bundle active for this episode (``env.map_name``).
    spawn_id:
        Named spawn point used for the focal agent in this episode.  ``None``
        if spawn was random or unnamed.
    episode_id:
        Unique string identifier for this episode (``run_id + "_ep{N}"``).
    step_idx:
        Zero-based index of this decision within the current episode.
    agent_id:
        ID of the agent that produced this transition (e.g. ``"car_0"``).
    """

    obs: np.ndarray
    action_norm: np.ndarray
    action_phys: np.ndarray
    reward: float
    reward_components: Dict[str, float]
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    global_state: np.ndarray
    map_id: Optional[str]
    spawn_id: Optional[str]
    episode_id: str
    step_idx: int
    agent_id: str
    lap_crossed: bool = False
    lap_count: int = 0
    target_laps: int = 1
    race_completed: bool = False
    terminal_reason: Optional[str] = None
    lifecycle_status: str = "active"
    finish_position: Optional[int] = None
    lifecycle_masks: Dict[str, np.ndarray] = field(default_factory=dict)
