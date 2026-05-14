"""Shared environment data structures.

These types are intentionally lightweight and framework-agnostic so they can be
used by online trainers, offline dataset tooling, and heuristic policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


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
    """Deterministic collection of spawn states for a reset."""

    states: Tuple[SpawnState, ...]
    plan_id: Optional[str] = None


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
