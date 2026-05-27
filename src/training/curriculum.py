"""Curriculum learning support — phase-gated spawn selection.

A :class:`CurriculumManager` holds an ordered list of :class:`CurriculumPhase`
objects.  After each episode it tracks whether the agent succeeded (based on
:class:`~metrics.outcomes.EpisodeOutcome`) in a rolling window.  When the
rolling success-rate exceeds the phase's ``success_threshold`` the manager
advances to the next phase automatically.

Each phase carries a pool of **named spawn points** (strings that must exist in
the map YAML ``annotations.spawn_points`` section).  Before every episode the
trainer calls :meth:`CurriculumManager.next_spawn_plan` which returns a
:class:`~env.types.SpawnPlan` built from the current phase's pool.  The env
applies it at highest priority (overriding random / centerline spawning).

Usage::

    phases = [
        CurriculumPhase("easy", spawn_names=["straight_a", "straight_b"],
                         success_threshold=0.6, window_size=50),
        CurriculumPhase("hard", spawn_names=["straight_a", "corner_a", "corner_b"],
                         success_threshold=0.75, window_size=100),
    ]
    curriculum = CurriculumManager(phases)

    # — wired into the trainer via spawn_plan_fn:
    def spawn_plan_fn():
        pts = env._spawn_manager.spawn_points
        return curriculum.next_spawn_plan(pts, agent_ids)

    trainer = OnPolicyTrainer(..., spawn_plan_fn=spawn_plan_fn,
                              hooks=[..., CurriculumHook(curriculum)])
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

import numpy as np

from env.types import SpawnPlan, SpawnState

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumPhase:
    """One curriculum stage with a named spawn-point pool and advancement gate.

    Parameters
    ----------
    name:
        Human-readable label (used in logs / metadata).
    spawn_names:
        Pool of named spawn points for this phase.  Names must appear in the
        map's ``annotations.spawn_points`` YAML section.  An empty list means
        "no override — use env's normal spawn policy".
    success_threshold:
        Rolling success rate (0.0–1.0) required to advance to the next phase.
        Ignored for the final phase.
    window_size:
        Number of most-recent episodes used to compute the rolling success rate.
    """

    name: str
    spawn_names: Sequence[str] = field(default_factory=list)
    success_threshold: float = 0.7
    window_size: int = 50

    def __post_init__(self) -> None:
        if not (0.0 <= self.success_threshold <= 1.0):
            raise ValueError(
                f"CurriculumPhase '{self.name}': success_threshold must be in "
                f"[0, 1], got {self.success_threshold}"
            )
        if self.window_size < 1:
            raise ValueError(
                f"CurriculumPhase '{self.name}': window_size must be >= 1, "
                f"got {self.window_size}"
            )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class CurriculumManager:
    """Tracks curriculum progress and produces :class:`~env.types.SpawnPlan` objects.

    Parameters
    ----------
    phases:
        Ordered list of phases, from easiest to hardest.  Must be non-empty.
    start_phase:
        Index of the phase to start in (default 0).
    """

    def __init__(
        self,
        phases: Sequence[CurriculumPhase],
        start_phase: int = 0,
    ) -> None:
        if not phases:
            raise ValueError("CurriculumManager requires at least one phase.")
        self._phases: List[CurriculumPhase] = list(phases)
        self._phase_idx: int = max(0, min(int(start_phase), len(self._phases) - 1))
        self._window: Deque[bool] = deque(maxlen=self._phases[self._phase_idx].window_size)
        self._episode_count: int = 0   # total episodes seen (never resets)
        self._phase_episodes: int = 0  # episodes in current phase (resets on advance)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def on_episode_end(self, outcome: Any) -> bool:
        """Record episode result and possibly advance phase.

        Parameters
        ----------
        outcome:
            :class:`~metrics.outcomes.EpisodeOutcome` instance **or** its
            ``.value`` string.  Any value whose string form is ``"target_crash"``
            counts as success.  All other values count as failure.

        Returns
        -------
        bool
            ``True`` if the curriculum advanced to the next phase this call.
        """
        # Normalise to string so callers can pass the enum or its .value
        outcome_str = str(getattr(outcome, "value", outcome)).lower()
        success = outcome_str == "target_crash"

        self._window.append(success)
        self._episode_count += 1
        self._phase_episodes += 1

        if self.is_final_phase:
            return False

        phase = self._phases[self._phase_idx]
        if len(self._window) >= phase.window_size and self.success_rate >= phase.success_threshold:
            self._advance()
            return True
        return False

    # ------------------------------------------------------------------
    # Spawn plan production
    # ------------------------------------------------------------------

    def next_spawn_plan(
        self,
        spawn_points: Mapping[str, np.ndarray],
        agent_ids: Sequence[str],
    ) -> Optional[SpawnPlan]:
        """Build a :class:`~env.types.SpawnPlan` for the current phase.

        Returns ``None`` when the current phase has no ``spawn_names`` or none
        of the named points exist in *spawn_points* (so the env falls back to
        its normal spawning).

        Each agent is assigned the spawn point at index
        ``(episode_count + agent_index) % len(valid_names)`` for variety while
        keeping spawning deterministic given the same curriculum state.

        Parameters
        ----------
        spawn_points:
            Named spawn point map from the env's :class:`~env.spawn_manager.SpawnManager`
            (``env._spawn_manager.spawn_points``).
        agent_ids:
            Ordered list of agent IDs to include in the plan.
        """
        phase = self._phases[self._phase_idx]
        valid_names = [n for n in phase.spawn_names if n in spawn_points]
        if not valid_names:
            return None

        states: List[SpawnState] = []
        for j, agent_id in enumerate(agent_ids):
            name = valid_names[(self._episode_count + j) % len(valid_names)]
            raw = np.asarray(spawn_points[name], dtype=np.float32).reshape(-1)
            pose = np.zeros(3, dtype=np.float32)
            pose[: min(raw.shape[0], 3)] = raw[: min(raw.shape[0], 3)]
            states.append(SpawnState(agent_id=agent_id, pose=pose, spawn_id=name))

        return SpawnPlan(
            states=tuple(states),
            plan_id=f"{phase.name}_ep{self._episode_count}",
            map_id=None,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> CurriculumPhase:
        """The currently active :class:`CurriculumPhase`."""
        return self._phases[self._phase_idx]

    @property
    def phase_index(self) -> int:
        """Zero-based index of the current phase."""
        return self._phase_idx

    @property
    def success_rate(self) -> float:
        """Rolling success rate in [0.0, 1.0].  Zero when window is empty."""
        if not self._window:
            return 0.0
        return float(sum(self._window)) / float(len(self._window))

    @property
    def episode_count(self) -> int:
        """Total number of episodes processed by this manager."""
        return self._episode_count

    @property
    def phase_episodes(self) -> int:
        """Number of episodes processed in the current phase."""
        return self._phase_episodes

    @property
    def is_final_phase(self) -> bool:
        """``True`` when no further phases remain."""
        return self._phase_idx >= len(self._phases) - 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _advance(self) -> None:
        """Move to the next phase and reset the rolling window."""
        old = self._phases[self._phase_idx].name
        self._phase_idx += 1
        new_phase = self._phases[self._phase_idx]
        self._window = deque(maxlen=new_phase.window_size)
        self._phase_episodes = 0
        _log.info(
            "Curriculum: advanced from '%s' → '%s' (episode %d, rate %.2f)",
            old,
            new_phase.name,
            self._episode_count,
            self.success_rate,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a serialisable status dict for logging / W&B."""
        return {
            "curriculum/phase_index": self._phase_idx,
            "curriculum/phase_name": self.current_phase.name,
            "curriculum/success_rate": round(self.success_rate, 4),
            "curriculum/phase_episodes": self._phase_episodes,
            "curriculum/total_episodes": self._episode_count,
            "curriculum/is_final": self.is_final_phase,
        }
