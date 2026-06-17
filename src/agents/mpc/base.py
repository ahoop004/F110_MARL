"""Shared MPC utility scaffolding.

This module provides deterministic candidate generation and scoring helpers.
It does not register any controller with ``AgentFactory``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agents.mpc.costs import CostWeights, trajectory_cost
from agents.mpc.rollout import DEFAULT_DT, DEFAULT_WHEELBASE, rollout_kinematic_bicycle


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 10
    dt: float = DEFAULT_DT
    wheelbase: float = DEFAULT_WHEELBASE
    steering_min: float = -0.4189
    steering_max: float = 0.4189
    speed_min: float = 0.0
    speed_max: float = 3.0
    steering_samples: int = 5
    speed_samples: int = 5
    max_candidate_sequences: int = 256


@dataclass(frozen=True)
class MPCPlanResult:
    action_sequence: np.ndarray
    trajectory: np.ndarray
    cost: float
    candidate_count: int

    @property
    def first_action(self) -> np.ndarray:
        if self.action_sequence.shape[0] == 0:
            return np.zeros(2, dtype=np.float32)
        return self.action_sequence[0].astype(np.float32, copy=True)


def make_action_grid(config: MPCConfig) -> np.ndarray:
    """Create a deterministic grid of ``[steering, speed]`` actions."""
    steering_count = max(1, int(config.steering_samples))
    speed_count = max(1, int(config.speed_samples))
    steering = np.linspace(
        float(config.steering_min),
        float(config.steering_max),
        steering_count,
        dtype=np.float32,
    )
    speed = np.linspace(
        float(config.speed_min),
        float(config.speed_max),
        speed_count,
        dtype=np.float32,
    )
    return np.array([[s, v] for s in steering for v in speed], dtype=np.float32)


def generate_action_sequences(
    config: MPCConfig,
    *,
    action_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate bounded, deterministic candidate action sequences.

    The helper starts with constant-action sequences, then adds simple
    one-switch sequences when the candidate budget allows.  This gives small
    horizons a useful deterministic search set without building the full
    combinatorial grid.
    """
    horizon = max(1, int(config.horizon))
    max_sequences = max(1, int(config.max_candidate_sequences))
    grid = _normalize_action_grid(action_grid if action_grid is not None else make_action_grid(config))
    if grid.shape[0] == 0:
        grid = np.zeros((1, 2), dtype=np.float32)

    sequences: list[np.ndarray] = []
    for action in grid:
        sequences.append(np.repeat(action.reshape(1, 2), horizon, axis=0))
        if len(sequences) >= max_sequences:
            return np.stack(sequences).astype(np.float32)

    switch_step = max(1, horizon // 2)
    for first in grid:
        for second in grid:
            if np.array_equal(first, second):
                continue
            seq = np.empty((horizon, 2), dtype=np.float32)
            seq[:switch_step] = first
            seq[switch_step:] = second
            sequences.append(seq)
            if len(sequences) >= max_sequences:
                return np.stack(sequences).astype(np.float32)

    return np.stack(sequences).astype(np.float32)


def evaluate_action_sequences(
    pose: Optional[np.ndarray],
    action_sequences: Optional[np.ndarray],
    *,
    centerline: Optional[np.ndarray] = None,
    target_speed: Optional[float] = None,
    config: MPCConfig = MPCConfig(),
    weights: CostWeights = CostWeights(),
) -> MPCPlanResult:
    """Roll out and score candidate sequences, returning the best plan."""
    sequences = _normalize_sequences(action_sequences, max(1, int(config.horizon)))
    best_cost = float("inf")
    best_sequence = sequences[0]
    best_trajectory = rollout_kinematic_bicycle(
        pose,
        best_sequence,
        dt=config.dt,
        wheelbase=config.wheelbase,
        horizon=best_sequence.shape[0],
    )

    for sequence in sequences:
        trajectory = rollout_kinematic_bicycle(
            pose,
            sequence,
            dt=config.dt,
            wheelbase=config.wheelbase,
            horizon=sequence.shape[0],
        )
        cost = trajectory_cost(
            trajectory,
            sequence,
            centerline=centerline,
            target_speed=target_speed,
            weights=weights,
        )
        if cost < best_cost:
            best_cost = cost
            best_sequence = sequence
            best_trajectory = trajectory

    return MPCPlanResult(
        action_sequence=best_sequence.astype(np.float32, copy=True),
        trajectory=best_trajectory.astype(np.float32, copy=True),
        cost=float(best_cost),
        candidate_count=int(sequences.shape[0]),
    )


def plan_grid_search(
    pose: Optional[np.ndarray],
    *,
    centerline: Optional[np.ndarray] = None,
    target_speed: Optional[float] = None,
    config: MPCConfig = MPCConfig(),
    weights: CostWeights = CostWeights(),
) -> MPCPlanResult:
    """Generate candidate sequences and return the lowest-cost plan."""
    sequences = generate_action_sequences(config)
    return evaluate_action_sequences(
        pose,
        sequences,
        centerline=centerline,
        target_speed=target_speed,
        config=config,
        weights=weights,
    )


def _normalize_action_grid(action_grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(action_grid, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    result = np.zeros((arr.shape[0], 2), dtype=np.float32)
    cols = min(2, arr.shape[1])
    result[:, :cols] = arr[:, :cols]
    result[~np.isfinite(result)] = 0.0
    return result


def _normalize_sequences(action_sequences: Optional[np.ndarray], horizon: int) -> np.ndarray:
    horizon = max(1, int(horizon))
    if action_sequences is None:
        return np.zeros((1, horizon, 2), dtype=np.float32)
    arr = np.asarray(action_sequences, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((1, horizon, 2), dtype=np.float32)

    result = np.zeros((arr.shape[0], horizon, 2), dtype=np.float32)
    steps = min(horizon, arr.shape[1])
    cols = min(2, arr.shape[2])
    result[:, :steps, :cols] = arr[:, :steps, :cols]
    result[~np.isfinite(result)] = 0.0
    return result


__all__ = [
    "MPCConfig",
    "MPCPlanResult",
    "evaluate_action_sequences",
    "generate_action_sequences",
    "make_action_grid",
    "plan_grid_search",
]
