"""Kinematic rollout helpers for fixed-policy MPC controllers."""
from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit


DEFAULT_DT = 0.1
DEFAULT_WHEELBASE = 0.3302


def normalize_pose(pose: Optional[np.ndarray]) -> np.ndarray:
    """Return a finite ``[x, y, theta]`` pose, defaulting to the origin."""
    if pose is None:
        return np.zeros(3, dtype=np.float32)
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    result = np.zeros(3, dtype=np.float32)
    if arr.size:
        n = min(3, arr.size)
        result[:n] = arr[:n]
    result[~np.isfinite(result)] = 0.0
    return result


def normalize_actions(actions: Optional[np.ndarray], horizon: int) -> np.ndarray:
    """Return a finite ``(horizon, 2)`` action array.

    Actions follow the environment convention ``[steering, speed]``.
    """
    horizon = max(0, int(horizon))
    result = np.zeros((horizon, 2), dtype=np.float32)
    if actions is None or horizon == 0:
        return result

    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return result

    rows = min(horizon, arr.shape[0])
    cols = min(2, arr.shape[1])
    if rows > 0 and cols > 0:
        result[:rows, :cols] = arr[:rows, :cols]
    result[~np.isfinite(result)] = 0.0
    return result


def kinematic_bicycle_step(
    pose: Optional[np.ndarray],
    action: Optional[np.ndarray],
    *,
    dt: float = DEFAULT_DT,
    wheelbase: float = DEFAULT_WHEELBASE,
) -> np.ndarray:
    """Advance one kinematic bicycle step.

    Parameters
    ----------
    pose:
        ``[x, y, theta]``.
    action:
        ``[steering, speed]``.
    dt:
        Planning step duration in seconds.
    wheelbase:
        Vehicle wheelbase in metres.
    """
    state = normalize_pose(pose).astype(np.float32, copy=True)
    act = normalize_actions(action, 1)[0]
    steer = float(act[0])
    speed = float(act[1])
    dt = max(float(dt), 0.0)
    wheelbase = max(float(wheelbase), 1e-6)

    theta = float(state[2])
    state[0] += speed * np.cos(theta) * dt
    state[1] += speed * np.sin(theta) * dt
    state[2] = _wrap_angle(theta + speed / wheelbase * np.tan(steer) * dt)
    return state


def rollout_kinematic_bicycle(
    pose: Optional[np.ndarray],
    actions: Optional[np.ndarray],
    *,
    dt: float = DEFAULT_DT,
    wheelbase: float = DEFAULT_WHEELBASE,
    horizon: Optional[int] = None,
) -> np.ndarray:
    """Roll out a kinematic bicycle trajectory.

    Returns an array of shape ``(horizon + 1, 3)`` containing the initial pose
    followed by one state for each action.
    """
    if horizon is None:
        if actions is None:
            horizon = 0
        else:
            arr = np.asarray(actions)
            horizon = 1 if arr.ndim == 1 else int(arr.shape[0])
    horizon = max(0, int(horizon))
    action_array = normalize_actions(actions, horizon)

    return _rollout_normalized(
        normalize_pose(pose), action_array, max(float(dt), 0.0), max(float(wheelbase), 1e-6),
    )


@njit(cache=True, error_model="numpy")
def _rollout_normalized(pose, actions, dt, wheelbase):
    """Roll out normalized inputs without per-step Python calls and copies."""
    trajectory = np.empty((len(actions) + 1, 3), dtype=np.float32)
    trajectory[0] = pose
    for idx in range(len(actions)):
        # The public step normalizes each incoming state, including any
        # nonfinite values produced by an extreme previous action.
        x, y, theta = (np.float64(trajectory[idx, 0]), np.float64(trajectory[idx, 1]),
                       np.float64(trajectory[idx, 2]))
        if not np.isfinite(x):
            x = 0.0
        if not np.isfinite(y):
            y = 0.0
        if not np.isfinite(theta):
            theta = 0.0
        steer, speed = np.float64(actions[idx, 0]), np.float64(actions[idx, 1])
        trajectory[idx + 1, 0] = x + speed * np.cos(theta) * dt
        trajectory[idx + 1, 1] = y + speed * np.sin(theta) * dt
        angle = theta + speed / wheelbase * np.tan(steer) * dt
        trajectory[idx + 1, 2] = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return trajectory


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


__all__ = [
    "DEFAULT_DT",
    "DEFAULT_WHEELBASE",
    "kinematic_bicycle_step",
    "normalize_actions",
    "normalize_pose",
    "rollout_kinematic_bicycle",
]
