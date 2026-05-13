"""Shared utilities for observation and action wrappers."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np


_DEFAULT_DTYPE = np.float32


def to_numpy(
    data: Any,
    *,
    dtype: np.dtype = _DEFAULT_DTYPE,
    copy: bool = False,
    flatten: bool = False,
) -> np.ndarray:
    """Convert arbitrary array-like ``data`` into a ``np.ndarray``.

    Parameters
    ----------
    data:
        Input array-like payload.
    dtype:
        Target dtype (``np.float32`` by default).
    copy:
        When ``True`` returns a copy even if ``data`` is already an ndarray.
    flatten:
        When ``True`` the result is flattened to 1D.
    """

    arr = np.asarray(data, dtype=dtype)
    if copy:
        arr = arr.copy()
    if flatten and arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def ensure_index(value: Any) -> int:
    """Coerce a scalar or zero-d array to ``int`` for discrete actions."""

    if np.isscalar(value):
        return int(value)
    return int(np.asarray(value).item())


def downsample_lidar(
    scan: Iterable[float],
    target_beams: Optional[int],
    *,
    pad_value: float = 0.0,
    dtype: np.dtype = _DEFAULT_DTYPE,
) -> np.ndarray:
    """Downsample or pad LiDAR scans to ``target_beams`` samples."""

    scan_array = to_numpy(scan, dtype=dtype, copy=False, flatten=True)
    if target_beams is None or target_beams <= 0:
        return scan_array

    target = int(target_beams)
    size = int(scan_array.size)

    if size == target:
        return scan_array.copy()

    if size > target:
        indices = np.linspace(0, size - 1, target, dtype=np.int32)
        return scan_array[indices]

    padded = np.full((target,), pad_value, dtype=dtype)
    padded[:size] = scan_array
    return padded


# ---------------------------------------------------------------------------
# Sector / radial geometry helpers (used by f110ParallelEnv for forcing reward)
# ---------------------------------------------------------------------------

_SECTOR_DEGREES = (
    ("front",       -22.5,   22.5),
    ("front_right",  22.5,   67.5),
    ("right",        67.5,  112.5),
    ("back_right",  112.5,  157.5),
    ("back",        157.5, -157.5),
    ("back_left",  -157.5, -112.5),
    ("left",       -112.5,  -67.5),
    ("front_left",  -67.5,  -22.5),
)
_SECTOR_NAMES = tuple(name for name, *_ in _SECTOR_DEGREES)


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _sector_from_angle(angle_deg: float) -> str:
    angle_deg = _wrap_degrees(angle_deg)
    for name, start, end in _SECTOR_DEGREES:
        if name == "back":
            if angle_deg >= 157.5 or angle_deg < -157.5:
                return name
        elif start <= end and start <= angle_deg < end:
            return name
        elif start > end and (angle_deg >= start or angle_deg < end):
            return name
    return "front"


def _radial_gain(
    distance: float,
    preferred: float,
    inner_tol: float,
    outer_tol: float,
    falloff: str,
) -> float:
    preferred = max(float(preferred), 0.0)
    inner_tol = max(float(inner_tol), 0.0)
    outer_tol = max(float(outer_tol), 0.0)
    lower = max(0.0, preferred - inner_tol)
    upper = preferred + outer_tol

    if falloff == "binary":
        return 1.0 if lower <= distance <= upper else 0.0

    if distance < lower:
        return 1.0 if inner_tol > 0.0 else 0.0
    if distance > upper:
        if outer_tol == 0.0:
            return 0.0
        ratio = (upper - distance) / outer_tol
        return max(0.0, min(1.0, ratio))
    if falloff == "gaussian":
        sigma = (inner_tol + outer_tol) / 2.0 or 1.0
        return float(np.exp(-((distance - preferred) ** 2) / (2.0 * sigma ** 2)))
    return 1.0
