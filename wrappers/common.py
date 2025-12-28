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
