"""Fixed-size relative Frenet observations for nearby agents."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from wrappers.observations.base import ObservationComponent


_FIELDS = ("delta_s", "delta_d", "delta_vs", "delta_vd")
_DEFAULT_MAXIMA = {
    "delta_s": 20.0,
    "delta_d": 5.0,
    "delta_vs": 20.0,
    "delta_vd": 10.0,
}


class FrenetNeighborsComponent(ObservationComponent):
    """Nearest-agent slots ``[Δs, Δd, Δvs, Δvd, present]``.

    Neighbors are ordered by absolute wrapped longitudinal distance, with
    agent ID used only as a deterministic tie-breaker. Missing slots are zero.
    """

    def __init__(
        self,
        *,
        max_neighbors: int,
        maxima: Mapping[str, float] | None = None,
        clip: bool = False,
    ) -> None:
        self.max_neighbors = max(int(max_neighbors), 1)
        configured = dict(_DEFAULT_MAXIMA)
        configured.update(dict(maxima or {}))
        self._maxima = np.asarray(
            [max(abs(float(configured[field])), 1e-6) for field in _FIELDS],
            dtype=np.float32,
        )
        self.clip = bool(clip)

    @property
    def dim(self) -> int:
        return 5 * self.max_neighbors

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        out.fill(0.0)
        neighbors = info.get("frenet_neighbors", []) if isinstance(info, dict) else []
        if not isinstance(neighbors, (list, tuple)):
            return
        for slot, neighbor in enumerate(neighbors[: self.max_neighbors]):
            if not isinstance(neighbor, Mapping):
                continue
            start = 5 * slot
            values = np.asarray(
                [_finite_number(neighbor.get(field)) for field in _FIELDS],
                dtype=np.float32,
            )
            out[start : start + 4] = values / self._maxima
            out[start + 4] = 1.0
        np.nan_to_num(out, copy=False)
        if self.clip:
            np.clip(out, -1.0, 1.0, out=out)


def _finite_number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


__all__ = ["FrenetNeighborsComponent"]
