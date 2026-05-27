"""Base class for observation components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class ObservationComponent(ABC):
    """Single named slice of the flat observation vector.

    Subclasses must implement :meth:`dim` and either override
    :meth:`compute_into` (zero-allocation fast path used by
    :class:`~wrappers.observations.composer.ObservationComposer`) **or**
    override :meth:`compute` (legacy path that allocates a fresh array).

    The default :meth:`compute` calls :meth:`compute_into` on a fresh buffer,
    and the default :meth:`compute_into` calls :meth:`compute` and copies into
    the output slice — so it is safe to override just one of them.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Number of scalar values this component contributes."""

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        """Return a fresh 1-D float32 array of length ``self.dim``.

        The default implementation delegates to :meth:`compute_into`, which
        subclasses may override for zero-allocation behaviour.
        """
        out = np.empty(self.dim, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        """Write this component's values into *out* (a pre-allocated float32 slice).

        Parameters
        ----------
        raw_obs:
            Dict of arrays keyed by obs field name (``lidar``, ``pose``,
            ``velocity``, …).
        info:
            Step info dict from ``env.step()`` or ``env.reset()``.
        out:
            Pre-allocated float32 buffer of length :attr:`dim`.  Values are
            written in-place; no new array is allocated by the default path.

        The default implementation calls :meth:`compute` and copies the result.
        Override this to write directly and avoid any intermediate allocation.
        """
        result = self.compute(raw_obs, info)
        np.copyto(out, result)

    @property
    def name(self) -> str:
        return self.__class__.__name__
