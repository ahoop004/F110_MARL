"""Compatibility helpers around torch serialization defaults."""
from __future__ import annotations

from typing import Any

import torch


def safe_load(path: str, *, map_location: Any | None = None, weights_only: bool | None = None) -> Any:
    """Call ``torch.load`` while gracefully handling version-specific kwargs.

    PyTorch 2.6 flipped the default for ``weights_only`` to ``True`` which breaks
    checkpoints containing optimizer state dictionaries (they rely on pickled
    Python objects). Passing ``weights_only=False`` restores the legacy behaviour,
    but older PyTorch releases reject that keyword entirely. This helper attempts
    the new signature first and falls back to the legacy call when needed.
    """

    load_kwargs = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location

    # Explicitly request the legacy behaviour unless the caller overrides it.
    if weights_only is None:
        desired_weights_only = False
    else:
        desired_weights_only = weights_only

    try:
        return torch.load(path, weights_only=desired_weights_only, **load_kwargs)
    except TypeError:
        # Older torch versions (<2.0) do not accept the weights_only kwarg.
        return torch.load(path, **load_kwargs)


__all__ = ["safe_load"]
