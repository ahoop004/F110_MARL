"""Compatibility helpers around torch serialization defaults."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import torch


def validate_checkpoint_compatibility(checkpoint, expected, *, states=()) -> None:
    """Reject semantic or tensor-layout mismatches before loading any weights."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")

    def compare(stored, current, field):
        if isinstance(current, Mapping):
            if not isinstance(stored, Mapping) or set(stored) != set(current):
                raise ValueError(f"Incompatible checkpoint field {field}: mapping keys differ")
            for key in current:
                compare(stored[key], current[key], f"{field}.{key}")
        elif isinstance(current, (list, tuple)):
            if not isinstance(stored, (list, tuple)) or len(stored) != len(current):
                raise ValueError(f"Incompatible checkpoint field {field}: sequence differs")
            for index, value in enumerate(current):
                compare(stored[index], value, f"{field}[{index}]")
        elif isinstance(current, torch.Tensor) or hasattr(current, "shape"):
            try:
                matches = torch.equal(torch.as_tensor(stored).cpu(), torch.as_tensor(current).cpu())
            except (TypeError, ValueError, RuntimeError):
                matches = False
            if not matches:
                raise ValueError(f"Incompatible checkpoint field {field}: values differ")
        elif stored != current:
            raise ValueError(f"Incompatible checkpoint field {field}: checkpoint={stored!r}, expected={current!r}")

    for field, value in expected.items():
        if field not in checkpoint:
            raise ValueError(f"Checkpoint missing required field {field}")
        compare(checkpoint[field], value, field)
    for name, stored, current in states:
        if not isinstance(stored, Mapping) or set(stored) != set(current):
            raise ValueError(f"Incompatible checkpoint {name}: state keys differ")
        for key, value in current.items():
            source = stored[key]
            if not isinstance(source, torch.Tensor) or source.shape != value.shape:
                raise ValueError(f"Incompatible checkpoint {name}.{key}: tensor shape differs")


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


def resolve_device(preferred: Optional[Iterable[Any]] = None) -> torch.device:
    """Resolve torch device from preferred values, environment, then availability."""

    candidates = []
    if preferred:
        for value in preferred:
            if value is None:
                continue
            candidates.append(str(value))

    env_value = os.environ.get("F110_DEVICE")
    if env_value:
        candidates.append(env_value)

    for value in candidates:
        choice = value.strip().lower()
        if not choice:
            continue
        if choice == "cpu":
            return torch.device("cpu")
        if choice.startswith("cuda") or choice == "gpu":
            if torch.cuda.is_available():
                return torch.device(choice if choice.startswith("cuda") else "cuda")
            print(
                f"[WARN] Requested device '{value}' but CUDA is unavailable; falling back to CPU."
            )
            return torch.device("cpu")
        print(f"[WARN] Unknown device '{value}'; expected 'cpu' or 'cuda'.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


__all__ = ["safe_load", "resolve_device", "validate_checkpoint_compatibility"]
