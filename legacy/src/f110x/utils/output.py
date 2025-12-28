"""Utilities for resolving output directories/files under a common root."""
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def resolve_output_dir(path_like: PathLike, root: Path) -> Path:
    """Resolve ``path_like`` against ``root`` and ensure the directory exists."""

    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_file(path_like: PathLike, root: Path) -> Path:
    """Resolve ``path_like`` against ``root`` and ensure the parent directory exists."""

    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

__all__ = ["resolve_output_dir", "resolve_output_file"]
