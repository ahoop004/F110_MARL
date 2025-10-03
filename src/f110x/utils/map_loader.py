"""Utility for loading map metadata and image information for F110 environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, MutableMapping

import numpy as np
import yaml
from PIL import Image


@dataclass
class MapData:
    metadata: Dict[str, Any]
    image_path: Path
    image_size: Tuple[int, int]
    yaml_path: Path
    track_mask: np.ndarray | None


class MapLoader:
    """Handles map YAML/image resolution with sensible fallbacks."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.cwd()
        self._cache: MutableMapping[tuple[str, str], _CachedMap] = {}

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def load(self, env_cfg: Dict[str, Any]) -> MapData:
        map_dir = self._resolve_path(Path(env_cfg.get("map_dir", "")))
        map_yaml_name = env_cfg.get("map_yaml") or env_cfg.get("map")
        if map_yaml_name is None:
            raise ValueError("env configuration must define 'map_yaml' or 'map'")

        map_yaml_path = (map_dir / map_yaml_name).expanduser().resolve()
        if not map_yaml_path.exists():
            raise FileNotFoundError(f"Map YAML not found: {map_yaml_path}")

        cache_key = (str(map_dir), map_yaml_name)
        cached = self._cache.get(cache_key)

        yaml_mtime = map_yaml_path.stat().st_mtime_ns

        if cached is None or cached.yaml_mtime != yaml_mtime:
            with map_yaml_path.open("r") as yaml_file:
                metadata = yaml.safe_load(yaml_file)
            if not isinstance(metadata, dict):
                raise TypeError(f"Map YAML must define a mapping at root: {map_yaml_path}")
            metadata = dict(metadata)
        else:
            metadata = cached.metadata

        image_rel = metadata.get("image")
        fallback_image = env_cfg.get("map_image")
        if image_rel:
            image_path = (map_yaml_path.parent / image_rel).expanduser().resolve()
        elif fallback_image:
            image_path = (map_dir / fallback_image).expanduser().resolve()
        else:
            map_ext = env_cfg.get("map_ext", ".png")
            image_path = map_yaml_path.with_suffix(map_ext)

        if not image_path.exists():
            raise FileNotFoundError(f"Map image not found: {image_path}")

        image_mtime = image_path.stat().st_mtime_ns

        if (
            cached is None
            or cached.yaml_mtime != yaml_mtime
            or cached.image_mtime != image_mtime
            or cached.image_path != image_path
        ):
            with Image.open(image_path) as image:
                converted = image.convert("L")
                try:
                    image_size = converted.size
                    gray = np.asarray(converted, dtype=np.uint8)
                finally:
                    converted.close()
            gray.setflags(write=False)
            cached = _CachedMap(
                metadata=metadata,
                image_path=image_path,
                image_size=image_size,
                yaml_path=map_yaml_path,
                yaml_mtime=yaml_mtime,
                image_mtime=image_mtime,
                grayscale=gray,
            )
            self._cache[cache_key] = cached
        threshold = env_cfg.get("track_threshold")
        gray_source = cached.grayscale
        if threshold is None:
            threshold = 200 if gray_source.mean() > 127 else 127
        invert = bool(env_cfg.get("track_inverted", False))
        mask_key = (int(threshold), invert)
        track_mask = cached.track_masks.get(mask_key)
        if track_mask is None:
            mask = gray_source >= mask_key[0]
            if invert:
                mask = ~mask
            mask.setflags(write=False)
            cached.track_masks[mask_key] = mask
            track_mask = mask

        metadata_view = dict(cached.metadata)

        return MapData(
            metadata=metadata_view,
            image_path=cached.image_path,
            image_size=cached.image_size,
            yaml_path=cached.yaml_path,
            track_mask=track_mask,
        )


@dataclass
class _CachedMap:
    metadata: Dict[str, Any]
    image_path: Path
    image_size: Tuple[int, int]
    yaml_path: Path
    yaml_mtime: int
    image_mtime: int
    grayscale: np.ndarray
    track_masks: Dict[tuple[int, bool], np.ndarray] = field(default_factory=dict)
