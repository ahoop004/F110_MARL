"""Utility for loading map metadata and image information for F110 environment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

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

        with map_yaml_path.open("r") as yaml_file:
            metadata = yaml.safe_load(yaml_file)

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

        image = Image.open(image_path).convert("L")
        gray = np.asarray(image, dtype=np.uint8)
        image_size = image.size

        threshold = env_cfg.get("track_threshold")
        if threshold is None:
            threshold = 200 if gray.mean() > 127 else 127
        invert = bool(env_cfg.get("track_inverted", False))
        track_mask = gray >= threshold
        if invert:
            track_mask = ~track_mask

        return MapData(
            metadata=metadata,
            image_path=image_path,
            image_size=image_size,
            yaml_path=map_yaml_path,
            track_mask=track_mask,
        )

    def augment_config(self, env_cfg: Dict[str, Any]) -> Dict[str, Any]:
        data = self.load(env_cfg)
        augmented = dict(env_cfg)
        augmented["map_meta"] = data.metadata
        augmented["map_image_path"] = str(data.image_path)
        augmented["map_image_size"] = data.image_size
        augmented["map_yaml_path"] = str(data.yaml_path)
        return augmented
