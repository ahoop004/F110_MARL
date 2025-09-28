"""Utility for loading map metadata and image information for F110 environment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
from PIL import Image


@dataclass
class MapData:
    metadata: Dict[str, Any]
    image_path: Path
    image_size: Tuple[int, int]
    yaml_path: Path


class MapLoader:
    """Handles map YAML/image resolution with sensible fallbacks."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.cwd()

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def _load_metadata(self, map_yaml_path: Path) -> Dict[str, Any]:
        with map_yaml_path.open("r") as yaml_file:
            return yaml.safe_load(yaml_file)

    def load(self, env_cfg: Dict[str, Any]) -> MapData:
        map_dir = self._resolve_path(Path(env_cfg.get("map_dir", "")))
        map_yaml_name = env_cfg.get("map_yaml") or env_cfg.get("map")
        if map_yaml_name is None:
            raise ValueError("env configuration must define 'map_yaml' or 'map'")

        map_yaml_path = (map_dir / map_yaml_name).expanduser().resolve()
        if not map_yaml_path.exists():
            raise FileNotFoundError(f"Map YAML not found: {map_yaml_path}")

        metadata = self._load_metadata(map_yaml_path)
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

        with Image.open(image_path) as img:
            image_size = img.size

        return MapData(metadata=metadata, image_path=image_path, image_size=image_size, yaml_path=map_yaml_path)

    def augment_config(self, env_cfg: Dict[str, Any]) -> Dict[str, Any]:
        data = self.load(env_cfg)
        augmented = dict(env_cfg)
        augmented["map_meta"] = data.metadata
        augmented["map_image_path"] = str(data.image_path)
        augmented["map_image_size"] = data.image_size
        augmented["map_yaml_path"] = str(data.yaml_path)
        return augmented
