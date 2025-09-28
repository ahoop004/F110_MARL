"""Utility for loading map metadata and image information for F110 environment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml
from PIL import Image


@dataclass
class MapData:
    metadata: Dict[str, Any]
    image_path: Path
    image_size: tuple[int, int]


class MapLoader:
    """Handles map YAML/image resolution with sensible fallbacks."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir or Path.cwd()

    def resolve_map_paths(self, env_cfg: Dict[str, Any]) -> tuple[Path, Path]:
        map_dir = Path(env_cfg.get("map_dir", ""))
        map_yaml = env_cfg.get("map_yaml") or env_cfg.get("map")
        if map_yaml is None:
            raise ValueError("env configuration must define 'map_yaml' or 'map'")

        map_yaml_path = (map_dir / map_yaml).expanduser().resolve()
        if not map_yaml_path.exists():
            raise FileNotFoundError(f"Map YAML not found: {map_yaml_path}")

        with open(map_yaml_path, "r") as yaml_file:
            map_meta = yaml.safe_load(yaml_file)

        image_rel = map_meta.get("image")
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

        return MapData(metadata=map_meta, image_path=image_path, image_size=image_size), map_yaml_path

    def inject_map_data(self, env_cfg: Dict[str, Any]) -> Dict[str, Any]:
        map_data, _ = self.resolve_map_paths(env_cfg)
        env_cfg = dict(env_cfg)
        env_cfg["map_meta"] = map_data.metadata
        env_cfg["map_image_path"] = str(map_data.image_path)
        env_cfg["map_image_size"] = map_data.image_size
        return env_cfg
