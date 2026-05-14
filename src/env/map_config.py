"""Runtime map path and metadata resolution for F110 environments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from PIL import Image

from src.env.types import MapRuntimeConfig


def normalize_map_identifier(identifier: Optional[Any]) -> Optional[str]:
    if identifier is None:
        return None
    identifier = str(identifier)
    return identifier if Path(identifier).suffix else f"{identifier}.yaml"


def resolve_map_runtime_config(
    cfg: Mapping[str, Any],
    map_data: Optional[Any] = None,
) -> MapRuntimeConfig:
    """Resolve one active map into concrete paths and metadata.

    This function intentionally does not select train/eval splits or schedule
    maps across episodes; that belongs in core map selection.
    """

    map_dir_value = cfg.get("map_dir")
    if map_dir_value is not None:
        map_dir = Path(map_dir_value)
    elif map_data is not None:
        map_dir = Path(map_data.yaml_path).parent  # type: ignore[attr-defined]
    else:
        map_dir = Path.cwd()

    map_ext_value = cfg.get("map_ext")
    if map_ext_value is not None:
        map_ext = str(map_ext_value)
    elif map_data is not None:
        map_ext = map_data.image_path.suffix or ".png"  # type: ignore[attr-defined]
    else:
        map_ext = ".png"

    map_name = normalize_map_identifier(cfg.get("map"))
    map_yaml = normalize_map_identifier(cfg.get("map_yaml"))
    if map_name is None and map_yaml is not None:
        map_name = map_yaml
    elif map_yaml is None and map_name is not None:
        map_yaml = map_name

    map_path = (map_dir / f"{map_name}").resolve()
    yaml_path = (map_dir / f"{map_yaml}").resolve()

    metadata = cfg.get("map_meta")
    if metadata is None and map_data is not None:
        metadata = dict(map_data.metadata)  # type: ignore[attr-defined]
    elif isinstance(metadata, Mapping):
        metadata = dict(metadata)
    if metadata is None:
        with open(map_path, "r") as handle:
            metadata = yaml.safe_load(handle) or {}

    preloaded_image_path = cfg.get("map_image_path")
    if preloaded_image_path is None and map_data is not None:
        preloaded_image_path = map_data.image_path  # type: ignore[attr-defined]

    image_rel = metadata.get("image")
    if preloaded_image_path is not None:
        image_path = Path(preloaded_image_path).resolve()
    elif image_rel:
        image_path = (map_path.parent / image_rel).resolve()
    else:
        img_filename = cfg.get("map_image")
        if img_filename is not None:
            image_path = (map_dir / img_filename).resolve()
        elif map_data is not None:
            image_path = Path(map_data.image_path).resolve()  # type: ignore[attr-defined]
        else:
            image_path = map_path.with_suffix(map_ext)

    image_size = cfg.get("map_image_size")
    if image_size is None and map_data is not None:
        image_size = map_data.image_size  # type: ignore[attr-defined]
    if image_size is not None:
        width, height = map(int, image_size)
    else:
        with Image.open(image_path) as img:
            width, height = img.size

    return MapRuntimeConfig(
        map_dir=map_dir,
        map_ext=map_ext,
        map_name=map_name,
        map_yaml=map_yaml,
        map_path=map_path,
        yaml_path=yaml_path,
        metadata=dict(metadata),
        image_path=image_path,
        image_size=(width, height),
    )
