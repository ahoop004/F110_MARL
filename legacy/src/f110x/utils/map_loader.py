"""Utility for loading map metadata and image information for F110 environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, MutableMapping, Optional

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
    centerline_path: Optional[Path] = None
    centerline: Optional[np.ndarray] = None
    spawn_points: Dict[str, np.ndarray] = field(default_factory=dict)


class MapLoader:
    """Handles map YAML/image resolution with sensible fallbacks."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.cwd()
        self._cache: MutableMapping[tuple[str, str], _CachedMap] = {}

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def load(self, env_cfg: Dict[str, Any]) -> MapData:
        map_root_value = env_cfg.get("map_dir") or env_cfg.get("map_root") or "maps"
        map_dir = self._resolve_path(Path(str(map_root_value)))
        env_cfg["map_dir"] = str(map_dir)

        map_yaml_name = env_cfg.get("map_yaml")
        map_value = env_cfg.get("map")
        bundle_value = env_cfg.get("map_bundle")

        if map_yaml_name is None:
            candidate = bundle_value if bundle_value is not None else map_value
            if candidate is None:
                raise ValueError("env configuration must define a 'map' bundle or explicit 'map_yaml'")
            map_yaml_path = self._resolve_bundle_yaml(map_dir, candidate)
            map_yaml_name = self._relative_yaml_name(map_dir, map_yaml_path)
            env_cfg["map_yaml"] = map_yaml_name
            env_cfg["map"] = map_yaml_name
            env_cfg["map_bundle"] = str(candidate)
        else:
            yaml_candidate = Path(str(map_yaml_name)).expanduser()
            if not yaml_candidate.is_absolute():
                yaml_candidate = (map_dir / yaml_candidate)
            map_yaml_path = yaml_candidate.resolve()

        if not map_yaml_path.is_file():
            raise FileNotFoundError(f"Map YAML not found: {map_yaml_path}")

        cache_key = (str(map_dir), str(map_yaml_name))
        cached = self._cache.get(cache_key)

        yaml_mtime = map_yaml_path.stat().st_mtime_ns

        if cached is None or cached.yaml_mtime != yaml_mtime:
            with map_yaml_path.open("r") as yaml_file:
                metadata = yaml.safe_load(yaml_file)
            if not isinstance(metadata, dict):
                raise TypeError(f"Map YAML must define a mapping at root: {map_yaml_path}")
            metadata = dict(metadata)
            spawn_points = self._parse_spawn_points(metadata)
        else:
            metadata = cached.metadata
            spawn_points = cached.spawn_points

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
                spawn_points=spawn_points,
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

        if cached.spawn_points is not spawn_points:
            cached.spawn_points = spawn_points

        centerline_path: Optional[Path] = None
        explicit = env_cfg.get("centerline_csv")
        auto_flag_raw = env_cfg.get("centerline_autoload", True)
        auto_flag = True if auto_flag_raw is None else bool(auto_flag_raw)
        centerline_enabled = auto_flag or explicit is not None
        if centerline_enabled:
            if explicit:
                candidate_path = Path(explicit)
                if candidate_path.is_absolute():
                    candidate = candidate_path.resolve()
                else:
                    candidate = (map_dir / candidate_path).resolve()
            else:
                candidate = cached.yaml_path.with_name(f"{cached.yaml_path.stem}_centerline.csv")
            if candidate.exists():
                centerline_path = candidate

        centerline: Optional[np.ndarray] = None
        if centerline_path is not None:
            cl_mtime = centerline_path.stat().st_mtime_ns
            cached_cl = cached.centerline
            if (
                cached.centerline_path != centerline_path
                or cached.centerline_mtime != cl_mtime
                or cached_cl is None
            ):
                centerline = self._load_centerline(centerline_path)
                if centerline is not None:
                    centerline.setflags(write=False)
                cached.centerline = centerline
                cached.centerline_path = centerline_path
                cached.centerline_mtime = cl_mtime
            else:
                centerline = cached_cl

        return MapData(
            metadata=metadata_view,
            image_path=cached.image_path,
            image_size=cached.image_size,
            yaml_path=cached.yaml_path,
            track_mask=track_mask,
            centerline_path=centerline_path,
            centerline=centerline,
            spawn_points={name: value.copy() for name, value in cached.spawn_points.items()},
        )

    def _resolve_bundle_yaml(self, map_dir: Path, bundle: Any) -> Path:
        bundle_str = str(bundle).strip()
        if not bundle_str:
            raise ValueError("map bundle identifier cannot be empty")

        candidate_path = Path(bundle_str)
        # Absolute reference
        if candidate_path.is_absolute():
            resolved = candidate_path
            if resolved.is_file():
                return resolved
            if resolved.with_suffix(".yaml").is_file():
                return resolved.with_suffix(".yaml")
            raise FileNotFoundError(f"Map YAML not found for bundle '{bundle_str}': {resolved}")

        # Relative explicit path (with extension or nested directories)
        if candidate_path.suffix:
            resolved = (map_dir / candidate_path).resolve()
            if resolved.is_file():
                return resolved

        resolved = (map_dir / candidate_path).resolve()
        if resolved.is_file():
            return resolved

        yaml_with_suffix = resolved.with_suffix(".yaml")
        if yaml_with_suffix.is_file():
            return yaml_with_suffix

        if resolved.is_dir():
            yaml_files = sorted(resolved.glob("*.yaml"))
            if yaml_files:
                return yaml_files[0].resolve()

        search_name = candidate_path.name
        matches = sorted(map_dir.rglob(f"{search_name}.yaml"))
        if matches:
            return matches[0].resolve()

        raise FileNotFoundError(f"Map YAML not found for bundle '{bundle_str}' within {map_dir}")

    @staticmethod
    def _relative_yaml_name(map_dir: Path, yaml_path: Path) -> str:
        try:
            return yaml_path.relative_to(map_dir).as_posix()
        except ValueError:
            return str(yaml_path)

    @staticmethod
    def _load_centerline(path: Path) -> np.ndarray:
        def _read(skip_header: int) -> np.ndarray:
            return np.genfromtxt(
                path,
                delimiter=",",
                comments="#",
                skip_header=skip_header,
                usecols=(0, 1, 2),
                dtype=np.float32,
            )

        data = _read(1)
        if data.size == 0:
            data = _read(0)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if np.isnan(data).any():
            data = data[~np.isnan(data).any(axis=1)]
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _parse_spawn_points(metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        annotations = metadata.get("annotations")
        if not isinstance(annotations, dict):
            return {}

        points_raw = annotations.get("spawn_points")
        if not points_raw:
            return {}

        spawn_points: Dict[str, np.ndarray] = {}
        for entry in points_raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("id") or entry.get("label")
            if not name:
                continue
            pose_raw = entry.get("pose") or entry.get("poses") or entry.get("position")
            if pose_raw is None:
                continue
            pose_arr = np.asarray(pose_raw, dtype=np.float32)
            if pose_arr.ndim != 1:
                pose_arr = pose_arr.reshape(-1)
            if pose_arr.size < 2:
                continue
            if pose_arr.size == 2:
                pose_arr = np.concatenate([pose_arr, np.zeros(1, dtype=np.float32)])
            elif pose_arr.size > 3:
                pose_arr = pose_arr[:3]
            pose_arr = pose_arr.astype(np.float32, copy=True)
            pose_arr.setflags(write=False)
            spawn_points[str(name)] = pose_arr
        return spawn_points


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
    centerline_path: Optional[Path] = None
    centerline_mtime: Optional[int] = None
    centerline: Optional[np.ndarray] = None
    spawn_points: Dict[str, np.ndarray] = field(default_factory=dict)
