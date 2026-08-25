"""Scenario-level map bundle discovery and split selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import math

import numpy as np
import yaml


def coerce_bundle_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("environment.map_bundles cannot be empty")
        return [value]
    if isinstance(value, (list, tuple)):
        bundles = [str(item).strip() for item in value]
        bundles = [item for item in bundles if item]
        if not bundles:
            raise ValueError("environment.map_bundles cannot be empty")
        return bundles
    raise TypeError("environment.map_bundles must be a string or list of strings")


def resolve_bundle_yaml(map_dir: Path, bundle: str) -> Path:
    bundle_str = str(bundle).strip()
    if not bundle_str:
        raise ValueError("map bundle identifier cannot be empty")

    candidate_path = Path(bundle_str)
    if candidate_path.is_absolute():
        resolved = candidate_path
        if resolved.is_file():
            return resolved
        if resolved.with_suffix(".yaml").is_file():
            return resolved.with_suffix(".yaml")
        raise FileNotFoundError(f"Map YAML not found for bundle '{bundle_str}': {resolved}")

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


def discover_map_bundles(env_config: Dict[str, Any]) -> List[str]:
    map_root = env_config.get("map_dir") or env_config.get("map_root") or "maps"
    map_dir = Path(str(map_root)).expanduser()
    if not map_dir.is_absolute():
        map_dir = (Path.cwd() / map_dir).resolve()

    bundles: List[str] = []
    for entry in sorted(map_dir.iterdir()):
        if not entry.is_dir():
            continue
        yaml_files = sorted(entry.glob("*.yaml"))
        if not yaml_files:
            continue
        yaml_path = yaml_files[0]
        try:
            metadata = yaml.safe_load(yaml_path.read_text())
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        image_field = metadata.get("image")
        if image_field:
            image_path = (yaml_path.parent / image_field).expanduser().resolve()
        else:
            image_path = None
            for ext in (".png", ".pgm", ".jpg", ".jpeg"):
                candidate = yaml_path.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is None or not image_path.exists():
            continue
        stem = yaml_path.stem
        centerline_path = yaml_path.with_name(f"{stem}_centerline.csv")
        walls_path = yaml_path.with_name(f"{stem}_walls.csv")
        if not centerline_path.exists() or not walls_path.exists():
            continue
        bundles.append(entry.name)

    return bundles


def relative_yaml_name(map_dir: Path, yaml_path: Path) -> str:
    try:
        return yaml_path.relative_to(map_dir).as_posix()
    except ValueError:
        return str(yaml_path)


def apply_map_bundle(env_config: Dict[str, Any], bundle: str) -> Dict[str, Any]:
    map_root = env_config.get("map_dir") or env_config.get("map_root") or "maps"
    map_dir = Path(str(map_root)).expanduser()
    if not map_dir.is_absolute():
        map_dir = (Path.cwd() / map_dir).resolve()

    yaml_path = resolve_bundle_yaml(map_dir, bundle)
    env_config["map_dir"] = str(map_dir)
    env_config["map_yaml"] = relative_yaml_name(map_dir, yaml_path)
    env_config["map"] = env_config["map_yaml"]
    env_config["map_bundle"] = str(bundle)
    return env_config


def normalize_maps_key(env_config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate `maps:` into the legacy map/map_bundles representation."""

    maps_raw = env_config.get("maps")
    if maps_raw is None:
        return env_config

    env_config = dict(env_config)
    env_config.pop("maps")

    if isinstance(maps_raw, str) and maps_raw.strip().lower() in {"auto", "all"}:
        env_config["map_bundles"] = True
        return env_config

    maps_list = [maps_raw] if isinstance(maps_raw, str) else list(maps_raw)

    if len(maps_list) == 1:
        bundle = str(maps_list[0]).strip()
        map_dir = Path(str(env_config.get("map_dir", "maps"))).expanduser()
        if not map_dir.is_absolute():
            map_dir = (Path.cwd() / map_dir).resolve()
        try:
            yaml_path = resolve_bundle_yaml(map_dir, bundle)
            env_config["map"] = str(yaml_path)
            env_config.setdefault("map_dir", str(map_dir))
        except FileNotFoundError:
            env_config["map"] = bundle
    else:
        env_config["map_bundles"] = maps_list

    return env_config


def apply_map_split(
    env_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    env_config = normalize_maps_key(env_config)
    explicit_train = coerce_bundle_list(env_config.get("map_bundles_train"))
    explicit_eval = coerce_bundle_list(env_config.get("map_bundles_eval"))
    if explicit_train or explicit_eval:
        train_bundles = explicit_train or explicit_eval or []
        eval_bundles = explicit_eval or train_bundles
        is_eval = str(mode).lower() in {"eval", "evaluation", "test"}
        active_bundles = eval_bundles if is_eval else train_bundles
        if not active_bundles:
            raise ValueError(f"No explicit map bundles configured for {mode} mode")
        env_config = dict(env_config)
        env_config["map_bundles_train"] = list(train_bundles)
        env_config["map_bundles_eval"] = list(eval_bundles)
        env_config["map_bundle_active"] = active_bundles[0]
        env_config["map_split_mode"] = "eval" if is_eval else "train"
        return apply_map_bundle(env_config, active_bundles[0])

    map_bundles_raw = env_config.get("map_bundles")
    if map_bundles_raw is None:
        map_bundles = coerce_bundle_list(map_bundles_raw)
    elif (
        map_bundles_raw is True
        or (isinstance(map_bundles_raw, str) and map_bundles_raw.strip().lower() in {"auto", "all"})
    ):
        map_bundles = discover_map_bundles(env_config)
        env_config = dict(env_config)
        env_config["map_bundles"] = list(map_bundles)
    else:
        map_bundles = coerce_bundle_list(map_bundles_raw)
    if not map_bundles:
        return env_config

    split_cfg = env_config.get("map_split") or {}
    if not isinstance(split_cfg, dict):
        raise TypeError("environment.map_split must be a mapping when provided")

    train_ratio = split_cfg.get("train_ratio", 0.8)
    try:
        train_ratio = float(train_ratio)
    except (TypeError, ValueError):
        train_ratio = 0.8
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("map_split.train_ratio must be between 0 and 1")

    seed = split_cfg.get("seed")
    if seed is None:
        seed = experiment_config.get("seed", env_config.get("seed", 0))
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0

    shuffle = split_cfg.get("shuffle", True)
    rng = np.random.default_rng(seed)
    bundles = list(map_bundles)
    if shuffle:
        rng.shuffle(bundles)

    total = len(bundles)
    if total == 1:
        train_bundles = bundles
        eval_bundles: List[str] = []
    else:
        train_count = int(math.floor(train_ratio * total))
        train_count = max(1, min(total - 1, train_count))
        train_bundles = bundles[:train_count]
        eval_bundles = bundles[train_count:]

    is_eval = str(mode).lower() in {"eval", "evaluation", "test"}
    active_bundles = eval_bundles if is_eval else train_bundles
    if not active_bundles:
        active_bundles = train_bundles

    pick_key = "eval_pick" if is_eval else "train_pick"
    pick_strategy = split_cfg.get(pick_key, split_cfg.get("pick", "first"))
    if str(env_config.get("map_cycle", "")).lower() == "per_episode":
        pick_strategy = env_config.get("map_pick", pick_strategy)
    if pick_strategy not in {"first", "random"}:
        pick_strategy = "first"
    if pick_strategy == "random":
        chosen = active_bundles[int(rng.integers(0, len(active_bundles)))]
    else:
        chosen = active_bundles[0]

    env_config = dict(env_config)
    env_config["map_bundles_train"] = list(train_bundles)
    env_config["map_bundles_eval"] = list(eval_bundles)
    env_config["map_bundle_active"] = chosen
    env_config["map_split_mode"] = "eval" if is_eval else "train"
    return apply_map_bundle(env_config, chosen)
