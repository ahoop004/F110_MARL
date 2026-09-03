#!/usr/bin/env python3
"""Benchmark track-preview construction, cache lookup, and preview sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from utils.map_loader import MapLoader
from utils.track_preview import (
    TRACK_PREVIEW_PREPROCESSING_VERSION,
    TrackPreviewGeometry,
    TrackPreviewGeometryCache,
    build_track_preview_cache_key,
)


DEFAULT_MAPS = (
    "Budapest_map",
    "circle_map",
    "Melbourne_map",
    "Montreal_map",
    "Shanghai_map",
    "Silverstone_map",
    "Spa_map",
    "Spielberg_map",
)


def _measure(function: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = function()
    return time.perf_counter() - started, result


def _median_spread(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def benchmark_map(
    loader: MapLoader,
    map_name: str,
    *,
    spacing: float,
    preview_points: int,
    sample_calls: int,
    repetitions: int,
) -> dict[str, Any]:
    map_data = loader.load(
        {
            "map_dir": str(ROOT / "maps"),
            "map_bundle": map_name,
            "centerline_autoload": True,
            "walls_autoload": True,
        }
    )
    if map_data.centerline is None:
        raise ValueError(f"Map {map_name!r} has no centerline")

    construction: list[float] = []
    for _ in range(repetitions):
        seconds, geometry = _measure(
            lambda: TrackPreviewGeometry.build(
                map_data.centerline, map_data.walls, spacing=spacing
            )
        )
        if geometry is None:
            raise ValueError(f"Map {map_name!r} produced no preview geometry")
        construction.append(seconds)

    cache = TrackPreviewGeometryCache(max_entries=1)
    key = build_track_preview_cache_key(
        map_identity=map_data.yaml_path,
        centerline=map_data.centerline,
        walls=map_data.walls,
        spacing=spacing,
    )
    cached = cache.get_or_build(key, map_data.centerline, map_data.walls)
    assert cached is not None
    cache_lookup: list[float] = []
    for _ in range(repetitions):
        seconds, reused = _measure(
            lambda: cache.get_or_build(
                build_track_preview_cache_key(
                    map_identity=map_data.yaml_path,
                    centerline=map_data.centerline,
                    walls=map_data.walls,
                    spacing=spacing,
                ),
                map_data.centerline,
                map_data.walls,
            )
        )
        assert reused is cached
        cache_lookup.append(seconds)

    sample_positions = cached.points[
        np.linspace(0, len(cached.points) - 1, sample_calls, dtype=np.int64)
    ]
    def sample() -> None:
        last_index = -1
        for position in sample_positions:
            last_index = cached.nearest_index(position, last_index=last_index)
            cached.preview(
                position, preview_points, start_index=last_index
            )

    sampling: list[float] = []
    for _ in range(repetitions):
        seconds, _ = _measure(sample)
        sampling.append(seconds / sample_calls)

    payload_digest = hashlib.sha256()
    last_index = -1
    for position in sample_positions:
        last_index = cached.nearest_index(position, last_index=last_index)
        preview = cached.preview(position, preview_points, start_index=last_index)
        payload_digest.update(np.asarray(preview["curvature"]).tobytes())
        payload_digest.update(np.asarray(preview["width"]).tobytes())

    return {
        "map": map_name,
        "centerline_points": len(map_data.centerline),
        "wall_points": sum(len(wall) for wall in (map_data.walls or {}).values()),
        "preview_geometry_points": len(cached.points),
        "construction_seconds": _median_spread(construction),
        "cache_lookup_seconds": _median_spread(cache_lookup),
        "preview_sample_seconds_per_call": _median_spread(sampling),
        "preview_payload_sha256": payload_digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+", default=DEFAULT_MAPS)
    parser.add_argument("--spacing", type=float, default=0.3)
    parser.add_argument("--preview-points", type=int, default=20)
    parser.add_argument("--sample-calls", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default="/tmp/f110_track_geometry_benchmark.json")
    args = parser.parse_args()
    if args.spacing <= 0.0:
        parser.error("--spacing must be positive")
    if args.preview_points <= 0 or args.sample_calls <= 0 or args.repetitions <= 0:
        parser.error("preview points, sample calls, and repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    loader = MapLoader(base_dir=ROOT)
    report = {
        "benchmark_version": "1.0",
        "preprocessing_version": TRACK_PREVIEW_PREPROCESSING_VERSION,
        "spacing": args.spacing,
        "preview_points": args.preview_points,
        "sample_calls": args.sample_calls,
        "repetitions": args.repetitions,
        "results": [
            benchmark_map(
                loader,
                map_name,
                spacing=args.spacing,
                preview_points=args.preview_points,
                sample_calls=args.sample_calls,
                repetitions=args.repetitions,
            )
            for map_name in args.maps
        ],
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"results: {output}")


if __name__ == "__main__":
    main()
