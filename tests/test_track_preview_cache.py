from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from PIL import Image

from env.f110ParallelEnv import F110ParallelEnv
from env.map_schedule import MapScheduler
from utils.track_preview import (
    TrackPreviewGeometry,
    TrackPreviewGeometryCache,
    build_track_preview_cache_key,
)


def _circle(radius: float, count: int = 80) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles))).astype(
        np.float32
    )


def _key(
    map_name: str,
    centerline: np.ndarray,
    walls: dict[int, np.ndarray],
    spacing: float = 0.3,
):
    return build_track_preview_cache_key(
        map_identity=map_name,
        centerline=centerline,
        walls=walls,
        spacing=spacing,
    )


def test_geometry_cache_builds_once_and_reuses_immutable_result() -> None:
    centerline = _circle(10.0)
    walls = {0: _circle(9.0), 1: _circle(11.0)}
    cache = TrackPreviewGeometryCache(max_entries=2)
    calls = 0

    def builder(*args, **kwargs):
        nonlocal calls
        calls += 1
        return TrackPreviewGeometry.build(*args, **kwargs)

    key = _key("map_a.yaml", centerline, walls)
    first = cache.get_or_build(key, centerline, walls, builder=builder)
    second = cache.get_or_build(key, centerline, walls, builder=builder)

    assert first is second
    assert calls == 1
    assert len(cache) == 1
    assert first is not None
    assert not first.points.flags.writeable
    assert not first.curvature.flags.writeable
    assert not first.width.flags.writeable
    assert not first.projection_geometry.arc_lengths.flags.writeable


def test_geometry_cache_key_tracks_every_preprocessing_input() -> None:
    centerline = _circle(10.0)
    walls = {0: _circle(9.0), 1: _circle(11.0)}
    baseline = _key("map_a.yaml", centerline, walls)

    changed_centerline = centerline.copy()
    changed_centerline[0, 0] += 0.01
    changed_walls = {key: value.copy() for key, value in walls.items()}
    changed_walls[1][0, 0] += 0.01

    assert _key("map_b.yaml", centerline, walls) != baseline
    assert _key("map_a.yaml", changed_centerline, walls) != baseline
    assert _key("map_a.yaml", centerline, changed_walls) != baseline
    assert _key("map_a.yaml", centerline, walls, spacing=0.4) != baseline
    assert replace(baseline, preprocessing_version=baseline.preprocessing_version + 1) != baseline


def test_geometry_cache_is_lru_bounded() -> None:
    centerline = _circle(10.0)
    walls = {0: _circle(9.0), 1: _circle(11.0)}
    cache = TrackPreviewGeometryCache(max_entries=2)
    keys = [_key(f"map_{index}.yaml", centerline, walls) for index in range(3)]

    first = cache.get_or_build(keys[0], centerline, walls)
    cache.get_or_build(keys[1], centerline, walls)
    assert cache.get_or_build(keys[0], centerline, walls) is first
    cache.get_or_build(keys[2], centerline, walls)

    calls = 0

    def builder(*args, **kwargs):
        nonlocal calls
        calls += 1
        return TrackPreviewGeometry.build(*args, **kwargs)

    cache.get_or_build(keys[1], centerline, walls, builder=builder)
    assert calls == 1
    assert len(cache) == 2


def test_environment_map_revisit_reuses_geometry_without_cursor_state() -> None:
    centerline_a = _circle(10.0)
    walls_a = {0: _circle(9.0), 1: _circle(11.0)}
    centerline_b = _circle(12.0)
    walls_b = {0: _circle(11.0), 1: _circle(13.0)}
    env = F110ParallelEnv.__new__(F110ParallelEnv)
    env._track_preview_agents = frozenset({"car_0"})
    env._track_preview_spacing = 0.3
    env._track_preview_geometry_cache = TrackPreviewGeometryCache(2)
    env._track_preview_last_indices = {"car_0": 37}

    env.yaml_path = Path("map_a.yaml")
    env.walls = walls_a
    first_a = env._build_track_preview_geometry(centerline_a)
    env.yaml_path = Path("map_b.yaml")
    env.walls = walls_b
    geometry_b = env._build_track_preview_geometry(centerline_b)
    env.yaml_path = Path("map_a.yaml")
    env.walls = walls_a
    second_a = env._build_track_preview_geometry(centerline_a)

    assert first_a is second_a
    assert geometry_b is not first_a
    assert env._track_preview_last_indices == {"car_0": 37}
    assert second_a is not None
    before = second_a.preview(centerline_a[5], 8)
    after = first_a.preview(centerline_a[5], 8)
    np.testing.assert_array_equal(before["curvature"], after["curvature"])
    np.testing.assert_array_equal(before["width"], after["width"])


def test_centerline_change_resets_agent_cursors_outside_cache() -> None:
    env = F110ParallelEnv.__new__(F110ParallelEnv)
    env.possible_agents = ["car_0", "car_1"]
    env._track_preview_last_indices = {"car_0": 12, "car_1": 34}
    env._invalidate_global_state_cache = lambda: None
    env._centerline_state = type(
        "CenterlineStateStub",
        (),
        {"set_centerline": lambda self, centerline, path=None: None},
    )()
    env._build_track_preview_geometry = lambda centerline: object()
    env._update_renderer_centerline = lambda: None

    env.set_centerline(_circle(10.0), path=Path("map_b_centerline.csv"))

    assert env._track_preview_last_indices == {"car_0": -1, "car_1": -1}


def _write_map_bundle(root: Path, centerline_x: float) -> Path:
    bundle = root / "test_map"
    bundle.mkdir()
    Image.new("L", (4, 4), color=255).save(bundle / "map.png")
    (bundle / "map.yaml").write_text(
        "image: map.png\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n"
    )
    (bundle / "map_centerline.csv").write_text(
        f"x,y,w\n{centerline_x},0,0\n1,0,0\n1,1,0\n0,1,0\n"
    )
    (bundle / "map_walls.csv").write_text(
        "wall_id,x,y\n0,-1,-1\n0,2,-1\n0,2,2\n0,-1,2\n"
    )
    return bundle


def test_scheduler_revalidates_changed_geometry_sources(tmp_path: Path) -> None:
    bundle = _write_map_bundle(tmp_path, centerline_x=0.0)
    scheduler = MapScheduler(
        {
            "map_dir": str(tmp_path),
            "map_bundles": ["test_map"],
            "map_bundles_train": ["test_map"],
            "map_bundle_active": "test_map",
        },
        rng=np.random.default_rng(7),
    )
    first = scheduler.load_bundle(
        "test_map",
        map_ext=".png",
        centerline_render=False,
        centerline_features=True,
    )
    centerline_path = bundle / "map_centerline.csv"
    previous_mtime = centerline_path.stat().st_mtime_ns
    centerline_path.write_text(
        "x,y,w\n0.25,0,0\n1,0,0\n1,1,0\n0,1,0\n"
    )
    os.utime(centerline_path, ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000))

    second = scheduler.load_bundle(
        "test_map",
        map_ext=".png",
        centerline_render=False,
        centerline_features=True,
    )

    assert first.centerline is not None
    assert second.centerline is not None
    assert first.centerline[0, 0] == 0.0
    assert second.centerline[0, 0] == 0.25

    walls_path = bundle / "map_walls.csv"
    walls_mtime = walls_path.stat().st_mtime_ns
    walls_path.write_text(
        "wall_id,x,y\n0,-2,-1\n0,2,-1\n0,2,2\n0,-2,2\n"
    )
    os.utime(walls_path, ns=(walls_mtime + 1_000_000, walls_mtime + 1_000_000))
    third = scheduler.load_bundle(
        "test_map",
        map_ext=".png",
        centerline_render=False,
        centerline_features=True,
    )

    assert second.walls is not None
    assert third.walls is not None
    assert second.walls[0][0, 0] == -1.0
    assert third.walls[0][0, 0] == -2.0
    assert scheduler.configured_bundle_count == 1
