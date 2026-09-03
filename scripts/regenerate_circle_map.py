#!/usr/bin/env python3
"""Regenerate the circle-map walls and occupancy images from its centerline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = REPO_ROOT / "maps" / "circle_map"
MAP_NAME = "circle_map"
TRACK_WIDTH_M = 2.70
BACKGROUND_GRAY = 205
ROAD_GRAY = 255
WALL_GRAY = 0
WALL_WIDTH_PX = 3
SUPERSAMPLE = 4


def _closed_offsets(points: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lengths, 1e-9)
    left_normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    signed_area = 0.5 * np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    # For a counter-clockwise centerline the left normal points inward.
    if signed_area > 0.0:
        outer = points - half_width * left_normals
        inner = points + half_width * left_normals
    else:
        outer = points + half_width * left_normals
        inner = points - half_width * left_normals
    return outer, inner


def _image_points(
    points: np.ndarray,
    *,
    origin: tuple[float, float],
    resolution: float,
    height: int,
    scale: int,
) -> list[tuple[float, float]]:
    cols = (points[:, 0] - origin[0]) / resolution
    rows = height - 1 - (points[:, 1] - origin[1]) / resolution
    return list(zip((cols * scale).tolist(), (rows * scale).tolist()))


def _write_walls(path: Path, walls: tuple[np.ndarray, np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("wall_id", "vertex_id", "x", "y"))
        for wall_id, wall in enumerate(walls):
            writer.writerow((wall_id, "BEGIN", "nan", "nan"))
            for vertex_id, (x, y) in enumerate(wall):
                writer.writerow((wall_id, vertex_id, f"{x:.6f}", f"{y:.6f}"))
            writer.writerow((wall_id, "END", "nan", "nan"))


def main() -> None:
    yaml_path = MAP_DIR / f"{MAP_NAME}.yaml"
    centerline_path = MAP_DIR / f"{MAP_NAME}_centerline.csv"
    metadata = yaml.safe_load(yaml_path.read_text())
    centerline = np.genfromtxt(
        centerline_path,
        delimiter=",",
        skip_header=1,
        usecols=(0, 1),
        dtype=np.float64,
    )
    outer, inner = _closed_offsets(centerline, TRACK_WIDTH_M / 2.0)
    _write_walls(MAP_DIR / f"{MAP_NAME}_walls.csv", (outer, inner))

    source = Image.open(MAP_DIR / f"{MAP_NAME}.pgm")
    width, height = source.size
    source.close()
    scale = SUPERSAMPLE
    canvas = Image.new("L", (width * scale, height * scale), BACKGROUND_GRAY)
    draw = ImageDraw.Draw(canvas)
    origin = (float(metadata["origin"][0]), float(metadata["origin"][1]))
    resolution = float(metadata["resolution"])
    outer_px = _image_points(
        outer, origin=origin, resolution=resolution, height=height, scale=scale
    )
    inner_px = _image_points(
        inner, origin=origin, resolution=resolution, height=height, scale=scale
    )
    draw.polygon(outer_px, fill=ROAD_GRAY)
    draw.polygon(inner_px, fill=BACKGROUND_GRAY)
    draw.line(outer_px + [outer_px[0]], fill=WALL_GRAY, width=WALL_WIDTH_PX * scale)
    draw.line(inner_px + [inner_px[0]], fill=WALL_GRAY, width=WALL_WIDTH_PX * scale)
    image = canvas.resize((width, height), Image.Resampling.LANCZOS)
    image.save(MAP_DIR / f"{MAP_NAME}.pgm")
    image.save(MAP_DIR / f"{MAP_NAME}.png")


if __name__ == "__main__":
    main()
