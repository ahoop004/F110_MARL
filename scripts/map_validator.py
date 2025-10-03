#!/usr/bin/env python3
"""Validate F110 map assets for common consistency issues."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

import numpy as np
import yaml
from PIL import Image


DEFAULT_TRACK_THRESHOLD_DARK = 127
DEFAULT_TRACK_THRESHOLD_LIGHT = 200
DEFAULT_WALL_ADJACENT_RATIO = 0.05


@dataclass
class Issue:
    severity: str
    message: str


@dataclass
class MapReport:
    yaml_path: Path
    image_path: Optional[Path] = None
    issues: List[Issue] = field(default_factory=list)

    def add(self, severity: str, message: str) -> None:
        self.issues.append(Issue(severity=severity, message=message))

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "ERROR" for issue in self.issues)


@dataclass(frozen=True)
class ValidationSettings:
    """Runtime knobs controlling validation sensitivity and whitelisting."""

    wall_adjacent_threshold: float = DEFAULT_WALL_ADJACENT_RATIO
    whitelist: Set[str] = field(default_factory=set)


def _resolve_image_path(yaml_path: Path, metadata: dict) -> Path:
    image_rel = metadata.get("image")
    if image_rel:
        return (yaml_path.parent / str(image_rel)).expanduser().resolve()

    map_ext = metadata.get("map_ext", ".png")
    return yaml_path.with_suffix(map_ext)


def _load_whitelist_file(path: Path) -> Set[str]:
    entries: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.split("#", 1)[0].strip()
            if entry:
                entries.add(entry)
    return entries


def _load_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def _compute_track_mask(gray: np.ndarray, metadata: dict) -> np.ndarray:
    threshold = metadata.get("track_threshold")
    if threshold is None:
        threshold = DEFAULT_TRACK_THRESHOLD_LIGHT if float(gray.mean()) > 127.0 else DEFAULT_TRACK_THRESHOLD_DARK
    invert = bool(metadata.get("track_inverted", False))
    mask = gray >= float(threshold)
    if invert:
        mask = ~mask
    return mask


def _adjacent_background_ratio(track_mask: np.ndarray) -> float:
    if track_mask.size == 0:
        return 0.0

    edge = np.zeros_like(track_mask, dtype=bool)

    # Compare each pixel with its 4-neighbourhood without wrap-around.
    edge[:-1, :] |= track_mask[:-1, :] & ~track_mask[1:, :]
    edge[1:, :] |= track_mask[1:, :] & ~track_mask[:-1, :]
    edge[:, :-1] |= track_mask[:, :-1] & ~track_mask[:, 1:]
    edge[:, 1:] |= track_mask[:, 1:] & ~track_mask[:, :-1]

    adjacent = edge.sum(dtype=np.int64)
    total = track_mask.sum(dtype=np.int64)
    if total == 0:
        return 0.0
    return float(adjacent) / float(total)


def _estimate_wall_presence(gray: np.ndarray, track_mask: np.ndarray, threshold: float) -> bool:
    if gray.size == 0:
        return False

    unique_vals = np.unique(gray)
    if unique_vals.size <= 2:
        return False

    adjacent_ratio = _adjacent_background_ratio(track_mask)
    return adjacent_ratio > threshold


def _validate_origin(report: MapReport, metadata: dict, width: int, height: int) -> None:
    origin = metadata.get("origin")
    if origin is None:
        report.add("ERROR", "Missing 'origin' in YAML metadata")
        return
    if not isinstance(origin, (list, tuple)) or len(origin) < 3:
        report.add("ERROR", "'origin' must be a sequence of length 3")
        return

    try:
        x0, y0, _ = [float(v) for v in origin[:3]]
    except (TypeError, ValueError):
        report.add("ERROR", "'origin' values must be numeric")
        return

    resolution = metadata.get("resolution")
    if resolution is None:
        report.add("ERROR", "Missing 'resolution' in YAML metadata")
        return
    try:
        resolution = float(resolution)
    except (TypeError, ValueError):
        report.add("ERROR", "'resolution' must be numeric")
        return
    if resolution <= 0:
        report.add("ERROR", "'resolution' must be positive")
        return

    width_m = width * resolution
    height_m = height * resolution
    if width_m < 1.0 or height_m < 1.0:
        report.add("WARN", f"Map dimensions are very small ({width_m:.2f}m x {height_m:.2f}m)")

    aspect_ratio_px = width / height if height else math.inf
    world_ratio = abs(width_m / height_m) if height_m else math.inf
    if height and abs(aspect_ratio_px - world_ratio) > 0.05 * aspect_ratio_px:
        report.add("WARN", "Aspect ratio mismatch between pixels and world dimensions")

    if not (-500.0 <= x0 <= 500.0) or not (-500.0 <= y0 <= 500.0):
        report.add("WARN", f"Origin seems far from image bounds (origin=({x0}, {y0}))")


def _validate_thresholds(report: MapReport, metadata: dict) -> None:
    occupied = metadata.get("occupied_thresh")
    free = metadata.get("free_thresh")

    if occupied is None or free is None:
        report.add("WARN", "Missing 'occupied_thresh' or 'free_thresh'; using defaults may be risky")
        return

    try:
        occupied = float(occupied)
        free = float(free)
    except (TypeError, ValueError):
        report.add("ERROR", "Threshold values must be numeric")
        return

    if not (0.0 <= free < occupied <= 1.0):
        report.add("ERROR", "Thresholds must satisfy 0 <= free < occupied <= 1")


def _check_track_distribution(report: MapReport, track_mask: np.ndarray) -> None:
    coverage = float(track_mask.mean()) if track_mask.size else 0.0
    if coverage == 0.0:
        report.add("ERROR", "Track threshold produced zero track pixels")
    elif coverage < 0.01:
        report.add("WARN", f"Track coverage very low ({coverage:.3%})")
    elif coverage > 0.95:
        report.add("WARN", f"Track coverage very high ({coverage:.3%}); threshold may be incorrect")


def _check_wall_presence(
    report: MapReport,
    gray: np.ndarray,
    track_mask: np.ndarray,
    *,
    threshold: float,
) -> None:
    has_walls = _estimate_wall_presence(gray, track_mask, threshold)
    if not has_walls:
        report.add("WARN", "No distinct wall/background band detected; map may only contain track vs out-of-bounds")


def _check_start_metadata(report: MapReport, metadata: dict) -> None:
    start_keys = [key for key in metadata.keys() if "start" in key.lower()]
    if not start_keys:
        report.add("INFO", "No start/finish metadata found in map YAML (expected in env config)")


def _apply_whitelist(report: MapReport, whitelist: Set[str]) -> None:
    if not whitelist:
        return

    name = report.yaml_path.name
    stem = report.yaml_path.stem
    if name not in whitelist and stem not in whitelist:
        return

    adjusted: List[Issue] = []
    for issue in report.issues:
        if issue.severity == "WARN":
            adjusted.append(Issue(severity="INFO", message=f"{issue.message} [whitelisted]"))
        else:
            adjusted.append(issue)
    report.issues = adjusted


def validate_map(yaml_path: Path, settings: ValidationSettings) -> MapReport:
    report = MapReport(yaml_path=yaml_path)

    try:
        metadata = _load_yaml(yaml_path)
    except Exception as exc:  # pragma: no cover - defensive
        report.add("ERROR", f"Failed to parse YAML: {exc}")
        return report

    try:
        image_path = _resolve_image_path(yaml_path, metadata)
    except Exception as exc:
        report.add("ERROR", f"Unable to resolve image path: {exc}")
        return report

    report.image_path = image_path
    if not image_path.exists():
        report.add("ERROR", f"Image file not found: {image_path}")
        return report

    try:
        with Image.open(image_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
            width, height = img.size
    except Exception as exc:
        report.add("ERROR", f"Failed to load image: {exc}")
        return report

    _validate_origin(report, metadata, width, height)
    _validate_thresholds(report, metadata)

    track_mask = _compute_track_mask(gray, metadata)
    _check_track_distribution(report, track_mask)
    _check_wall_presence(report, gray, track_mask, threshold=settings.wall_adjacent_threshold)
    _check_start_metadata(report, metadata)

    _apply_whitelist(report, settings.whitelist)

    return report


def iter_map_yaml_files(map_dir: Path) -> Iterable[Path]:
    for path in sorted(map_dir.rglob("*.yaml")):
        if path.is_file() and path.suffix.lower() == ".yaml":
            yield path


def format_report(report: MapReport) -> str:
    header = f"Map: {report.yaml_path.name}"
    lines = [header]
    for issue in report.issues:
        lines.append(f"  [{issue.severity}] {issue.message}")
    if not report.issues:
        lines.append("  [OK] No issues detected")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate map YAML/image pairs for the F110 environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "map_dir",
        type=Path,
        help="Directory containing map YAML files",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Return a non-zero exit code if any warnings are found",
    )
    parser.add_argument(
        "--wall-threshold",
        type=float,
        default=DEFAULT_WALL_ADJACENT_RATIO,
        help="Adjacent background ratio required to emit the wall-band warning",
    )
    parser.add_argument(
        "--whitelist",
        action="append",
        default=[],
        metavar="NAME",
        help="Map filename or stem whose warnings should be downgraded to info (repeatable)",
    )
    parser.add_argument(
        "--whitelist-file",
        type=Path,
        help="Path to file containing newline-separated map names to whitelist",
    )

    args = parser.parse_args(argv)
    map_dir = args.map_dir.expanduser().resolve()

    if not map_dir.is_dir():
        parser.error(f"Map directory not found: {map_dir}")

    whitelist: Set[str] = set(args.whitelist or [])
    if args.whitelist_file:
        if not args.whitelist_file.exists():
            parser.error(f"Whitelist file not found: {args.whitelist_file}")
        whitelist.update(_load_whitelist_file(args.whitelist_file))

    settings = ValidationSettings(
        wall_adjacent_threshold=float(args.wall_threshold),
        whitelist={entry.strip() for entry in whitelist if entry.strip()},
    )

    reports = [validate_map(path, settings) for path in iter_map_yaml_files(map_dir)]

    any_errors = False
    any_warnings = False
    for report in reports:
        print(format_report(report))
        print()
        if report.has_errors:
            any_errors = True
        if any(issue.severity == "WARN" for issue in report.issues):
            any_warnings = True

    if not reports:
        print(f"No YAML maps found in {map_dir}")
        return 1

    if any_errors:
        return 2
    if args.fail_on_warn and any_warnings:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
