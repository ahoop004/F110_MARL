#!/usr/bin/env python3
"""Audit map geometry in metres and plot local-width versus normal-ray estimates.

Run: PYTHONPATH=src venv/bin/python scripts/audit_map_geometry.py
Plots require the analysis extra (matplotlib). No map assets are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.map_loader import MapLoader
from utils.track_preview import TrackPreviewGeometry, _cross_2d

ROOT = Path(__file__).resolve().parents[1]


def normal_ray_width(points, walls, closed):
    """Previous width estimator, retained here solely for diagnosis/comparison."""
    arrays = [np.asarray(w, dtype=np.float32)[:, :2] for w in walls.values()]
    starts = np.vstack(arrays)
    vectors = np.vstack([np.roll(w, -1, axis=0) - w for w in arrays])
    tangent = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
               if closed else np.gradient(points, axis=0))
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    result = []
    for point, normal in zip(points, normals):
        relative = starts - point
        denom = _cross_2d(normal, vectors)
        valid = abs(denom) > 1e-8
        t = _cross_2d(relative[valid], vectors[valid]) / denom[valid]
        u = _cross_2d(relative[valid], normal) / denom[valid]
        hits = t[(u >= 0) & (u <= 1) & np.isfinite(t)]
        positive, negative = hits[hits > 0], hits[hits < 0]
        width = (positive.min() - negative.max() if len(positive) and len(negative)
                 else 2 * np.linalg.norm(starts - point, axis=1).min())
        result.append(float(width))
    return np.asarray(result)


def audit(name, output, loader):
    data = loader.load(dict(map_dir=str(ROOT / 'maps'), map_bundle=name,
                            centerline_autoload=True, walls_autoload=True))
    geometry = TrackPreviewGeometry.build(data.centerline, data.walls)
    if geometry is None or not data.walls:
        raise ValueError(f'{name}: missing geometry')
    old = normal_ray_width(geometry.points, data.walls, geometry.closed)
    if not all(np.isfinite(a).all() for a in (geometry.points, geometry.width, geometry.curvature)):
        raise ValueError(f'{name}: nonfinite geometry')
    raster = np.flipud(np.asarray(Image.open(data.image_path).convert('L')))
    origin = np.asarray(data.metadata['origin'])
    if abs(origin[2]) > 1e-9:
        raise ValueError(f'{name}: raster audit currently requires zero map rotation')
    resolution = float(data.metadata['resolution'])
    pixels = np.floor((geometry.points - origin[:2]) / resolution).astype(int)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < raster.shape[1]) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < raster.shape[0]))
    occupancy = raster.astype(float) / 255.
    if not data.metadata.get('negate', 0):
        occupancy = 1 - occupancy
    free = np.zeros(len(pixels), dtype=bool)
    free[inside] = occupancy[pixels[inside, 1], pixels[inside, 0]] < data.metadata.get('free_thresh', .196)
    worst = int(np.argmax(old - geometry.width))
    row = dict(map=name, length_m=float(geometry.projection_geometry.total_length),
               resolution_m_per_pixel=resolution, closed=bool(geometry.closed),
               centerline_nonfree_samples=int((~free).sum()), samples=len(free),
               local_width_min_m=float(geometry.width.min()),
               local_width_median_m=float(np.median(geometry.width)),
               local_width_max_m=float(geometry.width.max()),
               normal_ray_width_max_m=float(old.max()),
               maximum_width_change_m=float(np.max(abs(old - geometry.width))),
               abs_curvature_max_per_m=float(abs(geometry.curvature).max()),
               worst_location_m=geometry.points[worst].tolist())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(old)) * geometry.spacing
    axes[0].plot(x, old, label='Previous normal-ray width', alpha=.7)
    axes[0].plot(x, geometry.width, label='Local boundary width')
    axes[0].set(xlabel='Track distance (m)', ylabel='Width (m)', title=name)
    axes[0].legend()
    ax = axes[1]
    extent = [origin[0], origin[0] + raster.shape[1] * resolution,
              origin[1], origin[1] + raster.shape[0] * resolution]
    ax.imshow(raster, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=255)
    for key, wall in data.walls.items():
        wall = np.vstack((wall, wall[0]))
        ax.plot(wall[:, 0], wall[:, 1], linewidth=1, label=f'Wall {key}')
    ax.plot(geometry.points[:, 0], geometry.points[:, 1], color='green', linewidth=1)
    point = geometry.points[worst]
    ax.scatter(*point, color='red', s=30)
    radius = max(5., float(geometry.width[worst]) * 2)
    ax.set(xlim=(point[0] - radius, point[0] + radius),
           ylim=(point[1] - radius, point[1] + radius), aspect='equal',
           title='Largest change: geometry over occupancy image')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f'{name}.png', dpi=150)
    plt.close(fig)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/map_audit')
    args = parser.parse_args()
    names = args.maps or sorted(p.name for p in (ROOT / 'maps').glob('*_map') if p.is_dir())
    args.output.mkdir(parents=True, exist_ok=True)
    loader = MapLoader(base_dir=ROOT)
    results = []
    for name in names:
        row = audit(name, args.output, loader)
        results.append(row)
        print(f"{name}: peak {row['normal_ray_width_max_m']:.3f} -> {row['local_width_max_m']:.3f} m; "
              f"nonfree samples {row['centerline_nonfree_samples']}/{row['samples']}", flush=True)
    (args.output / 'report.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
