# Map geometry audit and local track width

The closed racing maps have two enclosing wall contours. The previous width
estimator intersected an infinite centerline-normal ray with all walls. Near
sharp inner hairpin corners that ray can miss the adjacent wall and hit a distant
track section. Width spikes therefore came from estimation, not map-unit scaling.

Track-preview preprocessing version 2 uses the sum of shortest point-to-segment
distances to the two closed boundaries. Duplicate wall vertices are safe. This
is a local lane-width estimate, not necessarily an exact normal cross-section.
It retains physically wider corners without clipping widths to an arbitrary cap.
Other wall layouts retain the previous estimator. Track limits still use a
symmetric half-width about the centerline; this assumes a roughly centred path
and is not a substitute for polygon/vehicle-footprint containment.

Map assets, physical units, observation scales, and input ordering are unchanged.
Corrected widths feed both observations and existing geometric track limits.
Previously trained policies can still load, but see corrected widths near affected
corners; evaluation is appropriate before comparing performance with old runs.

## Audit results

All sampled centerline positions were inside the raster and classified free in
all ten maps. This checks registration along the centerline, not every wall pixel.
The plots overlay walls and centerline on the occupancy image at the largest
width change. Shanghai retains a genuinely wider hairpin visible in that image.

| Map | Previous maximum width (m) | Local maximum width (m) |
|---|---:|---:|
| Budapest_map | 2.801 | 2.780 |
| Hockenheim_map | 5.346 | 2.477 |
| L_map | 1.000 | 1.000 |
| Melbourne_map | 3.475 | 2.749 |
| Montreal_map | 2.194 | 2.128 |
| Shanghai_map | 20.103 | 4.589 |
| Silverstone_map | 2.648 | 2.346 |
| Spa_map | 7.353 | 2.418 |
| Spielberg_map | 19.202 | 2.600 |
| circle_map | 2.705 | 2.700 |

## Reproduce

```bash
PYTHONPATH=src venv/bin/python scripts/audit_map_geometry.py
```

The script writes `outputs/map_audit/report.json` and one PNG per map. Use
`--maps Shanghai_map circle_map` to select maps or `--output PATH` to choose
another destination. It reports widths, curvature, map resolution, raster checks,
and the location of the largest change. Plotting requires matplotlib.
