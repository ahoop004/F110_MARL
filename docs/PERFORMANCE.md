# Performance Benchmarking

Use the fixed-work MAPPO benchmark before and after optimization changes. It
runs each repetition in a fresh process with one scenario, map, deterministic
spawn plan, seed, device, and deterministic shared-policy action sequence.
The result includes stage timings, work counts, action hashes, throughput,
peak RSS/CUDA memory, dependency versions, configuration hashes, and git state.

Default three-repetition CPU benchmark:

```bash
PYGLET_HEADLESS=true python3 scripts/benchmark_complete4.py \
  --scenario scenarios/complete_4.yaml \
  --map Budapest_map --seed 42 --physics-substeps 256 \
  --device cpu --repetitions 3 \
  --output /tmp/f110_complete4_benchmark.json
```

Run the same command for these scenario arms:

```text
scenarios/complete_4.yaml
scenarios/complete_4_frenet.yaml
scenarios/complete_4_frenet_neighbors.yaml
```

On a CUDA host, use `--device cuda`. The benchmark synchronizes CUDA around
each measured stage and reports peak allocated CUDA bytes. This synchronization
is intentional for attribution, so compare primary total throughput only with
other runs from this harness.

Optional additional profiled repetition:

```bash
PYGLET_HEADLESS=true python3 scripts/benchmark_complete4.py \
  --scenario scenarios/complete_4.yaml \
  --map Budapest_map --seed 42 --physics-substeps 256 \
  --device cpu --repetitions 3 \
  --profile /tmp/f110_complete4.prof \
  --output /tmp/f110_complete4_profiled.json
```

This writes both the raw `/tmp/f110_complete4.prof` file and a cumulative-time
report at `/tmp/f110_complete4.prof.txt`. The profiled worker is stored as
`profiled_result_excluded_from_summary` and never contributes to primary
throughput statistics.

The report's `fixed_work_verified` and `action_sequence_verified` fields must
both be true before comparing runs. The workload intentionally disables
collision termination and uses an unreachable lap target so every repetition
executes the requested number of physics substeps; this is a throughput
benchmark, not an episode-outcome evaluation.

## Track-preview geometry

Measure preview preprocessing separately from the per-step sampling path:

```bash
python3 scripts/benchmark_track_geometry.py \
  --repetitions 3 --sample-calls 100 \
  --output /tmp/f110_track_geometry_benchmark.json
```

The report separates uncached construction, content-keyed cache lookup, and
one nearest-index-plus-preview sample. Geometry is cached only in memory for
the lifetime of an environment. Disk persistence is intentionally deferred:
the one-time build cost does not currently justify adding a persistent schema
and another source-invalidation boundary.

### Projection allocation review

P6 compared 2,600 Budapest preview calls under `cProfile` and `tracemalloc`.
The baseline rebuilt three closed-track interpolation tails per call: 7,800
`numpy.append` calls consumed 0.109 cumulative seconds. Precomputing those
immutable arrays removed all 7,800 calls; diagnostic runtime changed from
2.509 s to 2.295 s and traced peak memory from 60,583 to 49,967 bytes. These
instrumented numbers identify allocation sources and are not primary
throughput measurements.

Progress projection cannot safely seed or replace preview projection. Progress
uses the original map centerline, while preview uses a uniformly resampled
polyline; map-wide and off-track checks found different segment indices and arc
lengths. The preview-specific nearest-index cursor therefore remains in place.

Numba is already used and warmed in the physics and LiDAR paths, but there is
no existing centerline-projection kernel to reuse. A new JIT kernel was not
introduced because it would add compilation latency and a new numerical
equivalence surface. Revisit that option only with a separately warmed,
map-wide benchmark if projection remains dominant after the allocation change.
