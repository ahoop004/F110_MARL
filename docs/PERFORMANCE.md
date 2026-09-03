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

## MAPPO update benchmark

P7 uses one synthetic but shape-accurate four-agent rollout for every batch
size. The rollout contains 2,048 steps per agent, and every candidate performs
the configured 10 PPO epochs. Model weights, rollout bytes, shuffle seed, loss
definitions, clipping, and coefficients remain fixed.

```bash
python3 scripts/benchmark_mappo_update.py \
  --device cuda --batch-sizes 64 128 256 512 \
  --n-steps 2048 --n-epochs 10 --repetitions 3 \
  --profile-batch-size 512 \
  --output /tmp/f110_mappo_update_benchmark.json
```

The profiler trace and readable CPU/CUDA tables are written beside the JSON
result. On the Quadro RTX 5000 target, the optimized update measured:

| Batch | Median samples/s | Median time | Peak CUDA MiB |
| ---: | ---: | ---: | ---: |
| 64 | 17,235 | 4.753 s | 38.3 |
| 128 | 36,512 | 2.244 s | 38.3 |
| 256 | 68,459 | 1.197 s | 38.7 |
| 512 | 133,716 | 0.613 s | 39.8 |

The implementation also replaces per-scalar CUDA GAE reads with one bulk copy
per agent, gathers each minibatch from one packed tensor, reuses the optimizer
parameter tuple and agent-identity basis, and transfers aggregate metrics only
once. Against the legacy implementation at batch size 64, the initial study
improved throughput from approximately 13,525 to 15,653 samples/s before any
batch-size change.

Batch size 512 is now the default under update version
`p7-packed-batch512-v1`. It uses the same samples and epochs but reduces the
number of optimizer steps and changes minibatch composition. Existing learning
curves are therefore not seed-trajectory comparable; start new runs when using
this default. AMP and `torch.compile` remain disabled.
