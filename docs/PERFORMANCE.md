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
