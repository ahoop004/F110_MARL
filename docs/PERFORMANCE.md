# Performance Benchmarking

Use the fixed-work MAPPO benchmark before and after optimization changes. It
runs each repetition in a fresh process with one scenario, map, deterministic
spawn plan, seed, device, and deterministic shared-policy action sequence.
The result includes stage timings, work counts, action hashes, throughput,
peak RSS/CUDA memory, dependency versions, configuration hashes, and git state.

Default three-repetition CPU benchmark:

```bash
PYGLET_HEADLESS=true python3 scripts/benchmark_complete4.py \
  --scenario scenarios/legacy/complete_4.yaml \
  --map Budapest_map --seed 42 --physics-substeps 256 \
  --device cpu --repetitions 3 \
  --output /tmp/f110_complete4_benchmark.json
```

Run the same command for these scenario arms:

```text
scenarios/legacy/complete_4.yaml
scenarios/legacy/complete_4_frenet.yaml
scenarios/legacy/complete_4_frenet_neighbors.yaml
```

On a CUDA host, use `--device cuda`. The benchmark synchronizes CUDA around
each measured stage and reports peak allocated CUDA bytes. This synchronization
is intentional for attribution, so compare primary total throughput only with
other runs from this harness.

Optional additional profiled repetition:

```bash
PYGLET_HEADLESS=true python3 scripts/benchmark_complete4.py \
  --scenario scenarios/legacy/complete_4.yaml \
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


## PPO/MAPPO cleanup measurements

The September 2026 cleanup shares one Python-double GAE recurrence in
`agents.common` on CPU and CUDA. For 2,048 seeded transitions, five warmed
measurements against revision `cfd19fb` gave median PPO GAE times of 42.07 to
2.56 ms on CPU and 217.89 to 2.62 ms on a Quadro RTX 5000 (PyTorch 2.8.0).
Advantages and returns were exactly equal, including terminal and truncation
boundaries. These are isolated GAE timings, not full-training speedups.

Waypoint controllers cache the original closure and curvature calculations,
checking centerline contents to detect replacements and in-place edits. Three
full Budapest traversals (2,016 actions each) measured median per-action times
of 559.7 to 101.1 microseconds for pure pursuit and 397.2 to 76.9 microseconds
for Stanley. Action arrays matched the previous implementation exactly at every
waypoint on Budapest, Silverstone, and circle maps. Closure thresholds, curvature
horizons, and nearest-point cursors retain their existing behavior.

The environment now copies observation state vectors from its immutable global
snapshot after lifecycle updates. This removes the second centralized-vector
construction while keeping observations writable and snapshot masks current.
For 128 physics steps plus reset, profiling confirmed 258 to 129 vector
constructions with the original MAPPO action sequence and workload preserved.
Regression coverage includes time-limit termination and observation mutation.
No reward, action, observation dimension, seed, or optimizer settings changed.


## Headless PPO on HPC

PPO supports opt-in synchronous CPU collectors through the existing training
entry point. Start with one GPU and eight environment workers, allocating at
least nine CPU cores for the workers and learner:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYGLET_HEADLESS=true python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --num-envs 8 --torch-threads 1 --no-render --seed 42
```

Use the cluster's Python environment with CUDA-enabled PyTorch. The scheduler
should assign GPU visibility; a run uses one visible GPU. Run other seeds in
separate GPU allocations. W&B remains optional via `--no-wandb`. The equivalent
scenario settings are `experiment.num_envs` and `experiment.torch_threads`.
Defaults preserve one-environment training and the existing PyTorch thread count.

Each spawned process owns its environment, opponents, composers, and CPU rollout
buffer and reuses `OnPolicyTrainer.train`. Only the parent owns the policy,
optimizer, CUDA context, and output hooks. It batches inference requests, waits
for every live collector to finish its fragment, then performs one pooled PPO
update. No transitions are collected with stale weights during an update.
Workers stop at episode ends or their fragment capacity; shorter episodes can
leave workers waiting at the update barrier.

`n_steps` is the maximum pooled capacity: 2,048 with eight workers means at most
256 decisions per worker per update. It must be divisible by `num_envs`.
`experiment.episodes` is the exact total episode count, divided across workers,
and must be at least `num_envs`. Final worker quotas and early episode endings
can produce smaller updates. GAE is calculated per fragment using the final
observation for truncation bootstrap before pooling and advantage normalization.
The actor still uses local observations. Rewards, action repeat, action bounds,
network dimensions, minibatch size, and epochs retain their configured values.

Worker RNG seeds are `(experiment.seed + worker_id) % 2**32`; an explicit
environment seed is offset separately in the same way. Work is processed in
worker-ID order to keep scheduling from changing action/event order. Checkpoint
and dataset provenance records the worker count, seed lists, and fragment size;
dataset episode IDs include the worker ID. Evaluation and checkpoint selection
remain in the parent, at the configured total completed-episode cadence.

Parallel collection changes sampling order, rollout horizons, and pooled
advantage normalization. Compare learning quality on new runs; do not expect the
single-environment learning trajectory. Rendering and curriculum remain
single-environment paths. MAPPO supports grouped parallel collectors; see
[Parallel MAPPO](PARALLEL_MAPPO.md) for the 400-environment configuration and
its separate per-environment rollout horizon. Checkpoint actors
remain loadable for evaluation and MAPPO initialization. For `--eval`, retain the
training collection CLI settings when checking provenance; evaluation itself is
serial. Workers report errors to the parent and are reaped on failure.

Benchmark 1, 4, 8, and 16 environments on the target node. Record decisions/s,
physics steps/s, update time, evaluation time, and wall time including startup;
allow for process imports and Numba warmup. Very short runs can be slower with
workers. No L40 throughput claim is made from the local smoke tests. Larger PPO
minibatches (256/512) are a separate experiment because they change optimization.


A preliminary local comparison used the pretraining actor on circle_map with
8,192 decisions, four updates, 2,048 pooled steps, 10 epochs, and minibatches of
64 in each run. Collision termination was disabled and the lap target raised to
fix the work count; evaluation and logging were excluded. On a Quadro RTX 5000:

| Collectors | Training-call wall time, including worker startup | Decisions/s after first update |
| ---: | ---: | ---: |
| 1 | 34.15 s | 360.6 |
| 4 | 15.50 s | 806.5 |

These are single measurements with different worker seeds and trajectories, not
learning-equivalence or L40 scaling results. Parent environment setup is excluded
from both wall times. Measure longer runs on the target node before selecting a
worker count. CSV/W&B episode rows carry worker identity; W&B component totals
are kept separately for interleaved worker episodes.

Standard W&B logging aggregates metrics inside each PPO worker and sends one
summary per episode. Full transition records still cross the process boundary
when dataset or custom step hooks require them. This removes per-decision W&B
record serialization, but workers still construct records for local aggregation.
PPO checkpoint-selection and standalone evaluation use actor-only deterministic
inference; training still computes both actions and critic values. These changes
preserve actions, metrics, and dataset records; measure throughput on the target
node with the intended logging settings.

A local evaluation microbenchmark (115 inputs, `[256, 256]` MLP, one CPU thread,
100 warmup calls, median of three 2,000-call repetitions) measured 341 → 134 µs
per action on CPU and 577 → 313 µs on a Quadro RTX 5000 when replacing full
deterministic `act()` with `predict()`. This includes the NumPy result transfer,
but excludes environment stepping and is not an end-to-end or L40 speed claim.

## Combined-slip environment review (September 2026)

An environment-only review used the current
`ppo_lap_completion_pretrain_combined_slip.yaml`: 0.01 s physics steps, 0.001 s
RK4 integration substeps, 108 LiDAR beams, and 20 track-preview points. A separate
four-car scaling workload copied the same vehicle/sensor configuration into one
environment; it did not run a multiple-PPO trainer. Fixed physical commands,
named spawn plans, and seed 42 were reused. Original collision/termination logic
remained enabled, with deterministic resets when the joint episode ended.

Local hardware: Intel Xeon Silver 4214, Python 3.12, one Numba/BLAS thread.
Each arm warmed 32 steps, then measured three repetitions of 1,024 joint steps
in the same process. Separate trajectory-capture and cProfile repetitions were
excluded from the throughput numbers. Timings below exclude reset, setup/JIT,
policy inference, wrapper composition, PPO updates, rendering, and logging.

| Workload | Current env-step time | Compiled RK4 prototype | Speedup |
| --- | ---: | ---: | ---: |
| circle_map, one car | 1.169 s | 0.772 s | 1.51× |
| circle_map, four cars | 4.145 s | 2.424 s | 1.71× |
| Budapest_map, one car | 1.170 s | 0.761 s | 1.54× |

The prototype compiled the complete `CombinedSlipVehicle.advance` numerical
loop and actuator sampling together, using the existing force/actuator kernels
and no fastmath. It was injected only into benchmark processes; production
physics was not edited. Physics-state, LiDAR, global-state-vector, and
termination traces were bitwise identical across the baseline/prototype on all
three workloads. The prototype also passed the 76 existing combined-slip
physics/environment and friction-protocol tests. This is evidence for a narrow
optimization candidate, not full learning equivalence or an HPC speed claim.

A separate integration-only probe compared a compiled serial batch against
`prange` with four Numba threads. It used heterogeneous states from seed 42,
30 warmup calls, and five repetitions of 200 calls per batch size. Compilation
was excluded, and outputs were bitwise identical to the serial batch.

| Vehicles in batch | Serial kernel | Four-thread kernel | Kernel speedup |
| ---: | ---: | ---: | ---: |
| 1 | 34.0 µs | 43.3 µs | 0.79× |
| 4 | 132.8 µs | 52.3 µs | 2.54× |
| 8 | 262.8 µs | 86.3 µs | 3.04× |
| 32 | 1062.6 µs | 338.3 µs | 3.14× |

These are integration-only measurements, not additional end-to-end speedups.
One-car threading lost to scheduling overhead. Larger batches provide independent
work, but geometry/snapshots/scans remain outside this kernel. When combining
Numba threading with environment processes, budget CPU cores across both layers
to avoid oversubscription. The parallel prototype uses numeric parameter arrays;
passing the existing nested rate-limit tuples directly failed Numba's parallel
wrapper compilation, so a production batch kernel needs an explicit packed input
contract. `batch_probe.py` and its JSON/log are beside the environment profiles.

Priorities from the baseline cProfile runs (inclusive timings overlap):

1. **Compile the full RK4 loop first.** `CombinedSlipVehicle.advance` accounted
   for 31.7% of one-car and 38.5% of four-car profiled env time. Each car-step
   currently invokes 32 Python actuator samplings and 41 dynamics evaluations.
   RK4 stages and successive substeps are sequential dependencies; parallelize
   independent cars only after removing Python orchestration overhead.
2. **Compile/batch centerline projection.** `project_to_centerline` accounted
   for 14.1% / 15.9%. Progress and preview use different geometries and must keep
   independent indices. Fuse the small search/projection array operations and
   batch agents over each geometry, preserving candidate order, tie-breaking,
   float precision, and seam behavior. Preview also performs a separate nearest
   sample search; changing it requires the existing projection-equivalence tests.
3. **Cache immutable episode metadata.** Recursive global-state metadata freezing
   consumed 10.9% / 3.8%; copying physics metadata into public infos consumed
   another 4.1% / 4.2%. Cache the already-frozen episode metadata at reset/map or
   friction changes, while preserving independent mutable public info payloads
   and immutable historical snapshots. Avoid copying LiDAR arrays through
   `Simulator.current_observation` solely to construct the centralized vector.
4. **Batch scans and opponent ray casting after those changes.** Scanning was
   about 4.9% in both baselines; opponent ray casting added 5.0% with four cars.
   These numerical kernels already use Numba. Batched car/scan rows are a better
   candidate than launching threads separately for just 108 beams. Preserve
   per-car noise streams and the current float beam-index recurrence. Pairwise
   collision writes need deterministic reduction if parallelized; several pairs
   can write the same car's collision index. The `opp_poses` packing loop has no
   reader in the current source and is a smaller deletion candidate.
5. **Parallel evaluation and worker transport are separate candidates.** Fixed
   evaluation episodes can use isolated environments and a frozen policy, with
   deterministic result ordering. Training currently exchanges observations and
   actions through pipes every decision. Shared-memory buffers could reduce that
   transport cost without changing fragment boundaries or policy-update barriers.
   Neither opportunity was timed by this environment-only benchmark.

Raw profiles, trace arrays, JSON timings, prototype test output, and the portable
benchmark harness are in `/tmp/f110_env_review` for this review session. Run the
harness from the repository root after copying it to the target node:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYGLET_HEADLESS=true venv/bin/python /tmp/f110_env_review/benchmark.py \
  --agents 1 --map circle_map --steps 1024 --repetitions 3
# Repeat with --compiled, and separately with --agents 4.
```

### Implemented environment optimizations

The measured environment work is now implemented in the active path:

- `physics.dynamic_models.integrate_combined_slip` compiles the complete RK4
  loop and exact held-reference actuator sampling. `CombinedSlipVehicle.advance`
  validates the timestep and commits both subsystems only after successful
  integration and final validation. Stage order and arithmetic are retained;
  fastmath and additional worker threads are disabled.
- `utils.centerline` and `utils.track_preview` compile the closest-segment and
  nearest-sample searches. They avoid temporary candidate/gather arrays while
  retaining float32 arithmetic, first-candidate ties, local windows, and seam
  wrapping. Progress and preview retain their separate geometries and cursors.
- `env.state_views.FrozenSnapshotMapping` detaches metadata once and permits
  sharing it between immutable global-state snapshots. The environment drops
  this cache on reset and map changes, so newly sampled friction and spawn
  metadata enter the next snapshot. Public info dictionaries still get separate
  mutable physics metadata; a schema-specific copy replaces generic recursive
  deepcopy, including an independent evaluation-grid list.
- `Simulator.current_observation(include_scans=False)` lets central-state
  construction omit the LiDAR copy. Its default still returns the full copied
  observation.

Final measurements used the same baseline harness, seeds, resolved configurations,
actions, warmup, and three unprofiled repetitions described above. Times are
median wall seconds for 1,024 joint environment steps, excluding reset, JIT,
policy inference, PPO updates, logging, and rendering:

| Workload | Before | Implemented | Speedup |
| --- | ---: | ---: | ---: |
| Circle, one car | 1.169 s | 0.555 s | 2.11× |
| Circle, four cars | 4.145 s | 1.830 s | 2.27× |
| Budapest, one car | 1.170 s | 0.566 s | 2.07× |

Physics states, scans, central vectors, and termination traces were **bitwise
identical** to the saved pre-optimization baseline for every workload. Regression
tests also compare compiled integration to the previous Python RK4 driver across
three timestep lengths and three grip values; compare projection/search results
to NumPy references; and check metadata mutation isolation and cache refresh.
No reward, observation shape, action bound, rollout horizon, or learning
hyperparameter was changed by this optimization pass. This evidence covers the
tested numerical workloads, not learning convergence or an HPC throughput claim.

Final timing JSON, traces, and cProfile output are in
`/tmp/f110_env_implemented_final`; `comparison.json` records exact trace and
configuration checks against `/tmp/f110_env_review`. To measure production code
with that harness, **omit `--compiled`**: that switch installs the old integration
prototype. The retained `_baseline` filename suffix identifies the harness arm,
not the implementation being measured.

Parallel evaluation, shared-memory PPO transport, and batched sensor/raycast
execution remain follow-up candidates. Profile the complete HPC training job
with these environment changes before choosing the next optimization.

Validation for this implementation (headless, with OMP/MKL/OpenBLAS threads
limited to one):

```bash
venv/bin/python -m compileall -q run.py src tests
PYGLET_HEADLESS=true venv/bin/python -m pytest tests/ -q
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/legacy/ppo.yaml --no-wandb --episodes 1 --quiet --output-dir /tmp/f110_optimization_smoke_ppo
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/legacy/mappo_gaplock.yaml --no-wandb --episodes 1 --quiet --output-dir /tmp/f110_optimization_smoke_mappo
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml --no-wandb --episodes 1 --quiet --output-dir /tmp/f110_optimization_smoke_slip
rg 'stable_baselines3|from gymnasium|from pettingzoo' run.py src configs scenarios
git diff --check
```

Compilation, all three smoke runs, and whitespace checks passed; the dependency
guard found no matches (normal `rg` exit code 1). The focused suite passed all
107 tests. The full suite finished with **570 passed, one failed**:
`tests/test_ppo_correctness.py::test_transfer_scenario_preserves_pretraining_contract`
still compares the transfer scenario to Frenet pretraining, while the existing
user-edited scenario inherits combined-slip pretraining. This failure was already
observed before the performance implementation and is unrelated to it.
