# 1v1 / racing MPC performance review

Reviewed on 2026-09-28 against `fdca65345e2b0f78093d291c6ba641aa8d6c321f`.
The local machine has a Xeon Silver 4214 (12 physical cores, 24 logical CPUs)
and Quadro RTX 5000. The intended HPC allocation is 128 cores.
Raw timings, commands, versions and action/state hashes are saved in
[the benchmark evidence](benchmarks/1v1_mpc_performance_review.json).

## Findings and changes

1. **MPC dominates steady collection.** Each decision solves a 12-variable
   finite-difference optimization over 30 prediction steps. Traffic can trigger
   three optimization starts instead of one. Local warmed controller means were
   about 32–34 ms, with 79–82 ms p95 in the 200-step probes. These spikes exceed
   the 50 ms simulation interval and can interrupt smooth desktop playback.
   The plant, observations and environment bookkeeping took about 2.6–2.9 ms.
   The optimizer's `max_evaluations: 260` is per start, not a hard total budget
   for the whole decision; candidate and braking checks add further work.

2. **Reduced temporary arrays in MPC prediction.** Tire-force derivatives now
   return scalar tuples, and midpoint integration reuses one buffer across its
   substeps. The solver settings, integration substeps, footprint samples,
   candidate selection, rewards and policy inputs remain the same. Three
   alternating before/after probes showed the following medians:

   | Scenario | MPC before | MPC after | Controller + environment before | After |
   | --- | ---: | ---: | ---: | ---: |
   | MAPPO 1v1 LoRA attack | 33.73 ms | 31.81 ms | 27.38 steps/s | 28.82 steps/s |
   | PPO 1v1 racing MPC | 32.01 ms | 30.04 ms | 28.89 steps/s | 30.64 steps/s |

   These are modest, approximately 5–6% throughput gains. Each probe uses seed
   42, 20 warmup decisions and 200 timed decisions, with a stationary learner.
   All action and simulator-pose hashes match across each scenario's six runs.
   These measurements isolate the prediction change, before distance-field
   sharing was added. They exclude learner inference, learning, evaluation,
   startup and resets; local desktop activity and some concurrent validation
   also limit timing precision.

3. **Share live MPC map distance fields within each worker.** Every active map
   here is 2000 × 2000 pixels, so a float64 field occupies 30.5 MiB per MPC.
   Previously each controller independently loaded the image and computed two
   distance transforms. Controllers now share an immutable field when the map
   file/version, resolution, occupancy convention and threshold match. Weak
   references release unused maps instead of accumulating an eight-map cache
   in every worker. Four environments on the same map can save about 91.6 MiB
   per worker in MPC fields alone, or about 8.9 GiB across 100 workers. Savings
   decrease when the environments use different maps. A cold map still needs
   its distance transform; one initial local MPC map build took about 0.5 s.
   A four-controller probe reduced map-build time from 1.97 s to 0.50 s and
   unique field storage from 128 MB to 32 MB. Subsequent shared-field lookups
   including controller geometry setup took less than 1 ms each.

4. **Remove the fixed 5 ms delay in desktop rendering.** A warmed 100-step
   `--render` probe after this change measured 0.59 ms mean / 0.77 ms p95 in
   `env.render`, and about 35 displayed joint steps/s overall. That probe
   excludes policy inference and updates. Training updates, map changes and
   first-use compilation still pause the synchronous display loop. The simulation
   advances 0.05 s per decision; this renderer has no interpolation between
   decisions or explicit real-time pacing.

5. **Accelerate RGB capture.** Pyglet's RGBA-to-RGB conversion used a per-pixel
   regex and took about 470 ms/frame at 1000 × 800 locally. Reading native RGBA,
   then dropping alpha and flipping rows with NumPy reduced the same static
   capture probe to about 8 ms/frame. This affects `rgb_array` capture, not
   ordinary `--render` or headless training. Pixel order and ownership are tested.

6. **Set process count and rollout size explicitly on HPC.** Both collectors
   default to one worker per environment. `--num-envs 400` alone therefore
   launches 400 processes on 128 cores. Grouped workers already support several
   environments per process. PPO's active `n_steps: 1024` also does not divide
   evenly across 400 environments; use `--rollout-steps-per-env` to set a valid
   pooled batch. MAPPO already has a parallel horizon of 256, independently of
   its serial 2048-step buffer.

## Running on the 128-core allocation

A starting layout is 400 environments in 100 worker processes (four per worker),
with one parent learner. Benchmark readiness scheduling against synchronous
scheduling on the allocated node. Readiness scheduling can avoid waiting on an
expensive MPC solve at every action barrier, but all environments still meet at
the policy-update barrier. Arrival order changes stochastic policy sampling.

```bash
python3 -u run.py --scenario scenarios/mappo_1v1_attack_lora.yaml \
  --num-envs 400 --num-workers 100 --torch-threads 1 \
  --collector-scheduling ready --rollout-steps-per-env 256 --no-render
```

Use the same collection settings for the full-actor arm. The PPO scenario can
use these flags too; its pooled rollout then becomes 102400 transitions, which
changes update frequency relative to its serial configuration. Process count
alone is not an optimizer change. Do not shorten the MPC horizon, reduce solver
budgets, increase action repeat or coarsen physics integration merely to match a
throughput target: those changes need separate controller/learning validation.

Compare two balanced process layouts at the same work budget:

```bash
python3 scripts/benchmark_collectors.py \
  --scenario scenarios/mappo_1v1_attack_lora.yaml \
  --num-envs 400 --workers 80 100 --scheduling synchronous ready \
  --rollout-steps-per-env 256 --total-steps 307200 --repetitions 3 \
  --set evaluation.enabled=false --output-dir /tmp/1v1_hpc_benchmark
```

This keeps 400 environments and 256 decisions per environment in every rollout,
with three full collection/update rounds per run. The harness excludes the first
round from steady timings and disables W&B. Evaluation is explicitly disabled
for this throughput probe; retain the scenario's evaluation protocol in training.
Compare round steps/s, whole-process wall time, startup time and peak memory,
not GPU utilization alone. Reserve CPU and memory for the parent and measure on
the actual allocation before choosing the final process count. On a shared
filesystem, a node-local `NUMBA_CACHE_DIR` can avoid repeatedly reading JIT cache
files over the network; warm it using the same Python environment on the node.

The local end-to-end LoRA smoke benchmark used eight environments, 64 decisions
per environment per rollout, three updates (1536 joint decisions), default
optimizer settings and the real pretrained actor. All four runs completed:

| Workers | Scheduler | Collection + update steps/s | Whole-process seconds |
| --- | --- | ---: | ---: |
| 4 | synchronous | 55.0 | 51.6 |
| 4 | ready | 79.8 | 37.2 |
| 8 | synchronous | 75.1 | 36.5 |
| 8 | ready | 112.1 | 28.3 |

These are one-repetition local checks, with some concurrent validation during
the first runs, not a scaling prediction for 128 cores. Startup and the first
round are excluded from the rate column. Scheduling changes trajectories, so
these are equal work budgets, not identical races or a learning-quality comparison.

## Reproducing the local controller/render split

The existing benchmark now accepts scenario overrides and desktop rendering,
reports setup/reset/wall time separately, and defaults to the current PPO file
`scenarios/ppo_1v1_racing_mpc.yaml`.

```bash
PYGLET_HEADLESS=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  venv/bin/python scripts/benchmark_mpc_traffic.py \
  --scenario scenarios/mappo_1v1_attack_lora.yaml --warmup 20 --steps 200

env -u PYGLET_HEADLESS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  venv/bin/python scripts/benchmark_mpc_traffic.py \
  --scenario scenarios/mappo_1v1_attack_lora.yaml --warmup 20 --steps 200 --render
```

These park the learner and exercise the real fixed controller and environment.
Use the collector benchmark for policy inference, observations, reward processing,
IPC and optimizer costs. Changing the map lists via `--set` permits the same
probe on other tracks. Report map, seed, warmup and step count with results.

## Validation limitations already present in the repository

The broader focused suite initially reported 54 passes, three failures and five
setup errors. Six of the unsuccessful cases reference the absent
`scenarios/ppo_1v1_racing_mpc_circle.yaml`; the current filename omits `_circle`.
The other two are configuration assertions in `test_racing_mpc.py`: they expect
every 2v2 opponent profile to match the shared config and specify 5 m/s, while
`mappo_2v2_asymmetric.yaml` currently specifies an MPC `max_speed` of 20. The
controller clamps this to the shared physical limit at setup. These scenario
files and existing assertions were not changed by the performance work.

After the changes, 59 focused cases passed: MPC physics/trajectory/traffic tests,
attack transfer/training/evaluation, map-cache lifetime/invalidation, pixel layout
and four midpoint-reference cases. The two unrelated 2v2 configuration assertions
were excluded from that run; the stale-file pursuit suite was not rerun.
Final comparisons including distance-field sharing preserved every action and
simulator-pose hash for both PPO and LoRA on Budapest and circle maps.

During the review, a separate workspace edit changed the LoRA training
`max_steps` from 1200 to 12000. The performance changes preserve that edit.
It occurred after the above test and collector runs; it breaks the existing
full/LoRA matched-scenario assertion because the full arm still uses 1200.
That single assertion was rerun and confirmed failing. The final controller
comparisons stay below either timeout.
