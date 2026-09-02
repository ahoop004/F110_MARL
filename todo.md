# F110 MARL Performance Optimization Plan

Completed engineering history lives in `done.md`. This file is intentionally
limited to active performance work for the pure-PyTorch training path.

## Objective

Increase training throughput for the four-agent MAPPO scenarios without
changing environment dynamics, rewards, observations, action bounds, CTDE
behavior, lifecycle semantics, seeds, or experiment outcomes.

Primary scenarios:

```text
scenarios/complete_4.yaml
scenarios/complete_4_frenet.yaml
scenarios/complete_4_frenet_neighbors.yaml
```

The optimizations must also preserve the single-agent PPO and off-policy paths.

## Measured Baseline

Profile date: 2026-09-02

Workload:

```bash
PYGLET_HEADLESS=true F110_LOG_EVERY=100000 F110_SUMMARY_EVERY=100000 \
  /usr/bin/time -f 'wall=%e user=%U sys=%S maxrss_kb=%M' \
  venv/bin/python run.py \
  --scenario scenarios/complete_4.yaml \
  --no-wandb --no-render --episodes 1 --quiet \
  --output-dir /tmp/f110_profile_complete4_wall
```

Observed result:

```text
device:                    Quadro RTX 5000
observation dimension:    115
global-state dimension:   48
policy decisions:         4,497
physics substeps:         8,993
PPO updates:              3
wall time:                59.02 s
peak RSS:                 1,834,940 KiB
decision throughput:      ~76 decisions/s
physics throughput:       ~152 substeps/s
```

Deterministic `cProfile` run:

```bash
PYGLET_HEADLESS=true F110_LOG_EVERY=100000 F110_SUMMARY_EVERY=100000 \
  venv/bin/python -m cProfile -o /tmp/f110_complete4.prof run.py \
  --scenario scenarios/complete_4.yaml \
  --no-wandb --no-render --episodes 1 --quiet \
  --output-dir /tmp/f110_profile_complete4
```

The profiled run took 70.74 s. Important cumulative costs were:

```text
F110ParallelEnv.step                 28.86 s / 8,993 calls
Simulator.step                      10.38 s / 8,993 calls
MAPPOAgent.update                    9.58 s / 3 calls
MAPPO actor action selection         9.26 s / 11,590 calls
centerline projection                7.87 s / 71,960 calls
track-preview injection              7.53 s / 8,995 calls
track-preview construction           4.19 s / 2 calls
get_global_state                     4.09 s / 48,258 calls
reward composition                   0.81 s / 23,176 calls
observation composition              0.41 s / 11,594 calls
```

Cumulative profiler times overlap. Use them to prioritize work, not to predict
the exact wall-time reduction from adding percentages.

## Non-Negotiable Correctness Gates

Every optimization must preserve:

- `run.py` as the single training entry point.
- The `reset`, `step`, `get_global_state`, and `get_agent_state` contracts.
- Shared MAPPO actor with local observations only.
- Centralized, agent-conditioned critic behavior for `complete_4`.
- One reward composer and factual reward stream per trainable agent.
- One transition per active agent decision, with no post-terminal records.
- Separate termination and truncation handling for GAE.
- Existing observation dimensions and numerical values for each scenario.
- Existing reward totals and component breakdowns under a fixed trajectory.
- Existing map, spawn, lap, collision, and terminal-vehicle behavior.
- Explicit seeds and reproducible scenario expansion.

Do not improve speed by reducing LiDAR beams, preview points, network size,
rollout length, PPO epochs, map set, agent count, or physics fidelity. Those are
experiment changes, not implementation optimizations.

---

## P0 - Build a Repeatable Benchmark Harness

**Goal:** compare changes on identical work rather than variable-length random
episodes.

- [x] Add a benchmark script under `scripts/` or a focused benchmark test that
  runs a fixed number of MAPPO decisions and physics substeps.
- [x] Use an explicit scenario, map, spawn plan, seed, action sequence, device,
  warm-up period, and measured interval.
- [x] Separate timings for:

  ```text
  environment reset/map setup
  policy action selection
  centralized value estimation
  physics/environment stepping
  reward-context assembly
  observation composition
  rollout storage
  PPO update
  total wall time
  ```

- [x] Report decisions/s, physics substeps/s, update samples/s, peak RSS, and
  peak CUDA memory.
- [x] Support all three `complete_4` observation variants.
- [x] Run at least three measured repetitions and report median plus spread.
- [x] Store benchmark metadata alongside results: commit, Python, NumPy,
  PyTorch, CUDA, GPU, CPU, scenario hash, and resolved config hash.
- [x] Keep profiling optional so instrumentation overhead is excluded from the
  primary throughput number.
- [x] Add a documented command for generating `cProfile` output and a readable
  top-function report.

Exit criteria:

- [ ] Repeated unchanged runs have sufficiently low variance to detect a 5%
  throughput change.
- [x] Baseline and optimized runs execute the same number of agent decisions,
  environment substeps, transitions, and optimizer samples.

---

## P1 - Gate Unused Frenet Preview and Neighbor Work

**Why first:** `complete_4.yaml` pays for track-preview and relative-neighbor
construction despite using only LiDAR, ego state, progress, and previous action.
Track-preview work consumed 7.53 profiled seconds per episode, plus 4.19 seconds
of geometry construction across initial map setup and reset.

Primary files:

```text
run.py
src/core/env_builder.py
src/env/f110ParallelEnv.py
src/utils/track_preview.py
src/wrappers/observations/composer.py
src/wrappers/observations/frenet_vehicle_track.py
src/wrappers/observations/frenet_neighbors.py
```

- [x] Derive explicit environment feature requirements from every active
  observation and reward config during setup.
- [x] Distinguish these requirements rather than treating all centerline users
  as equivalent:

  ```text
  centerline progress/facts
  Frenet vehicle state
  track preview
  relative Frenet neighbors
  centerline rendering
  ```

- [x] Generate `track_preview` only when a configured consumer requires it.
- [x] Generate `frenet_neighbors` only when a configured consumer requires it.
- [x] Preserve centerline progress facts for `complete_4.yaml`; its observation
  and lap-completion reward depend on them.
- [x] Keep requirements aggregated across all agents so heterogeneous scenarios
  remain valid.
- [ ] Fail clearly during setup when an enabled component requires unavailable
  geometry.
- [x] Add contract tests proving each scenario requests the intended features.
- [x] Add numerical-equivalence tests for Frenet preview and neighbor payloads.

Exit criteria:

- [x] `complete_4.yaml` performs no track-preview projection or neighbor sorting.
- [x] `complete_4_frenet.yaml` receives unchanged preview arrays and no unused
  neighbor payload.
- [x] `complete_4_frenet_neighbors.yaml` receives unchanged preview and neighbor
  payloads.
- [ ] Reward totals, lap facts, observations, and terminal outcomes are
  unchanged for fixed trajectories.

Research implication:

This changes computation cost only. It must not change the information exposed
to any policy. Keep runtime comparisons separate from learning-quality claims.

---

## P2 - Compute Global State Once per Substep

**Why:** the profile recorded 48,258 `get_global_state()` calls for 8,993
physics substeps. The environment, per-agent reward contexts, transition
lifecycle fields, and trainer reconstruct overlapping state views.

Primary files:

```text
src/env/f110ParallelEnv.py
src/env/types.py
src/training/marl_trainer.py
src/training/reward_context.py
src/training/on_policy_trainer.py
src/training/off_policy_trainer.py
```

- [ ] Define one authoritative post-step `GlobalState` snapshot.
- [ ] Reuse that snapshot when building `StepFacts`.
- [ ] Pass the snapshot into reward-context assembly rather than calling the
  environment once per agent.
- [ ] Pass its lifecycle masks into transition construction rather than calling
  the environment again per transition.
- [ ] Reuse one pre-decision and one post-decision global vector in MAPPO.
- [ ] Avoid exposing mutable internal arrays; cached public state must remain an
  immutable snapshot for the current step.
- [ ] Invalidate the cache on reset, step, map change, lifecycle transition, and
  any public state mutation such as initial-speed application.
- [ ] Apply the same safe reuse pattern to single-agent trainers where useful.
- [ ] Add call-count instrumentation to prevent accidental regressions.

Exit criteria:

- [ ] Global-state reconstruction is O(1) per environment substep, not O(number
  of trainable agents).
- [ ] `GlobalState.vector`, masks, metadata, and per-agent lifecycle fields are
  byte-for-byte or numerically identical to the baseline at each fixed step.
- [ ] Dataset records retain independent copies where required by schema.

Research implication:

Never reuse a pre-step state as a post-step state. That would corrupt CTDE critic
targets and offline datasets even if it improved throughput.

---

## P3 - Batch MAPPO Rollout Inference

**Why:** the shared actor and centralized critic are currently invoked once per
agent, producing many tiny CUDA launches and CPU/GPU synchronizations.

Primary files:

```text
src/agents/mappo/__init__.py
src/agents/common/networks.py
src/training/marl_trainer.py
```

- [ ] Add a batched action API accepting ordered agent IDs and stacked local
  observations.
- [ ] Run the shared actor once per joint decision.
- [ ] Return actions and log probabilities mapped back to the original agent
  IDs without reordering transitions.
- [ ] Add a batched centralized-value API.
- [ ] For `agent_conditioned`, append the correct one-hot identity to each
  repeated global state before the single critic call.
- [ ] Preserve `shared_team` critic behavior.
- [ ] Transfer the action/log-probability batch to CPU once per joint decision,
  not once per agent.
- [ ] Keep deterministic evaluation supported by the batched API.
- [ ] Handle shrinking active-agent sets and one-agent batches.
- [ ] Add fixed-seed equivalence tests using controlled PyTorch RNG state.

Exit criteria:

- [ ] One actor forward and one critic forward occur per joint decision.
- [ ] Agent IDs, actions, log probabilities, values, and stored transitions stay
  correctly aligned.
- [ ] Batched and scalar inference agree within floating-point tolerance when
  given identical samples.
- [ ] CTDE remains intact: actor input contains no global state.

Research implication:

Sampling a batch can consume random numbers in a different order than four
scalar calls. Treat exact seeded trajectory reproduction separately from
distributional equivalence, and start new learning curves under a new run
version if trajectories change.

---

## P4 - Reduce Rollout Storage Synchronization and Allocation

**Why:** per-agent storage currently converts and assigns individual NumPy
objects to CUDA tensors every decision. Transition records are also fully built
even when no dataset hook consumes them.

Primary files:

```text
src/agents/mappo/__init__.py
src/training/marl_trainer.py
src/training/hooks.py
src/replay/dataset_writer.py
src/env/types.py
```

- [ ] Add batched rollout-buffer insertion for all active trainable agents.
- [ ] Minimize repeated `torch.as_tensor` calls and scalar device assignments.
- [ ] Avoid implicit CUDA synchronization from repeated Python `float(tensor)`
  conversions in the hot path.
- [ ] Determine which hooks require full `TransitionRecord` objects.
- [ ] Skip dataset-only copies and lifecycle payload construction when dataset
  recording is disabled, while preserving generic hook behavior.
- [ ] Do not weaken the dataset schema or omit required transition fields when
  recording is enabled.
- [ ] Measure host allocations and CUDA memory before and after the change.

Exit criteria:

- [ ] Buffer contents match the baseline for every agent and timestep.
- [ ] Dataset-enabled runs remain schema-compatible and complete.
- [ ] Dataset-disabled runs avoid dataset-specific state copies.
- [ ] Terminal and truncated transitions remain correct.

---

## P5 - Cache Track Geometry Across Resets

**Why:** track-width construction performs an expensive point-by-wall-segment
intersection pass. A map cycle can construct the same immutable geometry more
than once across training episodes and evaluation runs.

Primary files:

```text
src/utils/track_preview.py
src/env/f110ParallelEnv.py
src/env/map_schedule.py
src/utils/map_loader.py
```

- [ ] Key cached geometry by map identity plus centerline, wall, spacing, and
  preprocessing version.
- [ ] Reuse immutable preview geometry when returning to an unchanged map.
- [ ] Bound cache size to the configured map set.
- [ ] Keep per-agent nearest-index cursors outside the shared geometry cache and
  reset them every episode.
- [ ] Invalidate cached geometry when source files or relevant config change.
- [ ] Consider persisting preprocessed geometry only if invalidation remains
  explicit and auditable.
- [ ] Benchmark construction separately from per-step preview sampling.

Exit criteria:

- [ ] The first load builds geometry once per unique cache key.
- [ ] Later resets reuse it without changing preview values.
- [ ] Map switching cannot leak indices or geometry between maps.

---

## P6 - Optimize Track Projection Without Changing Geometry

Begin only after P1-P5 are measured. Centerline projection is important, but
algorithmic changes carry more numerical and research risk than eliminating
unused or duplicated work.

Primary files:

```text
src/utils/centerline.py
src/utils/track_preview.py
src/env/centerline_state.py
tests/test_centerline_projection.py
tests/test_frenet_vehicle_track_observation.py
```

- [ ] Confirm whether preview projection can reuse the already-computed
  centerline/Frenet projection for each agent.
- [ ] Avoid duplicate nearest-index search when the preview and progress
  geometries have a proven index mapping.
- [ ] Reuse interpolation arrays for closed tracks rather than appending them on
  every preview call.
- [ ] Profile NumPy allocation hot spots before introducing new kernels.
- [ ] Evaluate existing Numba paths and warm-up behavior before adding any new
  dependency or implementation.
- [ ] Preserve seam handling, search windows, wrong-way detection, and uniform
  arc-length sampling exactly.

Exit criteria:

- [ ] Projection results match the baseline across every configured map,
  including seam-adjacent, off-track, reverse-heading, and invalid-input cases.
- [ ] No map-specific tolerance adjustment is required.

---

## P7 - Tune PPO Update Throughput

Begin after rollout hot paths are improved. This phase must distinguish
implementation tuning from algorithm/hyperparameter changes.

Primary files:

```text
src/agents/mappo/__init__.py
configs/training/mappo.yaml
```

- [ ] Use `torch.profiler` to measure CPU launch time, CUDA kernels, memory
  copies, and synchronization during `MAPPOAgent.update`.
- [ ] Benchmark batch sizes 64, 128, 256, and 512 with the same stored rollout.
- [ ] Report optimizer samples/s and peak CUDA memory.
- [ ] Keep `n_steps`, `n_epochs`, shuffling, loss definitions, advantage
  normalization, clipping, and coefficients unchanged during the batch-size
  implementation study.
- [ ] Check whether preallocated agent-identity tensors reduce update overhead.
- [ ] Check whether pooled tensors can be assembled without repeated temporary
  allocations.
- [ ] Consider AMP only as a separate research/configuration arm with numerical
  validation; do not silently enable it.
- [ ] Do not use `torch.compile` by default until compile latency, dynamic active
  sets, checkpoint behavior, and reproducibility are measured.

Exit criteria:

- [ ] Selected defaults improve update throughput on the target GPU.
- [ ] Losses, gradients, KL, entropy, and parameter updates match within defined
  tolerances for the same rollout and minibatch ordering.
- [ ] Any batch-size default change is recorded as an experiment-version change,
  because minibatch composition can affect learning even with the same data.

---

## P8 - End-to-End Regression and Research Validation

- [ ] Run the fixed-work benchmark for all three primary scenarios.
- [ ] Compare baseline and optimized profiles by function call count and time.
- [ ] Run at least five fixed seeds for episode-level outcome checks.
- [ ] Verify identical observation dimensions:

  ```text
  complete_4                    115
  complete_4_frenet             158
  complete_4_frenet_neighbors   173
  ```

- [ ] Verify identical transition counts, terminal causes, lap counts, finish
  positions, reward components, and dataset contents for fixed scripted runs.
- [ ] Confirm W&B-disabled and dataset-disabled benchmarks do no external I/O.
- [ ] Measure with and without dataset recording to quantify its intentional
  overhead.
- [ ] Run long enough to include multiple map cycles and multiple PPO updates.
- [ ] Record final before/after throughput and memory results in a performance
  document under `docs/`.

Target acceptance criteria:

- [ ] At least 25% higher decision throughput on `complete_4.yaml` on the same
  hardware and fixed workload.
- [ ] No throughput regression greater than 5% for either Frenet scenario.
- [ ] No increase greater than 5% in peak host or CUDA memory unless justified.
- [ ] No changes to environment, observation, reward, lifecycle, dataset, or
  CTDE contracts.

---

## Validation Sequence

Run the smallest relevant checks after each slice.

Static and focused tests:

```bash
venv/bin/python -m compileall -q run.py src tests
venv/bin/python -m pytest tests/test_centerline_projection.py -q
venv/bin/python -m pytest tests/test_frenet_vehicle_track_observation.py -q
venv/bin/python -m pytest tests/test_mappo_terminal_handling.py -q
venv/bin/python -m pytest tests/test_reward_motion_components.py -q
```

Full regression and dependency guard:

```bash
venv/bin/python -m pytest tests/ -q
rg "stable_baselines3|from gymnasium|from pettingzoo" run.py src configs scenarios
```

Headless smoke tests:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/complete_4.yaml \
  --no-wandb --episodes 1 --quiet

PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/complete_4_frenet.yaml \
  --no-wandb --episodes 1 --quiet

PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/complete_4_frenet_neighbors.yaml \
  --no-wandb --episodes 1 --quiet
```

Single-agent regression smokes:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo.yaml \
  --no-wandb --episodes 1 --quiet
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/sac.yaml \
  --no-wandb --total-steps 10 --quiet
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/td3.yaml \
  --no-wandb --total-steps 10 --quiet
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/dqn.yaml \
  --no-wandb --total-steps 10 --quiet
```

## Delivery Order

```text
1. Repeatable fixed-work benchmark harness
2. Feature-gated previews and neighbors
3. One global-state snapshot per substep
4. Batched MAPPO actor and critic inference
5. Batched storage and lazy transition payloads
6. Map-geometry cache
7. Projection micro-optimizations
8. PPO update tuning
9. Full equivalence and research-readiness report
```

Each delivery should include:

1. Before/after benchmark results.
2. Files changed and contract implications.
3. Focused and full validation commands.
4. Any failures, variance, or hardware limitations.
5. A clear statement about whether experiment comparability is preserved.

## Deferred Until Performance Work Is Stable

- Calibrate final race `max_steps` from controller duration distributions.
- Finish per-agent/team terminal-reason logging across console, CSV, and W&B.
- Complete multi-map deterministic lap validation.
- Run calibrated multi-seed learning comparisons.
- Add new RL/MARL algorithms or stronger opponents.
- Add `--verbose` and `--debug` CLI flags.

These remain worthwhile, but should not be mixed into performance patches or
benchmarks because they make attribution and reproducibility harder.
