- [ ] Phase 0 – Shared Extraction Milestones
  - [x] Freeze current behaviour with smoke tests/notes so post-migration regressions are obvious. (`docs/smoke_tests.md`)
  - [x] Lift `_resolve_config_input`/`_load_experiment_config` into `utils/config.py` (`load_config(path, experiment=None)`), keeping scenario manifests + `F110_CONFIG`/`F110_EXPERIMENT` overrides intact.
  - [x] Extract `_build_curriculum_schedule`, `_resolve_reward_mode`, and `_build_reward_wrapper` into `engine/reward.py`, exposing a small surface that both legacy loops and new runners can call.
  - [x] Extract `TrajectoryBuffer`, idle-speed truncation logic, and best-model bookkeeping from `run_training` into `engine/rollout.py`, returning neutral data so trainers/runners stay lean.
  - [x] Introduce a shim in `experiments/train.py`/`experiments/eval.py` that delegates to the new helpers without changing their public API (first green refactor checkpoint).

- [ ] Phase 1 – Core Architecture Setup
  - [x] Create packages: `engine/`, `runner/`, `trainer/`, `utils/` (reuse existing modules where possible).
  - [x] Flesh out `runner/context.py` as a runner-tailored context object (config, env handles, map data, trainers, reward hooks, IO paths) without PPO-specific fields.
  - [x] Migrate environment setup + agent building into `engine/builder.py`, leaving thin adapter shells in the old entrypoints until callers move.
  - [x] Ensure `ExperimentConfig` + scenario YAML compatibility by funnelling everything through the extracted config loader.

- [ ] Phase 2 – Trainer Layer
  - [x] Define abstract `Trainer` base (`trainer/base.py`) with `__init__`, `select_action`, `observe`, `update`, `save`, `load`.
  - [x] Split concrete trainers into `on_policy.py` (PPO, A2C, REINFORCE) and `off_policy.py` (DQN, TD3, SAC), reusing existing trainer implementations.
  - [x] Add `trainer/registry.py` for algorithm lookup so runner/context picks trainers without hard-coded `cfg.ppo` branches.

- [x] Phase 3 – Engine Utilities
  - [x] Finalise `engine/rollout.py` with shared stepping helpers (`run_episode()`, `collect_trajectory()`), idle handling, curriculum triggers, and checkpoint callbacks.
  - [x] Finalise `engine/reward.py` with curriculum-aware wrapper construction + mode resolution, backing onto the extracted helpers.

- [x] Phase 4 – Runner Layer
  - [x] Implement `runner/train_runner.py` to compose context + trainers, run training episodes via engine rollouts, handle logging/checkpoint hooks.
  - [x] Implement `runner/eval_runner.py` to reuse context, load checkpoints, run deterministic evaluation, and optional rollout recording.
  - [x] Ensure runners expose algorithm-agnostic agent interfaces (primary agent, opponents, roster metadata) and allow heuristic-only evaluation rosters.
  - [x] Replace interim TrainRunner/EvalRunner shims (currently delegate to legacy loops) with engine-native execution.

- [x] Phase 5 – CLI Integration & Migration
  - [x] Refactor `experiments/main.py` to keep only CLI parsing, seeding, and mode dispatch; runners handle execution.
  - [x] Update `experiments/train.py`/`experiments/eval.py` to delegate into the new runner modules (maintaining backwards-compatible CLI flags/env vars until removed).
  - [x] Rework `run.py` to source presets from the trainer registry instead of `_DEFAULT_CONFIGS`, and stream scenarios through `utils/config.load_config`.
  - [x] Confirm CLI and YAML manifest compatibility end-to-end.
  - [x] Ensure `run.py` flattens scenario manifests when supplied via `--config`/grid specs so `experiments/main.py` receives legacy-style configs for overrides.
  - [x] Populate or remap algorithm preset files (e.g. `scenarios/ppo.yaml`, `scenarios/dqn.yaml`) so trainer registry discovery aligns with shipped scenarios.

- [x] Logging & Monitoring Refresh
  - [x] Build `utils/logger.py` as the unified façade for console summaries and W&B pushes (structured events, pluggable sinks, lifecycle hooks).
  - [x] Swap `runner/train_runner.py`/`runner/eval_runner.py` onto the new logger API so prints and `wandb.log` calls flow through a single path.
  - [x] Define the console emission cadence + field set (rewards, collisions, curriculum state, timings) and match those keys to W&B metrics.
  - [x] Standardise metric keys (`train/return_*`, `eval/return_*`, curriculum state, collision stats) and thread them through engine/runner hooks.
  - [x] Document W&B configuration expectations (project/entity inputs, run-id resume behaviour, summary vs step metrics) for the logger adapter (see `experiments/main.py`).
  - [x] Remove legacy logging paths once runners adopt the unified façade (direct prints, ad-hoc W&B usage, stray CSV writers).

- [ ] Cleanup & Testing
  - [x] Delete legacy helpers once shims point to the new modules and the import-time `CTX = create_training_context()` side effect is removed.
  - [ ] Add unit/integration coverage for config loader, runner context creation, builder functions, rollout loop, and registry resolution.
  - [ ] Run regression passes with baseline PPO/DQN scenarios to validate the extraction approach.

- [ ] Stretch Goals
  - [ ] Asynchronous/distributed training support once rollouts live in the engine package.
  - [ ] Checkpoint resume + experiment continuation workflow leveraging the new context + logger utilities.
  - [ ] Simplified CLI entrypoints (`python main.py train --algo ppo --episodes 1000`, etc.) after legacy flags are retired.
