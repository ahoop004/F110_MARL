# F110 MARL - Completed Work

## P0 - Env Refactor (Complete)

All P0 extractions complete. Env reduced from 1513 → 1100 lines. 111 tests green.
Full PPO/SAC/TD3/DQN smoke matrix passed.

### P0.1 — Centerline And Progress Extraction

- [x] `src/env/centerline_state.py`
  - Finish-line parsing, crossing detection, finish-line info injection.
  - Centerline render progress selection helpers.
  - Centerline projection, progress deltas, wrong-way flags, centerline feature helpers.
    - `CenterlineProgressTracker` computes `progress`, `progress_delta`, `d`, `vs`, `vd`, `heading_error`, `wrong_way` each step.
    - `step()` injects `info[agent_id]["centerline"]` before `filter_info_payloads`.
    - `build_agent_state` accepts optional `centerline_facts` and populates `ProgressState`.
    - `_last_centerline_facts` stored on env; passed to `get_agent_state()`.
    - Two bugs fixed: missing `import yaml` in `update_map`, uninitialized `_episode_step_count`/`_lock_speed_steps`/`_locked_velocities`.
  - `CenterlineRuntimeState` owns centerline arrays, path, render/feature flags, render point selection, render connect policy.
  - `register_centerline_usage`, `set_centerline`, `_update_renderer_centerline` logic moved out of env; thin compatibility wrappers remain.
  - Centerline autoload policy absorbed into `MapScheduler.build_load_config`.
  - Tests: 9 in `tests/test_centerline_state.py` — fact keys, forward `vs`, lateral deviation, wrong-way flag, first-step delta=0, delta positive after advance, reset clears state, None centerline returns `{}`.
  - PPO headless smoke passes; full PPO/SAC/TD3/DQN smoke matrix passed.

### P0.2 — Spawn Contract And Compatibility

- [x] `src/env/spawn.py`
  - Named spawn-point parsing, random spawn sampling, centerline-relative spawn.
  - `SpawnRequest`, `SpawnResult`, normalized spawn metadata types.
  - `resolve_reset_spawn(...)` centralizes reset pose selection.
  - `_apply_spawn_plan(plan, request)` → highest-precedence path in `resolve_reset_spawn`.
    - `plan_id` and per-agent `spawn_id` forwarded in result metadata.
    - Plan-based spawns set `update_start_poses=False`.

- [x] `src/core/spawn_config.py`
  - `normalize_spawn_config(env_config)` accepts nested `spawn:` block or legacy flat keys.
  - Nested block: `policy`, `enabled`, `allow_reuse`, `centerline/offsets/target/ego`, `points`, `start_poses`.
  - Nested takes field-level precedence; flat fills gaps.
  - `build_env_kwargs` and `create_environment` call it; all existing scenarios unchanged.

- [x] `src/env/spawn_manager.py`
  - `SpawnManager` owns spawn policy config, random spawn state, named spawn points, `_centerline_index`, and last-episode metadata/mapping.
  - `resolve(options, *, centerline, walls, start_poses, spawn_plan)` builds `SpawnRequest` and delegates to `resolve_reset_spawn`.
  - `init_metadata` param: env passes pre-loaded YAML meta (spawn points live in `annotations.spawn_points`).
  - `update_map_data`, `reseed`, `reset_episode` handle lifecycle.
  - Removed `_extract_spawn_points`, `_sample_random_spawn`, `_sample_centerline_spawn` from env (~80 lines replaced).
  - Tests: 24 in `tests/test_spawn_config.py` (normalization, SpawnPlan integration); 18 in `tests/test_spawn_manager.py` (init, lifecycle, resolve precedence, random named-point spawn).

### P0.3 — Info Builder And Step Facts

- [x] `src/env/info_builder.py`
  - `add_step_info_fields`, `build_reset_info_payloads`, `build_step_facts`, `filter_info_payloads`.
  - Levels: `minimal` (fast training), `training` (reward/metrics), `debug` (diagnostics).
  - `STABLE_STEP_INFO_KEYS` and `STABLE_RESET_INFO_KEYS` frozensets document the guaranteed stable contract.
  - Tests verify `add_step_info_fields` output covers all required stable keys.

### P0.4 — Map Runtime And Env Coordinator

- [x] `src/env/map_config.py` — runtime map path and metadata resolution for one selected map.
- [x] `src/core/map_selection.py` — scenario-level bundle discovery, split selection, `maps: auto`, train/eval/test split, map schedule setup.
- [x] `src/env/map_schedule.py` — `MapScheduler`:
  - Bundle cycle state, round-robin/random/first selection, epoch shuffle.
  - `build_load_config` — centerline autoload policy (deciding which centerline/walls keys to pass to `MapLoader`).
  - `load_from_path(map_path, map_ext)` — raw YAML load for `update_map` hot-swaps.
  - `_cache: Dict[Tuple, MapData]` keyed by `(bundle, map_ext, centerline_render, centerline_features)`; `invalidate_cache`.
  - `update_map` on env reduced to 2 lines.
- [x] `src/env/render_adapter.py`
  - `compute_relative_snapshot`, `build_render_observations`, `flush_render_state`.
  - Removed `_compute_relative_snapshot`, `_apply_reward_ring_to_renderer` from env.
  - `render()` reduced to renderer init + `flush_render_state` + dispatch.

### P0.5 — Refactor Guardrails

- [x] `tests/test_import_boundary.py` — parametrized AST scan across all `src/env/*.py`; 17 tests, all green. Forbidden: `src.training`, `src.agents`, `wandb`, `torch`, `tensorflow`, `jax`, `stable_baselines3`, `gymnasium`, `pettingzoo`, wrapper composers.
- [x] `tests/test_obs_assembly.py` — 13 tests: lidar/pose/velocity/lap/collision/acceleration/target sensors, fallback collisions, missing scans, per-agent spec override.
- [x] Full PPO/SAC/TD3/DQN smoke matrix passed after P0.1 + P0.4 extractions.
- [x] `todo.md` kept in sync with each extraction.

---

## Completed Cleanup (pre-P0)

- [x] Extract sector/radial helpers from old observation wrapper into `src/wrappers/common.py`.
- [x] Delete old `src/rewards/` reward system and consolidate on `src/wrappers/rewards/`.
- [x] Remove stale SB3/Gymnasium/PettingZoo dependencies from dependency metadata.
- [x] Update README to document the current `run.py` architecture.
- [x] Remove obsolete `src/wrappers/action.py`; active action code lives under `src/wrappers/actions/`.
- [x] Update sweeps from old `run_sb3.py` / `run_v2.py` entry points to `run.py`.
- [x] Convert remaining `sb3_*` scenario algorithm names to current pure PyTorch names.
- [x] Remove tracked SB3 model outputs; add `sb3_models/` to `.gitignore`.
- [x] Remove generated local artifacts: `outputs/`, `wandb/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`.
- [x] Remove map YAML backup files; ignore future `*.bak` files.
- [x] Remove stale eval/checkpoint tooling.
- [x] Remove dead core helpers tied to old preset flattening and metadata checkpoint flow.
- [x] Normalize older MARL scenario files to current observation/reward config file references.

Removed stale files:

```text
eval.py
docs/training_flow_diagram.md
tools/watch_best_model.py
tools/average_phase_checkpoints.py
tools/obs_probe.py
src/core/observations.py
src/core/obs_flatten.py
src/core/checkpoint_manager.py
src/core/run_metadata.py
src/core/best_model_tracker.py
src/wrappers/normalize.py
src/utils/reward_utils.py
src/utils/spawn_generator.py
src/render/setup_extensions.py
src/curriculum/training_integration.py
```

---

## P1 - Env Contract And Multi-Agent Readiness (Complete)

173 tests green. PPO smoke clean.

### P1.1 — HeuristicPolicy Protocol

- [x] `HeuristicPolicy` in `src/core/protocol.py` — `act(obs, info, *, deterministic)` + `reset(agent_id, info)`, `is_heuristic_policy()` helper.
- [x] Added no-op `reset()` to `FollowTheGapPolicy` so `FTGAgent` satisfies the protocol.
- [x] All four heuristic classes (`FTGAgent`, `PurePursuitAgent`, `StanleyAgent`, `HybridPPFTGAgent`) pass `isinstance(x, HeuristicPolicy)`.
- [x] `tests/test_heuristic_protocol.py` — 20 tests.

### P1.2 — Explicit Agent Roles

- [x] `trainable: bool` field on agent configs; explicit field takes priority over algorithm inference.
- [x] `is_trainable_agent`, `get_trainable_agent_ids`, `get_fixed_agent_ids`, `split_agent_roles` in `agent_builder.py`.
- [x] `PYTORCH_RL_ALGOS` / `HEURISTIC_ALGOS` promoted to `frozenset`.
- [x] `find_rl_agent` in `run.py` uses role helpers; raises on zero or multiple trainable agents.
- [x] `fixed_policy_agents` on env; passthrough in `env_builder.py`.
- [x] `tests/test_agent_roles.py` — 30 tests.

### P1.3 — Per-Agent Composer Dicts

- [x] `build_obs_composers` / `build_reward_composers` in `run.py` → `Dict[str, Composer]`.
- [x] `run.py main()` builds dicts, extracts single-agent entry for current trainers.
- [x] `tests/test_multi_agent_composers.py` — 12 tests.

---

## P2 - Scenario Setup And Validation (Complete)

202 tests green. PPO smoke clean.

- [x] Trimmed `src/core/config.py` to `AgentFactory` + `register_builtin_agents` only. Removed `load_yaml`, `resolve_paths`, `EnvironmentFactory`.
- [x] `src/core/__init__.py` updated — exports `HeuristicPolicy`, `is_heuristic_policy`; removed dead exports.
- [x] Strengthened `validate_scenario` in `src/core/scenario.py`:
  - Required sections + `experiment.name` + map field + non-empty agents.
  - All agent algorithms must be in `PYTORCH_RL_ALGOS | HEURISTIC_ALGOS`; unknown → clear error listing known options.
  - Trainable (RL) agents require `observation:` and `reward:`; heuristics exempt.
  - `trainable: false` override exempts even RL agents from obs/reward requirement.
- [x] `tests/test_scenario_validation.py` — 29 tests.

---

## P3 - Render And CLI Output (Complete)

293 tests green.

- [x] `environment.rendering:` nested block passes through `env_builder.py` (`passthrough_keys`).
- [x] Scenario-configurable vehicle colors via `environment.rendering.vehicle_colors`:
  - Accepts `"#rrggbb"` / `"#rrggbbaa"`, RGB `[r,g,b]`, RGBA `[r,g,b,a]`.
  - `_parse_vehicle_colors()` + `_color_to_rgba()` in `f110ParallelEnv.py`.
  - Applied via `renderer.set_agent_colors()` on first render call.
- [x] Heatmap `print()` → `logging.getLogger(__name__).debug()`.
- [x] Stray `print()` in `env_builder.py` → `logging.warning()`.
- [x] Startup banner: `algorithm`, `trainable=(...)`, `fixed=(...)`, `map`, `seed`, `device`, `obs_dim`, `action_dim`.
- [x] Rich markup escaping fixed — agent IDs in parentheses (square brackets consumed as Rich tags).
- [x] `ConsoleLogger` is the single terminal output path; no raw `print()` in training path.
- [x] `tests/test_render_config.py` — 24 tests.

---

## P4 - Efficiency Tweaks (Complete)

321 tests green.

- [x] Replay buffer stores transitions on CPU; `sample()` moves batches to training device — no GPU memory pinned for 1 M-transition buffers. `tests/test_replay_buffer_cpu.py` — 13 tests.
- [x] Reward accumulated across all `action_repeat` sub-steps in all three trainers. `tests/test_reward_accumulation.py` — 8 tests.
- [x] Checkpoint cadence configurable via `params.checkpoint_every` / `F110_CHECKPOINT_EVERY` env var.
- [x] Eliminate `np.asarray` / copy churn in observation and action composers:
  - `ObservationComponent.compute_into(raw_obs, info, out)` — writes into pre-allocated slice; default bridges to `compute()`.
  - `ObservationComposer` pre-allocates `(obs_dim,)` buffer; one `buf.copy()` per `wrap()` instead of N+2 allocs.
  - `LidarComponent`: `np.multiply` + `np.minimum` (not `np.clip`) — 2.3× faster on 108-beam scans.
  - `RelativePoseComponent`: float32 + `math.sin/cos/sqrt` — 4.4× faster.
  - `ActionComposer`: removed forced `.copy()`; `PreventReverseComponent` writes in-place.
  - Measured: `wrap()` 16.96 → 9.77 µs (1.7×), `process()` 3.55 → 2.52 µs (1.4×).
  - `tests/test_obs_action_perf.py` — 27 tests.

---

## P5 - Offline RL / Dataset Support (Complete)

321 tests green.

- [x] `TransitionRecord` frozen dataclass (`src/env/types.py`) — `obs`, `action_norm`, `action_phys`, `reward`, `reward_components`, `next_obs`, `terminated`, `truncated`, `info`, `global_state`, `map_id`, `spawn_id`, `episode_id`, `step_idx`, `agent_id`.
- [x] `src/replay/dataset_writer.py` — `DatasetWriter` (chunked `.npz`, `metadata.json`, `DATASET_SCHEMA_VERSION = "1.0"`) + `DatasetHook(TrainingHook)`.
- [x] `TrainingHook.on_step(record)` no-op added to base; all three trainers emit `TransitionRecord` via hooks.
- [x] `on_policy_trainer` / `off_policy_trainer`: `run_id`, `_episode_id()`, `_map_id()`, `_spawn_id()` helpers; `TransitionRecord` emitted per decision.
- [x] `SpawnPlan` enriched: `map_id`, `vehicle_params`, `scenario_hash`, `plan_id` optional fields.
- [x] `run.py`: `--dataset-dir` / `--dataset-chunk-size`; SHA-256 scenario hash in `metadata.json`.
- [x] `tests/test_dataset_writer.py` — 28 tests.

---

## P6 - MAPPO / MARL (Complete)

321 tests green. MAPPO headless smoke passes.

- [x] `src/agents/mappo/__init__.py` — shared actor + centralized critic, per-agent rollout buffers, pooled PPO update.
- [x] `src/training/marl_trainer.py` — multi-agent episode loop, global state for centralized critic, per-agent reward composers, buffer-full trigger.
- [x] `configs/training/mappo.yaml` + `scenarios/mappo_gaplock.yaml`.
- [x] `run.py`: MARL detection before `find_rl_agent`, `_run_mappo()` dispatch, full `trainable_ids` dict for composers, `other_agents` excludes all trainable IDs.
- [x] `tests/test_mappo_agent.py` — 31 tests. `tests/test_marl_trainer.py` — 15 tests.

---

## P7 - Curriculum Learning (Complete)

366 tests green.

- [x] `src/training/curriculum.py` — `CurriculumPhase` (frozen dataclass) + `CurriculumManager`:
  - Phase pool of named spawn points referencing map YAML `annotations.spawn_points`.
  - Rolling success window; auto-advances when `success_rate ≥ success_threshold` over `window_size` episodes.
  - `next_spawn_plan(spawn_points, agent_ids)` → `SpawnPlan` (highest-priority env override); never mutates env internals.
- [x] `CurriculumHook(TrainingHook)` — updates manager from `info["outcome"]`; logs advances; emits W&B `curriculum/` metrics.
- [x] `env.reset()` extracts `options["spawn_plan"]` and forwards to `spawn_manager.resolve()`.
- [x] `on_policy_trainer` / `off_policy_trainer`: `spawn_plan_fn: Optional[Callable]`; `_reset_env()` injects plan via `options`.
- [x] `run.py`: parses `curriculum: phases:` YAML block; builds manager, hook, closure; wired into both dispatchers.
- [x] `scenarios/ppo_curriculum.yaml` — 3-phase example (close_straight → extended_straight → full_track).
- [x] `tests/test_curriculum.py` — 45 tests.
