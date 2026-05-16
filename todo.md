# F110 MARL — Project Todo

## Project Goals

- Pure PyTorch training: no SB3, Gymnasium, or PettingZoo in the training path.
- Single `run.py` entry point; algorithm dispatched from scenario `algorithm:`.
- Clean config hierarchy: environment, vehicle, training, observation, reward.
- MVP: single-agent PPO done → Phase 2 off-policy done → Phase 3 MAPPO next.

---

## Current State

**Working:**

- `python3 run.py --scenario scenarios/ppo.yaml --no-wandb --episodes N`
- `python3 run.py --scenario scenarios/sac.yaml --no-wandb --total-steps N`
- `python3 run.py --scenario scenarios/td3.yaml --no-wandb --total-steps N`
- `python3 run.py --scenario scenarios/dqn.yaml --no-wandb --total-steps N`
- `--render` flag works when local display/render dependencies are available.
- Main training path has no SB3, Gymnasium, or PettingZoo imports.

**Key source layout:**

```text
run.py                                  # single training entry point
src/env/spaces.py                       # SpaceSpec / DictSpaceSpec
src/env/f110ParallelEnv.py              # physics env, no ParallelEnv base
src/agents/common/networks.py           # shared Actor/Critic/MLP networks
src/agents/ppo/__init__.py              # RolloutBuffer, PPOAgent
src/agents/sac/__init__.py              # SACAgent
src/agents/td3/__init__.py              # TD3Agent
src/agents/dqn/__init__.py              # DQNAgent
src/replay/replay_buffer.py             # pure PyTorch replay buffer
src/wrappers/observations/              # ObservationComponent + composer
src/wrappers/rewards/                   # RewardComponent + composer
src/wrappers/actions/                   # ActionComponent + composer
src/training/on_policy_trainer.py       # episode-based PPO/A2C loop
src/training/off_policy_trainer.py      # step-based SAC/TD3/DQN loop
src/training/hooks.py                   # console, W&B, checkpoint hooks
configs/env/default.yaml                # timestep, lidar, render defaults
configs/vehicle/default.yaml            # vehicle physics params
configs/training/on_policy.yaml         # PPO/A2C training defaults
configs/training/off_policy.yaml        # SAC/TD3/DQN training defaults
configs/observations/                   # observation configs
configs/reward/                         # reward configs
scenarios/                              # full experiment configs
```

---

## Cleanup

- [x] Extract sector/radial helpers from old observation wrapper into `src/wrappers/common.py`.
- [x] Delete old `src/rewards/` reward system and consolidate on `src/wrappers/rewards/`.
- [x] Remove stale SB3/Gymnasium/PettingZoo dependencies from dependency metadata.
- [x] Update README to document the current `run.py` architecture.
- [x] Remove obsolete `src/wrappers/action.py`; active action code lives under `src/wrappers/actions/`.
- [x] Update sweeps from old `run_sb3.py` / `run_v2.py` entry points to `run.py`.
- [x] Convert remaining `sb3_*` scenario algorithm names to current pure PyTorch names where equivalent support exists.
- [x] Remove tracked SB3 model outputs and add `sb3_models/` to `.gitignore`.
- [x] Remove generated local artifacts: `outputs/`, `wandb/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`.
- [x] Remove map YAML backup files and ignore future `*.bak` files.
- [x] Remove stale eval/checkpoint tooling that depended on missing old evaluator infrastructure.
- [x] Remove dead core helpers tied to old preset flattening and metadata checkpoint flow.
- [x] Normalize older MARL scenario files to current observation/reward config file references before MAPPO work.

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

## Refactor Phase — Environment Architecture

Goal: make the environment a stable runtime boundary for online RL, offline RL
dataset generation/replay, heuristic policies, and future multi-agent trainers
such as MAPPO/QMIX/MADDPG. The env should own physics and world state only; agent
logic, reward shaping, observation shaping, action shaping, and trainer control
flow should stay outside the env.

### Target Contract

- [ ] Keep the core env API trainer-agnostic:
  - `reset(options=None) -> (obs_dict, info_dict)`
  - `step(action_dict) -> (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)`
  - `possible_agents`, `agents`, `action_spaces`, `observation_spaces`
  - `get_global_state()` for centralized critics and offline dataset records
  - `get_agent_state(agent_id)` for per-agent diagnostics and dataset records
- [ ] Keep env rewards minimal and factual.
  - Env may expose raw facts in `info`: collisions, progress, target collision, timeouts, poses, velocities, map/spawn metadata.
  - Task rewards remain in `RewardComposer` so online/offline/MARL trainers can reuse the same reward definitions.
- [ ] Keep env observations factual and unshaped.
  - Env emits raw observation dicts.
  - `ObservationComposer` owns policy-specific flattening/normalization.
  - This preserves support for heuristic policies that need raw LiDAR and RL policies that need compact tensors.
- [ ] Keep action semantics explicit.
  - Env accepts physical actions only.
  - `ActionComposer` owns normalized continuous actions, discrete lookup actions, and constraints.
  - Heuristic agents should output physical actions directly or use the same action adapter interface.

### Module Split Plan

Dependency rule: `src/env/*` should not import trainers, agents, scenario parsing,
W&B, checkpoint hooks, or reward/observation/action composers. `src/core/*` may
translate scenario YAML into env/trainer objects. `src/training/*` may consume
env APIs but should not know map YAML internals.

- [x] `src/env/types.py`
  - Own small dataclasses/protocols used by multiple env modules.
  - Initial types: `MapRuntimeConfig`, `SpawnState`, `SpawnPlan`, `ProgressState`, `AgentState`, `GlobalState`, `StepFacts`.
  - Keep these plain Python/NumPy structures; no Torch, W&B, trainer, or policy dependencies.
- [x] `src/env/map_config.py`
  - Own runtime map path and metadata resolution for one selected map.
  - Move `_configure_map_paths`, `_load_map_metadata`, image path selection, map extension handling, and preloaded `MapData` adaptation.
  - Input: selected map config/path plus optional preloaded map data.
  - Output: immutable `MapRuntimeConfig` with map id, image path, yaml path, origin/resolution, centerline/wall references, and render metadata.
  - Must not own train/eval split selection or per-episode map scheduling.
- [x] `src/core/map_selection.py`
  - Own scenario-level map selection before env construction.
  - Move bundle discovery, split selection, `maps: auto`, train/eval/test split handling, and per-episode map cycling policy.
  - Output: selected map configs or a `MapSchedule` that env setup can consume.
  - Shared by online training, offline dataset collection, evaluation, and future curriculum scheduling.
- [ ] `src/env/spawn.py`
  - Own spawn point parsing, deterministic spawn plans, and stochastic spawn sampling.
  - [x] Move named spawn-point parsing from env/map metadata.
  - [x] Move random spawn sampling from `F110ParallelEnv`.
  - [x] Move named spawn-point YAML loading from `src/core/env_builder.py`.
  - [x] Add `SpawnRequest`, `SpawnResult`, and normalized spawn metadata types.
  - [x] Add `resolve_reset_spawn(...)` to centralize reset pose selection.
  - [x] Move centerline-relative spawn helpers.
  - [x] Move start pose resolution and spawn metadata bookkeeping.
  - Output: `SpawnState` per agent plus stable `spawn_id` metadata for logs/datasets.
  - Support modes: fixed start pose, YAML spawn points, random centerline spawn, deterministic replay/eval plan.
  - Must not choose train/eval maps or decide task rewards.

### Spawn Architecture

Goal: support hand-authored map YAML spawn points, centerline-relative training
spawns, curriculum-generated random spawns, deterministic evaluation, offline RL
dataset collection, and future multi-agent training with one consistent reset
result.

- [ ] Split spawn logic into three layers.
  - Spawn sources: `map_yaml`, `centerline_relative`, `curriculum`, `replay_plan`.
  - Spawn samplers: `fixed`, `random_named`, `round_robin`, `centerline_random`, `curriculum_stage`.
  - Spawn result: normalized poses, optional initial speeds, per-agent spawn ids, and metadata.
- [ ] Keep map YAML spawn points as named anchors.
  - Map editor can hard-code stable `annotations.spawn_points` in map YAML.
  - Use named anchors for eval, debugging, reproducible baselines, and scenario-specific starts.
  - Avoid making hand-authored YAML points the only training spawn mechanism.
- [ ] Support centerline-relative online training spawns.
  - Generate starts from centerline progress `s`, lateral offset `d`, longitudinal offset, and role assignment.
  - Respect lap start/finish exclusion windows so random starts do not accidentally begin on terminal boundaries.
  - Optionally clamp lateral offsets against wall distance when wall data is loaded.
  - Emit `spawn_s`, `spawn_d`, `spawn_source`, `spawn_mode`, and role/agent placement metadata.
- [ ] Support curriculum-generated spawn requests.
  - Curriculum should produce a `SpawnRequest`, not mutate env internals directly.
  - Request fields should include progress range, offset range, speed range, fixed/random mode, role placement, and optional phase/stage id.
  - Env/spawn resolver should turn that request into the same `SpawnResult` used by regular online training.
- [ ] Support deterministic evaluation/offline replay plans.
  - Add deterministic `SpawnPlan` support with ordered reset entries.
  - Include map id, spawn id, seed, plan index, vehicle params, and scenario hash in metadata.
  - Offline dataset writer should consume public spawn metadata only.
- [ ] Proposed scenario config shape for named map YAML spawns:

  ```yaml
  environment:
    spawn:
      source: map_yaml
      mode: random_named
      allow_reuse: true
  ```

- [ ] Proposed scenario config shape for deterministic eval:

  ```yaml
  environment:
    spawn:
      source: map_yaml
      mode: fixed
      agents:
        car_0: spawn_2
        car_1: spawn_1
  ```

- [ ] Proposed scenario config shape for centerline training:

  ```yaml
  environment:
    spawn:
      source: centerline_relative
      mode: random
      progress_range: [0.05, 0.95]
      avoid_finish: true
      ego:
        agent: car_0
        s_offset: 0.0
        d_offset_range: [-0.5, 0.5]
        speed_range: [0.2, 0.8]
      target:
        agent: car_1
        s_offset: 0.0
        d_offset: 0.0
        speed_range: [0.2, 0.8]
  ```

- [ ] Proposed scenario config shape for curriculum-driven spawns:

  ```yaml
  environment:
    spawn:
      source: curriculum
      curriculum_key: gaplock_line2
  ```

- [ ] Preserve backward compatibility during migration.
  - Keep existing `random_spawn`, `spawn_policy: centerline_relative`, `spawn_centerline`, `spawn_offsets`, `spawn_target`, and `spawn_ego` working until scenarios are migrated.
  - Add config normalization in `src/core/env_builder.py` or a future `src/core/spawn_config.py`.
  - Emit deprecation notes only in verbose/debug mode, not during quiet sweeps.

- [ ] `src/env/centerline_state.py`
  - Own centerline registration and progress/lap state.
  - [x] Move finish-line parsing, crossing detection, and finish-line info injection.
  - [x] Move centerline render progress selection helpers.
  - [ ] Move centerline projection, progress deltas, lap counting, wrong-way/finish flags, and centerline feature helpers.
  - Output: factual progress payloads for `info`, `get_agent_state()`, and `get_global_state()`.
  - Must not encode reward weights, terminal reward decisions, or policy-specific features.
- [ ] `src/env/collision_state.py`
  - Own collision and terminal-condition facts that are independent of reward shaping.
  - [x] Track persistent collision flags and collision steps.
  - [x] Build collision/lap terminations and global collision termination.
  - [x] Build timeout truncation facts.
  - [ ] Track richer wall/agent-agent collision breakdowns when simulator exposes them.
  - Output: collision/termination fields for `StepFacts`, `info`, and offline transition records.
- [x] `src/env/spaces_builder.py`
  - Own raw env action/observation space construction.
  - Keep `SpaceSpec` / `DictSpaceSpec` as the public representation.
  - Build spaces from env runtime config only: agent count, lidar beams, pose fields, physical action bounds.
  - Must not depend on RL observation flattening, normalization, discrete action tables, or trainer config.
- [ ] `src/env/state_views.py`
  - Own conversion from simulator/raw env state into stable public views.
  - Implement `get_agent_state(agent_id)`, `get_global_state()`, mask assembly, and ordered multi-agent state vectors.
  - Keep agent ordering stable by `possible_agents`.
  - Centralized critics and offline datasets should use this module rather than scraping simulator internals.
- [ ] `src/env/info_builder.py`
  - Own `info` payload assembly and `info_level` filtering.
  - [x] Move collision, target collision, target finish, and speed-lock step info fields.
  - Levels: `minimal` for fast training, `training` for reward/metrics, `debug` for diagnostics/render overlays.
  - Keep field names stable so online trainers, offline writers, and heuristic policies can rely on them.
- [ ] `src/render/render_state.py`
  - Own render-only overlay bookkeeping.
  - Move reward ring, reward overlay, reward heatmap, ticker, and render callback state out of `F110ParallelEnv`.
  - Rendering observes env state and `StepFacts`; it must not affect reset/step behavior.
- [ ] `src/env/f110ParallelEnv.py`
  - Keep as the coordinator and public env class.
  - Constructor wires map runtime, spawn manager, progress tracker, collision tracker, spaces, info builder, and render state.
  - `reset()` delegates map/spawn/progress/state reset and returns raw observations plus reset info.
  - `step()` delegates action application, simulator step, factual state updates, info assembly, and termination/truncation assembly.
  - Target size after split: under 900 lines before deeper physics/render cleanup.
- [x] `src/core/env_builder.py`
  - Own translating scenario/environment/vehicle config into env constructor kwargs.
  - Build `F110ParallelEnv` from selected map schedule, vehicle params, lidar config, render config, and multi-agent config.
  - Keep trainer-specific validation outside the env constructor.
- [x] `src/core/agent_builder.py`
  - Own fixed-policy and trainable-agent role construction.
  - Normalize heuristic aliases, fixed-policy defaults, controlled/trainable/fixed agent sets, and per-agent policy config.
  - Current single-agent trainers require exactly one trainable RL agent; MAPPO can consume multiple trainable agents later.
- [ ] `src/replay/dataset_writer.py`
  - Add after `StepFacts`, `get_agent_state()`, and `get_global_state()` are stable.
  - Own chunked offline dataset writes and schema metadata.
  - Must consume public env outputs only; no simulator internals.

Recommended extraction order:

1. [x] Add characterization tests for current `reset()` / `step()` behavior.
2. [x] Extract `types.py`, `map_config.py`, and `core/map_selection.py`.
3. [ ] Extract `spawn.py`.
   - [x] Named spawn parsing.
   - [x] Random spawn sampling.
   - [x] Named spawn YAML loading.
   - [x] Reset pose resolution.
   - [x] Centerline-relative spawn.
4. [ ] Extract `centerline_state.py`.
   - [x] Finish-line parsing/progress helpers.
   - [x] Centerline render progress helpers.
   - [ ] Centerline registration and lap/progress helpers.
5. [ ] Extract `collision_state.py`, `state_views.py`, and `info_builder.py`.
   - [x] Collision persistence/termination helpers.
   - [x] Step info collision/target/speed-lock fields.
   - [ ] State views and global state.
   - [ ] Info level filtering.
6. [x] Extract `spaces_builder.py`.
7. [ ] Extract `render_state.py`.
8. [x] Add `core/env_builder.py` and `core/agent_builder.py`.
   - [x] `core/env_builder.py`
   - [x] `core/agent_builder.py`
9. [ ] Add offline `dataset_writer.py` once env facts are stable.

### Online RL Support

- [ ] Preserve fast online stepping for on-policy PPO/A2C and off-policy SAC/TD3/DQN.
  - Avoid allocating large temporary dicts/arrays inside the hot `step()` path where practical.
  - Keep info payload configurable: `info_level: minimal|training|debug`.
- [ ] Make action repeat handling trainer-owned or expose env-level repeat as an explicit wrapper.
  - Trainers must be able to accumulate reward/info across repeated env steps.
  - Offline collection must record each physical env step, not only each decision step.

### Offline RL / Dataset Support

- [ ] Add a transition-record interface independent of trainer classes.
  - `TransitionRecord`: `obs_raw`, `action_phys`, `reward_components`, `next_obs_raw`, `terminated`, `truncated`, `info`, `global_state`, `map_id`, `spawn_id`.
  - Online trainers can ignore it; dataset collectors can write it.
- [ ] Add `src/replay/dataset_writer.py` later, after env facts are stable.
  - Start with simple chunked `.npz` or `.pt` files.
  - Keep schema version in each dataset.
- [ ] Ensure deterministic replay inputs.
  - Seed, map bundle, spawn selection, vehicle params, and scenario hash should be present in dataset metadata.

### Heuristic Policy Support

- [ ] Define a small policy interface for fixed drivers:
  - `reset(agent_id, info=None)`
  - `act(raw_obs, info=None) -> physical_action`
- [ ] Keep heuristic policies raw-observation compatible.
  - FTG/pure-pursuit/Stanley should not depend on RL observation composers.
  - Heuristic policies may optionally receive env/context handles for centerline access, but that dependency should be explicit.
- [x] Centralize fixed-policy construction in `src/core/agent_builder.py`.
  - Avoid scattering heuristic aliases and defaults across setup, scenarios, and trainers.

### Multi-Agent RL Support

- [ ] Add first-class controlled-agent sets.
  - Scenario should distinguish `controlled_agents`, `trainable_agents`, and `fixed_policy_agents`.
  - Current single-agent trainers can require exactly one `trainable_agent`.
  - MAPPO can train multiple controlled agents with shared or separate policies.
- [ ] Add `get_global_state()` as a stable centralized critic input source.
  - Include poses, velocities, collisions, progress/lap facts, and optionally map/spawn identifiers.
  - Keep ordering stable by `possible_agents`.
- [ ] Add per-agent masks.
  - `active_mask`, `terminated_mask`, `controlled_mask`, `trainable_mask`.
  - MAPPO needs masks for variable active agents and centralized value bootstrapping.
- [ ] Make reward composition multi-agent aware.
  - `RewardComposer` should support one composer per trainable agent.
  - Shared-policy MAPPO can aggregate per-agent rewards while retaining per-agent breakdowns.
- [ ] Make observation composition multi-agent aware.
  - Same component system, but composers should be created per policy/role, not only per single RL agent.

### Migration Order

1. [x] Add tests around current env contract before moving code.
   - Reset shape/key checks.
   - One-step smoke checks.
   - Collision/timeout info key checks.
   - `get_global_state()` characterization once added.
2. [x] Extract map config and map selection first.
   - Lowest trainer risk, easiest to verify with scenario expansion and PPO/SAC smoke tests.
3. [ ] Extract spawn handling.
   - Verify deterministic spawn plans and random spawn behavior.
4. [ ] Extract centerline/progress state.
   - Verify centerline reward scenarios still produce progress facts.
5. [ ] Extract render state.
   - Verify headless training remains unaffected and `--render` still starts.
6. [ ] Add multi-agent API primitives without changing trainers.
   - `get_global_state()`, masks, controlled-agent config parsing.
7. [ ] Update trainers to consume the new APIs.
   - Single-agent trainers first.
   - MAPPO trainer after single-agent compatibility is stable.

### Required Verification After Each Step

```bash
python3 -m compileall -q run.py src
python3 run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet --output-dir /tmp/f110_marl_smoke/ppo
python3 run.py --scenario scenarios/sac.yaml --no-wandb --total-steps 10 --quiet --output-dir /tmp/f110_marl_smoke/sac
python3 run.py --scenario scenarios/td3.yaml --no-wandb --total-steps 10 --quiet --output-dir /tmp/f110_marl_smoke/td3
python3 run.py --scenario scenarios/dqn.yaml --no-wandb --total-steps 10 --quiet --output-dir /tmp/f110_marl_smoke/dqn
```

---

## Refactor Phase — Setup Split

Goal: make `src/core/setup.py` a coordinator rather than a 500-line config
translator.

- [x] Extract scenario map normalization into `src/core/map_selection.py`.
  - Move `_normalize_maps_key`, `_coerce_bundle_list`, `_resolve_bundle_yaml`, `_discover_map_bundles`, `_apply_map_bundle`, and `_apply_map_split`.
- [x] Extract env kwargs construction into `src/core/env_builder.py`.
  - Build `env_kwargs` from `experiment`, `environment`, and vehicle config.
  - Own passthrough-key handling and centerline preload coordination.
- [x] Extract fixed-policy agent creation into `src/core/agent_builder.py`.
  - Keep RL algorithms skipped in setup and instantiated by `run.py`.
  - Keep heuristic aliases in one constant.
- [ ] Reduce `src/core/config.py` to only generic YAML/path helpers plus `AgentFactory`, or fold those helpers into the new builder modules.
- [ ] Make scenario validation stricter before env construction:
  - exactly one trainable RL agent for current single-agent trainers;
  - required observation/reward config for RL agents;
  - unsupported algorithms fail before env creation.

---

## Review Phase — Render and CLI Output

Goal: make visual debugging and terminal output predictable, scenario-configurable,
and useful for online RL, offline data collection, heuristic policy debugging, and
future multi-agent training.

### Render Review

- [ ] Define a scenario-level render config contract.
  - Support `environment.render: true|false` as the simple switch.
  - Add optional nested config: `environment.rendering`.
  - Keep render config separate from physics, rewards, observations, and trainers.
- [ ] Add scenario-configurable vehicle colors.
  - Proposed config shape:

    ```yaml
    environment:
      rendering:
        vehicle_colors:
          car_0: "#e8503c"
          car_1: "#48a7e8"
          car_2: [0.24, 0.78, 0.63, 1.0]
    ```

  - Accept hex strings, RGB/RGBA 0-1 floats, and optionally RGB/RGBA 0-255 ints.
  - Validate colors before renderer creation and fail with a clear config error.
  - Pass parsed colors from `src/core/env_builder.py` into `F110ParallelEnv`.
  - Apply colors through existing `EnvRenderer.set_agent_colors()`.
  - Ensure telemetry HUD swatches use the same scenario colors.
- [ ] Add default color policy documentation.
  - Current renderer has an internal palette keyed by trailing agent index.
  - Document role conventions: defenders cool colors, attackers warm colors, neutral/fallback palette for additional cars.
  - Keep stable defaults when no scenario colors are configured.
- [ ] Move render-only state out of `F110ParallelEnv`.
  - Continue planned `src/render/render_state.py` extraction.
  - Include reward ring, overlays, heatmap payloads, ticker, render metrics, lidar skip, and callback bookkeeping.
  - Env step/reset should update factual state only; render state should observe that state.
- [ ] Review render extension output.
  - Remove direct `print()` calls from render extensions or route them through the project logger/console.
  - Make heatmap parameter dumps opt-in with a debug flag.
  - Check HUD/telemetry text for overlap and readability with 2, 4, and 6 cars.
- [ ] Add render smoke verification.
  - Headless compile/training must still work with render disabled.
  - `--render` should still start locally when display dependencies are available.
  - Optional `rgb_array` smoke should verify non-empty frames when practical.

### CLI Output Review

- [ ] Define output modes consistently.
  - `--quiet`: minimal run summary and errors only.
  - default: concise per-episode/per-N-step progress.
  - future `--verbose`: config summary, component details, map/spawn details.
  - future `--debug`: detailed diagnostics and stack-friendly logs.
- [ ] Consolidate terminal output paths.
  - Prefer `ConsoleLogger` / Rich console wrappers over raw `print()`.
  - Audit remaining prints in agents, render extensions, curriculum, utils, metrics examples, and core smoke scripts.
  - Keep examples/docstrings as examples, but route runtime output through loggers.
- [ ] Review `run.py` startup output.
  - Show scenario name, algorithm, RL agents, fixed-policy agents, map bundle, seed, device, obs/action dims.
  - Keep it compact enough for sweeps and smoke tests.
- [ ] Review training progress output.
  - On-policy: episode, reward, moving mean, length, outcome, crash/timeout/success breakdown.
  - Off-policy: total steps, episode count, reward, moving mean, buffer size, update stats when available.
  - Avoid noisy per-step logs by default.
- [ ] Review Rich console dashboard.
  - Decide whether it is active, optional, or deprecated.
  - If kept, make it compatible with current `ConsoleLogger`, `TrainingHooks`, and `--quiet`.
  - Ensure color choices work on dark/light terminals and without Unicode-only assumptions where possible.
- [ ] Add CLI output tests or smoke checks.
  - `--quiet` should suppress progress chatter but preserve errors.
  - default mode should include one clear training start line and periodic progress.
  - invalid scenario/config should fail with clear, actionable messages.

---

## Efficiency Tweaks

- [ ] Move replay buffer storage to CPU by default and copy sampled batches to the selected training device.
  - Current large GPU replay buffers can reserve excessive GPU memory for SAC/TD3/DQN.
  - Add a config option such as `replay_device: cpu|train_device`.
- [ ] Accumulate rewards across `action_repeat` env steps instead of computing one composed reward from only the final repeated step.
  - This is more faithful to repeated control and avoids dropping intermediate collision/progress signals.
- [ ] Avoid repeated `np.asarray` / copy churn in observation and action composers.
  - Ensure components return `float32` arrays and preallocate concat buffers if profiling shows composer overhead.
- [ ] Cache static map-derived data per map bundle.
  - Avoid repeatedly parsing YAML/image/centerline/walls when cycling maps across episodes.
- [ ] Add optional checkpoint cadence controls for smoke tests and sweeps.
  - Current `CheckpointHook` saves episode 0 by default, which creates many tiny run artifact folders during short verification.

---

## Phase 3 — MARL: MAPPO

Design constraint: `Actor` in `src/agents/common/networks.py` is already MAPPO-ready.
Only the critic differs: `Critic(input_dim=n_agents * obs_dim)` for MAPPO vs
`Critic(input_dim=obs_dim)` for PPO.

- [ ] `src/agents/mappo/__init__.py`: shared actor + centralized critic; focal agent cycling; per-agent rollout buffers.
- [ ] `src/training/marl_trainer.py`: multi-agent episode loop; build global state for centralized critic.
- [ ] `configs/training/mappo.yaml`: MAPPO-specific training defaults.
- [ ] `scenarios/mappo_defender.yaml` and `scenarios/mappo_attacker.yaml`.

---

## Post-MVP: Evaluation & Curriculum

Deferred until the single-agent and off-policy paths are fully cleaned up.

- [ ] Eval config: spawn point names reference map YAML `annotations.spawn_points`.
- [ ] Curriculum config: phase advancement gates tied to eval success rate.

---

## Verification

```bash
python3 -m compileall -q run.py src
python3 run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet
python3 run.py --scenario scenarios/sac.yaml --no-wandb --total-steps 10 --quiet
python3 run.py --scenario scenarios/td3.yaml --no-wandb --total-steps 10 --quiet
python3 run.py --scenario scenarios/dqn.yaml --no-wandb --total-steps 10 --quiet
rg "stable_baselines3|from gymnasium|from pettingzoo|run_sb3|sb3_" run.py src configs scenarios sweeps requirements.txt pyproject.toml
```
