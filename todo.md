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

## Refactor Phase — Environment Split

Goal: reduce `src/env/f110ParallelEnv.py` from a large orchestration class into
small modules with explicit ownership. Keep public env behavior stable while
moving code behind private helper APIs.

- [ ] Extract map path and metadata handling into `src/env/map_config.py`.
  - Move `_configure_map_paths`, `_load_map_metadata`, map image path selection, and map extension handling.
  - Preserve support for preloaded `MapData`.
- [ ] Extract map bundle selection into `src/core/map_selection.py` or `src/env/map_selection.py`.
  - Move bundle discovery, split selection, per-episode cycling, and `maps: auto` behavior out of setup/env internals.
- [ ] Extract spawn-point parsing and sampling into `src/env/spawn.py`.
  - Move `_extract_spawn_points`, `_sample_random_spawn`, centerline spawn helpers, start pose resolution, and spawn metadata bookkeeping.
- [ ] Extract centerline/progress feature support into `src/env/centerline_state.py`.
  - Move centerline registration, progress state, finish-line parsing, and centerline feature payload assembly.
- [ ] Extract reward/render overlay state into `src/env/render_state.py` or keep under `src/render/`.
  - Move reward ring, reward overlay, reward heatmap, render ticker, and render callback bookkeeping.
- [ ] Extract observation/action space construction into `src/env/spaces_builder.py`.
  - Keep `SpaceSpec` as the public space representation.
- [ ] Add a small smoke test after each extraction:
  - `python3 run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet`
  - `python3 run.py --scenario scenarios/sac.yaml --no-wandb --total-steps 10 --quiet`

---

## Refactor Phase — Setup Split

Goal: make `src/core/setup.py` a coordinator rather than a 500-line config
translator.

- [ ] Extract scenario map normalization into `src/core/map_config.py`.
  - Move `_normalize_maps_key`, `_coerce_bundle_list`, `_resolve_bundle_yaml`, `_discover_map_bundles`, `_apply_map_bundle`, and `_apply_map_split`.
- [ ] Extract env kwargs construction into `src/core/env_builder.py`.
  - Build `env_kwargs` from `experiment`, `environment`, and vehicle config.
  - Own passthrough-key handling and centerline preload coordination.
- [ ] Extract fixed-policy agent creation into `src/core/agent_builder.py`.
  - Keep RL algorithms skipped in setup and instantiated by `run.py`.
  - Keep heuristic aliases in one constant.
- [ ] Reduce `src/core/config.py` to only generic YAML/path helpers plus `AgentFactory`, or fold those helpers into the new builder modules.
- [ ] Make scenario validation stricter before env construction:
  - exactly one trainable RL agent for current single-agent trainers;
  - required observation/reward config for RL agents;
  - unsupported algorithms fail before env creation.

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
