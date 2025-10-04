- [ ] Core Architecture Setup
  - [ ] Create new package structure for experiments, runner, trainer, engine, utils.
  - [ ] Move environment setup and config logic to `engine/builder.py`.
  - [ ] Add unified context class in `runner/context.py`.
  - [ ] Maintain compatibility with `ExperimentConfig` and scenario YAMLs.

- [ ] Configuration Handling
  - [ ] Create `utils/config.py` with `load_config(path, experiment=None)` helper.
  - [ ] Support CLI and W&B overrides in centralized YAML parsing and merging logic.
  - [ ] Route per-algorithm configuration through the registry to avoid hard-coded sections like `cfg.ppo`.

- [ ] Trainer Layer
  - [ ] Define abstract `Trainer` class (`base.py`) with core methods: `__init__`, `select_action`, `observe`, `update`, `save`, `load`.
  - [ ] Implement specialized trainer modules.
    - [ ] `on_policy.py` for PPO, A2C, REINFORCE.
    - [ ] `off_policy.py` for DQN, TD3, SAC.
  - [ ] Create `registry.py` for algorithm lookup (`ALGO_REGISTRY`).

- [ ] Engine Utilities
  - [ ] Build `rollout.py` with shared environment stepping helpers such as `run_episode()` and `collect_trajectory()`.
  - [ ] Build `reward.py` to combine reward construction and curriculum logic, porting `_build_reward_wrapper` and `_resolve_reward_mode`.

- [ ] Runner Layer
  - [ ] Implement `train_runner.py` to initialize context and trainer, execute episodes via `trainer.train_episode()`, and handle metric logging plus checkpointing.
  - [ ] Implement `eval_runner.py` to load trained models, run deterministic evaluation episodes, and support optional rollout recording.
  - [ ] Implement `context.py` to merge training and evaluation context, storing config, env, agents, map, reward, and output paths.
  - [ ] Ensure runners and contexts expose algorithm-agnostic primary agent interfaces instead of PPO-specific fields.
  - [ ] Allow evaluation contexts to run with heuristic-only rosters (no trainer adapters).

- [ ] Integration and Migration
  - [ ] Refactor `main.py` to keep only CLI, seeding, and mode dispatch, deferring to runners for execution.
  - [ ] Replace direct training/eval calls with runner abstractions.
  - [ ] Remove duplicate setup logic from `train.py` and `eval.py`.
  - [ ] Confirm backward compatibility with YAML configs and scenario manifests.
  - [ ] Drive default CLI algorithm presets from the algorithm registry instead of hard-coding DQN.
  - [ ] Rework `run.py` to source experiment configs and overrides from scenario manifests (via the registry) instead of maintaining local defaults.

- [ ] Logging & Monitoring
  - [ ] Move W&B setup and logging to `utils/logger.py`.
  - [ ] Standardize metric naming (e.g., `train/return_*`, `eval/return_*`).
  - [ ] Add CSV or TensorBoard fallback logging.

- [ ] Cleanup & Testing
  - [ ] Remove deprecated functions and repeated logic.
  - [ ] Add unit tests covering trainer classes, context creation, builder functions, scenario parsing, config parsing, and rollout loop correctness.
  - [ ] Run regression tests with baseline PPO and DQN experiments.

- [ ] Stretch Goals
  - [ ] Implement asynchronous or distributed training support.
  - [ ] Add checkpoint resume and experiment continuation.
  - [ ] Provide simplified CLI commands.
    - [ ] `python main.py train --algo ppo --episodes 1000`
    - [ ] `python main.py eval --config scenarios/gaplock_dqn.yaml`
