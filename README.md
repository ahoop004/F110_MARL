# F110_MARL

Multi-agent reinforcement learning stack for F1TENTH-style racing. The project wraps a high-fidelity simulator, PettingZoo-compatible environment, and a roster of training agents so you can stage adversarial racing scenarios (e.g., attacker vs. defender) and iterate on policies with PPO, TD3, DQN, or classic heuristics.

## Highlights
- **PettingZoo ParallelEnv** with customizable vehicle parameters, LiDAR pipelines, and rich collision handling (`src/f110x/envs`).
- **Physics core** backed by numba-accelerated vehicle, laser, and collision models (`src/f110x/physics`).
- **Flexible agent roster** builder that wires observation/action wrappers, controllers, and trainer adapters from YAML configs (`src/f110x/utils`).
- **Reward shaping** for pursuit/herding scenarios with curriculum support (`src/f110x/wrappers/reward.py`).
- **Training loop** in `experiments/train.py` and `experiments/main.py` that logs to Weights & Biases and TensorBoard.
- **Baseline policies** (gap-follow, random, heuristics) plus deep RL agents (PPO, TD3, DQN) under `src/f110x/policies`.

## Project Layout
```
configs/          Experiment definitions (env, agents, reward, main)
experiments/      CLI entrypoints for training/evaluation
scripts/          Automation helpers (e.g., launch_gaplock_sweeps.sh)
src/f110x/        Environment, physics, policies, trainers, utils
tests/            Env smoke tests and sample assets
```

## Getting Started
1. **Install dependencies** (editable install recommended):
   ```bash
   pip install -e .[dev]
   ```
2. **Verify the environment**:
   ```bash
   pytest tests/test_env.py
   ```
3. **Launch training** (defaults to `configs/config.yaml` and will auto-init wandb if enabled):
   ```bash
   python experiments/main.py --render        # optional flag for on-screen visualization
   python experiments/main.py --episodes 100  # override training episodes
   ```
4. **Run evaluation only** by setting `main.mode` to `eval` in your config or pushing that override through wandb.

## Configuration
- **Environment**: controlled via the `env` block (map assets, physics params, LiDAR, start poses).
- **Agents**: `agents.roster` declares each slot's algorithm, wrappers, and trainable flag. Unfilled slots receive gap-follow heuristics automatically.
- **Reward**: tuned through the `reward` section, with optional curricula (`reward_curriculum`) for staged difficulty.
- **Algorithm settings**: top-level `ppo`, `td3`, `dqn` blocks feed hyperparameters into their respective trainers.
- **Main**: toggles logging (Weights & Biases, TensorBoard), evaluation rollouts, and default checkpoints.

At runtime the drivers look for `F110_CONFIG` (path to a YAML file). If unset, `configs/config.yaml` is used. WandB sweeps can push overrides that are merged into the loaded config automatically.

## Running Sweeps
`scripts/launch_gaplock_sweeps.sh` spawns three `wandb agent` processes—one per algorithm—and pins them to GPUs 0/1/2:
```bash
scripts/launch_gaplock_sweeps.sh <TD3_SWEEP_ID> <PPO_SWEEP_ID> <DQN_SWEEP_ID> [WANDB_API_KEY]
```
Use this on a node with at least three visible GPUs; each agent will continue pulling runs until its sweep queue empties.

## Testing & Quality
- Current automated coverage focuses on environment resets and PettingZoo compliance (`tests/test_env.py`).
- Additional tests around trainers, config parsing, and reward wrappers are recommended (see `TODO.md`).

## License
Distributed under the terms of the project `LICENSE`.
