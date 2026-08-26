# F110_MARL

Pure PyTorch reinforcement learning experiments for adversarial F1TENTH racing.

The active training path is intentionally small: scenarios are YAML files, `run.py`
is the single entry point, and algorithm-specific agents are implemented directly
in `src/agents/` without Stable-Baselines3, Gymnasium, or PettingZoo in the
training loop.

## Current Capabilities

- Single-agent RL against fixed-policy opponents.
- MAPPO with a shared actor and configurable team or agent-conditioned critic.
- Pure PyTorch PPO/A2C, SAC/DDPG, TD3, and DQN implementations.
- Component-based observation, reward, and action pipelines.
- Map bundle loading, random spawn support, centerline features, console logging,
  W&B logging, and checkpoint hooks.
- Deterministic PPO/MAPPO checkpoint evaluation and curriculum utilities.

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a short PPO smoke test:

```bash
python3 run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1
```

Run the paired four-agent MAPPO reward/critic experiments:

```bash
python3 run.py --scenario scenarios/complete_4_individual.yaml --no-wandb
python3 run.py --scenario scenarios/complete_4_team_shared.yaml --no-wandb
```

The individual arm uses per-agent rewards and an agent-conditioned centralized
critic, `V_i(s)`. The team arm averages factual per-agent rewards with a fixed
team-size denominator and uses one shared team critic, `V(s)`. The actor remains
decentralized in both modes and receives only its local observation.

Run off-policy agents:

```bash
python3 run.py --scenario scenarios/sac.yaml --no-wandb --total-steps 1000
python3 run.py --scenario scenarios/td3.yaml --no-wandb --total-steps 1000
python3 run.py --scenario scenarios/dqn.yaml --no-wandb --total-steps 1000
```

Add `--render` for local visual debugging when a display is available.

## Active Architecture

```text
run.py
  -> core.scenario.load_and_expand_scenario()
  -> core.setup.create_training_setup()
  -> wrappers.observations.ObservationComposer
  -> wrappers.rewards.RewardComposer
  -> wrappers.actions.ActionComposer
  -> training.on_policy_trainer.OnPolicyTrainer
     or training.off_policy_trainer.OffPolicyTrainer
     or training.marl_trainer.MARLTrainer
```

Important directories:

```text
src/agents/                  Pure PyTorch RL agents and heuristic agents
src/agents/common/           Shared Actor/Critic/MLP networks
src/env/                     F110 parallel environment and lightweight spaces
src/replay/                  Pure PyTorch replay buffer
src/training/                On-policy/off-policy trainer loops and hooks
src/wrappers/actions/        Action processing components
src/wrappers/observations/   Observation components and composer
src/wrappers/rewards/        Reward components and composer
configs/                     Env, vehicle, training, observation, reward configs
scenarios/                   Full experiment definitions
sweeps/                      W&B sweep definitions
```

## Scenario Shape

Scenarios include shared config files and define one trainable RL agent plus any
fixed-policy opponents:

```yaml
includes:
  - ../configs/env/default.yaml
  - ../configs/vehicle/default.yaml
  - ../configs/training/on_policy.yaml
  - ../configs/wandb.yaml

experiment:
  name: ppo_gaplock
  episodes: 2000
  seed: 42

environment:
  maps: [line2]
  max_steps: 1500
  action_repeat: 2

agents:
  car_0:
    algorithm: ppo
    role: attacker
    target_id: car_1
    observation: ../configs/observations/rl_attacker.yaml
    reward: ../configs/reward/gaplock_attack.yaml
    params:
      learning_rate: 0.0003
      hidden_dims: [256, 256]
      gamma: 0.99
    action_constraints:
      prevent_reverse: true
      speed_index: 1

  car_1:
    algorithm: ftg
    role: defender
    observation: ../configs/observations/ftg.yaml
    params:
      max_distance: 10.0
```

`run.py` dispatches from the trainable agent's `algorithm:` field. Current
single-agent trainers expect one RL agent per scenario.

## Rewards

Rewards are configured in `configs/reward/*.yaml` and assembled by
`src/wrappers/rewards/composer.py`.

Available reward components include:

- `centerline`
- `collision`
- `speed`
- `gaplock_pressure`
- `gaplock_forcing`
- `terminal_success`
- `terminal_timeout`
- `terminal_self_crash`

Example:

```yaml
reward:
  gaplock_pressure:
    enabled: true
    weight: 1.0
  terminal_success:
    enabled: true
    bonus: 200.0
  terminal_timeout:
    enabled: true
    penalty: -100.0
```

## Observations And Actions

Observation configs live in `configs/observations/*.yaml` and are composed into
flat `float32` arrays by `ObservationComposer`.

Action processing lives in `src/wrappers/actions/`:

- Continuous agents output normalized actions in `[-1, 1]`; `DenormalizeComponent`
  maps them to physical env bounds.
- DQN outputs a discrete index; `DiscreteActionComponent` maps it to a configured
  physical action set.
- `PreventReverseComponent` can clamp the speed dimension to nonnegative values.

## Verification

Use these before and after cleanup or refactors:

```bash
python3 -m compileall -q run.py src
python3 run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet
python3 run.py --scenario scenarios/sac.yaml --no-wandb --total-steps 10 --quiet
python3 run.py --scenario scenarios/td3.yaml --no-wandb --total-steps 10 --quiet
python3 run.py --scenario scenarios/dqn.yaml --no-wandb --total-steps 10 --quiet
```

## Roadmap

- Finish cleanup after the SB3/Gymnasium/PettingZoo removal.
- Normalize remaining scenario and sweep files around `run.py`.
- Calibrate multi-lap episode limits with deterministic controller runs.
- Run multi-seed MAPPO reward/critic comparisons on explicit train/eval maps.
