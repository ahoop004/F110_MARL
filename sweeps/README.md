# Hyperparameter Sweeps

W&B sweep definitions for the current `run.py` entry point.

## Recommended Sweeps

- `algo_comparison_quick.yaml`: quick SAC, TD3, PPO comparison.
- `algo_comparison_sweep.yaml`: baseline SAC, TD3, DDPG, PPO, A2C comparison.
- `algo_comparison_tuning_sweep.yaml`: Bayesian tuning across SAC, TD3, PPO, A2C.
- `ppo_sweep.yaml` and `ppo_seed_sweep.yaml`: PPO seed/hyperparameter sweeps.
- `sac_sweep.yaml`, `td3_sweep.yaml`, `ddpg_sweep.yaml`: continuous off-policy sweeps.
- `dqn_sweep.yaml`: discrete action DQN sweep.

## Compatibility Sweeps

- `qrdqn_sweep.yaml` runs `scenarios/qrdqn.yaml`, which now uses the pure PyTorch
  DQN agent with the old QR-DQN action set. QR-DQN itself is not implemented yet.
- `tqc_sweep.yaml` runs `scenarios/tqc.yaml`, which now uses the pure PyTorch SAC
  agent with the old TQC sweep shape. TQC itself is not implemented yet.

## Usage

```bash
wandb sweep sweeps/sac_sweep.yaml
wandb agent <sweep-id>
```

Every sweep should use:

```yaml
program: run.py
command:
  - ${env}
  - python3
  - ${program}
```

Keep scenario paths under `parameters.scenario` or explicit `--scenario` command
arguments pointing at files in `scenarios/`.
