# Hyperparameter Sweeps

This directory contains WandB sweep configurations for hyperparameter tuning.

## Available Sweeps

### Full Grid Searches
- **sac_sweep.yaml** - SAC hyperparameter sweep (72 runs)
  - learning_rate: [0.0001, 0.0003, 0.0005]
  - gamma: [0.99, 0.995, 0.999]
  - tau: [0.005, 0.01]
  - batch_size: [128, 256]
  - hidden_dims: [[256,256], [512,512]]

- **ppo_sweep.yaml** - PPO hyperparameter sweep (162 runs)
  - learning_rate: [0.0001, 0.0003, 0.0005]
  - gamma: [0.99, 0.995, 0.999]
  - clip_range: [0.1, 0.2, 0.3]
  - ent_coef: [0.0, 0.01, 0.02]
  - n_steps: [1024, 2048]

- **td3_sweep.yaml** - TD3 hyperparameter sweep (108 runs)
  - learning_rate: [0.0001, 0.0003, 0.0005]
  - gamma: [0.99, 0.995, 0.999]
  - tau: [0.005, 0.01, 0.02]
  - batch_size: [128, 256]
  - policy_delay: [2, 3]

- **dqn_sweep.yaml** - DQN hyperparameter sweep (108 runs)
  - learning_rate: [0.0003, 0.0005, 0.001]
  - gamma: [0.99, 0.995, 0.999]
  - exploration_final_eps: [0.01, 0.05, 0.1]
  - exploration_fraction: [0.1, 0.2]
  - batch_size: [128, 256]

- **qrdqn_sweep.yaml** - QR-DQN hyperparameter sweep (162 runs)
  - learning_rate: [0.0003, 0.0005, 0.001]
  - gamma: [0.99, 0.995, 0.999]
  - exploration_final_eps: [0.01, 0.05, 0.1]
  - batch_size: [128, 256, 512]
  - hidden_dims: [[256,256], [512,512]]

### Quick Sweeps (8 runs each)
- **sac_sweep_quick.yaml** - Smaller SAC sweep for testing
- **ppo_sweep_quick.yaml** - Smaller PPO sweep for testing

## Usage

### 1. Initialize a sweep
```bash
wandb sweep sweeps/sac_sweep.yaml
```

This will output a sweep ID like: `ahoop004-old-dominion-university/marl-f110/abc123xyz`

### 2. Run sweep agents
```bash
# Run a single agent
wandb agent ahoop004-old-dominion-university/marl-f110/abc123xyz

# Run multiple agents in parallel (on different machines or terminals)
wandb agent ahoop004-old-dominion-university/marl-f110/abc123xyz  # Terminal 1
wandb agent ahoop004-old-dominion-university/marl-f110/abc123xyz  # Terminal 2
wandb agent ahoop004-old-dominion-university/marl-f110/abc123xyz  # Terminal 3
```

### 3. Monitor progress
View the sweep dashboard at:
```
https://wandb.ai/ahoop004-old-dominion-university/marl-f110/sweeps
```

## Tips

1. **Start with quick sweeps** to validate the setup before running full sweeps
2. **Use parallel agents** to speed up sweeps (one per GPU/CPU core)
3. **Monitor early** - check first few runs to catch configuration errors
4. **Reduce grid size** - Comment out parameter values to reduce total runs if needed
5. **Use random search** - Change `method: grid` to `method: random` and add `count: 20` for random sampling

## Modifying Sweeps

To reduce the number of runs, edit the YAML file and remove values:

```yaml
# Before (9 combinations)
learning_rate:
  values: [0.0001, 0.0003, 0.0005]
gamma:
  values: [0.99, 0.995, 0.999]

# After (4 combinations)
learning_rate:
  values: [0.0001, 0.0005]
gamma:
  values: [0.995, 0.999]
```

## Expected Results

Sweeps will optimize for `eval/success_rate` (evaluated periodically during training).

Key metrics to monitor:
- `eval/success_rate` - Success rate on evaluation episodes
- `eval/mean_reward` - Average reward on evaluation episodes
- `train/success_rate` - Training success rate (rolling window)
- `curriculum/phase` - Current curriculum phase
- `curriculum/success_rate` - Curriculum success rate

## Notes

- Each run trains for the full episode count specified in the scenario (typically 2500 episodes)
- The curriculum will adapt automatically based on performance
- Best hyperparameters may vary by curriculum phase
- Consider running sweeps at different curriculum phases for phase-specific tuning
