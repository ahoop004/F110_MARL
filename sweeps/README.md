# Hyperparameter Sweeps

This directory contains WandB sweep configurations for hyperparameter tuning.

## Available Sweeps

### Algorithm Comparison Sweeps 🆕

**Start here to identify the best algorithm for your task!**

- **algo_comparison_sweep.yaml** - Compare all major algorithms with baseline hyperparameters (6 runs)
  - Algorithms: SAC, TD3, DDPG, TQC, PPO, A2C
  - Fixed hyperparameters for fair comparison
  - **Use this first** to identify which algorithm works best

- **algo_comparison_tuning_sweep.yaml** - Compare algorithms + tune hyperparameters (Bayesian optimization)
  - Algorithms: SAC, TD3, PPO, A2C
  - Tunes learning_rate, gamma, tau, batch_size simultaneously
  - More thorough but more expensive than baseline comparison

- **algo_comparison_quick.yaml** - Quick algorithm test (3 runs, 500 episodes each)
  - Algorithms: SAC, TD3, PPO
  - For rapid prototyping and testing
  - Runs in ~1-2 hours per algorithm

### Algorithm-Specific Full Grid Searches
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

### Architecture Sweeps (TD3/TQC)
- **td3_mlp_sweep.yaml** - TD3 deep MLP width/depth sweep
- **td3_actor_critic_sweep.yaml** - TD3 actor/critic split sweep
- **td3_activation_sweep.yaml** - TD3 activation sweep
- **tqc_mlp_sweep.yaml** - TQC deep MLP width/depth sweep
- **tqc_actor_critic_sweep.yaml** - TQC actor/critic split sweep
- **tqc_activation_sweep.yaml** - TQC activation sweep

## Recommended Workflow

### Phase 1: Algorithm Selection
1. **Run algo_comparison_sweep.yaml** to identify best algorithm(s)
   ```bash
   wandb sweep sweeps/algo_comparison_sweep.yaml
   wandb agent <sweep-id>
   ```
2. Compare results on WandB dashboard
3. Select top 2-3 performing algorithms

### Phase 2: Algorithm-Specific Tuning
4. **Run algorithm-specific sweeps** for your top performers
   ```bash
   # If SAC performed best:
   wandb sweep sweeps/sac_sweep.yaml
   wandb agent <sweep-id>
   ```
5. Identify best hyperparameters for each algorithm

### Phase 3: Fine-tuning (Optional)
6. Create custom sweeps around best hyperparameters
7. Use Bayesian optimization for final refinement

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

Sweeps will optimize for `train/success_rate` (rolling window during training).

Key metrics to monitor:
- `train/success_rate` - Training success rate (rolling window)
- `train/reward_mean` - Rolling average reward
- `eval_agg/success_rate` - Success rate on evaluation episodes (if enabled)
- `eval_agg/avg_reward` - Average reward on evaluation episodes
- `curriculum/stage` - Current curriculum stage

## Notes

- Each run trains for the full episode count specified in the scenario (typically 1500 episodes)
- The curriculum will adapt automatically based on performance
- Best hyperparameters may vary by curriculum phase
- Consider running sweeps at different curriculum phases for phase-specific tuning
