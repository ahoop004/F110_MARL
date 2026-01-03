# Stable-Baselines3 Baselines for F110 Gaplock

This directory provides Stable-Baselines3 (SB3) baseline implementations for the F110 gaplock adversarial racing task. These baselines use proven, well-tested RL algorithms that should converge reliably.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-sb3.txt
```

### 2. Train a Baseline

```bash
# Train SAC baseline (recommended)
python run_sb3_baseline.py --algo sac --scenario scenarios/v2/gaplock_sac.yaml --wandb

# Train TD3 baseline
python run_sb3_baseline.py --algo td3 --scenario scenarios/v2/gaplock_td3.yaml --wandb

# Train PPO baseline
python run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_ppo.yaml --wandb
```

### 3. Monitor Training

With `--wandb` flag, training metrics are logged to Weights & Biases:
- Episode rewards
- Success rates
- Policy losses
- Value estimates

Without wandb, check TensorBoard logs in `./sb3_models/`

## Available Algorithms

| Algorithm | Best For | Key Hyperparameters |
|-----------|----------|-------------------|
| **SAC** | Continuous control, sample-efficient | `alpha` (entropy), `tau` (target update) |
| **TD3** | Continuous control, stable | `policy_noise`, `target_noise_clip` |
| **PPO** | General purpose, stable | `clip_range`, `ent_coef` |

## Command-Line Options

```bash
python run_sb3_baseline.py --help
```

**Main Options:**
- `--algo {sac,td3,ppo}` - Algorithm to use (required)
- `--scenario PATH` - Path to scenario YAML (required)
- `--episodes N` - Number of episodes to train (default: 2500)
- `--wandb` - Enable wandb logging
- `--seed N` - Random seed (default: 42)
- `--output-dir PATH` - Where to save models (default: ./sb3_models)

## Default Hyperparameters

All baselines use **well-tuned defaults** from SB3:

### SAC (Recommended for Continuous Control)
```python
learning_rate = 3e-4
buffer_size = 1,000,000
batch_size = 256
tau = 0.005  # Target network update rate
gamma = 0.995  # Discount factor
ent_coef = 'auto'  # Automatic entropy tuning
network = [256, 256]  # Hidden layer sizes
```

### TD3
```python
learning_rate = 3e-4
buffer_size = 1,000,000
batch_size = 256
tau = 0.005
gamma = 0.995
policy_delay = 2  # Delayed policy updates
target_policy_noise = 0.2  # Exploration noise
network = [256, 256]
```

### PPO
```python
learning_rate = 3e-4
n_steps = 2048  # Steps per update
batch_size = 256
gamma = 0.995
gae_lambda = 0.95  # Advantage estimation
clip_range = 0.2  # PPO clip parameter
ent_coef = 0.02  # Entropy coefficient
network = [256, 256]
```

## Output Structure

```
sb3_models/
├── sac/
│   └── seed_42/
│       ├── checkpoints/  # Periodic checkpoints
│       ├── models/       # Models for wandb
│       └── sac_final.zip # Final trained model
├── td3/
│   └── seed_42/
│       └── ...
└── ppo/
    └── seed_42/
        └── ...
```

## Loading Trained Models

```python
from stable_baselines3 import SAC

# Load trained model
model = SAC.load("sb3_models/sac/seed_42/sac_final.zip")

# Use for evaluation
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

## Comparison with Custom Agents

Use these baselines to:
1. **Validate environment**: If SB3 converges, your env/rewards are good
2. **Set benchmarks**: Custom agents should beat these baselines
3. **Debug custom agents**: If custom agents underperform, compare implementations

## Expected Performance

For the gaplock task (attacker vs FTG defender):
- **SAC**: Should reach **50-70% success rate** after 1000-1500 episodes
- **TD3**: Should reach **45-65% success rate** after 1000-1500 episodes
- **PPO**: Should reach **40-60% success rate** after 1500-2500 episodes

*Note: Exact performance depends on reward shaping and opponent strength.*

## Troubleshooting

### "ModuleNotFoundError: No module named 'stable_baselines3'"
```bash
pip install stable-baselines3
```

### "AttributeError: 'F110ParallelEnv' object has no attribute 'get_agent_obs'"
This is expected - the wrapper handles multi-agent coordination automatically.

### Training is slow
- Reduce `--episodes` for quick tests
- Use `--no-render` in scenario (set `render: false`)
- Check that CUDA is available for neural network training

### Wandb not logging
Make sure you're logged in:
```bash
wandb login
```

## Next Steps

After getting baselines:
1. Compare with your custom SAC/TD3/PPO implementations
2. Use baseline success rates as targets for custom architectures
3. Experiment with custom architectures (e.g., sequence-based, episodic memory)

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
