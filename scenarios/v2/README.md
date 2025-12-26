# V2 Scenario Examples

This directory contains example scenario files for the v2 training pipeline. Each scenario is a self-contained YAML configuration that defines a complete training setup.

## Available Scenarios

### Baseline Scenarios

**gaplock_ppo.yaml** - PPO Baseline
- Algorithm: Proximal Policy Optimization (PPO)
- Task: Train attacker to force FTG defender to crash
- Observations: 738 dims (gaplock preset)
- Rewards: Full gaplock rewards (all components)
- Episodes: 1500
- Best for: Initial baseline, stable training

**gaplock_td3.yaml** - TD3 Comparison
- Algorithm: Twin Delayed DDPG (TD3)
- Same task and configuration as PPO
- Best for: Comparing off-policy to on-policy

**gaplock_sac.yaml** - SAC Comparison
- Algorithm: Soft Actor-Critic (SAC)
- Same task and configuration as PPO
- Best for: Maximum entropy exploration

### Ablation Studies

**gaplock_simple.yaml** - Simplified Rewards
- Algorithm: PPO
- Rewards: Simple preset (no forcing components)
- Best for: Understanding reward component impact

**gaplock_custom.yaml** - Custom Configuration
- Demonstrates preset + overrides pattern
- Custom observation dims (360 LiDAR)
- Custom reward values
- Best for: Learning how to customize scenarios

## Quick Start

### Running a Scenario

```bash
# Basic usage
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml

# With W&B logging (override scenario config)
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb

# Override episodes
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --episodes 500

# Override seed
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --seed 123

# Quiet mode (minimal console output)
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --quiet

# Enable rendering
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --render
```

### Comparing Algorithms

Run multiple scenarios to compare algorithms:

```bash
# PPO
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb

# TD3
python v2/run.py --scenario scenarios/v2/gaplock_td3.yaml --wandb

# SAC
python v2/run.py --scenario scenarios/v2/gaplock_sac.yaml --wandb
```

All runs will log to the same W&B project (`f110-gaplock`) with different tags for easy comparison.

## Scenario Structure

### Minimal Example

```yaml
experiment:
  name: my_experiment
  episodes: 1000
  seed: 42

environment:
  map: maps/line2.yaml
  num_agents: 2
  max_steps: 5000

agents:
  car_0:
    role: attacker
    algorithm: ppo
    observation:
      preset: gaplock
    reward:
      preset: gaplock_full

  car_1:
    role: defender
    algorithm: ftg

wandb:
  enabled: true
  project: my-project
```

### Using Presets

The v2 system provides presets for observations and rewards:

**Observation Presets:**
- `gaplock` - 738 dims (720 LiDAR + ego + target + relative)
- `minimal` - 115 dims (108 LiDAR + ego only)
- `full` - 1098 dims (1080 LiDAR + ego + target + relative)

**Reward Presets:**
- `gaplock_simple` - Basic rewards (terminal + pressure + distance + heading + speed)
- `gaplock_full` - All rewards (includes forcing components)

### Customizing with Overrides

You can use presets and override specific values:

```yaml
agents:
  car_0:
    # Use gaplock preset but override LiDAR beams
    observation:
      preset: gaplock
      overrides:
        lidar:
          beams: 360

    # Use gaplock_full preset but modify rewards
    reward:
      preset: gaplock_full
      overrides:
        terminal:
          target_crash: 100.0  # Increase success reward
        pressure:
          distance_threshold: 1.5  # Bigger pressure zone
```

## Configuration Reference

### Experiment Section

```yaml
experiment:
  name: str              # Experiment name
  episodes: int          # Number of training episodes
  seed: int              # Random seed (optional)
```

### Environment Section

```yaml
environment:
  map: str               # Path to map YAML
  num_agents: int        # Number of agents
  max_steps: int         # Max steps per episode
  lidar_beams: int       # Number of LiDAR beams (optional)
  spawn_points: list     # Spawn point names (optional)
  timestep: float        # Simulation timestep (optional)
  render: bool           # Enable rendering (optional)
```

### Agent Section

```yaml
agents:
  <agent_id>:
    role: str            # 'attacker' or 'defender' (optional)
    algorithm: str       # Algorithm name (ppo, td3, sac, ftg, etc.)
    target_id: str       # Target agent ID (auto-resolved from roles)

    params:              # Algorithm-specific hyperparameters
      lr: float
      gamma: float
      # ... algorithm-specific

    observation:         # Observation configuration
      preset: str        # Preset name
      overrides: dict    # Override values (optional)

    reward:              # Reward configuration (optional, for trainable agents)
      preset: str        # Preset name
      overrides: dict    # Override values (optional)
```

### W&B Section

```yaml
wandb:
  enabled: bool          # Enable W&B logging
  project: str           # W&B project name
  name: str              # Run name (optional, defaults to experiment.name)
  tags: list             # Tags for this run (optional)
  group: str             # Group name (optional)
  notes: str             # Run notes (optional)
  mode: str              # 'online', 'offline', or 'disabled' (optional)
```

## Tips

1. **Start with a preset** - Use provided scenarios as templates
2. **Use consistent seeds** - For reproducible comparisons
3. **Tag your runs** - Makes W&B comparison easier
4. **Start simple** - Use `gaplock_simple` before `gaplock_full`
5. **Monitor metrics** - Watch success rate and avg reward in console
6. **Iterate quickly** - Use fewer episodes for hyperparameter search

## Troubleshooting

**Scenario not found:**
```bash
# Use absolute or relative path
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml
```

**Unknown preset error:**
```
# Check available presets in:
# - v2/rewards/presets.py (reward presets)
# - v2/core/observations.py (observation presets)
```

**Missing map file:**
```
# Ensure map file exists at path specified in scenario
# Default: maps/line2.yaml
```

## Creating Custom Scenarios

1. Copy an existing scenario:
```bash
cp scenarios/v2/gaplock_ppo.yaml scenarios/v2/my_experiment.yaml
```

2. Modify configuration:
- Change experiment name
- Adjust hyperparameters
- Customize rewards/observations
- Update W&B tags

3. Run your scenario:
```bash
python v2/run.py --scenario scenarios/v2/my_experiment.yaml --wandb
```

## Next Steps

- See [v2/README.md](../../v2/README.md) for full v2 documentation
- See [DESIGN_MASTER.md](../../v2/DESIGN_MASTER.md) for architecture overview
- See [rewards/DESIGN.md](../../v2/rewards/DESIGN.md) for reward configuration details
