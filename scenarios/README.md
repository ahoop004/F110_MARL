# Scenario Configurations

This directory contains training scenario configurations for the F110 MARL project.

## Directory Structure

```
scenarios/
├── v2/          # Active v2 scenario files (use these)
├── eval/        # Evaluation scenario configurations
└── README.md    # This file
```

## Active Scenarios

All current scenarios are in the **v2/** directory:

### Available Scenarios

**Gaplock (Adversarial Racing):**
- [v2/gaplock_sac.yaml](v2/gaplock_sac.yaml) - SAC for gaplock task
- [v2/gaplock_ppo.yaml](v2/gaplock_ppo.yaml) - PPO for gaplock task
- [v2/gaplock_td3.yaml](v2/gaplock_td3.yaml) - TD3 for gaplock task
- [v2/gaplock_rainbow.yaml](v2/gaplock_rainbow.yaml) - Rainbow DQN for gaplock
- [v2/gaplock_simple.yaml](v2/gaplock_simple.yaml) - Simplified gaplock (fewer components)
- [v2/gaplock_custom.yaml](v2/gaplock_custom.yaml) - Custom gaplock configuration
- [v2/gaplock_limo.yaml](v2/gaplock_limo.yaml) - Gaplock with LIMO dynamics

**Other:**
- [v2/test_render.yaml](v2/test_render.yaml) - Rendering test scenario

**Evaluation:**
- [eval/*.yaml](eval/) - Evaluation-specific configurations

## Usage

Run a scenario with the v2 training script:

```bash
# Basic training
python run_v2.py --scenario scenarios/v2/gaplock_sac.yaml

# With W&B logging
python run_v2.py --scenario scenarios/v2/gaplock_sac.yaml --wandb

# With rendering
python run_v2.py --scenario scenarios/v2/gaplock_sac.yaml --render

# Custom episodes
python run_v2.py --scenario scenarios/v2/gaplock_sac.yaml --episodes 2000
```

## Creating New Scenarios

Use existing v2 scenarios as templates. Key structure:

```yaml
experiment:
  name: my_experiment
  episodes: 1500
  seed: 42

environment:
  map: maps/line2/line2.yaml
  num_agents: 2
  max_steps: 2500
  timestep: 0.01

agents:
  car_0:
    role: attacker
    algorithm: sac
    observation:
      preset: gaplock
    reward:
      preset: gaplock_full
      overrides:
        terminal:
          target_crash: 100.0
          self_crash: -10.0

  car_1:
    role: target
    algorithm: ftg  # or ppg, pure_pursuit, etc.

wandb:
  enabled: true
  project: f110-marl
  entity: your-team
```

### Configuration Sections

**experiment:**
- `name`: Experiment name (used for output directories)
- `episodes`: Number of training episodes
- `seed`: Random seed for reproducibility

**environment:**
- `map`: Path to map YAML file
- `num_agents`: Number of agents (must match agents section)
- `max_steps`: Maximum steps per episode
- `timestep`: Physics timestep (typically 0.01)

**agents:**
- Each agent needs: `role`, `algorithm`, `observation`, `reward` (if trainable)
- Available algorithms: `sac`, `ppo`, `td3`, `rainbow`, `rec_ppo`
- Classical policies: `ftg`, `ppg`, `pure_pursuit`

**wandb:**
- `enabled`: Enable W&B logging
- `project`: W&B project name
- `entity`: W&B team/user name

### Reward Configuration

Use preset-based rewards with optional overrides:

```yaml
reward:
  preset: gaplock_full  # or gaplock_simple
  overrides:
    terminal:
      target_crash: 100.0
      self_crash: -10.0
    pressure:
      enabled: true
      distance_threshold: 2.5
    forcing:
      enabled: true
      pinch_pockets:
        enabled: true
        sigma: 0.3
```

Available presets:
- `gaplock_full` - Complete gaplock with all components
- `gaplock_simple` - Basic gaplock (terminal + distance shaping)

See [src/rewards/presets.py](../src/rewards/presets.py) for full preset definitions.

### Observation Configuration

```yaml
observation:
  preset: gaplock  # Standard gaplock observation space
```

## Examples

**Train SAC on gaplock:**
```bash
python run_v2.py --scenario scenarios/v2/gaplock_sac.yaml
```

**Evaluate trained agent:**
```bash
python run_v2.py --scenario scenarios/eval/gaplock_eval.yaml --checkpoint outputs/checkpoints/best.pt
```

**Custom reward tuning:**
Create a new YAML file with reward overrides:

```yaml
# my_custom_gaplock.yaml
experiment:
  name: custom_gaplock_high_crash_reward

environment:
  map: maps/line2/line2.yaml
  num_agents: 2

agents:
  car_0:
    role: attacker
    algorithm: sac
    observation:
      preset: gaplock
    reward:
      preset: gaplock_full
      overrides:
        terminal:
          target_crash: 200.0  # Higher crash reward
          self_crash: -50.0
        forcing:
          enabled: false  # Disable forcing rewards
```

Then run:
```bash
python run_v2.py --scenario scenarios/my_custom_gaplock.yaml
```

## Documentation

For more details:
- **Reward system**: [../docs/REWARD_SYSTEM_REMOVAL.md](../docs/REWARD_SYSTEM_REMOVAL.md)
- **Migration guide**: [../docs/MIGRATION_GUIDE.md](../docs/MIGRATION_GUIDE.md)
- **Main README**: [../README.md](../README.md)
- **V2 examples**: [../examples/](../examples/)

## Old Scenarios

Old scenario files (using deprecated task-based reward system) have been removed. All scenarios now use the modern component-based reward system in `src/rewards/`.

For migration from old scenarios, see [../docs/REWARD_SYSTEM_REMOVAL.md](../docs/REWARD_SYSTEM_REMOVAL.md).
