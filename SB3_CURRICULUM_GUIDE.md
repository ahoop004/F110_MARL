# SB3 Curriculum Training Guide

## Overview

The SB3 baseline script (`run_sb3_baseline.py`) now **fully supports curriculum learning** as specified in your scenario YAML files. Everything is controlled by the scenario configuration.

## What's New

### ✅ Features Now Supported

1. **Custom Reward Strategies** - Uses your gaplock reward config from scenarios
2. **Spawn Curriculum** - Environment-level difficulty progression (from `environment.spawn_curriculum`)
3. **Phased Curriculum** - Multi-stage training with advancement criteria (from `curriculum` section)
4. **FTG Opponent Scheduling** - Dynamic opponent difficulty based on curriculum stage
5. **WandB Integration** - Automatic curriculum metrics logging
6. **Scenario-Driven Config** - Episodes, seed, all parameters from YAML

### 🔧 Files Modified

1. **`src/baselines/sb3_wrapper.py`**
   - Added `reward_strategy` parameter
   - Computes custom rewards using RewardStrategy
   - Tracks episode state for reward computation

2. **`run_sb3_baseline.py`**
   - Reads all config from scenario files
   - Creates spawn curriculum if enabled
   - Creates phased curriculum callback
   - Integrates FTG schedules
   - Uses scenario values for episodes, seed, tags, notes

3. **`src/baselines/sb3_curriculum_callback.py`** (NEW)
   - Handles curriculum progression during training
   - Tracks episode completion and success rates
   - Advances through curriculum phases based on criteria
   - Updates FTG opponent parameters
   - Logs curriculum metrics to WandB

## Usage

### Basic Training (All from Scenario)

```bash
python3 run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_sb3_ppo.yaml --wandb
```

This will:
- Use 6000 episodes (from scenario)
- Use seed 42 (from scenario)
- Enable both spawn and phased curriculum
- Log to WandB with your tags and notes
- Use custom gaplock rewards
- Progress through 17 curriculum phases

### Override Specific Settings

```bash
# Custom episodes
python3 run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_sb3_ppo.yaml --wandb --episodes 10000

# Custom seed
python3 run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_sb3_ppo.yaml --wandb --seed 123
```

## What Gets Logged to WandB

### Episode Metrics (from SB3)
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/ep_len_mean` - Average episode length
- `train/policy_loss`, `train/value_loss`, etc. - Training metrics

### Curriculum Metrics (Custom)
- `curriculum/spawn/stage_index` - Current spawn curriculum stage
- `curriculum/spawn/success_rate` - Success rate for spawn curriculum
- `curriculum/phased/phase` - Current phased curriculum phase (0-16)
- `curriculum/phased/phase_episodes` - Episodes in current phase
- `curriculum/phased/success_rate` - Recent success rate
- `curriculum/phased/patience` - Patience counter (increases if not advancing)

### Reward Components (if tracked)
- Individual reward components from your gaplock config
- Stored in episode info as `reward_components`

## Curriculum Progression

### Spawn Curriculum
- Defined in `environment.spawn_curriculum`
- Progresses through stages: optimal_fixed → optimal_varied_speed → full_random
- Controlled by success rates and enable/disable rates
- FTG parameters updated via `ftg_schedule.by_stage`

### Phased Curriculum
- Defined in `curriculum.phases` (17 phases in your config)
- Each phase has:
  - **Criteria**: Success rate, avg reward, min episodes
  - **Patience**: Max episodes before forcing advance
  - **Spawn config**: Spawn points and speed ranges
  - **FTG config**: Opponent difficulty
  - **Overlap**: `keep_foundation`, `keep_previous` for mixing

- Phases advance when criteria met OR patience exceeded
- Progress logged to WandB in real-time

## Example Output

```
Loading scenario: scenarios/v2/gaplock_sb3_ppo.yaml
Creating environment...
Wrapping environment for SB3...
Using custom reward strategy for car_0

Creating spawn curriculum...
  Spawn curriculum: 3 stages, starting at 'optimal_fixed'
Loaded FTG schedules for 1 agents
Loaded phased curriculum with 17 phases

Curriculum initialized:
  Type: phased
  Phases: 17
  Starting phase: 0 - 1_foundation

Using cuda device
Curriculum callback enabled

Starting training:
  Algorithm: PPO
  Episodes: 6000
  Total timesteps: 15,000,000
  WandB run: your-run-name

============================================================
Advancing to phase 1: 2a_reduce_lock_600
  Success rate: 72.0% >= 70.0%
  Avg reward: 55.3 >= 50.0
============================================================
```

## Troubleshooting

### No Curriculum Progression
- Check that `spawn_curriculum.enabled: true` in scenario
- Verify `curriculum.phases` exists in scenario
- Check WandB for `curriculum/phased/phase` metric

### Wrong Rewards
- Verify "Using custom reward strategy for car_0" appears in output
- Check scenario has `agents.car_0.reward` section
- Monitor reward components in WandB

### FTG Not Updating
- Ensure `ftg_schedule.enabled: true` in agent config
- Check `by_stage` has entries for curriculum stages
- Look for "Applied FTG schedule" messages in output

## Scenario File Structure

Your scenario should have:

```yaml
experiment:
  episodes: 6000
  seed: 42

environment:
  spawn_curriculum:
    enabled: true
    stages: [...]
    spawn_configs: {...}

agents:
  car_0:
    algorithm: sb3_ppo
    reward:
      preset: gaplock_full
      overrides: {...}
  car_1:
    algorithm: ftg
    ftg_schedule:
      enabled: true
      by_stage: {...}

curriculum:
  type: phased
  phases:
    - name: "1_foundation"
      criteria: {...}
      spawn: {...}
      ftg: {...}

wandb:
  enabled: true
  project: marl-f110
  tags: [...]
  notes: "..."
```

## Next Steps

1. **Run training**: `python3 run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_sb3_ppo.yaml --wandb`
2. **Monitor WandB**: Check curriculum metrics, rewards, success rates
3. **Adjust scenario**: Modify curriculum criteria, FTG schedules, rewards as needed
4. **Iterate**: The scenario file controls everything - no code changes needed!
