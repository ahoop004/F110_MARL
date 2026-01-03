## Phased Curriculum Learning System

A modular curriculum learning system that progressively increases task difficulty across multiple orthogonal dimensions to improve training efficiency and final performance.

## Overview

The phased curriculum system automatically adjusts training difficulty based on agent performance, starting from simple scenarios and gradually introducing:

1. **Speed Lock Reduction** - Defender can adjust speed mid-episode
2. **Speed Variation** - Wider range of defender speeds
3. **Spatial Variation** - More diverse spawn positions
4. **FTG Resistance** - Increasingly competent defender

## Key Features

✅ **Automatic Progression** - Advances based on success rate and reward thresholds
✅ **Patience Mechanism** - Prevents getting stuck on difficult transitions
✅ **Discrete Phases** - Clear boundaries for debugging and analysis
✅ **WandB Integration** - Logs phase transitions and metrics
✅ **Checkpointing** - Saves and resumes curriculum state
✅ **Flexible Configuration** - Easy to customize phases and criteria

## Quick Start

### 1. Create a Curriculum Scenario

```yaml
# scenarios/v2/my_curriculum.yaml
curriculum:
  type: phased
  start_phase: 0  # Optional: start from specific phase

  phases:
    - name: "1_foundation"
      description: "Learn basics against weak defender"
      criteria:
        success_rate: 0.70
        avg_reward: 50.0
        min_episodes: 50
        patience: 200
      spawn:
        points: [spawn_pinch_right, spawn_pinch_left]
        speed_range: [0.44, 0.44]
      lock_speed_steps: 800
      ftg:
        steering_gain: 0.25
        bubble_radius: 2.0

    - name: "2_medium"
      description: "Handle speed variation"
      criteria:
        success_rate: 0.60
        min_episodes: 50
        patience: 250
      spawn:
        points: [spawn_pinch_right, spawn_pinch_left]
        speed_range: [0.35, 0.55]
      lock_speed_steps: 100
      ftg:
        steering_gain: 0.30
        bubble_radius: 2.2

    # ... more phases
```

### 2. Run Training

```bash
python run_v2.py --scenario scenarios/v2/my_curriculum.yaml --wandb
```

The curriculum will automatically:
- Start at phase 1
- Track success rate and reward
- Advance when criteria are met
- Log transitions to console and WandB

## Configuration

### Phase Configuration

Each phase defines:

```yaml
- name: "phase_identifier"           # Unique phase name
  description: "Human description"    # What this phase trains

  criteria:                           # Advancement criteria
    success_rate: 0.70               # Min success rate (0.0-1.0)
    avg_reward: 50.0                 # Optional: min avg reward
    min_episodes: 50                 # Min episodes before advance
    patience: 200                    # Max episodes before forced advance
    window_size: 100                 # Rolling window size

  spawn:                              # Spawn configuration
    points: [spawn_1, spawn_2]       # Or 'all' for all points
    speed_range: [0.44, 0.44]        # [min, max] defender speed

  lock_speed_steps: 800               # Steps defender speed is locked

  ftg:                                # FTG defender parameters
    steering_gain: 0.25
    bubble_radius: 2.0
    safety_margin: 0.05
    # ... any FTG parameter
```

### Advancement Criteria

Phases advance when:

```python
(success_rate >= threshold AND avg_reward >= threshold)
OR
episodes_in_phase >= patience
```

**Success Rate**: Fraction of `target_crash` outcomes in rolling window
**Avg Reward**: Mean episode reward in rolling window
**Min Episodes**: Safety threshold to ensure sufficient data
**Patience**: Forces advancement to prevent infinite loops

### Curriculum Metrics (WandB)

The following metrics are logged:

```
curriculum/phase_idx              - Current phase index
curriculum/phase_name             - Current phase name
curriculum/episodes_in_phase      - Episodes in current phase
curriculum/progress               - Overall progress (0.0-1.0)
curriculum/success_rate           - Rolling success rate
curriculum/avg_reward             - Rolling average reward
curriculum/advancement            - 1 when phase advances, else 0
curriculum/forced_advancement     - 1 if forced by patience
```

## Example Curriculum: Foundation → Expert

See [scenarios/v2/gaplock_ppo_phased_curriculum.yaml](../scenarios/v2/gaplock_ppo_phased_curriculum.yaml) for a complete 14-phase curriculum:

```
Phase 1: Foundation (800 step lock, 0.44 speed, 2 spawns, weak FTG)
  ↓ Success Rate: 70%
Phase 2a-2c: Reduce Lock (800→600→400→200 steps)
  ↓ Success Rate: 65% per phase
Phase 3a-3c: Speed Variation (tight→medium→wide range)
  ↓ Success Rate: 60%→55%→50%
Phase 4a-4c: Spatial Variation (add spawns gradually)
  ↓ Success Rate: 50%→45%→40%
Phase 5a-5c: FTG Resistance (weak→moderate→strong→expert)
  ↓ Success Rate: 35%→30%→25%
Complete!
```

## Testing

Test the curriculum system standalone:

```bash
python test_curriculum.py
```

This runs a simplified simulation to verify:
- Phase creation and configuration
- Metric tracking
- Advancement logic
- Patience mechanism

## Advanced Usage

### Starting from a Specific Phase

For ablation studies or debugging:

```yaml
curriculum:
  type: phased
  start_phase: 5  # Start from phase index 5
```

### Custom Advancement Logic

Modify `PhaseBasedCurriculum._check_advancement()` in [src/curriculum/phased_curriculum.py](../src/curriculum/phased_curriculum.py) to implement custom advancement logic.

### Curriculum + Sweeps

You can sweep curriculum parameters:

```yaml
# sweep config
parameters:
  curriculum.phases[0].criteria.success_rate:
    values: [0.60, 0.70, 0.80]
  curriculum.phases[0].criteria.patience:
    values: [150, 200, 250]
```

### Checkpointing Curriculum State

Curriculum state is automatically saved/loaded with model checkpoints. To manually manage:

```python
from src.curriculum.curriculum_env import (
    create_curriculum_checkpoint_data,
    restore_curriculum_from_checkpoint
)

# Save
checkpoint_data = create_curriculum_checkpoint_data(curriculum)

# Load
restore_curriculum_from_checkpoint(curriculum, checkpoint_data)
```

## Design Principles

1. **One-Way Progression** - No automatic regression to prevent instability
2. **Discrete Phases** - Clear boundaries for reproducibility
3. **Patience Over Perfection** - Forces advancement to avoid infinite loops
4. **Multi-Metric Gates** - Both success rate AND reward must improve
5. **Logging First** - Rich logging for analysis and debugging

## Troubleshooting

### Agent Stuck on a Phase

**Symptoms**: Episodes exceed patience repeatedly, forced advancements

**Solutions**:
- Lower success_rate threshold for that phase
- Increase patience to allow more learning time
- Check if phase jump is too large (add intermediate phase)
- Verify FTG parameters aren't too strong

### Advancing Too Quickly

**Symptoms**: Later phases have low success rates, agent struggles

**Solutions**:
- Increase min_episodes to require more data
- Raise success_rate thresholds
- Add intermediate phases for smoother transitions
- Check if early phases are too easy

### Performance Drops After Advancement

**Symptoms**: Success rate drops significantly in new phase

**Solutions**:
- This is normal for difficult transitions
- Patience will force advancement if truly stuck
- Consider adding intermediate phase
- May need to tune hyperparameters (learning rate, entropy)

## API Reference

### PhaseBasedCurriculum

Main curriculum class.

```python
from src.curriculum import PhaseBasedCurriculum

curriculum = PhaseBasedCurriculum.from_config(config_dict)

# Get current phase
phase = curriculum.get_current_phase()
config = curriculum.get_current_config()

# Update after episode
advancement_info = curriculum.update(
    episode_outcome='target_crash',
    episode_reward=150.0,
    episode_num=42
)

# Get metrics for logging
metrics = curriculum.get_metrics()

# Check if complete
if curriculum.is_complete():
    print("All phases completed!")
```

### Phase

Individual phase configuration.

```python
from src.curriculum import Phase, AdvancementCriteria

phase = Phase(
    name="1_foundation",
    description="Learn basics",
    criteria=AdvancementCriteria(
        success_rate=0.70,
        min_episodes=50,
        patience=200
    ),
    spawn_config={'points': ['spawn_1']},
    ftg_config={'steering_gain': 0.25},
    lock_speed_steps=800
)
```

### Training Integration

```python
from src.curriculum.training_integration import (
    setup_curriculum_from_scenario,
    add_curriculum_to_training_loop
)

# Automatic setup from scenario
curriculum = setup_curriculum_from_scenario(scenario, training_loop)

# Manual setup
add_curriculum_to_training_loop(training_loop, curriculum)
```

## Files

```
src/curriculum/
├── __init__.py                 - Public API exports
├── phased_curriculum.py        - Core curriculum classes
├── curriculum_env.py           - Environment integration
└── training_integration.py     - Training loop integration

scenarios/v2/
└── gaplock_ppo_phased_curriculum.yaml  - Example 14-phase curriculum

test_curriculum.py              - Standalone test script
docs/CURRICULUM.md             - This file
```

## Citation

If you use this curriculum system in your research, please cite:

```bibtex
@software{f110_phased_curriculum,
  title = {Phased Curriculum Learning for F110 Multi-Agent RL},
  year = {2025},
  url = {https://github.com/yourusername/F110_MARL}
}
```

## Future Enhancements

Potential extensions:
- [ ] Multi-agent curriculum (different phases per agent)
- [ ] Continuous parameter interpolation (vs discrete phases)
- [ ] Automatic phase discovery via meta-learning
- [ ] Curriculum visualization dashboard
- [ ] Phase-specific reward shaping
