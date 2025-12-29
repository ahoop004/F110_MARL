# Old Reward System Removal

**Status**: Removed
**Date**: December 29, 2024

## Summary

The old task-based reward system (`src/tasks/reward/`) has been completely removed in favor of the modern component-based reward system (`src/rewards/`).

## What Was Removed

### Source Code Deleted

**src/tasks/reward/** (10 files, ~5,000 lines):
- `gaplock.py` (1,655 lines) - Monolithic gaplock implementation
- `registry.py` - Task registration system
- `components.py` - Old component system
- `composite.py` - Reward composition
- `base.py` - Base classes
- `presets.py` - Old presets
- `progress.py` - Progress tracking rewards
- `fastest_lap.py` - Fastest lap rewards
- `kamikaze.py` - Kamikaze task rewards
- `__init__.py` - Module exports

**src/wrappers/reward.py** (141 lines):
- `RewardWrapper` class
- `RewardRuntimeContext` context class

### Files Updated

- `src/tasks/__init__.py` - Marked as deprecated with migration notes
- `src/wrappers/__init__.py` - Removed RewardWrapper exports
- `src/core/config.py` - Removed `WrapperFactory.wrap_reward()` method

### Scenario Files Deleted (8 files)

- `scenarios/gaplock_sac.yaml`
- `scenarios/gaplock_ppo.yaml`
- `scenarios/gaplock_td3.yaml`
- `scenarios/gaplock_rainbow_dqn.yaml`
- `scenarios/gaplock_rec_ppo.yaml`
- `scenarios/ftg_baseline.yaml`
- `scenarios/2_ftg_baseline.yaml`
- `scenarios/REWARD_PARAMETERS.yaml`

**Note**: All deleted scenarios have equivalent v2 versions in `scenarios/v2/`

---

## Replacement: The New System

Use the component-based reward system in `src/rewards/`:

### Architecture Comparison

| Aspect | Old System | New System |
|--------|-----------|------------|
| **Location** | `src/tasks/reward/` | `src/rewards/` |
| **Architecture** | Registry-based, monolithic | Component-based, composable |
| **Configuration** | Direct parameters | Preset-based with overrides |
| **Integration** | Environment wrapper | Direct instantiation |
| **Files** | 10 files, ~5,000 lines | 13 files, cleaner separation |
| **Maintainability** | Difficult (large classes) | Easy (small components) |

### Migration Examples

**OLD scenario format (no longer works):**
```yaml
agents:
  car_0:
    reward:
      task: gaplock
      ignore_non_trainable: true
      features:
        - gaplock_offense
      params:
        target_crash_reward: 90.0
        self_collision_penalty: -90.0
        pressure_distance_threshold: 2.5
        pressure_heading_tolerance: 0.785
        # ... 200+ more parameters
```

**NEW scenario format (current system):**
```yaml
agents:
  car_0:
    reward:
      preset: gaplock_full
      overrides:
        terminal:
          target_crash: 90.0
          self_crash: -90.0
        pressure:
          distance_threshold: 2.5
          heading_tolerance: 0.785
```

**Code migration:**

```python
# OLD (no longer works)
from tasks.reward import resolve_reward_task, RewardRuntimeContext
from wrappers.reward import RewardWrapper

context = RewardRuntimeContext(env=env, map_data=map_data, roster=roster)
strategy, config, notes = resolve_reward_task(context, config=reward_config)
wrapper = RewardWrapper(config=config, context=context)

# NEW (current system)
from rewards import build_reward_strategy

reward_strategy = build_reward_strategy(
    reward_config,
    agent_id='car_0',
    target_id='car_1'
)

# In training loop
reward_info = {...}  # Build step info
total_reward, components = reward_strategy.compute(reward_info)
```

---

## Why It Was Removed

1. **Duplication**: Two parallel reward systems existed (~10,000 lines total)
2. **Complexity**: Monolithic `gaplock.py` (1,655 lines) vs component-based (400 lines)
3. **Maintainability**: Registry pattern harder to extend than composition
4. **Active development**: All v2 code uses `src/rewards/` exclusively
5. **Feature parity**: New system has all features of old system
6. **Better architecture**: Protocol-based design, clean interfaces

---

## Active Pipeline

The v2 training pipeline uses only the new system:

```
run_v2.py
  ├─> load_and_expand_scenario()
  │   └─> expand_reward_preset()  # Load preset from src/rewards/presets.py
  ├─> create_training_setup()
  │   └─> build_reward_strategy()  # Create reward from src/rewards/builder.py
  └─> EnhancedTrainingLoop()
      └─> reward.compute()  # Compute rewards each step
```

---

## Migration Guide

### For Old Scenarios

**Option 1: Use v2 equivalents (recommended)**

All deleted scenarios have v2 equivalents:

| Old Scenario | New Equivalent |
|--------------|----------------|
| `gaplock_sac.yaml` | `v2/gaplock_sac.yaml` |
| `gaplock_ppo.yaml` | `v2/gaplock_ppo.yaml` |
| `gaplock_td3.yaml` | `v2/gaplock_td3.yaml` |
| `gaplock_rainbow_dqn.yaml` | `v2/gaplock_rainbow.yaml` |

**Option 2: Convert manually**

1. Change `reward.task` to `reward.preset`
2. Move `reward.params.*` to `reward.overrides.*` (nested by component)
3. Remove `reward.features` (handled by preset)

### For Custom Reward Code

If you implemented custom reward strategies in the old system:

1. Implement `RewardComponent` protocol (see `src/rewards/base.py`)
2. Add component to a preset in `src/rewards/presets.py`
3. Use preset in scenario config

See `src/rewards/gaplock/` for reference implementations.

---

## Rollback Instructions

If you need to temporarily access the old code:

```bash
# View removal commit
git log --all --oneline -- src/tasks/reward/

# Find commit before removal
git log --all -- src/tasks/reward/ | head -20

# Restore old code (not recommended)
git checkout <commit-before-removal> -- src/tasks/reward/
git checkout <commit-before-removal> -- src/wrappers/reward.py
```

**Note**: Restored code will not integrate with current codebase without additional changes.

---

## Testing

All tests updated to use new system:

```bash
# Run reward tests
pytest tests/rewards/ -v

# Run integration tests
pytest tests/test_reward_base.py -v

# Verify deprecation
pytest tests/test_deprecated.py -v
```

---

## Benefits of Removal

✅ **Simpler codebase**: 5,000 lines removed
✅ **Single source of truth**: Only `src/rewards/` for reward logic
✅ **Cleaner architecture**: Component-based > registry-based
✅ **Easier maintenance**: No duplicate systems to sync
✅ **Better testing**: Focused coverage on one system
✅ **Clearer for new users**: One obvious way to do things

---

## Questions?

- **Old scenarios won't load**: Use equivalent v2 scenarios in `scenarios/v2/`
- **Import errors**: Update imports to use `from rewards import ...`
- **Need old parameters**: Check `src/rewards/presets.py` for equivalents
- **Custom rewards**: Implement `RewardComponent` protocol (see `src/rewards/base.py`)

For detailed migration help, see:
- `docs/MIGRATION_GUIDE.md`
- `scenarios/v2/` for examples
- `src/rewards/presets.py` for available configurations
