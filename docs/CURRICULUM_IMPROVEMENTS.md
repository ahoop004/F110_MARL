# Curriculum Improvements v2

## Overview

Based on feedback about catastrophic forgetting and off-policy replay buffer staleness, we've implemented key improvements to the phased curriculum system.

## Key Changes

### 1. ✅ Overlap/Mixture Sampling

**Problem**: Completely switching to new configurations causes catastrophic forgetting.

**Solution**: Later phases now sample from multiple difficulty levels:

```yaml
- name: "3c_speed_wide"
  spawn:
    points: [spawn_pinch_right, spawn_pinch_left]
    speed_range: [0.30, 0.70]  # Current difficulty

  # NEW: Keep easier cases
  keep_foundation: 0.15  # 15% foundation (phase 0) episodes
  keep_previous: 0.25    # 25% previous phase episodes
  # Remaining 60% use current phase config
```

**Impact**:
- Prevents forgetting early skills
- Maintains diverse replay buffer for off-policy methods
- Smoother transitions between phases

### 2. ✅ Longer Patience for Off-Policy

**Problem**: SAC/TD3 need time to refill replay buffer after environment changes.

**Before**:
```yaml
patience: 200-300  # Too short for off-policy
```

**After**:
```yaml
Phase 1: patience: 500   # Foundation
Phase 2: patience: 400   # Lock reduction
Phase 3: patience: 450-500  # Speed variation
Phase 4: patience: 500-600  # Spatial variation
Phase 5: patience: 600-800  # FTG resistance
```

**Impact**:
- Off-policy methods get ~1M+ timesteps per phase
- Replay buffer can refill with new transitions
- Reduces forced advancements

### 3. ✅ More Granular FTG Progression

**Problem**: Large jumps in FTG difficulty (0.35 → 0.60) too abrupt.

**Before** (4 phases):
```
Phase 5a: steering_gain 0.35 → 0.40
Phase 5b: steering_gain 0.40 → 0.50
Phase 5c: steering_gain 0.50 → 0.60
```

**After** (7 phases):
```
Phase 5a: 0.37 (from 0.35 baseline)
Phase 5b: 0.40
Phase 5c: 0.43
Phase 5d: 0.46
Phase 5e: 0.50
Phase 5f: 0.55
Phase 5g: 0.60
```

**Impact**:
- Smoother difficulty ramp
- ~0.03-0.05 increments instead of 0.10
- More time to adapt at each level

### 4. ✅ Symmetric Speed Expansion

**Problem**: Avoid mean shifts in speed distribution.

**Implementation**:
```yaml
Phase 1: [0.44, 0.44]      # Optimal
Phase 3a: [0.42, 0.46]     # ±0.02 symmetric
Phase 3b: [0.35, 0.55]     # ±0.09 symmetric
Phase 3c: [0.30, 0.70]     # ±0.14 symmetric
```

**Impact**:
- No distribution shift, only variance increase
- Easier for value functions to generalize
- Maintains optimal speed in training mix

## Implementation Details

### Mixture Sampling Algorithm

```python
def get_current_config(sample_mixture=True):
    phase = get_current_phase()
    rand = random.random()

    if rand < phase.keep_foundation:
        return get_config(phase_idx=0)  # Foundation
    elif rand < (keep_foundation + keep_previous):
        return get_config(phase_idx=current_idx-1)  # Previous
    else:
        return phase.config  # Current
```

### Overlap Percentages

Based on curriculum stage:

| Phase Group | Foundation | Previous | Current |
|-------------|------------|----------|---------|
| 1-2 (Early) | 10%        | 15-20%   | 70-75%  |
| 3-4 (Mid)   | 15%        | 25%      | 60%     |
| 5 (Late)    | 20%        | 25%      | 55%     |

**Rationale**:
- Increase foundation overlap in later phases
- Maintain ~40-45% "easier" episodes to prevent forgetting
- Still give majority weight to current difficulty

## Performance Expectations

### Before Improvements

```
Common issues:
- Success rate drops after phase transition
- Performance oscillates
- Frequent forced advancements
- Off-policy methods struggle more than PPO
```

### After Improvements

```
Expected behavior:
- Smoother success rate curves
- Fewer forced advancements
- Off-policy methods (SAC/TD3) benefit from diverse replay
- Higher final performance on expert phases
```

## Example Scenarios

### For SAC/TD3/TQC (Off-Policy)
Use: `scenarios/v2/gaplock_sac_phased_curriculum_improved.yaml`

Features:
- Longer patience (500-800 episodes)
- 40-45% overlap in later phases
- 7 FTG sub-phases
- Total: ~6000 episodes

### For PPO (On-Policy)
Can use shorter patience:
```yaml
patience: 300-500  # On-policy doesn't need as long
```

PPO doesn't suffer from replay buffer staleness, so can progress faster.

## Monitoring in WandB

New metrics logged:

```python
curriculum/mixture_used        # Which difficulty level was sampled
curriculum/foundation_rate     # Fraction using foundation config
curriculum/previous_rate       # Fraction using previous config
curriculum/current_rate        # Fraction using current config
```

Look for:
- ✅ Smooth success_rate curves (not jagged)
- ✅ Fewer forced advancements
- ✅ Success rate maintains or slowly decreases after transition (not crashes)
- ⚠️ If success_rate drops >30% after transition: increase overlap or reduce difficulty jump

## Comparison: Original vs Improved

| Aspect | Original | Improved | Benefit |
|--------|----------|----------|---------|
| **Overlap** | None | 40-45% | Prevents forgetting |
| **Patience** | 200-300 | 400-800 | Off-policy friendly |
| **FTG Phases** | 3 | 7 | Smoother progression |
| **Speed Expansion** | Symmetric | Symmetric | ✓ (already good) |
| **Total Episodes** | 5000 | 6000 | More thorough training |
| **Phase Count** | 14 | 17 | More granular |

## Usage

### Train with Improved Curriculum

```bash
# SAC with improved curriculum
python run_v2.py \
  --scenario scenarios/v2/gaplock_sac_phased_curriculum_improved.yaml \
  --wandb

# Monitor key metrics
wandb: curriculum/success_rate
wandb: curriculum/advancement
wandb: curriculum/forced_advancement
```

### Compare Old vs New

```bash
# Old curriculum
wandb sweep sweeps/curriculum_baseline.yaml

# New curriculum
wandb sweep sweeps/curriculum_improved.yaml

# Compare in WandB:
# - Final success rate
# - Number of forced advancements
# - Learning curves smoothness
```

## Ablation Studies

Test individual components:

1. **Overlap only**: Set `keep_foundation=0.2, keep_previous=0.25`
2. **Patience only**: Set `patience=600`, no overlap
3. **FTG granularity only**: 7 phases, no overlap
4. **All combined**: Recommended configuration

Expected impact:
- Overlap: +10-15% final success rate
- Patience: -30% forced advancements
- FTG granularity: +5-10% in late phases
- Combined: +15-25% overall improvement

## Troubleshooting

### Still seeing catastrophic forgetting?

**Increase overlap**:
```yaml
keep_foundation: 0.25  # up from 0.20
keep_previous: 0.30    # up from 0.25
```

### Phases advancing too slowly?

**Reduce patience OR lower success thresholds**:
```yaml
patience: 500          # down from 600
success_rate: 0.60     # down from 0.65
```

### Performance plateau in late phases?

**Check if too much overlap** (stuck on easy cases):
```yaml
keep_foundation: 0.10  # reduce from 0.20
keep_previous: 0.15    # reduce from 0.25
```

Balance: Need enough overlap to prevent forgetting, but not so much that agent doesn't face hard cases.

## Future Work

Potential further improvements:

- [ ] **Adaptive overlap**: Automatically adjust based on performance
- [ ] **Temperature-based sampling**: Gradually shift from easy→hard over time
- [ ] **Multi-metric gates**: Require both success_rate AND novel achievements
- [ ] **Curriculum replay**: Dedicated buffer for foundation episodes
- [ ] **Phase-specific hyperparameters**: Different learning rates per phase

## References

Related work on curriculum learning:
- Narvekar et al. (2020) "Curriculum Learning for Reinforcement Learning Domains"
- Justesen et al. (2019) "Procedural Content Generation and Difficulty Adaptation"
- Akkaya et al. (2019) "Solving Rubik's Cube with a Robot Hand" (uses domain randomization + curriculum)

## Summary

The improved curriculum addresses key failure modes:

1. ✅ **Catastrophic forgetting** → Overlap sampling
2. ✅ **Replay staleness** → Longer patience
3. ✅ **Abrupt transitions** → Granular FTG progression
4. ✅ **Distribution shift** → Symmetric expansion

Expected improvement: **15-25% higher final performance** with **smoother learning curves**.
