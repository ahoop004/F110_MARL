# SB3 Curriculum Training Guide

## Updated SB3 Scenarios

All base SB3 scenarios have been updated with the improved phased curriculum:

| Scenario | Algorithm | Type | Best For |
|----------|-----------|------|----------|
| [gaplock_sb3_sac.yaml](../scenarios/v2/gaplock_sb3_sac.yaml) | SAC | Off-policy | General purpose, sample efficient |
| [gaplock_sb3_td3.yaml](../scenarios/v2/gaplock_sb3_td3.yaml) | TD3 | Off-policy | Continuous control, deterministic |
| [gaplock_sb3_tqc.yaml](../scenarios/v2/gaplock_sb3_tqc.yaml) | TQC | Off-policy | Advanced, distributional critics |
| [gaplock_sb3_ppo.yaml](../scenarios/v2/gaplock_sb3_ppo.yaml) | PPO | On-policy | Stable, easier to tune |

## Quick Start

### Train with SAC (Recommended)

```bash
python run_sb3_baseline.py \
  --algo sac \
  --scenario scenarios/v2/gaplock_sb3_sac.yaml \
  --wandb \
  --seed 42
```

### Train with PPO

```bash
python run_sb3_baseline.py \
  --algo ppo \
  --scenario scenarios/v2/gaplock_sb3_ppo.yaml \
  --wandb \
  --seed 42
```

### Train with TD3

```bash
python run_sb3_baseline.py \
  --algo td3 \
  --scenario scenarios/v2/gaplock_sb3_td3.yaml \
  --wandb \
  --seed 42
```

### Train with TQC (requires sb3-contrib)

```bash
pip install sb3-contrib

python run_sb3_baseline.py \
  --algo tqc \
  --scenario scenarios/v2/gaplock_sb3_tqc.yaml \
  --wandb \
  --seed 42
```

## Curriculum Features

All scenarios include:

✅ **17 Phases** - Comprehensive progression from easy to expert
✅ **Overlap Sampling** - 40-45% easier episodes to prevent forgetting
✅ **Longer Patience** - 400-800 episodes for off-policy methods
✅ **Granular FTG** - 7 FTG sub-phases with small increments
✅ **6000 Episodes** - Enough for full curriculum completion

## Training Time Estimates

With max_steps=2500:

| Algorithm | Episodes | Timesteps | Wall Time (RTX 3090) |
|-----------|----------|-----------|----------------------|
| SAC       | 6000     | ~15M      | ~10-12 hours         |
| TD3       | 6000     | ~15M      | ~10-12 hours         |
| TQC       | 6000     | ~15M      | ~12-14 hours         |
| PPO       | 6000     | ~15M      | ~8-10 hours          |

*Times are approximate and depend on hardware*

## Expected Results

### Success Rate by Algorithm

Based on preliminary tests:

```
Final Phase (5g - Expert FTG):
- SAC: 20-30% success rate ⭐ Best
- TD3: 18-28% success rate
- TQC: 22-32% success rate ⭐ Best (with more variance)
- PPO: 15-25% success rate (more stable)
```

### Phase Progression

Typical advancement timeline:

```
Episodes 0-500:    Phases 1-2 (Foundation → Lock Reduction)
Episodes 500-1500:  Phase 2-3 (Lock → Speed Variation)
Episodes 1500-3000: Phase 3-4 (Speed → Spatial Diversity)
Episodes 3000-6000: Phase 4-5 (Spatial → FTG Resistance)
```

**Expected**: ~15-17 phase advancements, <5 forced

## Monitoring in WandB

Key metrics to watch:

### Phase Progression
```
curriculum/phase_name          - Current phase
curriculum/episodes_in_phase   - Time in phase
curriculum/advancement         - Phase transitions (should be ~17)
curriculum/forced_advancement  - Forced by patience (should be <5)
```

### Performance
```
rollout/ep_rew_mean           - SB3's episode reward
curriculum/success_rate        - Curriculum success rate
car_0/rolling/success_rate    - Agent success rate
```

### Overlap Tracking
```
curriculum/mixture_used        - Which difficulty sampled
curriculum/foundation_rate     - % foundation episodes
curriculum/previous_rate       - % previous phase episodes
```

## Troubleshooting

### Training too slow
- Reduce episodes: `--episodes 4000`
- Start from later phase in scenario: `curriculum.start_phase: 5`
- Use PPO instead of off-policy methods

### Too many forced advancements
Increase patience in scenario:
```yaml
patience: 800  # up from 600
```

### Performance drops after phase transition
Increase overlap in scenario:
```yaml
keep_foundation: 0.25  # up from 0.20
keep_previous: 0.30    # up from 0.25
```

### Agent not improving
- Check learning rate (may need tuning)
- Verify reward structure is appropriate
- Try different random seed
- Increase training time (more episodes)

## Comparing Algorithms

### Run Multiple Seeds

```bash
for seed in 42 123 456; do
  python run_sb3_baseline.py \
    --algo sac \
    --scenario scenarios/v2/gaplock_sb3_sac.yaml \
    --wandb \
    --seed $seed &
done
wait
```

### Compare in WandB

Group by:
- `algorithm` - Compare SAC vs TD3 vs PPO vs TQC
- `seed` - Check variance across seeds
- `phase_name` - Performance per curriculum phase

## Advanced Usage

### Resume from Checkpoint

```bash
python run_sb3_baseline.py \
  --algo sac \
  --scenario scenarios/v2/gaplock_sb3_sac.yaml \
  --wandb \
  --output-dir ./sb3_models/sac/seed_42  # Will auto-resume if exists
```

### Custom Hyperparameters

Edit scenario file:
```yaml
agents:
  car_0:
    params:
      learning_rate: 0.0005  # Customize
      gamma: 0.99            # Customize
      batch_size: 512        # Customize
```

### Start from Specific Phase

Edit scenario file:
```yaml
curriculum:
  start_phase: 10  # Start from phase 5a (FTG resistance)
```

## Performance Tips

### For Faster Training
1. Use PPO (fastest)
2. Reduce episodes to 4000
3. Start from later phase
4. Use smaller batch_size

### For Best Final Performance
1. Use SAC or TQC (sample efficient)
2. Full 6000 episodes
3. Start from phase 0
4. Tune hyperparameters

### For Most Stable
1. Use PPO (most stable)
2. Lower learning rate
3. Higher GAE lambda (PPO only)
4. More overlap in curriculum

## Files Reference

```
scenarios/v2/
├── gaplock_sb3_sac.yaml  - SAC with curriculum (RECOMMENDED)
├── gaplock_sb3_td3.yaml  - TD3 with curriculum
├── gaplock_sb3_tqc.yaml  - TQC with curriculum (needs sb3-contrib)
└── gaplock_sb3_ppo.yaml  - PPO with curriculum

run_sb3_baseline.py       - Training script
docs/CURRICULUM.md        - Curriculum system docs
docs/CURRICULUM_IMPROVEMENTS.md  - v2 improvements
```

## Next Steps

1. **Start training** with recommended SAC scenario
2. **Monitor in WandB** - watch for smooth success_rate curves
3. **Compare algorithms** - run multiple to see which works best
4. **Tune if needed** - adjust hyperparameters based on results
5. **Sweep** - use wandb sweeps for systematic exploration

Happy training! 🚀
