# Hyperparameter Optimizations for TD3 Convergence

## Summary of Changes

### 1. Reward Structure (CRITICAL FIX)
**Problem**: Shaping rewards over 2500 steps (+600) dwarfed terminal timeout penalty (-100), making passive following optimal.

**Fix**: Reduced all shaping rewards by 80%
```yaml
# Before: Timeout = +495, Success = +319 (timeout farming!)
# After:  Timeout = -36, Success = +213 (success is +249 better!)
```

### 2. TD3 Hyperparameters

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `gamma` | 0.99 | **0.97** | Faster credit assignment in long episodes (500-2500 steps) |
| `hidden_dims` | [256, 256] | **[512, 512]** | More capacity for complex pinch behaviors |
| `success_buffer_ratio` | 0.1 | **0.3** | Learn heavily from rare successful episodes |
| `learning_starts` | 10000 | **20000** | Collect more diverse data before training |
| `exploration_noise` | 0.1 | **0.2** | Higher initial exploration to discover strategies |
| `exploration_noise_final` | 0.02 | **0.05** | Keep exploration throughout training |
| `exploration_noise_decay_steps` | 100000 | **150000** | More gradual decay (~75 episodes) |
| `target_noise` | 0.2 | **0.1** | More stable target policy estimates |
| `target_noise_clip` | 0.5 | **0.3** | Tighter clipping for precision control |

### 3. Spawn Curriculum

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `activation_samples` | 50 | **100** | More samples for stable stage decisions |
| `min_episode` | 100 | **200** | Build basic following skills before ramping difficulty |
| `enable_patience` | 7 | **10** | More patient before advancing stages |
| `disable_patience` | 3 | **4** | Slightly more patient when regressing |
| `cooldown` | 30 | **40** | Longer cooldown between transitions |

**Stage thresholds** (all increased):
- random_slow: enable_rate 0.65 → **0.70**
- random_medium: enable_rate 0.70 → **0.75**, disable_rate 0.50 → **0.55**
- close_challenges: enable_rate 0.75 → **0.80**, disable_rate 0.55 → **0.60**
- full_random: disable_rate 0.60 → **0.65**

## Expected Impact

### Reward Gradient
```
Old Structure:
  Timeout (2500 steps):  +495 return
  Success (500 steps):   +319 return
  Problem: Agent learns passive timeout farming

New Structure:
  Timeout (2500 steps):   -36 return  ✓
  Success (500 steps):   +213 return  ✓
  Advantage: Success is +249 better!
```

### Learning Dynamics

1. **Credit Assignment**: γ=0.97 means rewards 100 steps away are worth 0.97^100 = 4.8% of immediate reward (vs 36.6% with γ=0.99). Agent focuses on recent actions leading to crash.

2. **Exploration**: Starting at 0.2 noise gives strong random exploration for first ~400k steps (200 episodes), gradually reducing to 0.05 over 150k steps.

3. **Sample Efficiency**:
   - 20k learning_starts = ~8 episodes of random exploration
   - success_buffer_ratio=0.3 = 30% of each batch from successes
   - With 1% success rate: ~10 successes over 1000 episodes = ~25k transitions in success buffer

4. **Curriculum Pacing**:
   - First 200 episodes: random_slow stage (build following skills)
   - Need 70% success in 200-episode window to advance
   - Conservative progression ensures solid skills at each stage

## Files Updated

All TD3/SAC/PPO/Rainbow scenarios updated with:
- ✓ Reward rebalancing (80% reduction in shaping)
- ✓ Terminal penalties (timeout: -100)
- ✓ TD3-specific hyperparameters (gaplock_td3_easier.yaml only)

## Monitoring Recommendations

Track these metrics during training:

1. **Early signs of learning (episodes 0-200)**:
   - Avg return increasing from -36 (timeout) toward 0
   - Avg episode length decreasing from 2500 toward 1000
   - Any successes (even 0.5% is progress!)

2. **Convergence indicators (episodes 200-800)**:
   - Success rate > 5% in rolling 100 window
   - Avg return > +50 (mix of timeouts and successes)
   - Curriculum advancing to random_medium stage

3. **Target performance (episodes 800-1500)**:
   - Success rate > 20%
   - Avg return > +100
   - Some episodes in close_challenges stage

## Alternative Hyperparameters to Try (If Still Not Converging)

### If exploration seems insufficient:
```yaml
exploration_noise: 0.3           # Even higher
exploration_noise_decay_steps: 200000  # Slower decay
```

### If learning is unstable:
```yaml
lr_actor: 0.00005    # Even lower
lr_critic: 0.0001    # Even lower
batch_size: 512      # Larger batches
```

### If credit assignment is poor:
```yaml
gamma: 0.95          # Even lower (very aggressive)
```

### If network capacity is limiting:
```yaml
hidden_dims: [512, 512, 512]  # Deeper network
```

### If defender is too easy:
```yaml
# In car_1 params:
max_speed: 0.6       # Faster defender (current: 0.45)
```
