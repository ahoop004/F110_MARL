# HER Implementation + Algorithm Optimizations

## Summary of Changes

### 1. ✅ Hindsight Experience Replay (HER) - IMPLEMENTED

**What is HER?**
HER addresses sparse reward problems by relabeling failed episodes as successes for "easier" goals. For your pursuit task, when the agent gets close but doesn't crash, we treat those transitions as partial successes.

**Implementation:**
- Added `store_hindsight_transition()` method to TD3 and SAC agents
- Integrated into training loop with automatic distance calculation
- No architecture changes needed - works with existing code!

**HER Distance Thresholds:**
```python
if distance < 0.6m:   # Very close (< 1 car length)
    bonus = +100.0    # Large bonus
    → Store in success buffer

elif distance < 1.0m:  # Close approach
    bonus = +50.0     # Medium bonus
    → Store in success buffer

elif distance < 1.5m:  # Moderate approach
    bonus = +20.0     # Small bonus
    → Store in success buffer
```

**Expected Impact:**
- Dramatically speeds up learning from "near misses"
- Success buffer will fill up even at 0% success rate
- Agent learns: "getting close is good, crashing is better!"
- With success_buffer_ratio=0.3, ~30% of training batches will contain HER transitions

**Files Modified:**
- [src/agents/td3/td3.py](src/agents/td3/td3.py) - Added HER method
- [src/agents/sac/sac.py](src/agents/sac/sac.py) - Added HER method
- [src/core/enhanced_training.py](src/core/enhanced_training.py) - Integrated HER into training loop

---

### 2. ✅ Idle Penalty DISABLED

**Problem:**
- Agent velocity: 0.01 m/s
- Idle threshold: 0.12 m/s
- Penalty: -0.05 per step = -125 over 2500 steps!
- This was dominating the reward signal

**Fix:** Disabled behavior penalties entirely
```yaml
penalties:
  enabled: false
```

**Rationale:**
- `prevent_reverse` already prevents the agent from stopping/reversing
- Idle penalty was conflicting with prevent_reverse (0.01 min speed < 0.12 threshold)
- Agent was being punished for moving slowly, which is sometimes necessary for precise pinch maneuvers

**Expected Return Changes:**
```
Before (with idle penalty):
  Timeout: +60 shaping - 2.5 step - 125 idle - 100 terminal = -167.5

After (without idle penalty):
  Timeout: +60 shaping - 2.5 step - 100 terminal = -42.5
  Success: +12 shaping - 0.5 step + 200 terminal = +211.5

Advantage for success: +254 points!
```

---

### 3. ✅ TD3 Optimizations

All changes in [scenarios/v2/gaplock_td3_easier.yaml](scenarios/v2/gaplock_td3_easier.yaml):

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `gamma` | 0.99 | **0.97** | Faster credit assignment in long episodes |
| `hidden_dims` | [256, 256] | **[512, 512]** | More capacity for complex pinch behaviors |
| `success_buffer_ratio` | 0.1 | **0.3** | Learn heavily from successes + HER transitions |
| `learning_starts` | 10000 | **20000** | More diverse initial data |
| `exploration_noise` | 0.1 | **0.2** | Higher initial exploration |
| `exploration_noise_final` | 0.02 | **0.05** | Keep exploration throughout |
| `exploration_noise_decay_steps` | 100k | **150k** | More gradual decay |
| `target_noise` | 0.2 | **0.1** | More stable target estimates |
| `target_noise_clip` | 0.5 | **0.3** | Tighter clipping for precision |

**Curriculum also made more conservative:**
- `min_episode`: 100 → **200**
- `activation_samples`: 50 → **100**
- `enable_patience`: 7 → **10**
- Stage enable_rates increased by 5%

---

### 4. ✅ SAC Optimizations

All changes in [scenarios/v2/gaplock_sac_easier.yaml](scenarios/v2/gaplock_sac_easier.yaml):

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `gamma` | 0.995 | **0.97** | Faster credit assignment |
| `hidden_dims` | [256, 256] | **[512, 512]** | More capacity |
| `success_buffer_ratio` | 0.1 | **0.3** | Learn from HER transitions |
| `warmup_steps` | 5000 | **20000** | More initial exploration |

---

### 5. ✅ PPO Optimizations

All changes in [scenarios/v2/gaplock_ppo_easier.yaml](scenarios/v2/gaplock_ppo_easier.yaml):

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `gamma` | 0.995 | **0.97** | Faster credit assignment |
| `lam` (GAE) | 0.95 | **0.90** | Lower GAE lambda = less bias, faster credit |
| `hidden_dims` | [256, 256] | **[512, 512]** | More capacity |
| `update_epochs` | 10 | **20** | Learn more from each rollout |
| `batch_size` | 2048 | **4096** | Longer rollouts = more data |
| `minibatch_size` | 128 | **256** | Larger minibatches for stability |
| `ent_coef` | 0.01 | **0.05** | More exploration |
| `ent_coef_schedule.start` | 0.02 | **0.08** | Start with high exploration |

**PPO-specific notes:**
- GAE lambda reduction is critical for sparse rewards
- Larger batch_size (4096) means agent collects more experience before each update
- More update_epochs (20) means more learning from that experience
- Higher entropy = more random exploration early on

---

## Expected Training Dynamics

### Phase 1: Random Exploration (Episodes 0-200)
**What to expect:**
- Agent exploring randomly with high noise/entropy
- Lots of timeouts and crashes
- Return: -42 to -20
- **HER kicks in:** Even without successes, close approaches fill success buffer
- Success buffer accumulation: ~5-10 close approaches per episode → ~1000 HER transitions by episode 200

### Phase 2: Learning from HER (Episodes 200-500)
**What to expect:**
- Agent learns: "getting close = good"
- More aggressive pursuit behaviors
- First real successes start appearing (1-5%)
- Return: -20 to +50
- Success buffer contains mix of HER transitions + real successes
- Curriculum may advance to random_medium

### Phase 3: Success Refinement (Episodes 500-1000)
**What to expect:**
- Agent learns to convert close approaches into crashes
- Success rate climbing: 5% → 15%
- Return: +50 to +120
- HER transitions become less important as real successes dominate
- Curriculum advancing through stages

### Phase 4: Mastery (Episodes 1000-1500)
**What to expect:**
- Consistent pinch behaviors
- Success rate: 20%+
- Return: +120 to +180
- Full random spawns and speeds

---

## Monitoring HER Effectiveness

Track these metrics in W&B:

1. **Success buffer size:** Should grow steadily even at 0% success
2. **Average distance to target:** Should decrease over time
3. **Close approach frequency:** Episodes with distance < 1.0m
4. **Success rate progression:** Should show smooth improvement (not stuck at 0%)

---

## If Still Not Converging

### Try increasing HER aggressiveness:
```python
# In store_hindsight_transition():
if distance_to_target < 0.8:   # Was 0.6
    her_bonus = 150.0          # Was 100.0
elif distance_to_target < 1.5:  # Was 1.0
    her_bonus = 75.0           # Was 50.0
elif distance_to_target < 2.5:  # Was 1.5
    her_bonus = 30.0           # Was 20.0
```

### Try even higher exploration:
```yaml
# TD3:
exploration_noise: 0.3         # Was 0.2

# SAC:
alpha: 0.5                     # Was 0.2

# PPO:
ent_coef: 0.10                 # Was 0.05
```

### Try lower gamma:
```yaml
gamma: 0.95   # Was 0.97 (very aggressive discounting)
```

---

## Files Modified

### Core Algorithm Changes:
- ✅ [src/agents/td3/td3.py](src/agents/td3/td3.py)
- ✅ [src/agents/sac/sac.py](src/agents/sac/sac.py)
- ✅ [src/core/enhanced_training.py](src/core/enhanced_training.py)

### Scenario Configurations:
- ✅ [scenarios/v2/gaplock_td3_easier.yaml](scenarios/v2/gaplock_td3_easier.yaml)
- ✅ [scenarios/v2/gaplock_sac_easier.yaml](scenarios/v2/gaplock_sac_easier.yaml)
- ✅ [scenarios/v2/gaplock_ppo_easier.yaml](scenarios/v2/gaplock_ppo_easier.yaml)

### All Other Scenarios (reward rebalancing only):
- ✅ [scenarios/v2/gaplock_td3.yaml](scenarios/v2/gaplock_td3.yaml)
- ✅ [scenarios/v2/gaplock_sac.yaml](scenarios/v2/gaplock_sac.yaml)
- ✅ [scenarios/v2/gaplock_ppo.yaml](scenarios/v2/gaplock_ppo.yaml)
- ✅ [scenarios/v2/gaplock_rainbow_easier.yaml](scenarios/v2/gaplock_rainbow_easier.yaml)
- ✅ [scenarios/v2/gaplock_rainbow.yaml](scenarios/v2/gaplock_rainbow.yaml)

---

## Quick Test

Run a quick test to verify HER is working:

```bash
# TD3 with HER
python3 run_v2.py scenarios/v2/gaplock_td3_easier.yaml

# After ~50 episodes, check:
# 1. No more idle penalties in reward components
# 2. Agent moving faster (velocity > 0.1)
# 3. Success buffer filling up (check logs)
```

---

## Summary

**Three critical fixes applied:**

1. **HER:** Learn from near-misses, not just successes
2. **Idle penalty removed:** Agent can move slowly for precision
3. **Hyperparameters optimized:** Faster credit assignment, more exploration, more capacity

**Expected outcome:** Agent should start showing learning signals within 200 episodes and achieve 10%+ success rate by episode 500-800.
