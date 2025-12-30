# Final Reward Structure

## Terminal Rewards (Episode Outcomes)

```
┌─────────────────────────────────────────────────────────────┐
│                    TERMINAL REWARDS                          │
├─────────────────────────────────────────────────────────────┤
│ Success (target crash, attacker survives):    +200.0        │
│ Mutual Collision (both crash):                   0.0  ← NEW!│
│ Self-crash (attacker crashes alone):           -20.0        │
│ Timeout (max steps reached):                  -100.0        │
└─────────────────────────────────────────────────────────────┘
```

## Shaping Rewards (Per Step)

All reduced by 80% to prevent timeout farming:

```yaml
Pressure (stay close):
  bonus: 0.004                    # Every 5 steps when close
  streak_bonus: 0.004             # Accumulates up to 20x

Distance (proximity):
  reward_near: 0.004              # Within 1.2m
  penalty_far: 0.002              # Beyond 2.5m

Forcing (pinch maneuvers):
  pinch_pockets: 0.004            # In optimal pinch positions
  clearance: 0.003                # Target has limited clearance
  turn: 0.006                     # Target forced to turn

Heading: 0.001                    # Pointing toward target
Speed: 0.0005                     # Moving fast
Step: -0.001                      # Time penalty

Penalties: DISABLED               # No more idle/reverse penalties!
```

## Expected Episode Returns

### Scenario Analysis

```
Timeout Episode (2500 steps):
  Shaping:     ~60 (0.024/step avg)
  Step:        -2.5 (2500 × -0.001)
  Terminal:   -100
  ─────────────────
  Total:       -42.5  ← NEGATIVE (discourages timeout)

Success Episode (500 steps):
  Shaping:     ~12 (0.024/step avg)
  Step:        -0.5 (500 × -0.001)
  Terminal:   +200
  ─────────────────
  Total:      +211.5  ← POSITIVE (best outcome!)

Mutual Collision (500 steps):
  Shaping:     ~12
  Step:        -0.5
  Terminal:      0  ← NEUTRAL (neither won)
  ─────────────────
  Total:       +11.5  ← SLIGHTLY POSITIVE (better than self-crash)

Self-crash (300 steps):
  Shaping:      ~7
  Step:        -0.3
  Terminal:    -20
  ─────────────────
  Total:       -13.3  ← NEGATIVE (failure)
```

## Reward Gradient (Preference Order)

```
Best:    Success          +211.5  ← Agent strongly prefers this!
         ↑ +200 gap
Good:    Mutual Collision  +11.5  ← Aggressive play is okay
         ↑ +25 gap
Bad:     Self-crash        -13.3  ← But trying is better than...
         ↑ +29 gap
Worst:   Timeout           -42.5  ← ...giving up!
```

**Key Insight:** The gradient clearly guides the agent:
1. Try for clean success (+211)
2. If unsure, be aggressive even if mutual crash (+11)
3. Avoid passive timeout farming (-42)

## Hindsight Experience Replay (HER) Bonuses

HER augments rewards for close approaches:

```python
if distance < 0.6m:   # Very close (< 1 car length)
    bonus = +100.0
    → Store in success buffer with augmented reward

elif distance < 1.0m:  # Close approach
    bonus = +50.0
    → Store in success buffer with augmented reward

elif distance < 1.5m:  # Moderate approach
    bonus = +20.0
    → Store in success buffer with augmented reward
```

**Example HER Episode:**
```
Timeout episode that got within 0.8m:
  Original reward: -42.5
  HER augmented:   -42.5 + 100 = +57.5
  → Stored in success buffer!

Agent learns: "Getting close is valuable!"
```

## Design Rationale

### 1. Why Collision = 0.0 (Neutral)?

**Old thinking:** Mutual crash = -40 (penalize both dying)
**Problem:** Agent becomes too cautious, afraid to engage

**New thinking:** Mutual crash = 0.0 (neutral outcome)
**Rationale:**
- In pursuit tasks, mutual crash means attacker was aggressive enough
- It's not a success (target didn't crash alone)
- But it's not a failure (attacker didn't crash alone either)
- Encourages aggressive play without fear

**Gradient:**
```
Success:  +200  ← Still way better than mutual crash
Mutual:      0  ← Neutral
Self:      -20  ← Worse than mutual
Timeout:  -100  ← Worst
```

### 2. Why Timeout = -100 (Harsh)?

Without harsh timeout penalty, agent learns to "farm" shaping rewards:
```
Old (timeout = -10):
  2500 steps × 0.24 shaping = +600
  Timeout penalty = -10
  Total = +590  ← AGENT EXPLOITS THIS!

New (timeout = -100):
  2500 steps × 0.024 shaping = +60
  Timeout penalty = -100
  Total = -40  ← Timeout is bad!
```

### 3. Why Self-crash = -20 (Mild)?

Trying and failing should be better than not trying:
```
Timeline:
  Step 300: Agent tries aggressive pinch → crashes
  Return: -13.3

  vs.

  Step 2500: Agent passively follows → timeout
  Return: -42.5

Lesson: "Try harder next time, don't give up!"
```

### 4. Why Shaping Reduced 80%?

Original shaping dominated terminal rewards:
```
Old:
  Pressure: 0.02/step → 2500 steps = +50
  Distance: 0.02/step → 2500 steps = +50
  Forcing:  0.05/step → 2500 steps = +125
  Total shaping: ~+600/episode

New:
  All reduced 80% → ~+60/episode
  Now terminal rewards dominate!
```

## Monitoring During Training

Watch these reward components:

1. **Terminal rewards should dominate:**
   - Early: Mostly timeout (-100)
   - Learning: Mix of self-crash (-20), mutual (0), timeout (-100)
   - Success: More success (+200)

2. **Shaping should be small:**
   - Per-step rewards: < 0.05
   - Episode shaping total: < 100

3. **HER augmentation:**
   - Success buffer filling even at 0% success
   - Episodes with close approaches getting positive rewards

4. **Return progression:**
   ```
   Episode 0-200:   -42 → -20 (learning to get close)
   Episode 200-500: -20 → +50 (first successes)
   Episode 500-1000: +50 → +120 (consistent success)
   Episode 1000+:    +120 → +180 (mastery)
   ```

## Comparison to Old Structure

| Metric | Old | New | Change |
|--------|-----|-----|--------|
| Success reward | +100 | **+200** | +100 |
| Collision penalty | -40 | **0** | +40 |
| Self-crash penalty | -60 | **-20** | +40 |
| Timeout penalty | -10 | **-100** | -90 |
| Idle penalty | -0.05/step | **disabled** | +0.05/step |
| Pressure bonus | 0.02 | **0.004** | -80% |
| Distance reward | 0.02 | **0.004** | -80% |
| Forcing weight | 0.05 | **0.004** | -92% |

**Net effect:**
- **Success path:** Way more attractive (+211 vs +319, but cleaner gradient)
- **Timeout path:** Much worse (-42 vs +495!)
- **Aggressive play:** Encouraged (mutual crash = 0)
- **Learning signal:** Clearer (terminal rewards dominate)
