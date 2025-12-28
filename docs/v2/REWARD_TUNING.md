# Reward Parameter Tuning Guide

## Overview

All reward parameters can be tuned directly in scenario YAML files using the `overrides` section. This allows you to experiment with different reward shaping strategies without modifying preset files.

## Scenario Structure

```yaml
agents:
  car_0:
    reward:
      preset: gaplock_full  # Base preset

      overrides:            # Parameter overrides
        # Override any component here
```

## Available Parameters

### 1. Forcing Rewards (Gaussian Pinch Pockets)

Controls the Gaussian reward hotspots that guide the attacker to optimal positions.

```yaml
forcing:
  enabled: true

  pinch_pockets:
    enabled: true
    anchor_forward: 1.20  # Distance ahead of target (meters)
    anchor_lateral: 0.70  # Distance to side of target (meters)
    sigma: 0.50           # Gaussian width (meters, larger = wider hotspot)
    weight: 0.30          # Reward multiplier (larger = stronger attraction)
```

**Tuning Tips:**
- **anchor_forward**: Position the hotspot ahead (1.0-1.5m works well)
- **anchor_lateral**: Side offset for pinch position (0.5-0.9m typical)
- **sigma**: Controls how wide the reward zone is (0.3-0.7m range)
- **weight**: Overall strength of pinch attraction (0.1-0.5 typical)

**Visual Effect in Heatmap:**
- Smaller `sigma` → Tighter green hotspots (requires more precision)
- Larger `sigma` → Broader green zones (easier to find)
- Higher `weight` → Brighter green (stronger reward signal)

---

### 2. Clearance Reduction

Rewards the attacker when the target's wall clearance decreases.

```yaml
forcing:
  clearance:
    enabled: true
    weight: 0.80          # Clearance reduction multiplier
    band_min: 0.30        # Min clearance for reward (meters)
    band_max: 3.20        # Max clearance for reward (meters)
    clip: 0.25            # Max reward per step
    time_scaled: true     # Scale by timestep
```

**Tuning Tips:**
- **weight**: Higher values encourage forcing target toward walls
- **band_min/max**: Only reward within this clearance range
- **clip**: Prevents excessive rewards from rapid clearance changes

---

### 3. Turn Shaping

Rewards when the target turns (indicates reacting to pressure).

```yaml
forcing:
  turn:
    enabled: true
    weight: 2.0           # Turn shaping multiplier
    clip: 0.35            # Max reward per step
    time_scaled: true     # Scale by timestep
```

**Tuning Tips:**
- **weight**: Higher values encourage forcing target to turn
- Useful for learning aggressive forcing behavior

---

### 4. Distance-Based Shaping

Simple distance-based reward for staying in optimal attack range.

```yaml
distance:
  enabled: true
  near_distance: 1.0      # Close range threshold (meters)
  far_distance: 2.5       # Far range threshold (meters)
  reward_near: 0.12       # Reward when within near_distance
  penalty_far: 0.08       # Penalty when beyond far_distance
```

**Tuning Tips:**
- **near_distance**: Ideal attack distance (0.8-1.5m typical)
- **far_distance**: Max effective range (2.0-3.0m typical)
- Use smaller ranges for aggressive behavior
- Visualized as colored rings in reward ring extension

---

### 5. Heading Alignment

Rewards facing toward the target.

```yaml
heading:
  enabled: true
  coefficient: 0.08       # Alignment bonus coefficient
```

**Tuning Tips:**
- **coefficient**: Higher values encourage facing target (0.05-0.15 typical)
- Essential for learning to track and follow target

---

### 6. Speed Bonus

Rewards higher speed (encourages aggressive pursuit).

```yaml
speed:
  enabled: true
  bonus_coef: 0.05        # Speed bonus coefficient
```

**Tuning Tips:**
- **bonus_coef**: Higher values encourage faster movement (0.03-0.10 typical)
- Helps prevent passive/defensive behavior

---

### 7. Terminal Rewards

Large rewards/penalties for episode outcomes.

```yaml
terminal:
  target_crash: 60.0      # Target crashes (success)
  self_crash: -40.0       # Self crashes (failure)
  timeout: 0.0            # Episode timeout
```

**Tuning Tips:**
- **target_crash**: Main success signal (50-100 typical)
- **self_crash**: Discourages reckless behavior (-30 to -50 typical)
- **timeout**: Usually 0 (neutral) or small negative (encourages faster success)

---

## Example: Tuning Pinch Pockets

### Making Hotspots Tighter and Stronger

```yaml
forcing:
  pinch_pockets:
    anchor_forward: 1.20
    anchor_lateral: 0.70
    sigma: 0.35           # ← Reduced from 0.50 (tighter)
    weight: 0.50          # ← Increased from 0.30 (stronger)
```

**Effect**: Narrower, brighter green hotspots in heatmap. Agent must position more precisely but gets stronger reward signal.

### Moving Hotspots Closer

```yaml
forcing:
  pinch_pockets:
    anchor_forward: 0.90  # ← Reduced from 1.20
    anchor_lateral: 0.50  # ← Reduced from 0.70
    sigma: 0.50
    weight: 0.30
```

**Effect**: Pinch positions closer to target (more aggressive forcing).

---

## Testing Parameter Changes

### 1. Enable Heatmap Visualization

Set in environment section:
```yaml
environment:
  render: true
  visualization:
    heatmap:
      enabled: false  # Toggle with H key during training
```

### 2. Run Training

```bash
python3 v2/run.py scenarios/v2/gaplock_sac.yaml
```

### 3. Toggle Heatmap

Press **H** during training to visualize the reward field. You'll see:
- Parameter printout showing your override values
- 2D heatmap with green hotspots at pinch positions
- Red/yellow gradient showing distance rewards

### 4. Observe Behavior

Watch how the agent learns:
- Does it find the pinch pockets?
- Is it positioning correctly?
- Does it force the target effectively?

### 5. Iterate

Adjust parameters based on observed behavior:
- Hotspots too small → Increase `sigma`
- Agent ignores pinches → Increase `weight`
- Too aggressive/crashes → Decrease `weight`, increase `self_crash` penalty
- Too passive → Increase `speed.bonus_coef`, decrease `near_distance`

---

## Complete Example Scenario

See [scenarios/v2/gaplock_sac.yaml](../../scenarios/v2/gaplock_sac.yaml) for a fully annotated example with all tunable parameters exposed.

---

## Quick Reference: Key Parameters to Tune

For most experiments, focus on these:

| Parameter | What It Does | Typical Range |
|-----------|--------------|---------------|
| `forcing.pinch_pockets.weight` | Strength of pinch attraction | 0.1 - 0.5 |
| `forcing.pinch_pockets.sigma` | Width of pinch zone | 0.3 - 0.7 m |
| `distance.near_distance` | Optimal attack range | 0.8 - 1.5 m |
| `terminal.target_crash` | Success reward | 50 - 100 |
| `terminal.self_crash` | Crash penalty | -30 to -50 |
| `speed.bonus_coef` | Aggression level | 0.03 - 0.10 |

---

**Last Updated**: December 26, 2024
