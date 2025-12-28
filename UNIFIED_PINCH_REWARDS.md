# Unified Pinch Pocket Rewards

## Summary
Pinch pockets and potential field have been unified into a single `pinch_pockets` reward mechanism. These were previously separate rewards but computed values at the same spatial locations, so they've been merged for clarity and consistency.

## Changes Made

### 1. Code Changes

**File:** `src/rewards/gaplock/forcing.py`

**Before:**
- Two separate mechanisms:
  - `pinch_pockets`: Simple Gaussian rewards at pinch positions
  - `potential_field`: Gaussian field with peak/floor mapping
- Two separate reward components in telemetry:
  - `forcing/pinch`
  - `forcing/potential_field`

**After:**
- Single unified `pinch_pockets` mechanism
- Supports two modes:
  1. **Simple Gaussian** (default): Sum of Gaussians at left and right pockets
  2. **Potential Field**: Gaussian field with configurable peak/floor mapping
- Single reward component: `forcing/pinch`

### 2. Configuration Changes

**Before (separate sections):**
```yaml
forcing:
  pinch_pockets:
    enabled: true
    anchor_forward: 1.20
    anchor_lateral: 0.30
    sigma: 0.50
    weight: 0.30
  potential_field:
    enabled: true
    weight: 0.80
    sigma: 0.45
    peak: 1.0
    floor: -2.0
    power: 2.0
```

**After (unified):**
```yaml
forcing:
  pinch_pockets:
    enabled: true
    anchor_forward: 1.20
    anchor_lateral: 0.30
    sigma: 0.45
    weight: 0.80
    # Potential field mode (optional)
    peak: 1.0      # If specified, uses field mapping
    floor: -2.0    # instead of simple Gaussian
    power: 2.0
```

## How It Works

### Pinch Pocket Locations

The mechanism defines two optimal attack positions (pinch pockets) relative to the target:
- **Right pocket**: `(forward=1.20m, lateral=-0.30m)` in target's frame
- **Left pocket**: `(forward=1.20m, lateral=+0.30m)` in target's frame

These positions are strategically chosen to force the target into walls or obstacles.

### Two Modes

#### Mode 1: Simple Gaussian (default)
If `peak` and `floor` are **not specified** in the config:

```python
# Compute distance to each pocket
dist_right = distance_to_right_pocket()
dist_left = distance_to_left_pocket()

# Sum of Gaussians
reward = exp(-(dist_right^2) / (2*sigma^2)) + exp(-(dist_left^2) / (2*sigma^2))
reward *= weight
```

**Characteristics:**
- Rewards are always positive (0 to 2*weight)
- Highest reward when exactly at a pocket position
- Decays quickly as you move away

#### Mode 2: Potential Field
If `peak` and `floor` **are specified** in the config:

```python
# Use minimum distance to either pocket
d_min = min(dist_right, dist_left)

# Gaussian shaping with power exponent
ratio = d_min / sigma
shaped = exp(-0.5 * (ratio^power))

# Map to [floor, peak] range
reward = floor + (peak - floor) * shaped
reward *= weight
```

**Characteristics:**
- Rewards range from `floor*weight` (far away) to `peak*weight` (at pocket)
- Can be positive or negative depending on floor/peak values
- Smoother falloff controlled by `power` parameter
- More flexible for reward shaping

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable/disable pinch rewards | `true` |
| `anchor_forward` | Distance ahead of target (m) | `1.20` |
| `anchor_lateral` | Distance to side of target (m) | `0.70` |
| `sigma` | Gaussian width (m) | `0.50` |
| `weight` | Reward multiplier | `0.30` |
| `peak` | Max reward at optimal position | `None` (simple mode) |
| `floor` | Min reward when far from optimal | `None` (simple mode) |
| `power` | Field decay exponent | `2.0` |

## Example Configurations

### Simple Gaussian Mode
```yaml
pinch_pockets:
  enabled: true
  anchor_forward: 1.20
  anchor_lateral: 0.30
  sigma: 0.50
  weight: 0.30
  # No peak/floor = simple Gaussian mode
```

**Reward range:** `[0.0, 0.60]` (0.30 * 2)

### Potential Field Mode (Current)
```yaml
pinch_pockets:
  enabled: true
  anchor_forward: 1.20
  anchor_lateral: 0.30
  sigma: 0.45
  weight: 0.80
  peak: 1.0      # Max at pockets
  floor: -2.0    # Min when far
  power: 2.0     # Decay exponent
```

**Reward range:** `[-1.60, 0.80]` (weight * [floor, peak])

## Updated YAML Files

All v2 scenario files have been updated:
- ✅ `gaplock_sac.yaml` - Uses potential field mode (peak=1.0, floor=-2.0)
- ✅ `gaplock_ppo.yaml` - Uses potential field mode (peak=0.60, floor=-0.25)
- ✅ `gaplock_td3.yaml` - Uses potential field mode (peak=0.60, floor=-0.25)
- ✅ `gaplock_limo.yaml` - Uses potential field mode (peak=0.60, floor=-0.25)
- ✅ `gaplock_custom.yaml` - Uses potential field mode (peak=0.60, floor=-0.25)
- ⚠️ `gaplock_simple.yaml` - Forcing disabled (ablation study)

## Telemetry Output

**Before:**
```
=== Reward Components ===
car_0:
  forcing/pinch: +0.15
  forcing/potential_field: +0.32    # Duplicate/confusing
  forcing/clearance: +0.08
  forcing/turn: +0.05
```

**After:**
```
=== Reward Components ===
car_0:
  forcing/pinch: +0.32              # Unified component
  forcing/clearance: +0.08
  forcing/turn: +0.05
```

## Heatmap Visualization

The reward heatmap automatically visualizes the unified pinch pocket field:
- Uses the actual `reward_strategy.compute()` to evaluate rewards at different positions
- Shows the Gaussian potential field around the target
- Visualizes peak (hot colors) at pinch pockets and floor (cold colors) when far away

**No changes needed** - the heatmap extension automatically works with the unified reward.

## Migration Notes

### For Users

If you have custom scenarios with the old separate `potential_field` section:

**Before:**
```yaml
forcing:
  pinch_pockets:
    # ...
  potential_field:
    enabled: true
    peak: 1.0
    floor: -2.0
```

**After:**
```yaml
forcing:
  pinch_pockets:
    # ... (keep existing params)
    peak: 1.0      # Move peak/floor into pinch_pockets
    floor: -2.0
```

### Backwards Compatibility

The code maintains some backwards compatibility:
- If neither `peak` nor `floor` are specified, uses simple Gaussian mode
- Old configs without potential field will still work (default simple mode)

## Benefits of Unification

1. **Clarity**: Single mechanism instead of two overlapping ones
2. **Simplicity**: One config section instead of two
3. **Consistency**: One reward component in telemetry
4. **Flexibility**: Easy to switch between simple and field modes
5. **Performance**: Slightly faster (one computation instead of two)

## Testing

Verified with unit tests:
- ✓ Simple Gaussian mode works correctly
- ✓ Potential field mode matches previous implementation
- ✓ Reward values are identical to old potential_field computation
- ✓ Heatmap visualization works correctly
