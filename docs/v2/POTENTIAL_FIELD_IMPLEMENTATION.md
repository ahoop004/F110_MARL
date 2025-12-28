# Potential Field Reward Implementation

## Summary
The Gaussian potential field reward mechanism is now fully implemented in the v2 reward system. This mechanism was configured in the YAML scenario files but was not actually being computed.

## Changes Made

### 1. Updated `src/rewards/gaplock/forcing.py`

**Added potential_field configuration** (lines 73-83):
- `potential_field_enabled`: Enable/disable the mechanism
- `potential_field_weight`: Reward multiplier (default: 0.80)
- `potential_field_sigma`: Gaussian width (default: 0.45m)
- `potential_field_peak`: Max reward at optimal position (default: 0.60)
- `potential_field_floor`: Min reward when far from optimal (default: -0.25)
- `potential_field_power`: Field decay exponent (default: 2.0)
- `potential_field_time_scaled`: Scale by timestep (default: True)

**Added computation** (lines 145-149):
```python
# Potential field
if self.potential_field_enabled:
    potential_reward = self._compute_potential_field(obs, target_obs, step_info.get('timestep', 0.01))
    if potential_reward != 0.0:
        components['forcing/potential_field'] = potential_reward
```

**Implemented `_compute_potential_field()` method** (lines 293-360):
- Computes distance to both pinch pockets (left and right)
- Uses minimum distance: `d_min = min(dist_left, dist_right)`
- Applies Gaussian shaping: `shaped = exp(-0.5 * (d_min/sigma)^power)`
- Maps to [floor, peak] range: `value = floor + (peak - floor) * shaped`
- Multiplies by weight: `value *= weight`
- Optionally scales by timestep if `time_scaled=True`

## How It Works

The potential field creates a Gaussian reward field centered around the two pinch pockets:
- **Right pocket**: `(forward=1.20m, lateral=-0.30m)` relative to target
- **Left pocket**: `(forward=1.20m, lateral=+0.30m)` relative to target

### Reward Formula

```
d_min = min(distance_to_left_pocket, distance_to_right_pocket)
ratio = d_min / sigma
shaped = exp(-0.5 * ratio^power)
value = floor + (peak - floor) * shaped
reward = value * weight
```

### Example Values (from gaplock_sac.yaml)

With configuration:
- `weight: 0.80`
- `sigma: 0.45`
- `peak: 1.0`
- `floor: -2.0`
- `power: 2.0`

**At optimal position** (d_min ≈ 0):
- `shaped = exp(0) = 1.0`
- `value = -2.0 + (1.0 - (-2.0)) * 1.0 = 1.0`
- `reward = 1.0 * 0.80 = 0.80`

**At 1 sigma** (d_min = 0.45m):
- `shaped = exp(-0.5 * 1.0^2) = 0.6065`
- `value = -2.0 + 3.0 * 0.6065 = -0.18`
- `reward = -0.18 * 0.80 = -0.144`

**Far away** (d_min >> sigma):
- `shaped ≈ 0`
- `value ≈ floor = -2.0`
- `reward = -2.0 * 0.80 = -1.60`

## Expected Telemetry Output

You should now see `forcing/potential_field` in the reward components:

```
=== Reward Components ===
car_0:
  distance/far: -0.080
  distance/gradient: -0.000
  forcing/pinch: +0.000
  forcing/potential_field: +0.XXX    # <-- NOW APPEARS
  forcing/turn: +0.000
  heading/alignment: +0.074
  penalties/step: -0.010
  speed/bonus: +0.005
```

The value will be positive when near the pinch pockets and negative when far away, creating a shaping signal that guides the agent toward optimal attack positions.

## Configuration

All v2 gaplock scenarios now have potential_field configured:
- ✓ `gaplock_sac.yaml` - enabled with peak=1.0, floor=-2.0
- ✓ `gaplock_ppo.yaml` - enabled with peak=0.60, floor=-0.25
- ✓ `gaplock_td3.yaml` - enabled with peak=0.60, floor=-0.25
- ✓ `gaplock_limo.yaml` - enabled with peak=0.60, floor=-0.25
- ✓ `gaplock_custom.yaml` - enabled with peak=0.60, floor=-0.25
- ✓ `gaplock_simple.yaml` - forcing disabled (ablation study)

You can now adjust `peak`, `floor`, `weight`, `sigma`, and `power` in the YAML files to tune the reward signal without changing code.

## Testing

Verified with unit tests:
- ✓ Correct computation at optimal position (d_min ≈ 0)
- ✓ Correct computation at intermediate distance (d_min = 1 sigma)
- ✓ Correct computation far from pockets (d_min >> sigma)
- ✓ No computation when disabled

## Notes

The potential field uses the same anchor positions as pinch_pockets:
- `anchor_forward`: Distance ahead of target (default: 1.20m)
- `anchor_lateral`: Distance to side of target (default: 0.70m, overridden to 0.30m in configs)

If you change pinch_pockets anchor positions, the potential field pockets will move accordingly.
