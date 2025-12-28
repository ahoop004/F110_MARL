# Parameter Configuration Summary

## What Was Fixed

### Problem
- V2 was using unrealistic high-performance parameters (v_max=20 m/s)
- V1 used realistic Agilex Limo parameters (v_max=1.0 m/s)
- FTG policy parameters were not exposed in scenarios
- Vehicle behavior didn't match v1 or real Limo robot

### Solution
All gaplock scenarios now configured with proper Limo parameters matching v1.

---

## Updated Scenarios

### 1. `scenarios/v2/gaplock_sac.yaml`
✅ Added Limo vehicle_params
✅ Added FTG params matching v1
✅ v_max: 1.0 m/s (20x slower than before)

### 2. `scenarios/v2/gaplock_limo.yaml` (NEW)
Dedicated Limo configuration with full documentation

---

## Key Parameter Changes

### Vehicle Dynamics

| Parameter | Old Default | Limo (V1) | Impact |
|---|---|---|---|
| `v_max` | 20.0 m/s | **1.0 m/s** | 20x slower, realistic |
| `v_min` | -5.0 m/s | **-1.0 m/s** | Slower reverse |
| `a_max` | 9.51 m/s² | **2.0 m/s²** | More realistic acceleration |
| `s_min/s_max` | ±0.4189 rad | **±0.46 rad** | Slightly wider steering |
| `v_switch` | 7.319 m/s | **0.8 m/s** | Lower switching speed |

### FTG Policy

| Parameter | Old (implicit) | V1-Compatible | Impact |
|---|---|---|---|
| `min_speed` | 2.0 m/s | **0.2 m/s** | Lower minimum for Limo |
| `max_speed` | 20.0 m/s | **1.0 m/s** | Matches vehicle v_max |
| `max_steer` | (default) | **0.32 rad** | ~18.3° max steering |
| `gap_min_range` | (default) | **0.65 m** | Minimum safe gap |
| `use_disparity_extender` | (default) | **true** | Obstacle growth enabled |
| `target_mode` | (default) | **"farthest"** | Target farthest gap point |

---

## How to Use

### Training with Limo Parameters (Recommended)

```bash
# Use updated gaplock_sac.yaml (now has Limo params)
python3 v2/run.py --scenario scenarios/v2/gaplock_sac.yaml
```

### Training with High-Performance Parameters

If you need the old high-speed behavior for research:

```yaml
environment:
  vehicle_params:
    v_max: 20.0      # Fast racing
    a_max: 9.51      # High acceleration

agents:
  car_1:
    algorithm: ftg
    params:
      min_speed: 2.0
      max_speed: 20.0  # Match vehicle v_max
```

---

## Behavior Differences

### With Limo Parameters (v_max=1.0 m/s)
- ✅ Realistic sim-to-real transfer
- ✅ Matches v1 behavior
- ✅ Safer, more predictable
- ⏱️ Episodes take longer (more simulation time)
- ⏱️ Training may be slower (more steps needed)

### With High-Performance Parameters (v_max=20.0 m/s)
- ⚠️ Unrealistic for Limo robot
- ⚠️ Different from v1
- ✅ Faster episodes
- ✅ Faster training
- ⚠️ Poor sim-to-real transfer

---

## Verification

Run this to verify parameters are loaded correctly:

```python
from v2.core.setup import create_training_setup
from v2.core.scenario import load_and_expand_scenario

scenario = load_and_expand_scenario('scenarios/v2/gaplock_sac.yaml')
env, agents, _ = create_training_setup(scenario)

# Check vehicle params
car = env.sim.agents[0]
print(f"v_max: {car._v_max} m/s")  # Should be 1.0

# Check FTG params
ftg = agents['car_1']
print(f"max_speed: {ftg.max_speed} m/s")  # Should be 1.0
```

---

## Next Steps

1. ✅ **All scenarios updated with Limo params**
2. ✅ **FTG parameters exposed and configurable**
3. ✅ **Documentation created** ([VEHICLE_PARAMETERS.md](VEHICLE_PARAMETERS.md))
4. ⚠️ **Training will take longer** due to slower speeds (expected)
5. ✅ **Behavior now matches v1**

---

## Additional Resources

- Full parameter reference: [VEHICLE_PARAMETERS.md](VEHICLE_PARAMETERS.md)
- Agilex Limo specs: Official documentation
- V1 configuration reference: Check `plots/` directory for v1 run configs

---

**Last Updated**: December 26, 2024
**Status**: All scenarios configured with Limo parameters
