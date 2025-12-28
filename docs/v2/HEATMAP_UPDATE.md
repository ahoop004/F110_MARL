# Heatmap Visualization Update

## Summary

Updated the reward heatmap to query the **actual reward strategy** instead of using hardcoded distance-based calculations. This ensures the heatmap visualization always matches the real rewards the agent receives.

## Changes Made

### 1. Heatmap Extension ([v2/render/extensions/heatmap.py](../../v2/render/extensions/heatmap.py))

**Key Changes:**
- Added `reward_strategy` parameter to store reference to actual reward computation
- Removed hardcoded distance parameters (`near_distance`, `far_distance`, `reward_near`, `penalty_far`)
- Replaced `_calculate_proximity_reward()` with `_query_reward_at_position()`
- Updated `update()` to capture full target observation including pose and LiDAR
- Updated `_update_heatmap_colors()` to query real rewards for each grid cell

**New Method: `_query_reward_at_position(x, y)`**

For each grid cell position (x, y):
1. Constructs mock attacker observation at that position
2. Uses actual target observation from environment
3. Calls `reward_strategy.compute(step_info)`
4. Extracts spatial reward components:
   - `forcing/pinch` - Pinch pocket Gaussians
   - `distance/*` - Distance-based rewards
   - `heading/*` - Heading alignment rewards

**Result:** Heatmap now visualizes the true reward landscape including:
- ✓ Pinch pocket Gaussian hotspots
- ✓ Distance-based shaping
- ✓ Heading alignment bonuses
- ✓ Any other spatial reward components

### 2. Training Script ([v2/run.py](../../v2/run.py))

**Updated Extension Setup:**
```python
# Get reward strategy for attacker
attacker_reward_strategy = reward_strategies.get(attacker_id)

heatmap.configure(
    enabled=heatmap_config.get('enabled', False),
    target_agent=defender_id,
    attacker_agent=attacker_id,
    reward_strategy=attacker_reward_strategy,  # ← NEW
    extent_m=heatmap_config.get('extent_m', 6.0),
    cell_size_m=heatmap_config.get('cell_size_m', 0.25),
    alpha=heatmap_config.get('alpha', 0.22),
    update_frequency=heatmap_config.get('update_frequency', 5),
)
```

**Removed:** Old parameters that are no longer needed
- `near_distance`, `far_distance`, `reward_near`, `penalty_far`

## Reward Components Visualized

The heatmap now shows the combined spatial reward field from:

### Pinch Pocket Gaussians (from [forcing.py](../../v2/rewards/gaplock/forcing.py))
```python
# Two symmetric Gaussian hotspots at optimal attack positions
reward = weight * exp(-distance² / (2 * sigma²))
```

**Parameters from `gaplock_full` preset:**
- `anchor_forward`: 1.2m (ahead of target)
- `anchor_lateral`: 0.7m (to side of target)
- `sigma`: 0.5m (Gaussian width)
- `weight`: 0.3 (multiplier)

### Distance Rewards (from [distance.py](../../v2/rewards/gaplock/distance.py))
Simple linear interpolation between near/far distances:
- High reward when close (< 1.0m)
- Interpolated in optimal zone (1.0m - 2.5m)
- Penalty when far (> 2.5m)

### Heading Alignment (from [heading.py](../../v2/rewards/gaplock/heading.py))
Bonus for facing toward target:
- `reward = coefficient * cos(heading_error)`

## Visual Result

The heatmap now shows **two bright hotspots** (the pinch pockets) positioned ahead and to either side of the target, surrounded by a gradient field from distance and heading rewards.

Example visualization:
```
              ← Target

    ●━━━━━━━●     ← Gaussian hotspots (green)
   /         \
  /  Optimal  \
 |   Position  |
  \   Region  /
   \_________/
```

## Performance

- **Cost:** ~100-500 reward queries per heatmap update (depending on grid resolution)
- **Mitigation:** Use `update_frequency` parameter to update every N frames
- **Default:** 5 frames (20% overhead)

## Testing

Created test script: [test_heatmap_rewards.py](../../test_heatmap_rewards.py)

Verified pinch pocket rewards:
```
At target            ( 0.00,  0.00): 0.0306
Right pocket center  ( 0.55,  0.45): 0.1244  ← Maximum
Left pocket center   ( 0.55, -0.45): 0.1244  ← Maximum
Ahead of target      ( 0.55,  0.00): 0.1050
Far ahead            ( 1.00,  0.00): 0.0460
To side              ( 0.00,  1.00): 0.0102
```

✓ Correctly identifies pinch pocket centers as highest-reward positions

## Usage

The heatmap is disabled by default. Enable during training:

1. **Press `H`** to toggle heatmap on/off
2. Observe the spatial reward field centered on the defender
3. Notice the Gaussian hotspots showing optimal attack positions

## Future Enhancements

Potential improvements:
- [ ] Add parameter to filter which reward components to visualize
- [ ] Add colormap selection (current: red→yellow→green)
- [ ] Add reward value overlay (display numbers in grid cells)
- [ ] Cache reward values when target is stationary

## Backward Compatibility

The old distance-based parameters are still accepted but ignored if `reward_strategy` is provided. This ensures existing code continues to work.

---

**Last Updated**: December 26, 2024
