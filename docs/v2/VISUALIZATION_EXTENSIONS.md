# Visualization Extensions Guide

## Overview

The v2 renderer uses a modular extension system for optional visualization features. This allows you to customize what information is displayed during training without cluttering the core renderer.

## Available Extensions

### 1. TelemetryHUD
Enhanced telemetry display with multiple detail levels.

**Features:**
- Episode number and step count
- Real-time reward tracking (current and cumulative)
- Reward component breakdown
- Collision status indicators
- Observation snapshot (position, velocity, LiDAR stats)
- Agent-specific focus mode

**Display Modes:**
- **0 - OFF**: No telemetry
- **1 - MINIMAL**: Episode, step, FPS
- **2 - BASIC**: + rewards and collisions
- **3 - DETAILED**: + reward components
- **4 - FULL**: + observation snapshot

**Keyboard Controls:**
- `T` - Cycle through display modes
- `1-9` - Focus on specific agent
- `0` - Show all agents

**Usage:**
```python
from v2.render import TelemetryHUD

# Create telemetry extension
telemetry = TelemetryHUD(renderer)
telemetry.configure(
    enabled=True,
    mode=TelemetryHUD.MODE_BASIC  # or MODE_DETAILED, MODE_FULL
)
renderer.add_extension(telemetry)

# Update during training loop
telemetry.update_episode_info(episode=1, step=100)
telemetry.update_rewards(
    agent_id='car_0',
    reward=0.5,
    components={'proximity': 0.3, 'heading': 0.2},
    reset=False
)
telemetry.update_collision_status('car_0', collision=False)
```

### 2. RewardRingExtension
Visualizes circular reward zones around the target agent.

**Features:**
- Concentric rings showing reward zones
- Fill colors: green (high reward) → yellow (medium) → red (penalty)
- Distance-to-target display
- Configurable inner/outer radii

**Keyboard Controls:**
- `R` - Toggle reward ring display

**Usage:**
```python
from v2.render import RewardRingExtension

ring = RewardRingExtension(renderer)
ring.configure(
    enabled=True,
    target_agent='car_1',  # Agent to center rings on
    inner_radius=1.0,      # Near distance (meters)
    outer_radius=2.5,      # Far distance (meters)
    preferred_radius=1.5,  # Optional optimal radius
)
renderer.add_extension(ring)
```

### 3. RewardHeatmap
2D spatial reward field visualization.

**Features:**
- Shows reward value at each position
- Color-coded: red (low) → yellow (medium) → green (high)
- Configurable resolution and extent
- Centered on target agent
- Transparency control

**Keyboard Controls:**
- `H` - Toggle heatmap display

**Usage:**
```python
from v2.render import RewardHeatmap

heatmap = RewardHeatmap(renderer)
heatmap.configure(
    enabled=True,
    target_agent='car_1',
    attacker_agent='car_0',
    extent_m=6.0,           # Half-width in meters
    cell_size_m=0.25,       # Cell size (smaller = higher resolution)
    alpha=0.22,             # Transparency (0-1)
    near_distance=1.0,      # Reward parameters
    far_distance=2.5,
    reward_near=0.12,
    penalty_far=0.08,
    update_frequency=5      # Update every N frames
)
renderer.add_extension(heatmap)
```

## Complete Example

```python
from v2.env.f110ParallelEnv import F110ParallelEnv
from v2.render import TelemetryHUD, RewardRingExtension, RewardHeatmap
from v2.rewards.presets import load_preset

# Create environment
env = F110ParallelEnv(
    map_name='maps/line2/line2',
    num_agents=2,
    render_mode='human'
)

renderer = env.renderer

# Load reward config
reward_config = load_preset('gaplock_full')
dist_config = reward_config['distance']

# Configure telemetry
telemetry = TelemetryHUD(renderer)
telemetry.configure(enabled=True, mode=TelemetryHUD.MODE_DETAILED)
renderer.add_extension(telemetry)

# Configure reward ring
ring = RewardRingExtension(renderer)
ring.configure(
    enabled=True,
    target_agent='car_1',
    inner_radius=dist_config['near_distance'],
    outer_radius=dist_config['far_distance'],
)
renderer.add_extension(ring)

# Configure heatmap (start disabled)
heatmap = RewardHeatmap(renderer)
heatmap.configure(
    enabled=False,
    target_agent='car_1',
    attacker_agent='car_0',
    extent_m=6.0,
    cell_size_m=0.25
)
renderer.add_extension(heatmap)

# Training loop
observations, _ = env.reset()
for step in range(1000):
    # Get actions...
    observations, rewards, terms, truncs, infos = env.step(actions)

    # Update telemetry
    telemetry.update_episode_info(episode=1, step=step)
    for agent_id in rewards.keys():
        telemetry.update_rewards(agent_id, rewards[agent_id])
        telemetry.update_collision_status(agent_id, infos[agent_id].get('collision', False))

    env.render()
```

## Keyboard Controls Summary

| Key | Action |
|-----|--------|
| `T` | Cycle telemetry display mode |
| `R` | Toggle reward ring |
| `H` | Toggle reward heatmap |
| `F` | Toggle camera follow mode |
| `1-9` | Focus telemetry on specific agent |
| `0` | Show all agents in telemetry |
| Mouse scroll | Zoom camera |
| Mouse drag | Pan camera |

## Integration with Training

To use extensions in your training script:

1. **Get renderer from environment:**
   ```python
   env = F110ParallelEnv(..., render_mode='human')
   renderer = env.renderer
   ```

2. **Create and configure extensions:**
   ```python
   telemetry = TelemetryHUD(renderer)
   telemetry.configure(enabled=True)
   renderer.add_extension(telemetry)
   ```

3. **Update extensions in training loop:**
   ```python
   for episode in range(num_episodes):
       for step in range(max_steps):
           # ... step environment ...

           # Update telemetry
           telemetry.update_episode_info(episode, step)
           telemetry.update_rewards(agent_id, reward, components)

           env.render()
   ```

## Performance Considerations

- **Telemetry**: Minimal overhead (~0.1ms per frame)
- **Reward Ring**: Low overhead (~0.2ms per frame)
- **Heatmap**: Moderate overhead (~2-5ms per frame)
  - Use `update_frequency > 1` to reduce impact
  - Use larger `cell_size_m` for lower resolution

## Tips

1. **Start with telemetry only** - Add reward visualizations as needed
2. **Use keyboard shortcuts** - Toggle visualizations on/off dynamically
3. **Focus on one agent** - Press `1` or `2` to see detailed info for specific agent
4. **Disable heatmap during training** - Enable only when debugging
5. **Adjust heatmap resolution** - Balance between visual quality and performance

## See Also

- [examples/visualization_demo.py](../../examples/visualization_demo.py) - Complete working example
- [VEHICLE_PARAMETERS.md](VEHICLE_PARAMETERS.md) - Reward parameter reference
- [PARAMETER_CONFIGURATION_SUMMARY.md](PARAMETER_CONFIGURATION_SUMMARY.md) - Configuration guide

---

**Last Updated**: December 26, 2024
