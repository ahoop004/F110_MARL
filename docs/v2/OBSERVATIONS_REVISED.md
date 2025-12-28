# V2 Observation System - Revised Plan

## V1 Actual Configuration (from gaplock_ppo.yaml)

### Environment Settings
- **lidar_beams**: 720 (not 1080!)
- **max_scan**: 12.0 meters (not 30.0!)
- **observation_normalization**: running (enabled for trainable agents)

### Observation Components

**Lines 224-261 in gaplock_ppo.yaml**:

```yaml
components:
  - id: lidar
    type: lidar
    params:
      beams: 720
      max_range: 12.0
      normalize: true
      clip: 1.0

  - id: ego_pose
    type: ego_pose
    params:
      angle_mode: sin_cos

  - id: ego_vel
    type: velocity
    params:
      normalize: 2.0
      include_speed: true
      speed_scale: 1.0

  - id: target_pose
    type: target_pose
    params:
      angle_mode: sin_cos
    target:
      agent: car_1

  - id: target_vel
    type: velocity
    params:
      normalize: 2.0
      include_speed: true
      speed_scale: 1.0
    target:
      agent: car_1

  - id: relative_pose
    type: relative_pose
    params:
      angle_mode: sin_cos
    target:
      agent: car_1
```

### Observation Dimensions

| Component | Dims | Description |
|-----------|------|-------------|
| lidar | 720 | LiDAR beams, normalized to [0, 1] |
| ego_pose | 4 | (x, y, sin(θ), cos(θ)) |
| ego_vel | 3 | (vx, vy, speed) normalized by 2.0 |
| target_pose | 4 | (x, y, sin(θ), cos(θ)) |
| target_vel | 3 | (vx, vy, speed) normalized by 2.0 |
| relative_pose | 4 | (dx, dy, sin(Δθ), cos(Δθ)) |
| **Total** | **738** | |

---

## Simplified V2 Observation Presets

Based on your feedback: "ego state, target state, relative distance"

### Preset 1: Minimal (LiDAR only)

**Use case**: Pure reactive policy (FTG-like)

```python
'minimal': {
    'components': [
        {'type': 'lidar', 'params': {'beams': 720, 'normalize': True, 'max_range': 12.0}},
    ],
    'normalize_running': False,
}
```

**Dims**: 720

---

### Preset 2: Basic (Ego state only)

**Use case**: Self-aware reactive policy

```python
'basic': {
    'components': [
        {'type': 'lidar', 'params': {'beams': 720, 'normalize': True, 'max_range': 12.0}},
        {'type': 'ego_pose', 'params': {'angle_mode': 'sin_cos'}},
        {'type': 'velocity', 'params': {'normalize': 2.0, 'include_speed': True}},
    ],
    'normalize_running': False,
}
```

**Dims**: 720 + 4 + 3 = **727**

---

### Preset 3: Gaplock (Matches V1 exactly)

**Use case**: Adversarial task (current v1 configuration)

```python
'gaplock': {
    'components': [
        {'type': 'lidar', 'params': {'beams': 720, 'normalize': True, 'max_range': 12.0, 'clip': 1.0}},
        {'type': 'ego_pose', 'params': {'angle_mode': 'sin_cos'}},
        {'type': 'velocity', 'params': {'normalize': 2.0, 'include_speed': True}},
        {'type': 'target_pose', 'params': {'angle_mode': 'sin_cos'}},
        {'type': 'velocity', 'target': 'auto', 'params': {'normalize': 2.0, 'include_speed': True}},
        {'type': 'relative_pose', 'params': {'angle_mode': 'sin_cos'}},
    ],
    'normalize_running': True,  # For trainable agents only
    'normalize_clip': 10.0,
}
```

**Dims**: 720 + 4 + 3 + 4 + 3 + 4 = **738** (matches v1!)

---

### Preset 4: Simple Gaplock (Your suggestion)

**Use case**: Simplified adversarial (ego + target + distance)

```python
'gaplock_simple': {
    'components': [
        {'type': 'lidar', 'params': {'beams': 720, 'normalize': True, 'max_range': 12.0}},
        {'type': 'ego_pose', 'params': {'angle_mode': 'sin_cos'}},
        {'type': 'velocity', 'params': {'normalize': 2.0, 'include_speed': True}},
        {'type': 'target_pose', 'params': {'angle_mode': 'sin_cos'}},
        {'type': 'distance', 'params': {'normalize': 12.0}},  # Just distance, not full relative_pose
    ],
    'normalize_running': True,
    'normalize_clip': 10.0,
}
```

**Dims**: 720 + 4 + 3 + 4 + 1 = **732**

**Difference from gaplock**:
- ❌ No target velocity (simpler)
- ❌ No relative_pose (replaced with just distance)
- ✅ Fewer dims (732 vs 738)
- ✅ Simpler to understand

---

## Configuration in V2 Scenarios

### Example 1: Use exact v1 configuration

```yaml
experiment:
  name: gaplock_ppo_v1_match
  episodes: 1500

environment:
  map: maps/line2.yaml
  num_agents: 2
  max_steps: 5000
  lidar_beams: 720

agents:
  car_0:
    role: attacker
    algorithm: ppo
    params:
      lr: 0.0005
      gamma: 0.995

    observation:
      preset: gaplock  # Matches v1 exactly (738 dims)

    reward:
      preset: gaplock_full

  car_1:
    role: defender
    algorithm: ftg
    # No observation config (FTG doesn't use ObsWrapper)
```

### Example 2: Simplified version

```yaml
agents:
  car_0:
    role: attacker
    algorithm: ppo

    observation:
      preset: gaplock_simple  # Simplified (732 dims)
      # Auto-computed: obs_dim = 732

    reward:
      preset: gaplock_simple
```

### Example 3: Custom tweaks

```yaml
agents:
  car_0:
    observation:
      preset: gaplock
      overrides:
        components:
          - type: lidar
            params:
              beams: 360  # Reduce beams for faster training
```

---

## Running Normalization Strategy

**V1 behavior** (line 75 in scenario):
- `observation_normalization: running` - Enabled for PPO

**V2 approach**:
```python
# In scenario parser
if agent_config.get('trainable', True):  # Trainable agent
    if obs_config.get('normalize_running', False):
        obs_wrapper = RunningObsNormalizer(obs_wrapper, clip=obs_config.get('normalize_clip', 10.0))
else:  # Baseline agent (FTG)
    # Don't normalize
    pass
```

**Result**: Running normalization only for trainable agents (PPO, TD3, SAC), not for FTG baseline.

---

## Key Differences: V1 vs V2

### V1 (Complex)

```yaml
# 297 lines total!
wrappers:
  - factory: obs
    params:
      max_scan: 12.0
      components:
        - id: lidar
          type: lidar
          params:
            beams: 720
            max_range: 12.0
            normalize: true
            clip: 1.0
        # ... 5 more components with full config
```

### V2 (Simple)

```yaml
# 10-20 lines
observation:
  preset: gaplock  # All defaults handled
```

**OR** with tweaks:

```yaml
observation:
  preset: gaplock
  overrides:
    components:
      - type: lidar
        params:
          beams: 360  # Just override what you need
```

---

## Recommended Presets

Based on v1 and your feedback:

### 1. **minimal**
- LiDAR only (720 beams)
- **Dims**: 720
- **Use**: Pure reactive policies

### 2. **basic**
- LiDAR + ego state
- **Dims**: 727
- **Use**: Self-aware reactive

### 3. **gaplock_simple** (YOUR SUGGESTION)
- LiDAR + ego state + target pose + distance
- **Dims**: 732
- **Use**: Simplified adversarial
- **Difference from v1**: No target velocity, distance instead of relative_pose

### 4. **gaplock** (EXACT V1 MATCH)
- LiDAR + ego state + target pose + target vel + relative_pose
- **Dims**: 738
- **Use**: Full adversarial (proven in v1)

---

## Questions for You

### 1. Which preset should be the default?

**Options**:
- **A. gaplock_simple** (your suggestion, 732 dims) - Simpler, cleaner
- **B. gaplock** (v1 exact, 738 dims) - Proven, matches existing experiments

**My recommendation**: Start with **gaplock_simple**, but include both so you can A/B test.

### 2. Is gaplock_simple enough?

**What it has**:
- ✅ LiDAR (720 beams)
- ✅ Ego pose (x, y, sin(θ), cos(θ))
- ✅ Ego velocity (vx, vy, speed)
- ✅ Target pose (x, y, sin(θ), cos(θ))
- ✅ Distance to target (scalar)

**What it's missing** (vs v1):
- ❌ Target velocity (vx, vy, speed)
- ❌ Relative pose (dx, dy, sin(Δθ), cos(Δθ))

**Do you think target velocity is important?**
- If target is moving fast → harder to catch
- Agent could infer velocity from consecutive target poses
- But explicit velocity might help learning

**Do you think relative pose is important?**
- Already have target pose (absolute)
- Already have ego pose (absolute)
- Agent can compute relative in network
- But relative pose gives direct encoding of "where is target relative to me"

**My opinion**:
- Try **gaplock_simple** first
- If learning is slow, add back target velocity
- relative_pose is probably redundant (network can learn this)

### 3. Distance vs Relative Pose?

**distance component**:
- Just Euclidean distance: `sqrt((x_tgt - x_ego)^2 + (y_tgt - y_ego)^2)`
- 1 dimension

**relative_pose component**:
- dx, dy, sin(Δθ), cos(Δθ)
- 4 dimensions
- More information (direction + angle)

**Which is better?**
- Distance: Simpler, but loses directional info
- Relative pose: More complete, but agent can compute from ego + target poses

**Current v1 uses**: relative_pose

**Your suggestion**: "relative distance"

**My recommendation**: Try both:
- `gaplock_simple`: Uses **distance** (simpler)
- `gaplock`: Uses **relative_pose** (v1 match)

Then compare performance.

### 4. Is 720 beams too many?

**V1 uses**: 720 beams
**Simulator default**: 1080 beams
**Common in research**: 108-360 beams

**Trade-offs**:
- 720 beams: High resolution, slow training, large network input
- 360 beams: 2x faster, still good resolution
- 180 beams: 4x faster, decent resolution

**Questions**:
- Does v1 really need 720 beams, or is it overkill?
- Could we get same performance with 360?

**My recommendation**:
- Keep 720 as default (matches v1)
- Make it easy to experiment with fewer (360, 180)
- Maybe add `gaplock_fast` preset with 360 beams?

### 5. Presets Summary

Here's what I propose:

```python
OBSERVATION_PRESETS = {
    # Pure reactive
    'minimal': {
        'components': [lidar_720],
        'normalize_running': False,
    },

    # Self-aware reactive
    'basic': {
        'components': [lidar_720, ego_pose, ego_vel],
        'normalize_running': False,
    },

    # Simplified adversarial (YOUR SUGGESTION)
    'gaplock_simple': {
        'components': [lidar_720, ego_pose, ego_vel, target_pose, distance],
        'normalize_running': True,
    },

    # Full adversarial (V1 EXACT MATCH)
    'gaplock': {
        'components': [lidar_720, ego_pose, ego_vel, target_pose, target_vel, relative_pose],
        'normalize_running': True,
    },

    # Fast adversarial (360 beams)
    'gaplock_fast': {
        'components': [lidar_360, ego_pose, ego_vel, target_pose, distance],
        'normalize_running': True,
    },
}
```

**Are these 5 presets good? Too many? Too few?**

---

## Implementation Checklist

**Observation integration** (5 hrs):

- [ ] Define 5 observation presets in `v2/core/presets.py`
  - minimal, basic, gaplock_simple, gaplock, gaplock_fast

- [ ] Add observation config to scenario YAML schema
  - Support `preset` + `overrides`
  - Support `normalize_running` per agent

- [ ] Add agent roles to scenario schema
  - `role: attacker` / `role: defender`
  - Auto-resolve target_id from roles

- [ ] Auto-compute obs_dim
  - Create dummy env observation
  - Run through ObsWrapper
  - Extract dimension for agent creation

- [ ] Wire ObsWrapper into TrainingLoop
  - Create wrapper from config
  - Only apply running normalization to trainable agents
  - Handle multi-agent case (different wrappers per agent)

- [ ] Update example scenarios
  - gaplock_ppo.yaml with `observation: {preset: gaplock}`
  - Test with gaplock_simple too

- [ ] Test end-to-end
  - Verify dims match (738 for gaplock, 732 for gaplock_simple)
  - Verify running normalization only on trainable agents
  - Verify FTG baseline works without observation wrapper

---

## Your Feedback Needed

Please confirm:

1. ✅ **LiDAR beams: 720** (confirmed from v1 scenario)
2. ✅ **max_scan: 12.0** (not 30.0)
3. ✅ **Normalize for trainable only, not FTG** (confirmed)
4. ✅ **Explicit roles** (already in v1)
5. ✅ **Observation per agent** (confirmed)
6. ✅ **Auto-compute dims** (confirmed)

**New questions**:
7. **Default preset**: `gaplock_simple` or `gaplock` (v1 exact)?
8. **Is gaplock_simple enough** or do we need target velocity?
9. **Distance vs relative_pose**? (v1 uses relative_pose, you said "relative distance")
10. **Is 720 beams necessary** or can we reduce to 360 for faster training?
11. **5 presets okay** or too many/few?

Let me know your thoughts!
