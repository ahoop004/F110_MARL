# V2 Observation System

## Overview

V2 **already has** a sophisticated observation system ported from v1 ([v2/wrappers/observation.py](wrappers/observation.py:1)).

The system is:
- ✅ **Component-based** - Mix and match observation features
- ✅ **Extensible** - Registry pattern for custom components
- ✅ **Normalizable** - Running mean/std normalization
- ✅ **Flexible** - Per-component configuration

---

## Current Implementation

### ObsWrapper Class

**Core functionality**:
```python
from v2.wrappers.observation import ObsWrapper, RunningObsNormalizer

# Create wrapper
obs_wrapper = ObsWrapper(
    components=['lidar', 'ego_pose', 'target_pose', 'velocity'],
    max_scan=30.0,
    normalize=True,
    lidar_beams=108,
)

# Optionally wrap with running normalization
obs_wrapper = RunningObsNormalizer(obs_wrapper, clip=10.0)

# Use in training loop
raw_obs = env.reset()  # Dict[agent_id, obs_dict]
processed_obs = obs_wrapper(raw_obs, ego_id='car_0', target_id='car_1')
# Returns: flat numpy array ready for agent
```

### Available Components

**Sensor Data**:
- `lidar` - LiDAR scans (with downsampling)
- `collision` - Binary collision flag

**Ego State**:
- `ego_pose` - Own position (x, y, theta)
- `velocity` - Velocity vector (vx, vy, speed)
- `lap` - Lap count and time (for racing)

**Target Information**:
- `target_pose` - Opponent position
- `relative_pose` - Relative position to target
- `relative_sector` - Target in 8 directional sectors (front, front_right, right, etc.)
- `distance` - Euclidean distance to target

**Track Features**:
- `centerline` - Track progress, lateral error, waypoint lookahead

### Component Configuration

**Simple** (string name):
```yaml
components:
  - lidar
  - ego_pose
  - target_pose
  - velocity
```

**Advanced** (dict with params):
```yaml
components:
  - type: lidar
    params:
      beams: 54          # Downsample to 54 beams
      normalize: true    # Normalize by max_range
      max_range: 30.0

  - type: ego_pose
    params:
      normalize_xy: 30.0       # Normalize x,y by this value
      angle_mode: sin_cos      # Encode theta as [sin(θ), cos(θ)]
      include_xy: true

  - type: target_pose
    params:
      normalize_xy: 30.0
      angle_mode: sin_cos

  - type: velocity
    params:
      normalize: 10.0          # Normalize by max expected velocity
      include_speed: true      # Add ||v|| as extra feature

  - type: relative_pose
    target: car_1              # Which agent is the target
    params:
      normalize_xy: 30.0
      angle_mode: sin_cos

  - type: relative_sector
    target: car_1
    params:
      preferred_radius: 2.0    # Preferred distance
      inner_tolerance: 0.5
      outer_tolerance: 2.0
      falloff: linear          # or 'gaussian', 'binary'
```

### Running Normalization

**Purpose**: Stabilize training by normalizing observations to zero mean, unit variance.

**Usage**:
```python
obs_wrapper = ObsWrapper(components=[...])
normalized_wrapper = RunningObsNormalizer(
    obs_wrapper,
    eps=1e-8,      # Numerical stability
    clip=10.0,     # Clip normalized values to [-10, 10]
)
```

**How it works**:
- Maintains running mean and variance using Welford's algorithm
- Updates statistics every step
- Normalizes: `(obs - mean) / sqrt(var + eps)`
- Optional clipping to prevent extreme values

**Trade-offs**:
- ✅ Stabilizes training (less sensitive to scale)
- ✅ Faster convergence
- ❌ Non-stationary (stats change during training)
- ❌ Different behavior at start vs end of training

---

## Integration with V2 Pipeline

### Where Observations Fit

```
Environment.reset()
    ↓
Raw obs dict: {
    'car_0': {'pose': [x,y,θ], 'scans': [...], 'velocity': [vx,vy], ...},
    'car_1': {'pose': [x,y,θ], 'scans': [...], ...},
}
    ↓
ObsWrapper(obs, ego_id='car_0', target_id='car_1')
    ↓
Flat numpy array: [lidar_0, ..., lidar_107, x, y, sin(θ), cos(θ), ...]
    ↓
Agent.act(processed_obs)
    ↓
Action
```

### Configuration in Scenarios

**Scenario YAML**:
```yaml
experiment:
  name: gaplock_ppo_v1

environment:
  map: maps/line_map.yaml
  num_agents: 2

agents:
  car_0:  # Attacker
    algorithm: ppo

    observation:  # 🆕 Observation configuration
      components:
        - type: lidar
          params:
            beams: 108
            normalize: true
            max_range: 30.0

        - type: ego_pose
          params:
            angle_mode: sin_cos
            normalize_xy: 30.0

        - type: target_pose
          params:
            angle_mode: sin_cos
            normalize_xy: 30.0

        - type: velocity
          params:
            normalize: 10.0
            include_speed: true

      normalize_running: true  # Enable running normalization
      normalize_clip: 10.0

    reward:
      preset: gaplock_simple

  car_1:  # Defender
    algorithm: ftg
    # FTG doesn't use observations from wrapper
```

### Preset Observation Configurations

**Presets for common setups**:

```python
# v2/core/presets.py

OBSERVATION_PRESETS = {
    # Minimal: Just LiDAR (for pure reactive policies)
    'lidar_only': {
        'components': [
            {'type': 'lidar', 'params': {'beams': 108, 'normalize': True}},
        ],
        'normalize_running': False,
    },

    # Basic: LiDAR + ego state
    'basic': {
        'components': [
            {'type': 'lidar', 'params': {'beams': 108, 'normalize': True}},
            {'type': 'ego_pose', 'params': {'angle_mode': 'sin_cos', 'normalize_xy': 30.0}},
            {'type': 'velocity', 'params': {'normalize': 10.0}},
        ],
        'normalize_running': False,
    },

    # Gaplock: LiDAR + ego + target (for adversarial tasks)
    'gaplock': {
        'components': [
            {'type': 'lidar', 'params': {'beams': 108, 'normalize': True, 'max_range': 30.0}},
            {'type': 'ego_pose', 'params': {'angle_mode': 'sin_cos', 'normalize_xy': 30.0}},
            {'type': 'velocity', 'params': {'normalize': 10.0, 'include_speed': True}},
            {'type': 'target_pose', 'params': {'angle_mode': 'sin_cos', 'normalize_xy': 30.0}},
        ],
        'normalize_running': True,
        'normalize_clip': 10.0,
    },

    # Racing: LiDAR + ego + centerline (for time-trial racing)
    'racing': {
        'components': [
            {'type': 'lidar', 'params': {'beams': 108, 'normalize': True}},
            {'type': 'ego_pose', 'params': {'angle_mode': 'sin_cos', 'normalize_xy': 30.0}},
            {'type': 'velocity', 'params': {'normalize': 10.0, 'include_speed': True}},
            {'type': 'centerline', 'params': {
                'include_lateral': True,
                'include_longitudinal': True,
                'include_progress': True,
                'angle_mode': 'sin_cos',
            }},
        ],
        'normalize_running': True,
        'normalize_clip': 10.0,
    },

    # Full: Everything (for complex tasks)
    'full': {
        'components': [
            {'type': 'lidar', 'params': {'beams': 108, 'normalize': True}},
            {'type': 'ego_pose', 'params': {'angle_mode': 'sin_cos', 'normalize_xy': 30.0}},
            {'type': 'velocity', 'params': {'normalize': 10.0, 'include_speed': True}},
            {'type': 'target_pose', 'params': {'angle_mode': 'sin_cos', 'normalize_xy': 30.0}},
            {'type': 'relative_pose', 'params': {'normalize_xy': 30.0, 'angle_mode': 'sin_cos'}},
            {'type': 'distance', 'params': {'normalize': 30.0}},
            {'type': 'relative_sector'},
        ],
        'normalize_running': True,
        'normalize_clip': 10.0,
    },
}
```

**Usage in scenarios**:
```yaml
agents:
  car_0:
    algorithm: ppo
    observation:
      preset: gaplock  # Use preset
      overrides:       # Optional tweaks
        components:
          - type: lidar
            params:
              beams: 54  # Reduce beams for faster training
```

---

## Observation Dimensions

### Typical Sizes (Gaplock Task)

**LiDAR** (108 beams, normalized): `108 floats`

**Ego Pose** (x, y, sin(θ), cos(θ), normalized): `4 floats`

**Velocity** (vx, vy, speed, normalized): `3 floats`

**Target Pose** (x, y, sin(θ), cos(θ), normalized): `4 floats`

**Total**: `108 + 4 + 3 + 4 = 119 floats`

### Memory Considerations

**Per step**: 119 floats × 4 bytes = 476 bytes per observation

**Replay buffer** (1M steps): 476 MB

**Batch** (256 samples): 122 KB

→ Not a memory concern for typical setups.

### Downsample LiDAR?

**Options**:
- 1080 beams (simulator default): Full resolution
- 540 beams: 2x downsample
- 270 beams: 4x downsample
- 108 beams: 10x downsample (common choice)
- 54 beams: 20x downsample (faster, less detail)

**Trade-offs**:
- More beams = better obstacle detection, slower training
- Fewer beams = faster training, may miss small gaps

**Recommendation**: Start with 108 beams (good balance), reduce to 54 if too slow.

---

## Normalization Strategies

### Component-Level Normalization (Default)

**Pros**:
- ✅ Fixed scale (deterministic)
- ✅ Interpretable (values in known ranges)
- ✅ Stationary (same scale throughout training)

**Cons**:
- ❌ Requires domain knowledge (what's max_range?)
- ❌ May not match actual data distribution

**Example**:
```yaml
components:
  - type: lidar
    params:
      normalize: true
      max_range: 30.0  # Divide by 30 → [0, 1] range

  - type: ego_pose
    params:
      normalize_xy: 30.0  # ±30m track → approx [-1, 1]
      angle_mode: sin_cos  # Already in [-1, 1]
```

### Running Normalization (Optional)

**Pros**:
- ✅ Adaptive (learns true distribution)
- ✅ Zero mean, unit variance (good for neural nets)
- ✅ Handles unexpected values

**Cons**:
- ❌ Non-stationary (changes during training)
- ❌ Different scale at start vs end
- ❌ Harder to interpret

**When to use**:
- When you don't know the data distribution
- When training is unstable
- When using algorithms sensitive to scale (PPO, SAC)

**When not to use**:
- When you want deterministic behavior
- When you need interpretability
- When using algorithms robust to scale (DQN)

**Example**:
```yaml
observation:
  preset: gaplock
  normalize_running: true  # Enable
  normalize_clip: 10.0     # Clip to [-10, 10] std devs
```

---

## Common Configurations

### 1. Gaplock (Adversarial)

**Goal**: Attacker forces defender to crash

**Observations needed**:
- LiDAR (detect walls)
- Ego state (know your pose/velocity)
- Target state (track opponent)

```yaml
observation:
  preset: gaplock
```

**Output**: 119 floats (108 LiDAR + 4 ego pose + 3 velocity + 4 target pose)

### 2. Racing (Time Trial)

**Goal**: Complete lap as fast as possible

**Observations needed**:
- LiDAR (avoid walls)
- Ego state (speed control)
- Centerline features (stay on racing line)

```yaml
observation:
  preset: racing
```

**Output**: ~120-130 floats (LiDAR + ego + centerline)

### 3. Pure Reactive (Minimal)

**Goal**: Simple obstacle avoidance

**Observations needed**:
- LiDAR only (reactive policy)

```yaml
observation:
  preset: lidar_only
```

**Output**: 108 floats (just LiDAR)

### 4. Full Information (Research)

**Goal**: Maximum observability for experiments

**Observations needed**:
- Everything

```yaml
observation:
  preset: full
```

**Output**: ~140+ floats (all components)

---

## Integration Plan for Phase 8

### What Needs to Be Done

#### 1. Observation Configuration in Scenarios ✅ (Already works!)

The `ObsWrapper` already exists and works. We just need to:
- Add `observation` field to scenario YAML schema
- Load observation config in scenario parser
- Pass to training loop

**No changes to ObsWrapper itself needed!**

#### 2. Create Observation Presets (1 hr)

Add to `v2/core/presets.py`:
```python
OBSERVATION_PRESETS = {
    'lidar_only': {...},
    'basic': {...},
    'gaplock': {...},
    'racing': {...},
    'full': {...},
}
```

#### 3. Wire into Training Loop (1 hr)

**Current** (v2/core/training.py):
```python
# No observation processing - agents get raw obs
obs = env.reset()
action = agent.act(obs['car_0'])  # Raw dict
```

**Enhanced**:
```python
# Create obs wrapper from config
obs_wrapper = create_obs_wrapper(config['agents']['car_0']['observation'])

# Process observations
raw_obs = env.reset()
processed_obs = obs_wrapper(raw_obs, ego_id='car_0', target_id='car_1')
action = agent.act(processed_obs)  # Flat numpy array
```

#### 4. Update Agent Interface (2 hrs)

**Issue**: Current agents expect flat numpy arrays, but we're passing raw dicts.

**Solution**: Ensure all agents accept flat obs:
```python
class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, ...):
        # obs_dim comes from observation config
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        # obs is already flat numpy array (from ObsWrapper)
        pass
```

**Compute obs_dim**:
```python
# In scenario parser
obs_wrapper = create_obs_wrapper(obs_config)
dummy_obs = create_dummy_env_obs()
sample_output = obs_wrapper(dummy_obs, 'car_0', 'car_1')
obs_dim = sample_output.shape[0]  # Use this for agent creation
```

#### 5. Example Scenarios (1 hr)

Create scenarios with different observation configs:
- `scenarios/v2/gaplock_ppo.yaml` - Uses `observation: {preset: gaplock}`
- `scenarios/v2/racing_sac.yaml` - Uses `observation: {preset: racing}`
- `scenarios/v2/custom_obs.yaml` - Uses custom component list

---

## Questions for Discussion

### 1. Observation Presets

**Proposed presets**:
- `lidar_only` - Minimal
- `basic` - LiDAR + ego
- `gaplock` - LiDAR + ego + target (for adversarial)
- `racing` - LiDAR + ego + centerline (for time trial)
- `full` - Everything

**Are these sufficient? Missing any?**

### 2. Running Normalization

**Options**:
- Always enabled (simpler, may help training)
- Disabled by default (more predictable)
- Per-preset (enabled for some, disabled for others)

**Recommendation**: Disabled by default, enable in presets that need it (like `gaplock`).

**Do you agree?**

### 3. LiDAR Downsampling

**Common values**:
- 108 beams (v1 default)
- 54 beams (faster)

**Should we**:
- Keep 108 as default?
- Reduce to 54 for faster training?
- Make it configurable per scenario?

**Recommendation**: 108 as default, easy to override.

### 4. Observation Dimension Mismatch

**Problem**: Agent needs to know `obs_dim` at creation time.

**Solution Options**:

**A. Auto-compute from dummy observation**:
```python
# In scenario parser
obs_wrapper = create_obs_wrapper(obs_config)
dummy_obs = create_dummy_env_obs(env_config)
sample = obs_wrapper(dummy_obs, 'car_0', 'car_1')
obs_dim = sample.shape[0]
```

**B. Require explicit obs_dim in scenario**:
```yaml
agents:
  car_0:
    observation:
      preset: gaplock
      # Must specify dim manually
    params:
      obs_dim: 119  # Explicit
      act_dim: 2
```

**C. Agent computes from first observation**:
```python
# Agent builds network lazily on first call
def act(self, obs):
    if not self._initialized:
        self._build_network(obs.shape[0])
        self._initialized = True
    # ... rest of act
```

**Which approach? Recommendation: Option A (auto-compute).**

### 5. Multi-Agent Observations

**Current design**: Each agent can have different observations.

**Question**: Should both attacker and defender use the same observation wrapper?

**Options**:
- **A. Different per agent** (more flexible):
  ```yaml
  agents:
    car_0:
      observation: {preset: gaplock}  # Full obs
    car_1:
      observation: {preset: basic}    # Limited obs
  ```

- **B. Shared** (simpler, forces symmetry):
  ```yaml
  environment:
    observation: {preset: gaplock}  # All agents use same
  ```

**Recommendation**: Different per agent (more flexible, especially for baseline vs RL).

### 6. Target Agent Resolution

**Question**: How does attacker know which agent is the target?

**Current**: Hardcoded `target_id='car_1'` in training loop.

**Better**: Infer from agent roles:
```yaml
agents:
  car_0:
    role: attacker    # 🆕 Explicit role
    observation: {preset: gaplock}

  car_1:
    role: defender
    algorithm: ftg
```

Then in training loop:
```python
# Find attacker and defender
attacker_id = find_agent_by_role(config, 'attacker')  # 'car_0'
defender_id = find_agent_by_role(config, 'defender')  # 'car_1'

# Process obs for attacker
processed_obs = obs_wrapper(raw_obs, ego_id=attacker_id, target_id=defender_id)
```

**Should we add explicit roles?**

---

## Implementation Checklist for Phase 8

**Observation integration** (~5 hours total):

- [ ] **8.1**: Create observation presets in `v2/core/presets.py` (1 hr)
  - lidar_only, basic, gaplock, racing, full

- [ ] **8.2**: Add observation config to scenario schema (1 hr)
  - Parse `observation` field in scenario YAML
  - Support preset + overrides

- [ ] **8.3**: Auto-compute obs_dim from dummy observation (1 hr)
  - Create dummy env obs
  - Run through ObsWrapper
  - Extract dimension

- [ ] **8.4**: Wire ObsWrapper into TrainingLoop (1 hr)
  - Create wrapper from config
  - Process observations before passing to agent
  - Handle multi-agent case

- [ ] **8.5**: Update example scenarios (1 hr)
  - Add `observation` fields
  - Test different presets

- [ ] **8.6**: Test end-to-end (tests already exist for ObsWrapper)
  - Ensure obs dimensions match agent expectations
  - Verify running normalization works

---

## Summary

**Current State**:
- ✅ ObsWrapper already implemented and working
- ✅ Component-based, extensible
- ✅ Running normalization supported
- ❌ Not integrated with scenario system yet
- ❌ No presets defined yet

**Next Steps**:
1. Create observation presets
2. Add observation config to scenarios
3. Wire into training loop
4. Test end-to-end

**Estimated effort**: ~5 hours

**Key decisions needed**:
1. Which presets to include?
2. Running normalization: default on/off?
3. LiDAR beams: 108 or 54 default?
4. obs_dim: auto-compute or explicit?
5. Multi-agent: per-agent or shared obs config?
6. Add explicit agent roles (attacker/defender)?
