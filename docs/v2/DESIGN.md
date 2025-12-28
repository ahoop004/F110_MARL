# V2 Reward System Design

## What I Learned from V1

### V1 Gaplock Reward Analysis

The v1 gaplock reward ([src/f110x/tasks/reward/gaplock.py](../../src/f110x/tasks/reward/gaplock.py)) has **1,655 lines** with **106 parameters**.

**Key insight**: V1 implements partial credit through **dense reward shaping**.

#### Terminal Rewards (Episode End Only)
```python
# Only fire when episode terminates
target_crash_reward: 60.0           # Success
self_collision_penalty: -90.0       # Attacker crashed
truncation_penalty: -10.0           # Timeout
idle_truncation_penalty: -10.0      # Idle stop
```

#### Dense Shaping Rewards (Every Step)

Even in **failed episodes**, the agent receives rewards for good behavior:

**1. Pressure System** (lines 720-817)
- Small reward each step for being close to target (< 0.75m)
- Streak bonuses for staying close
- **Teaches**: "approach and stay near target"

**2. Distance Gradient** (lines 759-776)
- Smooth reward based on distance
- Closer = higher reward
- **Teaches**: "move toward target"

**3. Heading Alignment** (lines 782-784)
- Reward for pointing at target
- **Teaches**: "orient correctly"

**4. Speed Bonus** (lines 680-684)
- Reward for moving fast
- **Teaches**: "be aggressive"

**5. Commit/Escape** (lines 819-844)
- Bonus for committing to attack (close + aligned + fast)
- Penalty for backing off
- **Teaches**: "commit to attacks"

**6. Forcing Rewards** (lines 960-1212)
- Pinch pocket Gaussians (reward for specific positions)
- Clearance reduction (reward when target's wall clearance decreases)
- Turn shaping (reward when target turns away from wall)
- **Teaches**: "force target toward obstacles"

**7. Behavior Penalties**
- Idle, reverse, hard braking
- **Teaches**: "avoid bad behaviors"

### Example: Failed Episode with Partial Credit

```
Episode 237: TIMEOUT (failure)
Terminal:
  - timeout: -10.0

Dense (accumulated over 500 steps):
  - pressure_bonus: +12.4      (was close for 124 steps)
  - pressure_streak: +3.2      (maintained pressure)
  - distance_reward: +6.8      (approached target)
  - heading_reward: +4.1       (pointed at target)
  - speed_bonus: +15.3         (moved fast)
  - commit_bonus: +2.0         (committed once)
  - force_clearance: +5.7      (reduced target clearance)

Total: +39.5

Result: Episode FAILED but agent learned good behaviors!
```

This is why v1 works - the agent learns incrementally:
1. First: "how to approach target" (distance, heading)
2. Then: "how to pressure target" (stay close)
3. Then: "how to force target" (reduce clearance, pinch)
4. Finally: "how to make target crash" (terminal reward)

## V2 Design Goals

### 1. Preserve V1's Effectiveness

**Must keep**:
- Dense reward shaping for partial credit
- Terminal rewards for episode outcomes
- Component-based composition

**Must simplify**:
- 106 params → ~20-30 params organized in groups
- 1,655 lines → ~300-400 lines
- Flat params → grouped config

### 2. Make It Easier to Understand

V1 scenario files have 100+ reward params in flat lists:
```yaml
# V1 - hard to understand
reward:
  params:
    target_crash_reward: 60.0
    self_collision_penalty: -90.0
    pressure_distance: 0.75
    pressure_timeout: 0.5
    pressure_min_speed: 0.1
    pressure_heading_tolerance: 3.14
    pressure_bonus: 0.02
    pressure_bonus_interval: 1
    pressure_streak_bonus: 0.01
    pressure_streak_cap: 50
    distance_reward_near: 0.1
    distance_reward_near_distance: 0.5
    distance_reward_far_distance: 2.0
    distance_penalty_far: 0.05
    # ... 90+ more params
```

V2 groups by concept:
```yaml
# V2 - clear semantic groups
reward:
  preset: gaplock_simple  # Start simple
  overrides:
    terminal:
      target_crash: 100.0  # Increase success reward
    pressure:
      bonus_per_step: 0.03  # More aggressive pressure
```

### 3. Support Progressive Complexity

**Training Curriculum**:
1. **Early Training** (Episodes 0-500): Use `gaplock_simple`
   - Terminal + pressure + distance + heading + speed
   - No forcing rewards (too complex initially)

2. **Mid Training** (Episodes 500-1500): Use `gaplock_medium`
   - Add pinch pocket rewards
   - Add basic clearance shaping
   - Agent learns forcing positions

3. **Late Training** (Episodes 1500+): Use `gaplock_full`
   - Add turn shaping
   - Add commit/escape bonuses
   - Fine-tune all components

**Or**: Just use `gaplock_simple` if it works. Don't add complexity unnecessarily.

## Proposed V2 Architecture

### File Structure

```
v2/rewards/
├── README.md           # User-facing docs (created above)
├── DESIGN.md           # This file
├── __init__.py
├── base.py             # Base reward protocol
├── composer.py         # Reward composition system
├── presets.py          # Preset configurations
└── gaplock/
    ├── __init__.py
    ├── gaplock.py      # Main gaplock reward class
    ├── terminal.py     # Terminal rewards
    ├── pressure.py     # Pressure rewards
    ├── distance.py     # Distance shaping
    ├── heading.py      # Heading alignment
    ├── speed.py        # Speed bonuses
    ├── forcing.py      # Forcing rewards (pinch, clearance, turn)
    └── penalties.py    # Behavior penalties
```

### Base Protocol

```python
# v2/rewards/base.py
from typing import Protocol, runtime_checkable, Dict, Tuple

@runtime_checkable
class RewardComponent(Protocol):
    """Protocol for reward components."""

    def compute(self, step_info: dict) -> Dict[str, float]:
        """
        Compute reward components for this step.

        Args:
            step_info: Dict with keys:
                - obs: Agent observation
                - target_obs: Target observation (if available)
                - done: Whether episode is done
                - truncated: Whether episode was truncated
                - info: Additional info dict
                - timestep: Simulation timestep (dt)

        Returns:
            Dict of component names → reward values
        """
        ...

@runtime_checkable
class RewardStrategy(Protocol):
    """Protocol for complete reward strategies."""

    def reset(self) -> None:
        """Reset internal state for new episode."""
        ...

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward and components.

        Returns:
            (total_reward, components_dict)
        """
        ...
```

### Composer

```python
# v2/rewards/composer.py
from typing import Dict, List, Tuple
from .base import RewardComponent, RewardStrategy

class ComposedReward(RewardStrategy):
    """Composes multiple reward components."""

    def __init__(self, components: List[RewardComponent]):
        self.components = components

    def reset(self) -> None:
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        all_components = {}

        for component in self.components:
            component_rewards = component.compute(step_info)
            all_components.update(component_rewards)

        total = sum(all_components.values())
        return total, all_components
```

### Gaplock Reward

```python
# v2/rewards/gaplock/gaplock.py
from typing import Dict, Tuple, List
from ..base import RewardStrategy, RewardComponent
from ..composer import ComposedReward
from .terminal import TerminalReward
from .pressure import PressureReward
from .distance import DistanceReward
from .heading import HeadingReward
from .speed import SpeedReward
from .forcing import ForcingReward
from .penalties import BehaviorPenalties

class GaplockReward(RewardStrategy):
    """Gaplock adversarial task reward."""

    def __init__(self, config: dict):
        components: List[RewardComponent] = []

        # Terminal rewards
        if 'terminal' in config:
            components.append(TerminalReward(config['terminal']))

        # Pressure rewards
        if config.get('pressure', {}).get('enabled', True):
            components.append(PressureReward(config['pressure']))

        # Distance shaping
        if config.get('distance', {}).get('enabled', True):
            components.append(DistanceReward(config['distance']))

        # Heading alignment
        if config.get('heading', {}).get('enabled', True):
            components.append(HeadingReward(config['heading']))

        # Speed bonus
        if config.get('speed', {}).get('enabled', True):
            components.append(SpeedReward(config['speed']))

        # Forcing rewards (optional, disabled by default)
        if config.get('forcing', {}).get('enabled', False):
            components.append(ForcingReward(config['forcing']))

        # Behavior penalties
        if config.get('penalties', {}).get('enabled', True):
            components.append(BehaviorPenalties(config['penalties']))

        self.composer = ComposedReward(components)

    def reset(self) -> None:
        self.composer.reset()

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        return self.composer.compute(step_info)
```

### Example Component: Pressure Reward

```python
# v2/rewards/gaplock/pressure.py
import numpy as np
from typing import Dict

class PressureReward:
    """Reward for maintaining pressure on target (staying close)."""

    def __init__(self, config: dict):
        self.enabled = config.get('enabled', True)
        self.distance_threshold = config.get('distance_threshold', 0.75)
        self.bonus_per_step = config.get('bonus_per_step', 0.02)
        self.streak_bonus = config.get('streak_bonus', 0.01)
        self.streak_cap = config.get('streak_cap', 50)

        # State
        self.pressure_streak = 0

    def reset(self) -> None:
        self.pressure_streak = 0

    def compute(self, step_info: dict) -> Dict[str, float]:
        if not self.enabled:
            return {}

        obs = step_info['obs']
        target_obs = step_info.get('target_obs')

        if target_obs is None:
            return {}

        # Compute distance to target
        ego_pose = np.array(obs.get('pose', [0, 0, 0]))
        target_pose = np.array(target_obs.get('pose', [0, 0, 0]))
        distance = np.linalg.norm(target_pose[:2] - ego_pose[:2])

        components = {}

        if distance < self.distance_threshold:
            # Basic pressure bonus
            components['pressure/bonus'] = self.bonus_per_step

            # Streak bonus
            self.pressure_streak += 1
            if self.streak_bonus > 0 and self.pressure_streak > 1:
                capped_streak = min(self.pressure_streak, self.streak_cap)
                components['pressure/streak'] = self.streak_bonus * capped_streak
        else:
            # Lost pressure
            self.pressure_streak = 0

        return components
```

### Presets

```python
# v2/rewards/presets.py
from typing import Dict, Any

GAPLOCK_SIMPLE: Dict[str, Any] = {
    'terminal': {
        'target_crash': 60.0,
        'self_crash': -90.0,
        'collision': -90.0,
        'timeout': -10.0,
        'idle_stop': -10.0,
        'target_finish': -20.0,
    },
    'pressure': {
        'enabled': True,
        'distance_threshold': 0.75,
        'bonus_per_step': 0.02,
        'streak_bonus': 0.01,
        'streak_cap': 50,
    },
    'distance': {
        'enabled': True,
        'gradient': [
            (0.5, 0.1),
            (1.0, 0.05),
            (2.0, 0.0),
            (4.0, -0.05),
        ],
    },
    'heading': {
        'enabled': True,
        'coefficient': 0.03,
    },
    'speed': {
        'enabled': True,
        'coefficient': 0.02,
        'target_speed': 5.0,
    },
    'forcing': {
        'enabled': False,  # Disabled for simple preset
    },
    'penalties': {
        'enabled': True,
        'idle': -0.01,
        'reverse': -0.02,
        'brake': -0.05,
    },
}

GAPLOCK_MEDIUM: Dict[str, Any] = {
    **GAPLOCK_SIMPLE,
    'forcing': {
        'enabled': True,
        'pinch_pockets': {
            'weight': 0.03,
            'anchor_forward': 1.2,
            'anchor_lateral': 0.7,
            'sigma': 0.5,
        },
        'clearance': {
            'weight': 0.05,
            'band': (0.4, 3.0),
            'clip': 0.2,
        },
        'turn': {
            'enabled': False,  # Still disabled
        },
    },
}

GAPLOCK_FULL: Dict[str, Any] = {
    **GAPLOCK_MEDIUM,
    'forcing': {
        **GAPLOCK_MEDIUM['forcing'],
        'clearance': {
            'weight': 0.1,  # Increased weight
            'band': (0.4, 3.0),
            'clip': 0.2,
        },
        'turn': {
            'enabled': True,
            'weight': 0.05,
            'clip': 0.2,
        },
    },
}

PRESETS = {
    'gaplock_simple': GAPLOCK_SIMPLE,
    'gaplock_medium': GAPLOCK_MEDIUM,
    'gaplock_full': GAPLOCK_FULL,
}

def load_preset(name: str) -> Dict[str, Any]:
    """Load a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name].copy()

def merge_config(preset: dict, overrides: dict) -> dict:
    """Deep merge overrides into preset config."""
    result = preset.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result
```

## Usage Examples

### Simple Scenario (Using Preset)

```yaml
# scenarios/v2/gaplock_ppo_simple.yaml
experiment:
  name: gaplock_ppo_simple
  episodes: 1000

environment:
  map: maps/line_map.yaml
  num_agents: 2
  max_steps: 5000

agents:
  car_0:  # Attacker
    algorithm: ppo
    params:
      lr: 0.0003
      gamma: 0.995
    reward:
      preset: gaplock_simple

  car_1:  # Defender
    algorithm: ftg
```

**Benefits**:
- 10 lines vs 297 lines (v1)
- Clear and readable
- Easy to experiment: change `gaplock_simple` → `gaplock_medium`

### Custom Reward Composition

```yaml
# scenarios/v2/gaplock_custom.yaml
agents:
  car_0:
    algorithm: ppo
    reward:
      preset: gaplock_simple
      overrides:
        terminal:
          target_crash: 100.0  # Increase success reward
        pressure:
          bonus_per_step: 0.03  # More aggressive
        forcing:
          enabled: true  # Enable forcing even in simple preset
          pinch_pockets:
            weight: 0.02
```

### Ablation Study (Disable Components)

```yaml
# Test: Does forcing help or hurt?
agents:
  car_0:
    reward:
      preset: gaplock_full
      overrides:
        forcing:
          enabled: false  # Disable to test
```

## Implementation Plan

### Phase 8.1: Base Infrastructure (2 hrs)
- [ ] Create `v2/rewards/` directory
- [ ] Implement base protocols (`base.py`)
- [ ] Implement composer (`composer.py`)
- [ ] Implement presets (`presets.py`)
- [ ] Write tests

### Phase 8.2: Terminal Rewards (1 hr)
- [ ] Implement `gaplock/terminal.py`
- [ ] Handle all 6 outcome types
- [ ] Test with mock step_info

### Phase 8.3: Dense Shaping (3 hrs)
- [ ] Implement `pressure.py` (with streak tracking)
- [ ] Implement `distance.py` (gradient interpolation)
- [ ] Implement `heading.py` (alignment reward)
- [ ] Implement `speed.py` (speed bonus)
- [ ] Implement `penalties.py` (idle, reverse, brake)
- [ ] Test each component

### Phase 8.4: Forcing Rewards (3 hrs, OPTIONAL)
- [ ] Implement `forcing.py`:
  - [ ] Pinch pocket Gaussians
  - [ ] Clearance reduction (LiDAR-based)
  - [ ] Turn shaping
- [ ] This is complex - can defer if simple rewards work

### Phase 8.5: Integration (2 hrs)
- [ ] Implement `GaplockReward` main class
- [ ] Wire up to training loop
- [ ] Test end-to-end

### Phase 8.6: Presets & Scenarios (1 hr)
- [ ] Create 3 presets (simple, medium, full)
- [ ] Create example scenarios
- [ ] Document usage

## Testing Strategy

### Unit Tests

```python
# tests/test_rewards.py
def test_pressure_reward():
    config = {'distance_threshold': 0.75, 'bonus_per_step': 0.02}
    pressure = PressureReward(config)

    # Close to target
    step_info = {
        'obs': {'pose': [0, 0, 0]},
        'target_obs': {'pose': [0.5, 0, 0]},  # 0.5m away
    }
    components = pressure.compute(step_info)
    assert 'pressure/bonus' in components
    assert components['pressure/bonus'] == 0.02

    # Far from target
    step_info['target_obs']['pose'] = [2.0, 0, 0]  # 2m away
    components = pressure.compute(step_info)
    assert components == {}  # No reward

def test_terminal_rewards():
    config = {'target_crash': 60.0, 'self_crash': -90.0}
    terminal = TerminalReward(config)

    # Success
    step_info = {
        'done': True,
        'info': {'outcome': 'target_crash'},
    }
    components = terminal.compute(step_info)
    assert components['terminal/success'] == 60.0

    # Failure
    step_info['info']['outcome'] = 'self_crash'
    components = terminal.compute(step_info)
    assert components['terminal/crash'] == -90.0
```

### Integration Test

```python
def test_gaplock_reward_full():
    from v2.rewards.presets import load_preset
    from v2.rewards.gaplock import GaplockReward

    config = load_preset('gaplock_simple')
    reward = GaplockReward(config)

    # Simulate episode
    reward.reset()

    # Step 1: Approaching target
    step_info = {
        'obs': {'pose': [0, 0, 0], 'velocity': [2, 0]},
        'target_obs': {'pose': [1, 0, 0]},
        'done': False,
        'timestep': 0.01,
    }
    total, components = reward.compute(step_info)
    assert total > 0  # Should get distance + speed rewards
    assert 'speed/bonus' in components
    assert 'distance/gradient' in components

    # Step 2: Close to target
    step_info['obs']['pose'] = [0.5, 0, 0]  # Now 0.5m away
    total, components = reward.compute(step_info)
    assert 'pressure/bonus' in components

    # Step 3: Target crashes (success!)
    step_info['done'] = True
    step_info['info'] = {'outcome': 'target_crash'}
    total, components = reward.compute(step_info)
    assert 'terminal/success' in components
    assert components['terminal/success'] == 60.0
```

## Comparison: V1 vs V2

| Aspect | V1 | V2 |
|--------|----|----|
| Lines of code | 1,655 | ~300-400 (est.) |
| Parameters | 106 | ~25-30 (grouped) |
| Scenario YAML | 297 lines | 10-30 lines |
| Organization | Flat params | Semantic groups |
| Presets | No | Yes (simple/medium/full) |
| Partial credit | ✅ Yes | ✅ Yes (preserved) |
| Composability | Limited | Easy (enable/disable groups) |
| Testability | Hard | Easy (component-based) |
| Debuggability | Hard | Easy (components dict) |
| Learning curve | Steep | Gentle |

## Next Steps

1. **Get feedback** from user on this design
2. **Implement** base infrastructure first (protocols, composer, presets)
3. **Start simple**: Terminal + pressure + distance + heading
4. **Test early**: Run training with simple rewards
5. **Add complexity progressively**: Only add forcing if needed

## Questions for User

1. **Presets**: Do the 3 presets (simple/medium/full) make sense?
2. **Component organization**: Is the grouping clear?
3. **Forcing rewards**: Should we implement forcing rewards immediately, or defer until simple rewards work?
4. **Other tasks**: Besides gaplock, what other reward tasks do you need?
   - Pure racing (fastest lap)?
   - Blocking (prevent opponent from passing)?
   - Other adversarial scenarios?
