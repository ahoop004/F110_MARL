# V2 Reward System

## Philosophy

V2 rewards are organized into **composable groups** instead of 100+ flat parameters.

Each group can be enabled/disabled independently and has clear semantics.

## Reward Groups for Gaplock Task

### 1. Terminal Rewards (Episode End Only)

Sparse rewards that fire only when episode terminates:

```python
terminal = {
    'target_crash': 60.0,        # Success: target crashed
    'self_crash': -90.0,          # Failure: attacker crashed
    'collision': -90.0,           # Failure: both crashed
    'timeout': -10.0,             # Failure: time limit
    'idle_stop': -10.0,           # Failure: stopped moving
    'target_finish': -20.0,       # Failure: target finished race
}
```

### 2. Pressure Rewards (Dense Shaping)

Rewards for being close to target (teaches approach behavior):

```python
pressure = {
    'enabled': True,
    'distance_threshold': 0.75,   # meters
    'bonus_per_step': 0.02,       # small reward each step
    'streak_bonus': 0.01,         # increasing reward for consecutive steps
    'streak_cap': 50,             # max streak multiplier
}
```

**Partial Credit**: Even in failed episodes, agent learns to approach target.

### 3. Distance Shaping (Dense)

Smooth gradient based on distance to target:

```python
distance = {
    'enabled': True,
    'gradient': [
        (0.5, 0.1),    # (distance, reward) - very close = high reward
        (1.0, 0.05),   # close = medium reward
        (2.0, 0.0),    # medium = neutral
        (4.0, -0.05),  # far = small penalty
    ],
}
```

**Partial Credit**: Shapes movement toward target throughout episode.

### 4. Heading Rewards (Dense)

Reward for pointing toward target:

```python
heading = {
    'enabled': True,
    'coefficient': 0.03,  # reward = coef * alignment (0-1)
}
```

**Partial Credit**: Teaches orientation even before reaching target.

### 5. Speed Rewards (Dense)

Encourage aggressive movement:

```python
speed = {
    'enabled': True,
    'coefficient': 0.02,  # reward per m/s
    'target_speed': 5.0,  # cap speed bonus
}
```

**Partial Credit**: Rewards fast movement regardless of outcome.

### 6. Forcing Rewards (Dense, Advanced)

Reward for forcing target toward walls:

```python
forcing = {
    'enabled': False,  # disabled by default
    'pinch_pockets': {
        'weight': 0.05,
        'anchor_forward': 1.2,  # m in front of target
        'anchor_lateral': 0.7,  # m to side of target
        'sigma': 0.5,           # Gaussian width
    },
    'clearance': {
        'weight': 0.1,
        'band': (0.4, 3.0),  # only reward in this distance range
        'clip': 0.2,          # max reward per step
    },
    'turn': {
        'weight': 0.05,
        'clip': 0.2,
    },
}
```

**Partial Credit**: Rewards forcing behavior even if target doesn't crash.

### 7. Behavior Penalties (Dense)

Discourage bad behaviors:

```python
penalties = {
    'idle': -0.01,         # per step when speed < threshold
    'reverse': -0.02,      # per step when reversing
    'brake': -0.05,        # per hard brake event
}
```

## Preset Configurations

### Simple (Recommended for Early Training)

```python
GAPLOCK_SIMPLE = {
    'terminal': {...},    # Full terminal rewards
    'pressure': {...},    # Basic pressure shaping
    'distance': {...},    # Distance gradient
    'heading': {...},     # Heading alignment
    'speed': {...},       # Speed bonus
    'forcing': {'enabled': False},  # Disabled
    'penalties': {...},   # Basic penalties
}
```

**Use when**: Starting training, debugging, quick experiments.

### Medium (Balanced)

```python
GAPLOCK_MEDIUM = {
    'terminal': {...},
    'pressure': {...},
    'distance': {...},
    'heading': {...},
    'speed': {...},
    'forcing': {
        'enabled': True,
        'pinch_pockets': {'weight': 0.03},  # Reduced weight
        'clearance': {'weight': 0.05},
        'turn': {'enabled': False},
    },
    'penalties': {...},
}
```

**Use when**: Agent has learned basic approach, ready for forcing.

### Full (All Features)

```python
GAPLOCK_FULL = {
    'terminal': {...},
    'pressure': {...},
    'distance': {...},
    'heading': {...},
    'speed': {...},
    'forcing': {...},  # All forcing components enabled
    'penalties': {...},
}
```

**Use when**: Fine-tuning, complex environments, experienced agents.

## Usage in Scenarios

### Minimal YAML (using preset)

```yaml
reward:
  preset: gaplock_simple
  overrides:
    terminal:
      target_crash: 100.0  # increase success reward
```

### Custom Composition

```yaml
reward:
  groups:
    terminal:
      target_crash: 60.0
      self_crash: -90.0
      timeout: -10.0
    pressure:
      enabled: true
      distance_threshold: 0.75
      bonus_per_step: 0.02
    distance:
      enabled: true
      gradient:
        - [0.5, 0.1]
        - [1.0, 0.05]
        - [2.0, 0.0]
        - [4.0, -0.05]
    forcing:
      enabled: false  # Disable for early training
```

## Implementation

Rewards are computed in [v2/rewards/gaplock.py](gaplock.py):

```python
class GaplockReward:
    def __init__(self, config: dict):
        self.terminal = config['terminal']
        self.pressure = PressureReward(config['pressure'])
        self.distance = DistanceReward(config['distance'])
        # ... etc

    def compute(self, step_info: dict) -> tuple[float, dict]:
        """Returns (total_reward, components_dict)."""
        components = {}

        # Terminal rewards (only on done)
        if step_info['done']:
            if step_info['target_crashed']:
                components['terminal/success'] = self.terminal['target_crash']
            elif step_info['self_crashed']:
                components['terminal/crash'] = self.terminal['self_crash']
            # ... etc

        # Dense shaping (every step)
        if self.pressure['enabled']:
            components.update(self.pressure.compute(step_info))

        if self.distance['enabled']:
            components.update(self.distance.compute(step_info))

        # ... etc

        total = sum(components.values())
        return total, components
```

## Benefits vs V1

1. **Clarity**: 7 groups vs 100+ flat params
2. **Composability**: Enable/disable whole groups
3. **Presets**: Start simple, add complexity progressively
4. **Debugging**: Components dict shows what's contributing
5. **Experimentation**: Easy to compare "pressure only" vs "pressure + forcing"

## Partial Credit in Practice

Example failed episode (timeout):

```
Episode 1: Timeout (FAILURE)
  Terminal rewards:
    - timeout: -10.0
  Dense rewards (accumulated over 500 steps):
    - pressure: +15.3  (was close for 150 steps)
    - distance: +8.7   (approached target)
    - heading: +4.2    (pointed at target)
    - speed: +12.1     (moved fast)
  Total: +30.3
```

Even though episode **failed**, agent received **+30.3 reward** for good behavior.

This teaches:
- ✅ How to approach target
- ✅ How to stay close
- ✅ How to orient correctly
- ✅ How to move aggressively

Next episode, agent builds on this to eventually succeed.
