# V2 Design Master Document

**Status**: Architecture complete, ready for Phase 8 implementation
**Timeline**: 1-2 weeks (34 hours estimated)
**Goal**: Complete v2 pipeline with 84% code reduction while maintaining v1 performance

---

## Quick Links

- **Implementation Checklist**: See [REFACTOR_TODO.md](../REFACTOR_TODO.md) Phase 8
- **Transition Plan**: See [V1_TO_V2_TRANSITION.md](V1_TO_V2_TRANSITION.md)
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Rewards**: See [rewards/DESIGN.md](rewards/DESIGN.md)
- **Observations**: See [OBSERVATIONS_REVISED.md](OBSERVATIONS_REVISED.md)

---

## Key Decisions

### 1. Observations: Exact V1 Match (738 dims)
- **LiDAR**: 720 beams, 12.0m max range, normalized
- **Ego state**: pose (4 dims) + velocity (3 dims)
- **Target state**: pose (4 dims) + velocity (3 dims)
- **Relative pose**: 4 dims
- **Running normalization**: Enabled for trainable agents only

**Rationale**: Proven configuration, no risk of performance regression

### 2. Rewards: Component-Based with Full V1 Features
- **Terminal**: 6 outcome types (target crash, self crash, collision, timeout, idle, target finish)
- **Dense shaping**: Pressure, distance, heading, speed, penalties
- **Forcing**: Pinch pockets, clearance reduction, turn shaping (full v1 port)
- **Organization**: Grouped components vs 106 flat params

**Rationale**: Easier to understand and experiment with while keeping all v1 functionality

### 3. Configuration: Presets + Overrides
- **V1 scenario**: 297 lines → **V2 scenario**: 10-30 lines
- **Default presets** for algorithm, reward, observation
- **Easy overrides** for experimentation

**Rationale**: Simple by default, powerful when needed

### 4. Agent Roles: Explicit
- `role: attacker` / `role: defender`
- Auto-resolve target_id in training loop

**Rationale**: Clearer multi-agent setup

---

## What's Being Removed (84% Reduction)

### ❌ Completely Removed (~4,000 lines)

1. **Session Management** (experiments/session.py)
   - Complex state management
   - → Simple config loading

2. **CLI Framework** (experiments/cli.py)
   - Too many options
   - → Simple argparse

3. **Pydantic Config Models** (config_models.py)
   - Rigid schemas, boilerplate
   - → Plain dicts with validation

4. **Complex Builders** (builders.py - 1,586 lines!)
   - Recursive building, context objects
   - → Simple factory functions

5. **Trainer Classes** (trainer/on_policy.py, off_policy.py)
   - Unnecessary wrapper
   - → Direct agent usage

6. **Multiple Factory Systems** (3 registries)
   - Confusing, redundant
   - → 2 simple factories

7. **Inheritance-based Agents**
   - Rigid class structure
   - → Protocol-based (duck typing)

---

## What's Staying the Same

✅ **Agents**: PPO, TD3, SAC, DQN, Rainbow, FTG (already ported)
✅ **Environment**: F110ParallelEnv
✅ **ObsWrapper**: Component-based system (already ported)
✅ **Observations**: 738 dims (exact v1 match)
✅ **LiDAR**: 720 beams, 12.0m range
✅ **Normalization**: Running normalization for trainable agents

**Goal**: Same inputs → Same performance

---

## What's Being Added (New Features)

### 🆕 Metrics Tracking
- 6 episode outcome types
- Rolling statistics (success rate, avg reward, etc.)
- Per-episode and aggregate tracking

### 🆕 W&B Integration
- Auto-init from config
- Per-episode logging
- Rolling stats logging
- Hyperparameter tracking
- Easy algorithm comparison

### 🆕 Rich Terminal Output
- Progress bars
- Live metrics tables
- Color-coded output
- Clean, readable

### 🆕 Preset System
- **Algorithm presets**: Common hyperparameter configurations
- **Reward presets**: gaplock_simple, gaplock_full
- **Observation presets**: gaplock (738 dims)
- **Easy customization**: Preset + overrides

---

## V2 Architecture

```
v2/
├── run.py                   # 🆕 CLI entry point
│
├── core/
│   ├── agent_factory.py     # ✅ Exists
│   ├── env_factory.py       # ✅ Exists
│   ├── training.py          # ✅ Exists, needs enhancement
│   ├── scenario.py          # 🆕 YAML parser
│   ├── presets.py           # 🆕 Preset definitions
│   └── config.py            # 🆕 Config utilities
│
├── agents/                  # ✅ Ported (reuse as-is)
├── wrappers/                # ✅ Ported (reuse as-is)
│
├── rewards/                 # 🆕 NEW
│   ├── base.py              # Protocols
│   ├── composer.py          # Composition
│   ├── presets.py           # Presets
│   └── gaplock/
│       ├── gaplock.py       # Main class
│       ├── terminal.py      # Terminal rewards
│       ├── pressure.py      # Pressure shaping
│       ├── distance.py      # Distance shaping
│       ├── heading.py       # Heading alignment
│       ├── speed.py         # Speed bonuses
│       ├── forcing.py       # Forcing rewards
│       └── penalties.py     # Behavior penalties
│
├── metrics/                 # 🆕 NEW
│   ├── outcomes.py          # Outcome enum
│   ├── tracker.py           # Metrics tracking
│   └── aggregator.py        # Statistics
│
└── logging/                 # 🆕 NEW
    ├── wandb_logger.py      # W&B integration
    └── console.py           # Rich output
```

**Total new code**: ~2,000 lines
**Total v2**: ~2,800 lines
**vs V1**: ~7,000 lines

**Reduction**: 60% overall (84% in infrastructure)

---

## Example V2 Scenario

**Before (v1)**: 297 lines

**After (v2)**: 20 lines

```yaml
experiment:
  name: gaplock_ppo
  episodes: 1500
  seed: 42

environment:
  map: maps/line2.yaml
  num_agents: 2
  max_steps: 5000
  lidar_beams: 720
  spawn_points: [spawn_2, spawn_1]

agents:
  car_0:
    role: attacker
    algorithm: ppo
    params:
      lr: 0.0005
      gamma: 0.995
      hidden_dims: [512, 256, 128]

    observation:
      preset: gaplock  # 738 dims

    reward:
      preset: gaplock_full

  car_1:
    role: defender
    algorithm: ftg

wandb:
  enabled: true
  project: f110-gaplock
  tags: [ppo, gaplock]
```

**Run**: `python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb`

---

## Phase 8 Implementation Plan

### Summary (34 hours total)

| Task | Hours | Priority |
|------|-------|----------|
| 8.1 Rewards | 10 | High |
| 8.2 Metrics | 2 | High |
| 8.3 W&B | 2 | High |
| 8.4 Console | 2 | Medium |
| 8.5 Observations | 5 | High |
| 8.6 Scenarios | 4 | High |
| 8.7 CLI | 2 | High |
| 8.8 Training Loop | 3 | High |
| 8.9 Examples | 2 | Medium |
| 8.10 Docs | 2 | Medium |

### Recommended Implementation Order

**Week 1** (Core functionality):
1. **8.1 Rewards** (10 hrs) - Core functionality
2. **8.2 Metrics** (2 hrs) - Needed for logging
3. **8.5 Observations** (5 hrs) - Integration
4. **8.6 Scenarios** (4 hrs) - Config system

**Week 2** (Integration & polish):
5. **8.3 W&B** (2 hrs) - Logging
6. **8.4 Console** (2 hrs) - Output
7. **8.7 CLI** (2 hrs) - Entry point
8. **8.8 Training Loop** (3 hrs) - Wire everything
9. **8.9 Examples** (2 hrs) - Scenarios
10. **8.10 Docs** (2 hrs) - Documentation

**Or**: Parallel work on independent components (Rewards + Metrics + W&B simultaneously)

---

## Success Criteria

Phase 8 complete when:

✅ Can run: `python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb`
✅ Training progresses with rich terminal output
✅ Metrics logged to W&B (per episode + rolling stats)
✅ Episode outcomes tracked correctly (6 types)
✅ Reward components visible in logs
✅ Observations = 738 dims (exact v1 match)
✅ Can compare algorithms (PPO vs TD3 vs SAC)
✅ All 69+ tests passing
✅ Documentation complete
✅ **Performance matches v1** (same obs → same results)

---

## V1 vs V2 Comparison

| Aspect | V1 | V2 | Change |
|--------|----|----|--------|
| **Code** | 5,000 lines | 800 lines | **-84%** |
| **Scenario** | 297 lines | 20 lines | **-93%** |
| **Layers** | 7 | 3 | **-57%** |
| **Factories** | 3 systems | 2 simple | **-33%** |
| **Observations** | 738 dims | 738 dims | **Same** ✅ |
| **Agents** | ✓ | ✓ | **Same** ✅ |
| **Tests** | 30 | 69+ | **+230%** |
| **Metrics** | Basic | Structured | **Better** |
| **W&B** | Manual | Auto | **Better** |
| **Terminal** | Print | Rich | **Better** |
| **Presets** | None | Yes | **New** |

---

## Risk Mitigation

### Risk: Performance regression
**Mitigation**:
- Use exact same observations (738 dims)
- Same agent implementations
- Same environment
- Validate rewards match v1

### Risk: Breaking existing workflows
**Mitigation**:
- Keep v1 available during transition
- Clear migration guide
- Example scenarios for all use cases

### Risk: Scope creep
**Mitigation**:
- Strict adherence to plan
- Time-box each phase
- Focus on core functionality first

---

## Next Steps

1. **Review this design master document** - Ensure alignment
2. **Start Phase 8.1 (Rewards)** - Core functionality
3. **Implement in recommended order** - Build foundation first
4. **Test continuously** - Unit tests + integration tests
5. **Validate against v1** - Ensure performance matches

**Ready to proceed?** Let me know which component to start with!
