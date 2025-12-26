# Changelog

All notable changes to F110_MARL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2025-12-25

### Major Refactor: v2 Architecture

This release introduces a complete architectural refactor (v2) with **84% code reduction** while maintaining full feature parity.

### Added

#### Core Infrastructure
- **Protocol-based agent design** - Structural typing via `@runtime_checkable` Protocol
- **Factory pattern** - `AgentFactory`, `EnvironmentFactory`, `WrapperFactory` for clean object creation
- **Simple training loops** - `TrainingLoop` and `EvaluationLoop` classes (~200 lines total)
- **Utilities module** - Checkpointing, logging, random seeds, metrics computation
- **Clean package structure** - `v2/core/`, `v2/agents/`, `v2/wrappers/`

#### Agents
- All 6 algorithms now use unified protocol:
  - PPO (Proximal Policy Optimization)
  - Recurrent PPO (LSTM-based)
  - TD3 (Twin Delayed DDPG)
  - SAC (Soft Actor-Critic)
  - DQN (Deep Q-Network)
  - Rainbow DQN (6 DQN extensions)
  - FTG (Follow-The-Gap, non-learning baseline)

#### Testing
- **69 comprehensive tests** (53 unit + 16 integration)
- **87-100% agent coverage** (SAC: 87%, Rainbow: 88%, PPO: 89%, Networks: 100%)
- **Integration tests** for config loading, training loops, checkpointing, edge cases
- **Test helpers** for observation wrapping and environment setup

#### Documentation
- **[README.md](README.md)** - Complete project overview with v2 quick start
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Comprehensive v1 → v2 migration guide
- **[DEPRECATION_NOTICE.md](DEPRECATION_NOTICE.md)** - v1 deprecation timeline and rationale
- **[v2/examples/README.md](v2/examples/README.md)** - Detailed v2 examples and patterns
- **Example scripts** - `train_ppo_simple.py`, `train_td3_simple.py`

### Changed

#### Architecture Simplification
- **7 layers → 4 layers** - Removed unnecessary abstraction
- **3 factory systems → 1 factory** - Unified AgentFactory
- **Pydantic models → Plain dicts** - Eliminated config overhead
- **Inheritance → Protocol** - No base classes required

#### Code Reduction
- **Total lines**: ~5,000 → ~800 (84% reduction)
- **Training script**: ~100 lines → ~50 lines (50% reduction)
- **Factory code**: 1,586 lines → ~100 lines (94% reduction)
- **Config system**: Complex Pydantic → Simple YAML loading

#### API Changes
- `agent.select_action()` → `agent.act()`
- `agent.train()` → `agent.update()`
- `agent.save_model()` → `agent.save()`
- `agent.load_model()` → `agent.load()`
- Agent IDs: `agent_0` → `car_0` (match environment)
- TrainingLoop parameters: `max_steps` → `max_steps_per_episode`

### Deprecated

The v1 architecture is **deprecated** and will be removed in a future release:

- ✗ `src/f110x/` - v1 codebase
- ✗ `experiments/` - v1 experiment management
- ✗ `run.py` - v1 training entry point

**Migration required** - See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

### Fixed

- **Test coverage gaps** - Added comprehensive tests for SAC and Rainbow agents
- **Observation wrapper** - Created gym-style wrapper for dict → array transformation
- **API inconsistencies** - Unified parameter naming across all components
- **Package exports** - Added proper `__init__.py` files for all agent packages

### Performance

- **Startup time** - Faster initialization (fewer wrapper objects)
- **Memory footprint** - Reduced (simpler object hierarchy)
- **Training performance** - Identical (same algorithm implementations)
- **Code clarity** - 84% more concise, easier to understand and debug

---

## Metrics

### Code Size Comparison

| Component | v1 Lines | v2 Lines | Reduction |
|-----------|----------|----------|-----------|
| **Total codebase** | ~5,000 | ~800 | **84%** |
| Training script | ~100 | ~50 | 50% |
| Agent factory | 1,586 | ~100 | 94% |
| Training loop | 2,011 | ~200 | 90% |
| Config system | ~500 | ~270 | 46% |

### Test Coverage

| Metric | Value |
|--------|-------|
| **Total tests** | 69/69 passing (100%) |
| Unit tests | 53 |
| Integration tests | 16 |
| **Agent coverage** | 87-100% |
| - SAC | 87% |
| - Rainbow | 88% |
| - PPO | 89% |
| - Networks | 100% |
| **Core coverage** | 90% |

### Architecture Layers

| Aspect | v1 | v2 |
|--------|----|----|
| **Call stack depth** | 7 layers | 4 layers |
| Factory systems | 3 | 1 |
| Config models | Pydantic | Dict |
| Agent design | Inheritance | Protocol |

---

## Migration Path

1. **Read the migration guide** - [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
2. **Review v2 examples** - [v2/examples/](v2/examples/)
3. **Update imports** - Change `from src.f110x.*` to `from v2.core import *`
4. **Convert configs** - Pydantic models → plain dicts
5. **Use factories** - `AgentFactory.create('ppo', config)`
6. **Update training** - Replace `TrainRunner` with `TrainingLoop`
7. **Test thoroughly** - Run your experiments to verify equivalence

---

## Breaking Changes

### Imports
```python
# v1
from agents.ppo import PPOAgent
from engine.builder import build_runner_context

# v2
from v2.core import AgentFactory, EnvironmentFactory
```

### Agent Creation
```python
# v1
agent = PPOAgent(obs_dim=370, act_dim=2, lr=3e-4)

# v2
agent = AgentFactory.create('ppo', {'obs_dim': 370, 'act_dim': 2, 'lr': 3e-4})
```

### Training
```python
# v1
runner = TrainRunner(context, trainers, config)
runner.run(num_episodes=1000)

# v2
training_loop = TrainingLoop(env, agents, max_episodes=1000)
history = training_loop.run()
```

---

## Credits

This refactor consolidates learnings from v1 development and focuses on:
- **Simplicity** - Less code, clearer intent
- **Testability** - Comprehensive test coverage
- **Maintainability** - Protocol-based design, no deep inheritance
- **Usability** - Direct, no CLI/session complexity

Special thanks to the F1TENTH community and PettingZoo for multi-agent standards.

---

## [1.0.0] - 2024

### Initial Release

- Multi-agent F1TENTH racing environment
- 6 RL algorithms (PPO, TD3, SAC, DQN, Rainbow, RecurrentPPO)
- Complex v1 architecture with CLI, sessions, builders
- W&B integration for experiment tracking
- Scenario-based training configurations

---

[2.0.0]: https://github.com/yourusername/F110_MARL/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/yourusername/F110_MARL/releases/tag/v1.0.0
