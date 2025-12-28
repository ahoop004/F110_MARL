# Deprecation Notice: v1 Architecture

**Status:** DEPRECATED
**Date:** 2025-12-25
**Replacement:** v2 architecture (see [`v2/`](v2/) directory)

---

## Overview

The v1 architecture (`src/f110x/`, `experiments/`, `run.py`) is **deprecated** and will be removed in a future release.

All new development should use the v2 architecture.

---

## Migration Required

If you are currently using v1, please migrate to v2 as soon as possible.

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

---

## What's Deprecated

The following v1 components are deprecated:

### Code
- ✗ `src/f110x/` - v1 codebase (complex, 7-layer architecture)
- ✗ `experiments/` - v1 experiment management (CLI, session management)
- ✗ `run.py` - v1 training entry point

### Systems
- ✗ Complex Pydantic config models (`config_models.py`, `config_schema.py`)
- ✗ Multiple factory systems (`builders.py`, `trainer_registry.py`)
- ✗ Wrapper-based trainer classes (`OnPolicyTrainer`, `OffPolicyTrainer`)
- ✗ Session and CLI management (`experiments/session.py`, `experiments/cli.py`)

---

## What Replaces v1

The v2 architecture provides all v1 functionality with **84% less code**:

| v1 Component | v2 Replacement | Benefits |
|--------------|----------------|----------|
| `src/f110x/` (5,000 lines) | `v2/` (800 lines) | 84% reduction, clearer structure |
| `experiments/session.py` | `v2/core/config.py` | Simple factory pattern |
| `run.py` | `v2/examples/train_*.py` | Direct, no CLI complexity |
| `builders.py` (1,586 lines) | `AgentFactory` (~100 lines) | One registry, simple creation |
| Pydantic models | Plain dicts | Less boilerplate |
| Inheritance-based agents | Protocol-based agents | No inheritance required |

---

## Timeline

- **2025-12-25:** v2 released, v1 deprecated
- **2026-01-XX:** v1 archived (moved to `legacy/` directory)
- **2026-02-XX:** v1 removed entirely

We recommend migrating before the archive date to avoid disruption.

---

## Support

- **v2 Support:** Full support for bug fixes, features, and improvements
- **v1 Support:** Critical bug fixes only, no new features
- **Migration Help:** See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

---

## Why Deprecate v1?

The v1 architecture accumulated complexity over time:

1. **7 layers of abstraction** - Made debugging and understanding difficult
2. **3 factory systems** - Confusing, redundant object creation
3. **Pydantic overhead** - Unnecessary validation for research code
4. **Poor testability** - Complex initialization made testing hard
5. **Maintenance burden** - Too much code for the functionality provided

v2 fixes all of these issues while maintaining feature parity.

---

## Getting Started with v2

See the [README.md](README.md) for v2 quick start examples.

### Simple Example

```python
from v2.core import AgentFactory, EnvironmentFactory, TrainingLoop

# Create environment and agent
env = EnvironmentFactory.create({'map': 'maps/line_map.yaml', 'num_agents': 1})
agent = AgentFactory.create('ppo', {'obs_dim': 370, 'act_dim': 2, 'lr': 3e-4})

# Train!
training_loop = TrainingLoop(env, {'car_0': agent}, max_episodes=1000)
history = training_loop.run()
```

That's it. No CLI, no builders, no sessions.

---

## Questions?

- **Documentation:** [README.md](README.md), [v2/examples/README.md](v2/examples/README.md)
- **Migration Guide:** [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Issues:** [GitHub Issues](https://github.com/yourusername/F110_MARL/issues)

---

**Bottom Line:** Please migrate to v2. It's simpler, clearer, and more maintainable.
