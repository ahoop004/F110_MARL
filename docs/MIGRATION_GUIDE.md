# Migration Guide: v1 → v2

This guide helps you migrate from the v1 architecture (deprecated) to the streamlined v2 architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Changes](#key-changes)
3. [Breaking Changes](#breaking-changes)
4. [Migration Steps](#migration-steps)
5. [Code Comparisons](#code-comparisons)
6. [API Mapping](#api-mapping)
7. [Common Patterns](#common-patterns)
8. [FAQ](#faq)

---

## Overview

The v2 refactor simplifies the codebase by **84%**, removing unnecessary abstraction layers while maintaining full functionality.

### What Changed

| Aspect | v1 | v2 | Benefit |
|--------|----|----|---------|
| **Architecture** | 7 layers | 4 layers | Simpler call stack |
| **Agent Design** | Inheritance-based | Protocol-based | No inheritance required |
| **Factories** | 3 factory systems | 1 factory | Easier object creation |
| **Config System** | Pydantic models | Simple dicts | Less boilerplate |
| **Code Size** | ~5,000 lines | ~800 lines | 84% reduction |
| **Training Script** | ~100 lines | ~50 lines | 50% shorter |

### What Stayed the Same

- ✅ All 6 RL algorithms (PPO, RecPPO, TD3, SAC, DQN, Rainbow)
- ✅ F110 environment and physics
- ✅ Multi-agent support
- ✅ Checkpointing and logging
- ✅ Algorithm implementations (neural networks, update logic)

---

## Key Changes

### 1. Simplified Architecture

**v1 (7 layers):**
```
CLI → Session → Builder → RunnerContext → TrainerRegistry →
  OnPolicyTrainer → PPOAgent
```

**v2 (4 layers):**
```
Script → AgentFactory → TrainingLoop → Agent
```

### 2. Protocol-Based Agents

**v1:** Inheritance hierarchy
```python
PPOAgent(BaseAgent)
  ↓
BaseAgent(ABC)
```

**v2:** Protocol-based (structural typing)
```python
@runtime_checkable
class Agent(Protocol):
    def act(obs, deterministic): ...
    def store_transition(...): ...
    def update(): ...
    def save(path): ...
    def load(path): ...
```

No inheritance needed - just implement the methods.

### 3. Factory Pattern

**v1:** Multiple factory systems
- `builders.py` (1,586 lines)
- `config_models.py` (complex Pydantic models)
- `trainer_registry.py`

**v2:** Single factory
- [`v2/core/config.py`](v2/core/config.py) (~270 lines)
- Simple dict-based config
- One registry for all agents

### 4. Simplified Config

**v1:** Pydantic models with nested validation
```python
class AgentConfig(BaseModel):
    algorithm: str
    hyperparameters: HyperparameterConfig
    network: NetworkConfig
    ...
```

**v2:** Plain Python dicts
```python
config = {
    'obs_dim': 370,
    'act_dim': 2,
    'lr': 3e-4,
    'gamma': 0.99,
}
```

---

## Breaking Changes

### 1. Import Paths Changed

| v1 | v2 |
|----|-----|
| `from agents.ppo import PPOAgent` | `from v2.core import AgentFactory` |
| `from engine.builder import build_runner_context` | `from v2.core import EnvironmentFactory` |
| `from trainer.on_policy import OnPolicyTrainer` | `from v2.core import TrainingLoop` |
| `from experiments.session import Session` | ❌ Removed - use factories directly |

### 2. Agent Creation Changed

**v1:**
```python
from agents.ppo import PPOAgent

agent = PPOAgent(
    obs_dim=370,
    act_dim=2,
    device='cpu',
    lr=3e-4,
    # ... 20+ parameters
)
```

**v2:**
```python
from v2.core import AgentFactory

agent = AgentFactory.create('ppo', {
    'obs_dim': 370,
    'act_dim': 2,
    'lr': 3e-4,
    'gamma': 0.99,
})
```

### 3. Environment Creation Changed

**v1:**
```python
from engine.builder import build_runner_context
from config_models import EnvironmentConfig

config = EnvironmentConfig(
    map='maps/line_map.yaml',
    num_agents=1,
    ...
)
context = build_runner_context(config)
env = context.env
```

**v2:**
```python
from v2.core import EnvironmentFactory

env = EnvironmentFactory.create({
    'map': 'maps/line_map.yaml',
    'num_agents': 1,
})
```

### 4. Training Loop Changed

**v1:**
```python
from trainer.registry import create_trainer
from engine.runner import TrainRunner

trainers = {id: create_trainer(agent, config) for id, agent in agents.items()}
runner = TrainRunner(context, trainers, config)
runner.run(num_episodes=1000)
```

**v2:**
```python
from v2.core import TrainingLoop

training_loop = TrainingLoop(
    env=env,
    agents=agents,
    max_episodes=1000,
)
history = training_loop.run()
```

### 5. Agent IDs Changed

**v1:** Used `agent_0`, `agent_1`, etc.

**v2:** Environment uses `car_0`, `car_1`, etc.

```python
# v1
agents = {'agent_0': agent}

# v2
agents = {'car_0': agent}  # Match environment agent IDs
```

### 6. Training Loop Parameters Changed

| v1 | v2 |
|----|-----|
| `max_steps` | `max_steps_per_episode` |
| `num_episodes` | `max_episodes` |
| `update_interval` | `update_frequency` |

### 7. Return Values Changed

**TrainingLoop.run():**

**v1:** Returns aggregated metrics dict
```python
metrics = {
    'episode_rewards': [10.5, 12.3, ...],
    'episode_lengths': [100, 120, ...],
}
```

**v2:** Returns training history per agent
```python
history = {
    'car_0': [
        {'episode_reward': 10.5, 'episode_length': 100},
        {'episode_reward': 12.3, 'episode_length': 120},
        ...
    ],
    'car_1': [...],
}
```

---

## Migration Steps

### Step 1: Update Imports

**Before (v1):**
```python
from agents.ppo import PPOAgent
from agents.td3 import TD3Agent
from engine.builder import build_runner_context
from trainer.on_policy import OnPolicyTrainer
from config_models import AgentConfig, EnvironmentConfig
```

**After (v2):**
```python
from v2.core import (
    AgentFactory,
    EnvironmentFactory,
    TrainingLoop,
    save_checkpoint,
    load_checkpoint,
    SimpleLogger,
)
```

### Step 2: Convert Config to Dict

**Before (v1):**
```python
from config_models import AgentConfig

config = AgentConfig(
    algorithm='ppo',
    obs_dim=370,
    act_dim=2,
    lr=3e-4,
    gamma=0.99,
)
```

**After (v2):**
```python
config = {
    'obs_dim': 370,
    'act_dim': 2,
    'lr': 3e-4,
    'gamma': 0.99,
}
```

### Step 3: Use Factories

**Before (v1):**
```python
agent = PPOAgent(obs_dim=370, act_dim=2, lr=3e-4)
```

**After (v2):**
```python
agent = AgentFactory.create('ppo', {
    'obs_dim': 370,
    'act_dim': 2,
    'lr': 3e-4,
})
```

### Step 4: Update Agent IDs

**Before (v1):**
```python
agents = {'agent_0': ppo_agent, 'agent_1': td3_agent}
```

**After (v2):**
```python
agents = {'car_0': ppo_agent, 'car_1': td3_agent}
```

### Step 5: Replace TrainRunner with TrainingLoop

**Before (v1):**
```python
from engine.runner import TrainRunner

runner = TrainRunner(context, trainers, config)
runner.run(num_episodes=1000, max_steps=500)
```

**After (v2):**
```python
from v2.core import TrainingLoop

training_loop = TrainingLoop(
    env=env,
    agents=agents,
    max_episodes=1000,
    max_steps_per_episode=500,
)
history = training_loop.run()
```

### Step 6: Update Checkpoint Paths

**Before (v1):**
```python
# Checkpoint saved as: checkpoints/agent_0_episode_100.pt
```

**After (v2):**
```python
# Checkpoint saved as: checkpoints/checkpoint_episode_100/car_0.pt
save_checkpoint(agents, episode=100, checkpoint_dir='checkpoints/')
```

---

## Code Comparisons

### Complete Training Script

#### v1 (~100 lines)

```python
"""v1 training script"""
import yaml
from pathlib import Path

from config_models import (
    ExperimentConfig,
    EnvironmentConfig,
    AgentConfig,
    TrainingConfig,
)
from experiments.session import Session
from engine.builder import build_runner_context
from agents.ppo import PPOAgent
from trainer.registry import create_trainer
from engine.runner import TrainRunner

# Load config
with open('scenarios/ppo_racing.yaml', 'r') as f:
    raw_config = yaml.safe_load(f)

# Parse config
experiment_config = ExperimentConfig(**raw_config)
env_config = EnvironmentConfig(**raw_config['environment'])
agent_config = AgentConfig(**raw_config['agents']['agent_0'])
training_config = TrainingConfig(**raw_config['training'])

# Build context
context = build_runner_context(env_config)

# Create agents
agents = {}
for agent_id, cfg in raw_config['agents'].items():
    agent_params = AgentConfig(**cfg)
    if agent_params.algorithm == 'ppo':
        agents[agent_id] = PPOAgent(
            obs_dim=agent_params.obs_dim,
            act_dim=agent_params.act_dim,
            lr=agent_params.hyperparameters.lr,
            gamma=agent_params.hyperparameters.gamma,
            # ... 15 more parameters
        )

# Create trainers
trainers = {}
for agent_id, agent in agents.items():
    trainers[agent_id] = create_trainer(
        agent=agent,
        trainer_type='on_policy',
        config=training_config,
    )

# Create runner
runner = TrainRunner(
    context=context,
    trainers=trainers,
    config=training_config,
    experiment_config=experiment_config,
)

# Train
runner.run(
    num_episodes=training_config.num_episodes,
    max_steps=training_config.max_steps,
)
```

#### v2 (~50 lines)

```python
"""v2 training script"""
from v2.core import (
    AgentFactory,
    EnvironmentFactory,
    TrainingLoop,
    SimpleLogger,
    save_checkpoint,
    set_random_seeds,
)

# Set random seed
set_random_seeds(42)

# Create environment
env = EnvironmentFactory.create({
    'map': 'maps/line_map.yaml',
    'num_agents': 1,
    'timestep': 0.01,
})

# Create agent
agent = AgentFactory.create('ppo', {
    'obs_dim': 370,
    'act_dim': 2,
    'lr': 3e-4,
    'gamma': 0.99,
})

# Create logger
logger = SimpleLogger(log_dir='logs/ppo_training', verbose=True)

# Train
training_loop = TrainingLoop(
    env=env,
    agents={'car_0': agent},
    max_episodes=1000,
    max_steps_per_episode=500,
    log_callback=lambda ep, stats: logger.log(ep, stats),
    checkpoint_callback=lambda ep, agents: (
        save_checkpoint(agents, ep, 'checkpoints/')
        if ep % 100 == 0 else None
    ),
)

history = training_loop.run()
```

**Result:** 50% fewer lines, 100% clearer intent.

---

## API Mapping

### Agent Methods

| v1 Method | v2 Method | Notes |
|-----------|-----------|-------|
| `agent.select_action(obs)` | `agent.act(obs, deterministic=False)` | Renamed for clarity |
| `agent.store(...)` | `agent.store_transition(...)` | Same functionality |
| `agent.train()` | `agent.update()` | Returns metrics dict |
| `agent.save_model(path)` | `agent.save(path)` | Simplified name |
| `agent.load_model(path)` | `agent.load(path)` | Simplified name |

### Training Loop

| v1 | v2 | Notes |
|----|-----|-------|
| `TrainRunner.run(num_episodes=N)` | `TrainingLoop.run()` | Max episodes set in constructor |
| `runner.context.env` | `training_loop.env` | Direct access |
| `runner.trainers[id]` | `training_loop.agents[id]` | Agents, not trainers |

### Checkpointing

| v1 | v2 | Notes |
|----|-----|-------|
| `agent.save_model(f'agent_{id}.pt')` | `save_checkpoint(agents, episode, dir)` | Unified checkpoint system |
| `agent.load_model(path)` | `load_checkpoint(agents, path)` | Loads all agents at once |

### Logging

| v1 | v2 | Notes |
|----|-----|-------|
| `logger.log_metrics(metrics)` | `logger.log(episode, metrics)` | Simpler interface |
| `logger.get_stats()` | `logger.get_summary()` | Returns summary statistics |

---

## Common Patterns

### Pattern 1: Single-Agent Training

**v1:**
```python
# Create session
session = Session(config_path='scenario.yaml')

# Build context
context = session.build_context()

# Create agent
agent = session.create_agent('agent_0')

# Create trainer
trainer = create_trainer(agent, 'on_policy', config)

# Train
runner = TrainRunner(context, {'agent_0': trainer}, config)
runner.run(num_episodes=1000)
```

**v2:**
```python
env = EnvironmentFactory.create(env_config)
agent = AgentFactory.create('ppo', agent_config)

training_loop = TrainingLoop(env, {'car_0': agent}, max_episodes=1000)
history = training_loop.run()
```

### Pattern 2: Multi-Agent Training

**v1:**
```python
# Create multiple agents
agents = {
    'agent_0': PPOAgent(...),
    'agent_1': TD3Agent(...),
}

# Create trainers
trainers = {
    'agent_0': OnPolicyTrainer(agents['agent_0'], config),
    'agent_1': OffPolicyTrainer(agents['agent_1'], config),
}

# Train
runner = TrainRunner(context, trainers, config)
runner.run(num_episodes=1000)
```

**v2:**
```python
agents = {
    'car_0': AgentFactory.create('ppo', ppo_config),
    'car_1': AgentFactory.create('td3', td3_config),
}

training_loop = TrainingLoop(env, agents, max_episodes=1000)
history = training_loop.run()
```

### Pattern 3: Checkpoint and Resume

**v1:**
```python
# Save
agent.save_model(f'checkpoints/agent_0_ep_{episode}.pt')

# Load
agent.load_model('checkpoints/agent_0_ep_100.pt')
```

**v2:**
```python
# Save
save_checkpoint(
    agents=agents,
    episode=100,
    checkpoint_dir='checkpoints/',
    metrics={'reward': 42.5}
)

# Load
metadata = load_checkpoint(
    agents=agents,
    checkpoint_path='checkpoints/checkpoint_episode_100'
)
print(f"Resumed from episode {metadata['episode']}")
```

### Pattern 4: Custom Callbacks

**v1:**
```python
# No built-in callback support
# Had to subclass TrainRunner
```

**v2:**
```python
def log_callback(episode, stats):
    print(f"Episode {episode}: {stats}")

def checkpoint_callback(episode, agents):
    if episode % 100 == 0:
        save_checkpoint(agents, episode, 'checkpoints/')

training_loop = TrainingLoop(
    env, agents, max_episodes=1000,
    log_callback=log_callback,
    checkpoint_callback=checkpoint_callback,
)
training_loop.run()
```

---

## FAQ

### Q: Will v1 still work?

**A:** Yes, v1 code is still present in the repository but is **deprecated**. It will be removed in a future release. We recommend migrating to v2 as soon as possible.

### Q: Can I mix v1 and v2 code?

**A:** No. v1 and v2 use different import paths and APIs. You should migrate your entire training script to v2.

### Q: Are the algorithms identical between v1 and v2?

**A:** Yes! The core algorithm implementations (PPO, TD3, SAC, etc.) are nearly identical. The v2 refactor only changed the **architecture** around the algorithms, not the algorithms themselves.

### Q: What about my saved checkpoints from v1?

**A:** v2 uses the same PyTorch `.pt` format, but the checkpoint **structure** is different:
- **v1:** `checkpoints/agent_0_episode_100.pt` (single file per agent)
- **v2:** `checkpoints/checkpoint_episode_100/car_0.pt` (directory per checkpoint)

You'll need to manually reorganize v1 checkpoints to match v2 structure, or write a simple conversion script.

### Q: Is v2 faster than v1?

**A:** Training performance is identical (same algorithms). However:
- ✅ **Startup is faster** (less initialization overhead)
- ✅ **Code is clearer** (easier to debug and optimize)
- ✅ **Memory footprint is smaller** (fewer wrapper objects)

### Q: Can I contribute to v1?

**A:** We only accept contributions to v2. v1 is in maintenance mode and will be archived.

### Q: How do I report bugs?

**A:** Open an issue on GitHub with:
- v2 version
- Minimal reproduction script
- Error message/traceback
- Expected vs. actual behavior

### Q: Where can I find more examples?

**A:** See [`v2/examples/`](v2/examples/) directory:
- [`train_ppo_simple.py`](v2/examples/train_ppo_simple.py) - Single-agent PPO
- [`train_td3_simple.py`](v2/examples/train_td3_simple.py) - Off-policy TD3
- [`README.md`](v2/examples/README.md) - Detailed examples and patterns

---

## Need Help?

- **Examples:** [`v2/examples/`](v2/examples/)
- **Tests:** [`tests/`](tests/) (69 tests showing usage patterns)
- **Issues:** [GitHub Issues](https://github.com/yourusername/F110_MARL/issues)

---

**Happy Migrating!** The v2 way is simpler, clearer, and more maintainable.
