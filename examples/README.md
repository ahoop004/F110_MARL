# v2 Training Examples

This directory contains example training scripts demonstrating the v2 architecture.

---

## Quick Start

### Simple PPO Training

```bash
python v2/examples/train_ppo_simple.py
```

This runs a minimal PPO training demo:
- Single agent on line track
- 50 episodes
- Automatic checkpointing every 10 episodes
- CSV logging to `logs/ppo_simple/`

**What it demonstrates:**
- ✅ Factory-based agent creation
- ✅ Environment setup
- ✅ Training loop usage
- ✅ Checkpoint saving
- ✅ Metrics logging

---

## Example Scripts

### 1. `train_ppo_simple.py`

**Minimal PPO training example**

- Algorithm: PPO
- Agents: 1
- Episodes: 50
- Map: Line track
- Features: Checkpointing, logging

**Key code:**
```python
# Create agent
agent = AgentFactory.create('ppo', config)

# Create environment
env = EnvironmentFactory.create(env_config)

# Train!
training_loop = TrainingLoop(env, {'agent_0': agent}, max_episodes=50)
training_loop.run()
```

---

## v2 Architecture Overview

The examples demonstrate the simplified v2 architecture:

### Before (v1)
```
CLI → Session → Builder → RunnerContext → Trainer Registry →
  OnPolicyTrainer → PPOAgent
```
**7 layers, 3 factory systems, ~230 lines of wrapper code**

### After (v2)
```
Script → AgentFactory → TrainingLoop → PPOAgent
```
**4 layers, 1 factory, 0 wrappers**

---

## Code Comparison

### v1 (Old Way)
```python
# Complex multi-step setup
from experiments.session import Session
from engine.builder import build_runner_context
from trainer.registry import create_trainer

config = load_config('scenario.yaml')
context = build_runner_context(config)
agents = build_agents(config)
trainers = {id: create_trainer(agent, config) for id, agent in agents.items()}
runner = TrainRunner(context, trainers, config)
runner.run()
```

### v2 (New Way)
```python
# Simple, direct setup
from v2.core import AgentFactory, EnvironmentFactory, TrainingLoop

agent = AgentFactory.create('ppo', config)
env = EnvironmentFactory.create(env_config)
training_loop = TrainingLoop(env, {'agent_0': agent}, max_episodes=100)
training_loop.run()
```

**84% less code, 100% clearer!**

---

## Features Demonstrated

### ✅ Agent Protocol
All agents implement the same simple interface:
```python
agent.act(obs, deterministic=False)  # Select action
agent.store(...)                     # Store transition
agent.update()                       # Learn
agent.save(path) / load(path)        # Checkpointing
```

### ✅ Factory Pattern
Simple creation from config:
```python
# Agents
agent = AgentFactory.create('ppo', config)
# Available: ppo, td3, sac, dqn, rainbow

# Environment
env = EnvironmentFactory.create(env_config)

# Complete setup from YAML
setup = create_training_setup('scenario.yaml')
```

### ✅ Training Loop
Clean, focused training logic:
```python
training_loop = TrainingLoop(
    env=env,
    agents=agents,
    max_episodes=1000,
    log_callback=logger.log,
    checkpoint_callback=save_checkpoint,
)
history = training_loop.run()
```

### ✅ Utilities
```python
# Logging
logger = SimpleLogger(log_dir='logs/', verbose=True)
logger.log(episode=1, metrics={'reward': 10.5})

# Checkpointing
save_checkpoint(agents, episode=100, checkpoint_dir='checkpoints/')
load_checkpoint(agents, 'checkpoints/checkpoint_episode_100/')

# Random seeds
set_random_seeds(42)
```

---

## Adding Your Own Training Script

1. **Import v2 components:**
```python
from v2.core import (
    AgentFactory,
    EnvironmentFactory,
    TrainingLoop,
    SimpleLogger,
)
```

2. **Create environment and agents:**
```python
env = EnvironmentFactory.create({'map': 'maps/your_map.yaml', ...})
agent = AgentFactory.create('ppo', {...})
```

3. **Set up training:**
```python
logger = SimpleLogger(log_dir='logs/my_experiment')
training_loop = TrainingLoop(
    env,
    {'agent_0': agent},
    max_episodes=1000,
    log_callback=logger.log
)
```

4. **Train:**
```python
training_loop.run()
```

Done! No builders, no wrappers, no complexity.

---

## Output Structure

After running an example:

```
.
├── logs/
│   └── ppo_simple/
│       └── training_metrics.csv      # Episode metrics
└── checkpoints/
    └── ppo_simple/
        ├── ppo_episode_10/
        │   ├── agent_0.pt            # Agent weights
        │   └── metadata.json         # Episode info
        ├── ppo_episode_20/
        └── ...
```

---

## Next Steps

- Modify `train_ppo_simple.py` to try different hyperparameters
- Create your own training script for multi-agent scenarios
- Experiment with different algorithms (TD3, SAC, DQN, Rainbow)
- Add custom callbacks for evaluation, early stopping, etc.

---

## Comparison: Lines of Code

| Component | v1 | v2 | Reduction |
|-----------|----|----|-----------|
| Training script | ~100 lines | ~50 lines | -50% |
| Setup boilerplate | ~30 lines | ~10 lines | -67% |
| Factory imports | ~5 modules | ~1 module | -80% |
| **Clarity** | **Complex** | **Simple** | **✅** |

---

**The v2 way: Less code, more clarity, same power.**
