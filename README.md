# F110_MARL

**Multi-Agent Reinforcement Learning for F1TENTH Autonomous Racing**

A lightweight, modular research platform for training and evaluating MARL algorithms in adversarial F1TENTH racing scenarios.

---

## Features

- **6 RL Algorithms**: PPO, Recurrent PPO, TD3, SAC, DQN, Rainbow DQN
- **Protocol-Based Design**: Clean, minimal agent interface
- **Factory Pattern**: Simple agent and environment creation from config
- **Multi-Agent Support**: Train competitive or cooperative policies
- **Comprehensive Testing**: 69 unit and integration tests with 87-100% agent coverage
- **Production-Ready**: Checkpointing, logging, evaluation loops, and utilities

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/F110_MARL.git
cd F110_MARL

# Install dependencies (recommended: create a virtual environment first)
pip install -r requirements.txt
```

### Train a PPO Agent (v2 Architecture)

```python
from v2.core import AgentFactory, EnvironmentFactory, TrainingLoop, set_random_seeds

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

# Train!
training_loop = TrainingLoop(
    env=env,
    agents={'agent_0': agent},
    max_episodes=1000,
    max_steps_per_episode=500,
)
history = training_loop.run()
```

Run the complete example:
```bash
python v2/examples/train_ppo_simple.py
```

---

## Architecture Overview

### v2 (Current - Recommended)

The v2 architecture emphasizes **simplicity** and **clarity**:

```
Script → AgentFactory → TrainingLoop → Agent
```

**Key Components:**
- [`v2/core/config.py`](v2/core/config.py) - Factories for agents, environments, and wrappers
- [`v2/core/training.py`](v2/core/training.py) - TrainingLoop and EvaluationLoop
- [`v2/core/utils.py`](v2/core/utils.py) - Checkpointing, logging, and utilities
- [`v2/agents/`](v2/agents/) - Protocol-compliant RL agents

**Agent Protocol:**
```python
@runtime_checkable
class Agent(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray: ...
    def store_transition(self, obs, action, reward, next_obs, done): ...
    def update(self) -> Dict[str, float]: ...
    def save(self, path: str): ...
    def load(self, path: str): ...
```

All agents implement this simple interface - no inheritance required.

### v1 (Legacy - Deprecated)

The v1 architecture is **deprecated** and will be removed in a future release. See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for details on migrating to v2.

---

## Available Algorithms

| Algorithm | Type | Actions | Features |
|-----------|------|---------|----------|
| **PPO** | On-policy | Continuous | Stable, sample-efficient, clipped objectives |
| **Recurrent PPO** | On-policy | Continuous | LSTM for partial observability |
| **TD3** | Off-policy | Continuous | Twin Q-networks, target smoothing |
| **SAC** | Off-policy | Continuous | Maximum entropy, auto-tuning |
| **DQN** | Off-policy | Discrete | Experience replay, target network |
| **Rainbow DQN** | Off-policy | Discrete | 6 DQN extensions (categorical, noisy, n-step, PER, double, dueling) |

All algorithms support:
- ✅ Checkpointing and resuming
- ✅ Device selection (CPU/GPU)
- ✅ Configurable network architectures
- ✅ Replay buffers (off-policy)
- ✅ Action space handling (continuous/discrete)

---

## Project Structure

```
F110_MARL/
├── v2/                          # Current architecture (v2)
│   ├── core/
│   │   ├── config.py           # Factories and config loading
│   │   ├── training.py         # Training and evaluation loops
│   │   └── utils.py            # Utilities (logging, checkpoints, etc.)
│   ├── agents/                 # RL agent implementations
│   │   ├── ppo/
│   │   ├── td3/
│   │   ├── sac/
│   │   ├── dqn/
│   │   └── rainbow/
│   ├── wrappers/               # Environment wrappers
│   │   ├── observation.py
│   │   ├── action.py
│   │   └── reward.py
│   ├── env/                    # F110 environment
│   │   └── f110ParallelEnv.py
│   └── examples/               # Training examples
│       ├── train_ppo_simple.py
│       └── train_td3_simple.py
├── tests/                      # Test suite (69 tests)
│   ├── unit/                   # Agent unit tests
│   └── integration/            # Integration tests
├── maps/                       # Track configurations
├── logs/                       # Training logs (created at runtime)
└── checkpoints/                # Model checkpoints (created at runtime)
```

---

## Documentation

- **[v2/examples/README.md](v2/examples/README.md)** - Detailed v2 examples and code comparisons
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Guide for migrating from v1 to v2
- **[REFACTOR_TODO.md](REFACTOR_TODO.md)** - Complete refactor progress and metrics

---

## Examples

### Multi-Agent Training

```python
from v2.core import AgentFactory, EnvironmentFactory, TrainingLoop

# Create environment with 2 agents
env = EnvironmentFactory.create({
    'map': 'maps/line_map.yaml',
    'num_agents': 2,
    'start_poses': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
})

# Create different agents for each car
agents = {
    'car_0': AgentFactory.create('ppo', ppo_config),
    'car_1': AgentFactory.create('td3', td3_config),
}

# Train both agents
training_loop = TrainingLoop(env, agents, max_episodes=1000)
history = training_loop.run()
```

### Checkpoint Management

```python
from v2.core.utils import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(
    agents=agents,
    episode=100,
    checkpoint_dir='checkpoints/my_experiment',
    metrics={'mean_reward': 42.5}
)

# Load checkpoint
metadata = load_checkpoint(
    agents=agents,
    checkpoint_path='checkpoints/my_experiment/checkpoint_episode_100'
)
print(f"Resumed from episode {metadata['episode']}")
```

### Evaluation

```python
from v2.core import EvaluationLoop

# Evaluate trained agents
eval_loop = EvaluationLoop(
    env=env,
    agents=agents,
    num_episodes=10,
    deterministic=True  # Use deterministic actions
)
results = eval_loop.run()

print(f"Mean reward: {results['car_0']['mean_reward']:.2f}")
print(f"Std reward: {results['car_0']['std_reward']:.2f}")
```

### Custom Logging

```python
from v2.core.utils import SimpleLogger

logger = SimpleLogger(log_dir='logs/my_experiment', verbose=True)

training_loop = TrainingLoop(
    env=env,
    agents=agents,
    max_episodes=1000,
    log_callback=lambda ep, stats: logger.log(ep, stats)
)
training_loop.run()

# Get summary statistics
summary = logger.get_summary()
print(f"Average episode reward: {summary['mean_episode_reward']:.2f}")
```

---

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=v2 --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
```

**Current Coverage:**
- **Total Tests:** 69/69 passing (100%)
- **Agent Coverage:** 87-100% (SAC: 87%, Rainbow: 88%, PPO: 89%, Networks: 100%)
- **Core Coverage:** 90% (Training loops, utilities)

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{f110_marl,
  title = {F110_MARL: Multi-Agent Reinforcement Learning for F1TENTH Racing},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/F110_MARL}
}
```

---

## License

[Add your license here]

---

## Acknowledgments

- F1TENTH Gym for the racing simulator
- PettingZoo for multi-agent environment standards
- OpenAI Spinning Up for RL algorithm references

---

## Version History

- **v2.0** (2025-12-25) - Complete refactor with simplified architecture
  - 84% code reduction
  - Protocol-based agent design
  - Factory pattern for object creation
  - Comprehensive test suite (69 tests)
- **v1.0** (2024) - Initial implementation (deprecated)

---

**For v1 users:** See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for migrating to v2.
