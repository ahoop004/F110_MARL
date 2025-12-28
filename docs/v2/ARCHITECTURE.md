# V2 Architecture Design

## Overview

V2 is a **complete rewrite** focused on:
- ✅ **Simplicity**: 84% less code than v1
- ✅ **Clarity**: Protocol-based, 3 layers instead of 7
- ✅ **Composability**: Mix and match components
- ✅ **Research workflow**: Edit scenario → Run → Track in W&B → Iterate

---

## Current State (Phase 7 Complete)

### What Exists ✅

**Core Infrastructure** ([v2/core/](core/)):
- `agent_factory.py` - Creates agents from config (PPO, TD3, SAC, DQN, Rainbow, FTG)
- `env_factory.py` - Creates F110 environments from config
- `training.py` - Basic training loop (env.step → agent.act → agent.update)

**Agents** ([v2/agents/](agents/)):
- PPO, Recurrent PPO
- TD3 (off-policy)
- SAC (off-policy)
- DQN, Rainbow DQN
- FTG (Follow-The-Gap baseline)

**Wrappers** ([v2/wrappers/](wrappers/)):
- Observation filtering
- Frame stacking
- Action noise
- Episode recording

**Tests** ([tests/](../tests/)):
- 69 tests (Phase 6 complete)
- 230% of target coverage

### What's Missing ❌

1. **Reward system** (just designed, not implemented)
2. **Metrics tracking** (episode outcomes, success rates)
3. **W&B integration** (logging, hyperparameter tracking)
4. **CLI interface** (`v2/run.py` with scenario loading)
5. **Scenario parser** (YAML → config dict)
6. **Rich terminal output** (progress bars, live metrics)
7. **Rendering integration** (watch training visually)

---

## Target Architecture

### File Structure

```
v2/
├── __init__.py
├── ARCHITECTURE.md          # This file
├── run.py                   # 🆕 Main CLI entry point
│
├── core/                    # Core infrastructure
│   ├── __init__.py
│   ├── agent_factory.py     # ✅ Exists
│   ├── env_factory.py       # ✅ Exists
│   ├── training.py          # ✅ Exists
│   ├── config.py            # 🆕 Config loading & validation
│   ├── scenario.py          # 🆕 Scenario file parser
│   └── presets.py           # 🆕 Preset configurations
│
├── agents/                  # ✅ RL algorithms (complete)
│   ├── ppo/
│   ├── td3/
│   ├── sac/
│   ├── dqn/
│   ├── rainbow/
│   └── ftg/
│
├── wrappers/                # ✅ Environment wrappers (complete)
│   ├── observation.py
│   ├── frame_stack.py
│   ├── action_noise.py
│   └── recorder.py
│
├── rewards/                 # 🆕 Reward system (designed, not implemented)
│   ├── __init__.py
│   ├── README.md
│   ├── DESIGN.md
│   ├── base.py              # Reward protocols
│   ├── composer.py          # Reward composition
│   ├── presets.py           # Reward presets
│   └── gaplock/
│       ├── __init__.py
│       ├── gaplock.py       # Main gaplock reward
│       ├── terminal.py      # Terminal rewards
│       ├── pressure.py      # Pressure rewards
│       ├── distance.py      # Distance shaping
│       ├── heading.py       # Heading alignment
│       ├── speed.py         # Speed bonuses
│       ├── forcing.py       # Forcing rewards (optional)
│       └── penalties.py     # Behavior penalties
│
├── metrics/                 # 🆕 Metrics tracking
│   ├── __init__.py
│   ├── tracker.py           # Episode outcome tracking
│   ├── aggregator.py        # Rolling statistics
│   └── outcomes.py          # Outcome definitions
│
├── logging/                 # 🆕 W&B integration
│   ├── __init__.py
│   ├── wandb_logger.py      # W&B logging
│   └── console.py           # Rich terminal output
│
└── examples/                # ✅ Example scripts (exist)
    ├── train_ppo.py
    ├── train_td3.py
    └── README.md
```

### Scenario Files

```
scenarios/
└── v2/                      # 🆕 V2 scenarios
    ├── gaplock_ppo.yaml
    ├── gaplock_td3.yaml
    ├── gaplock_sac.yaml
    └── ablation/
        ├── no_forcing.yaml
        ├── simple_rewards.yaml
        └── full_rewards.yaml
```

---

## Component Design

### 1. Configuration System

**Goal**: Simple, flexible configuration with presets and overrides.

#### Config Structure

```python
# Complete experiment config
{
    'experiment': {
        'name': 'gaplock_ppo_v1',
        'episodes': 1000,
        'seed': 42,
    },

    'environment': {
        'map': 'maps/line_map.yaml',
        'num_agents': 2,
        'max_steps': 5000,
        'spawn_points': ['spawn_2', 'spawn_1'],  # [attacker, defender]
        'timestep': 0.01,
        'idle': {
            'speed_threshold': 0.12,
            'patience_steps': 120,
        },
    },

    'agents': {
        'car_0': {  # Attacker (trainable)
            'role': 'attacker',
            'algorithm': 'ppo',
            'params': {
                'lr': 0.0003,
                'gamma': 0.995,
                'hidden_dims': [256, 128],
            },
            'reward': {
                'preset': 'gaplock_simple',
                'overrides': {
                    'terminal': {'target_crash': 100.0},
                },
            },
        },
        'car_1': {  # Defender (baseline)
            'role': 'defender',
            'algorithm': 'ftg',
            'params': {},
        },
    },

    'wandb': {
        'enabled': True,
        'project': 'f110-gaplock',
        'entity': None,
        'tags': ['ppo', 'gaplock', 'simple-rewards'],
    },

    'rendering': {
        'enabled': False,
        'record_video': False,
    },
}
```

#### Preset System

**Algorithm Presets**:
```python
# v2/core/presets.py

ALGORITHM_PRESETS = {
    'ppo_gaplock': {
        'algorithm': 'ppo',
        'params': {
            'lr': 0.0003,
            'gamma': 0.995,
            'gae_lambda': 0.95,
            'hidden_dims': [256, 128],
            'clip_epsilon': 0.2,
        },
    },
    'td3_gaplock': {
        'algorithm': 'td3',
        'params': {
            'lr_actor': 0.0003,
            'lr_critic': 0.001,
            'gamma': 0.99,
            'hidden_dims': [400, 300],
        },
    },
    # ... etc
}

REWARD_PRESETS = {
    'gaplock_simple': { ... },  # From v2/rewards/presets.py
    'gaplock_medium': { ... },
    'gaplock_full': { ... },
}
```

**Usage in Scenarios**:
```yaml
# Ultra-minimal scenario using all presets
agents:
  car_0:
    preset: ppo_gaplock  # Uses algorithm + reward presets
    overrides:
      params:
        lr: 0.0005  # Just tweak what you need
```

#### Config Loading Flow

```
Scenario YAML
    ↓
Load & parse YAML
    ↓
Expand presets (algorithm, reward)
    ↓
Apply overrides
    ↓
Validate config
    ↓
Return config dict
```

### 2. Episode Outcomes & Metrics

**6 Mutually Exclusive Outcomes**:

```python
# v2/metrics/outcomes.py

from enum import Enum

class EpisodeOutcome(Enum):
    """Episode termination outcomes for gaplock task."""

    # Attacker success
    TARGET_CRASH = "target_crash"           # Target crashed, attacker survived

    # Attacker failures
    SELF_CRASH = "self_crash"               # Attacker crashed solo
    COLLISION = "collision"                 # Both crashed (mutual collision)
    TIMEOUT = "timeout"                     # Max steps reached
    IDLE_STOP = "idle_stop"                 # Attacker stopped moving
    TARGET_FINISH = "target_finish"         # Target crossed finish line

def determine_outcome(info: dict) -> EpisodeOutcome:
    """
    Determine episode outcome from info dict.

    Priority order (check in this order):
    1. Target finish (highest priority - immediate success for target)
    2. Collision (both crashed)
    3. Target crash (attacker success)
    4. Self crash (attacker solo crash)
    5. Idle stop (truncation due to idle)
    6. Timeout (max steps)
    """
    attacker_id = 'car_0'
    target_id = 'car_1'

    # Check if target finished
    if info.get(f'{target_id}/finished', False):
        return EpisodeOutcome.TARGET_FINISH

    # Check crash states
    attacker_crashed = info.get(f'{attacker_id}/collision', False)
    target_crashed = info.get(f'{target_id}/collision', False)

    if attacker_crashed and target_crashed:
        return EpisodeOutcome.COLLISION

    if target_crashed:
        return EpisodeOutcome.TARGET_CRASH

    if attacker_crashed:
        return EpisodeOutcome.SELF_CRASH

    # Check truncation reasons
    if info.get('idle_triggered', False):
        return EpisodeOutcome.IDLE_STOP

    if info.get('truncated', False):
        return EpisodeOutcome.TIMEOUT

    # Shouldn't reach here, but default to timeout
    return EpisodeOutcome.TIMEOUT
```

**Metrics Tracking**:

```python
# v2/metrics/tracker.py

from typing import Dict, List
from dataclasses import dataclass
from .outcomes import EpisodeOutcome

@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    outcome: EpisodeOutcome
    steps: int
    total_reward: float
    reward_components: Dict[str, float]  # From reward.compute()

    # Derived metrics
    @property
    def attacker_success(self) -> bool:
        return self.outcome == EpisodeOutcome.TARGET_CRASH

    @property
    def attacker_failure(self) -> bool:
        return self.outcome in [
            EpisodeOutcome.SELF_CRASH,
            EpisodeOutcome.COLLISION,
            EpisodeOutcome.TIMEOUT,
            EpisodeOutcome.IDLE_STOP,
            EpisodeOutcome.TARGET_FINISH,
        ]

    @property
    def target_success(self) -> bool:
        return self.outcome in [
            EpisodeOutcome.TARGET_FINISH,
            EpisodeOutcome.SELF_CRASH,
            EpisodeOutcome.IDLE_STOP,
        ]

    @property
    def target_failure(self) -> bool:
        return self.outcome in [
            EpisodeOutcome.TARGET_CRASH,
            EpisodeOutcome.COLLISION,
        ]

class MetricsTracker:
    """Tracks episode outcomes and rolling statistics."""

    def __init__(self, window: int = 100):
        self.window = window
        self.episodes: List[EpisodeMetrics] = []

    def add_episode(self, metrics: EpisodeMetrics) -> None:
        self.episodes.append(metrics)

    def recent(self, n: int = None) -> List[EpisodeMetrics]:
        """Get last N episodes (default: window)."""
        n = n or self.window
        return self.episodes[-n:]

    def success_rate(self, n: int = None) -> float:
        """Attacker success rate over last N episodes."""
        recent = self.recent(n)
        if not recent:
            return 0.0
        successes = sum(ep.attacker_success for ep in recent)
        return successes / len(recent)

    def outcome_counts(self, n: int = None) -> Dict[EpisodeOutcome, int]:
        """Count outcomes over last N episodes."""
        recent = self.recent(n)
        counts = {outcome: 0 for outcome in EpisodeOutcome}
        for ep in recent:
            counts[ep.outcome] += 1
        return counts

    def average_reward(self, n: int = None) -> float:
        """Average total reward over last N episodes."""
        recent = self.recent(n)
        if not recent:
            return 0.0
        return sum(ep.total_reward for ep in recent) / len(recent)

    def average_steps(self, n: int = None) -> float:
        """Average episode length over last N episodes."""
        recent = self.recent(n)
        if not recent:
            return 0.0
        return sum(ep.steps for ep in recent) / len(recent)
```

### 3. W&B Integration

**Goals**:
- Auto-initialize from scenario config
- Log per-episode metrics
- Log rolling statistics
- Log hyperparameters
- Tag runs for comparison

```python
# v2/logging/wandb_logger.py

import wandb
from typing import Dict, Optional
from v2.metrics.tracker import EpisodeMetrics

class WandbLogger:
    """Weights & Biases logger for experiments."""

    def __init__(self, config: dict, enabled: bool = True):
        self.enabled = enabled

        if not self.enabled:
            return

        wandb_config = config.get('wandb', {})

        # Initialize W&B
        wandb.init(
            project=wandb_config.get('project', 'f110-marl'),
            entity=wandb_config.get('entity'),
            name=config['experiment']['name'],
            config=self._flatten_config(config),
            tags=wandb_config.get('tags', []),
        )

    def log_episode(self, metrics: EpisodeMetrics) -> None:
        """Log single episode metrics."""
        if not self.enabled:
            return

        log_dict = {
            'episode': metrics.episode,
            'outcome': metrics.outcome.value,
            'steps': metrics.steps,
            'reward/total': metrics.total_reward,
            'success/attacker': int(metrics.attacker_success),
            'success/target': int(metrics.target_success),
        }

        # Log reward components
        for component, value in metrics.reward_components.items():
            log_dict[f'reward_components/{component}'] = value

        wandb.log(log_dict, step=metrics.episode)

    def log_rolling_stats(self, episode: int, stats: dict) -> None:
        """Log rolling statistics (e.g., last 100 episodes)."""
        if not self.enabled:
            return

        log_dict = {
            'rolling/success_rate': stats['success_rate'],
            'rolling/avg_reward': stats['avg_reward'],
            'rolling/avg_steps': stats['avg_steps'],
        }

        # Log outcome counts
        for outcome, count in stats['outcome_counts'].items():
            log_dict[f'rolling/outcomes/{outcome.value}'] = count

        wandb.log(log_dict, step=episode)

    def finish(self) -> None:
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()

    @staticmethod
    def _flatten_config(config: dict, prefix: str = '') -> dict:
        """Flatten nested config for W&B."""
        flat = {}
        for key, value in config.items():
            new_key = f'{prefix}{key}' if prefix else key
            if isinstance(value, dict):
                flat.update(WandbLogger._flatten_config(value, f'{new_key}/'))
            else:
                flat[new_key] = value
        return flat
```

### 4. Rich Terminal Output

**Goals**:
- Progress bar for episodes
- Live metrics (last 10-100 episodes)
- Success/failure counts
- Clean, readable output

```python
# v2/logging/console.py

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.live import Live
from typing import Dict
from v2.metrics.tracker import MetricsTracker, EpisodeOutcome

class ConsoleLogger:
    """Rich terminal output for training."""

    def __init__(self, total_episodes: int):
        self.console = Console()
        self.total_episodes = total_episodes
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        self.task = self.progress.add_task(
            "[cyan]Training...",
            total=total_episodes
        )

    def update(self, tracker: MetricsTracker, episode: int) -> None:
        """Update progress and display metrics."""
        self.progress.update(self.task, completed=episode)

        # Every 10 episodes, print summary
        if episode % 10 == 0:
            self._print_summary(tracker, episode)

    def _print_summary(self, tracker: MetricsTracker, episode: int) -> None:
        """Print summary table."""
        recent = tracker.recent(100)

        if not recent:
            return

        # Create table
        table = Table(title=f"Episode {episode}/{self.total_episodes}")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        # Success rates
        table.add_row(
            "Success Rate (last 100)",
            f"{tracker.success_rate():.1%}"
        )

        # Outcome counts
        counts = tracker.outcome_counts()
        table.add_row("Target Crashes", str(counts[EpisodeOutcome.TARGET_CRASH]))
        table.add_row("Self Crashes", str(counts[EpisodeOutcome.SELF_CRASH]))
        table.add_row("Collisions", str(counts[EpisodeOutcome.COLLISION]))
        table.add_row("Timeouts", str(counts[EpisodeOutcome.TIMEOUT]))
        table.add_row("Idle Stops", str(counts[EpisodeOutcome.IDLE_STOP]))
        table.add_row("Target Finishes", str(counts[EpisodeOutcome.TARGET_FINISH]))

        # Average metrics
        table.add_row("Avg Reward", f"{tracker.average_reward():.2f}")
        table.add_row("Avg Steps", f"{tracker.average_steps():.0f}")

        self.console.print(table)
```

### 5. Training Loop Integration

**Enhanced Training Loop**:

```python
# v2/core/training.py (enhanced version)

from typing import Dict
from v2.metrics.tracker import MetricsTracker, EpisodeMetrics
from v2.metrics.outcomes import determine_outcome
from v2.logging.wandb_logger import WandbLogger
from v2.logging.console import ConsoleLogger

class TrainingLoop:
    """Enhanced training loop with metrics and logging."""

    def __init__(
        self,
        env,
        agents: Dict[str, Agent],
        reward_strategy,  # From v2/rewards/
        config: dict,
    ):
        self.env = env
        self.agents = agents
        self.reward_strategy = reward_strategy
        self.config = config

        # Metrics tracking
        self.tracker = MetricsTracker(window=100)

        # Logging
        wandb_enabled = config.get('wandb', {}).get('enabled', False)
        self.wandb_logger = WandbLogger(config, enabled=wandb_enabled)
        self.console_logger = ConsoleLogger(config['experiment']['episodes'])

    def run(self) -> MetricsTracker:
        """Run training loop."""
        max_episodes = self.config['experiment']['episodes']

        for episode in range(max_episodes):
            # Run episode
            episode_reward, reward_components, steps, info = self._run_episode()

            # Determine outcome
            outcome = determine_outcome(info)

            # Create metrics
            metrics = EpisodeMetrics(
                episode=episode,
                outcome=outcome,
                steps=steps,
                total_reward=episode_reward,
                reward_components=reward_components,
            )

            # Track
            self.tracker.add_episode(metrics)

            # Log to W&B
            self.wandb_logger.log_episode(metrics)

            # Every 10 episodes, log rolling stats
            if episode % 10 == 0:
                rolling_stats = {
                    'success_rate': self.tracker.success_rate(),
                    'avg_reward': self.tracker.average_reward(),
                    'avg_steps': self.tracker.average_steps(),
                    'outcome_counts': self.tracker.outcome_counts(),
                }
                self.wandb_logger.log_rolling_stats(episode, rolling_stats)

            # Update console
            self.console_logger.update(self.tracker, episode)

        self.wandb_logger.finish()
        return self.tracker

    def _run_episode(self):
        """Run single episode (existing implementation enhanced)."""
        obs = self.env.reset()
        self.reward_strategy.reset()

        episode_reward = 0.0
        episode_components = {}
        steps = 0
        done = False

        while not done:
            # Agent acts
            actions = {}
            for agent_id, agent in self.agents.items():
                actions[agent_id] = agent.act(obs[agent_id])

            # Environment step
            next_obs, env_rewards, dones, infos = self.env.step(actions)

            # Compute custom reward (only for trainable agent)
            trainable_id = 'car_0'  # Attacker
            step_info = {
                'obs': obs[trainable_id],
                'target_obs': obs.get('car_1'),  # Defender
                'done': dones[trainable_id],
                'truncated': infos[trainable_id].get('truncated', False),
                'info': infos[trainable_id],
                'timestep': self.env.timestep,
            }

            reward, components = self.reward_strategy.compute(step_info)

            # Accumulate components
            for comp, value in components.items():
                episode_components[comp] = episode_components.get(comp, 0.0) + value

            # Agent update (only trainable agents)
            for agent_id, agent in self.agents.items():
                if hasattr(agent, 'update'):  # Skip FTG baseline
                    agent.update(
                        obs[agent_id],
                        actions[agent_id],
                        reward if agent_id == trainable_id else env_rewards[agent_id],
                        next_obs[agent_id],
                        dones[agent_id],
                    )

            episode_reward += reward
            steps += 1
            obs = next_obs
            done = any(dones.values())

        return episode_reward, episode_components, steps, infos[trainable_id]
```

### 6. CLI Interface

**Main Entry Point**:

```python
# v2/run.py

import argparse
from pathlib import Path
from v2.core.scenario import load_scenario
from v2.core.env_factory import EnvironmentFactory
from v2.core.agent_factory import AgentFactory
from v2.core.training import TrainingLoop
from v2.rewards.gaplock import GaplockReward
from v2.rewards.presets import load_preset, merge_config

def main():
    parser = argparse.ArgumentParser(description='V2 Training CLI')
    parser.add_argument('--scenario', type=str, required=True,
                        help='Path to scenario YAML file')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')

    args = parser.parse_args()

    # Load scenario
    config = load_scenario(args.scenario)

    # Apply CLI overrides
    if args.wandb:
        config['wandb']['enabled'] = True
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    if args.render:
        config['rendering']['enabled'] = True

    # Create environment
    env = EnvironmentFactory.create(config['environment'])

    # Create agents
    agents = {}
    for agent_id, agent_config in config['agents'].items():
        agents[agent_id] = AgentFactory.create(
            agent_config['algorithm'],
            agent_config['params']
        )

    # Create reward strategy (for trainable agent)
    trainable_id = 'car_0'  # Attacker
    reward_config = config['agents'][trainable_id]['reward']

    # Load preset and apply overrides
    if 'preset' in reward_config:
        preset = load_preset(reward_config['preset'])
        reward_config = merge_config(preset, reward_config.get('overrides', {}))

    reward_strategy = GaplockReward(reward_config)

    # Create training loop
    training_loop = TrainingLoop(
        env=env,
        agents=agents,
        reward_strategy=reward_strategy,
        config=config,
    )

    # Run!
    print(f"Starting training: {config['experiment']['name']}")
    print(f"Episodes: {config['experiment']['episodes']}")
    print(f"W&B: {'enabled' if config['wandb']['enabled'] else 'disabled'}")
    print()

    tracker = training_loop.run()

    # Print final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total Episodes: {len(tracker.episodes)}")
    print(f"Final Success Rate: {tracker.success_rate():.1%}")
    print(f"Final Avg Reward: {tracker.average_reward():.2f}")
    print("="*60)

if __name__ == '__main__':
    main()
```

### 7. Scenario Parser

```python
# v2/core/scenario.py

import yaml
from pathlib import Path
from typing import Dict, Any

def load_scenario(path: str) -> Dict[str, Any]:
    """
    Load and parse scenario YAML file.

    Returns fully-expanded config dict with presets resolved.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path) as f:
        raw_config = yaml.safe_load(f)

    # Apply defaults
    config = apply_defaults(raw_config)

    # Validate
    validate_config(config)

    return config

def apply_defaults(config: dict) -> dict:
    """Apply default values for missing fields."""
    defaults = {
        'experiment': {
            'name': 'unnamed_experiment',
            'episodes': 1000,
            'seed': 42,
        },
        'wandb': {
            'enabled': False,
            'project': 'f110-marl',
            'entity': None,
            'tags': [],
        },
        'rendering': {
            'enabled': False,
            'record_video': False,
        },
    }

    # Deep merge
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
        elif isinstance(default_value, dict):
            config[key] = {**default_value, **config[key]}

    return config

def validate_config(config: dict) -> None:
    """Validate config structure."""
    required = ['experiment', 'environment', 'agents']

    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # Validate agents
    if not config['agents']:
        raise ValueError("At least one agent must be specified")

    # Validate environment
    if 'map' not in config['environment']:
        raise ValueError("Environment must specify 'map'")
```

---

## Data Flow

### Configuration Flow

```
Scenario YAML
    ↓
load_scenario()
    ↓
Apply defaults
    ↓
Expand presets (algorithm, reward)
    ↓
Validate
    ↓
Config dict
    ↓
├─→ EnvironmentFactory.create(config['environment'])
├─→ AgentFactory.create(config['agents'])
├─→ GaplockReward(config['reward'])
├─→ WandbLogger(config)
└─→ TrainingLoop(env, agents, reward, config)
```

### Episode Flow

```
Episode Start
    ↓
env.reset() → obs
    ↓
reward_strategy.reset()
    ↓
┌─→ agent.act(obs) → action
│       ↓
│   env.step(action) → (next_obs, env_reward, done, info)
│       ↓
│   reward_strategy.compute(step_info) → (reward, components)
│       ↓
│   agent.update(obs, action, reward, next_obs, done)
│       ↓
│   obs = next_obs
│       ↓
└───[repeat until done]
    ↓
determine_outcome(info) → outcome
    ↓
EpisodeMetrics(outcome, steps, reward, components)
    ↓
├─→ MetricsTracker.add_episode(metrics)
├─→ WandbLogger.log_episode(metrics)
└─→ ConsoleLogger.update(tracker, episode)
```

---

## User Workflow

### Typical Research Loop

```bash
# 1. Create/edit scenario
vim scenarios/v2/gaplock_ppo.yaml

# 2. Run training
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb

# 3. Watch in terminal (rich output shows progress)
# Episodes: 342/1000 (34%)
# Success Rate: 12.3%
# Avg Reward: +15.7

# 4. Track in W&B dashboard
# - Compare different algorithms
# - Compare different reward configurations
# - Track hyperparameters

# 5. Iterate
# - Tweak hyperparameters
# - Adjust reward weights
# - Try different algorithms

# 6. Repeat
```

### Example Experiments

**Compare Algorithms**:
```bash
# Run PPO
python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb

# Run TD3
python v2/run.py --scenario scenarios/v2/gaplock_td3.yaml --wandb

# Run SAC
python v2/run.py --scenario scenarios/v2/gaplock_sac.yaml --wandb

# Compare in W&B dashboard (all logged to same project)
```

**Ablation Study** (test reward components):
```bash
# Simple rewards only
python v2/run.py --scenario scenarios/v2/ablation/simple_rewards.yaml --wandb

# Add forcing rewards
python v2/run.py --scenario scenarios/v2/ablation/full_rewards.yaml --wandb

# Compare which works better
```

**Hyperparameter Sweep** (later, Phase 9):
```bash
# Define sweep in W&B
# Launch sweep agents
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

---

## Implementation Phases

### Phase 8: Core Pipeline (Current)

**8.1 Rewards** (6 hrs):
- [ ] Base infrastructure (protocols, composer, presets)
- [ ] Terminal rewards
- [ ] Dense shaping (pressure, distance, heading, speed, penalties)
- [ ] Tests

**8.2 Metrics** (2 hrs):
- [ ] Outcome determination
- [ ] Episode metrics
- [ ] Rolling statistics
- [ ] Tests

**8.3 W&B Logging** (2 hrs):
- [ ] WandbLogger class
- [ ] Per-episode logging
- [ ] Rolling stats logging
- [ ] Hyperparameter tracking

**8.4 Console Output** (2 hrs):
- [ ] ConsoleLogger with rich
- [ ] Progress bar
- [ ] Summary table
- [ ] Pretty formatting

**8.5 Scenario System** (3 hrs):
- [ ] Scenario parser
- [ ] Preset system (algorithm, reward, environment)
- [ ] Config validation
- [ ] Override merging

**8.6 CLI** (2 hrs):
- [ ] v2/run.py main entry point
- [ ] Argument parsing
- [ ] Component wiring
- [ ] End-to-end integration

**8.7 Training Loop Enhancement** (2 hrs):
- [ ] Integrate reward system
- [ ] Integrate metrics tracking
- [ ] Integrate logging
- [ ] Multi-agent handling

**8.8 Example Scenarios** (2 hrs):
- [ ] gaplock_ppo.yaml
- [ ] gaplock_td3.yaml
- [ ] gaplock_sac.yaml
- [ ] Ablation scenarios

**8.9 Testing & Validation** (3 hrs):
- [ ] Integration tests
- [ ] End-to-end test (run full episode)
- [ ] Documentation
- [ ] Examples

**Total: ~24 hours**

### Phase 9: Advanced Features (Future)

- [ ] Rendering integration (watch training)
- [ ] Video recording
- [ ] Model checkpointing
- [ ] W&B sweeps integration
- [ ] Multi-run comparison
- [ ] Curriculum learning
- [ ] Other reward tasks (racing, blocking)

---

## Key Design Principles

### 1. **Simple by Default, Powerful When Needed**

- Default: 10-line scenario with presets
- Advanced: Full custom configuration

### 2. **Composable Components**

- Mix and match: environments, agents, rewards, metrics
- Protocol-based: easy to extend

### 3. **Clear Data Flow**

- Scenario → Config → Setup → Train → Metrics → Logs
- No hidden state, no complex factories

### 4. **Testable**

- Each component has unit tests
- Integration tests for data flow
- Mock environments for testing

### 5. **Observable**

- Rich terminal output (immediate feedback)
- W&B logging (long-term tracking)
- Clear metrics (know what's happening)

---

## Success Criteria

Phase 8 is complete when:

✅ User can write a 10-line scenario YAML
✅ User can run: `python v2/run.py --scenario <file> --wandb`
✅ Training runs with rich terminal output
✅ Metrics logged to W&B
✅ Episode outcomes tracked correctly
✅ Reward components visible
✅ Can compare algorithms easily
✅ All tests pass
✅ Documentation complete

---

## Questions for Discussion

1. **Reward System**:
   - Implement full forcing rewards now, or start with simple (terminal + pressure + distance + heading)?
   - Should we support other reward tasks (racing, blocking) in Phase 8?

2. **Metrics**:
   - Are the 6 outcome types correct and complete?
   - What other metrics should we track?

3. **Configuration**:
   - Is the preset system clear?
   - Should we support Python config files in addition to YAML?

4. **Logging**:
   - Is W&B + rich terminal sufficient?
   - Need any other logging (CSV files, TensorBoard)?

5. **Multi-Agent**:
   - Current design: car_0 = attacker (trainable), car_1 = defender (FTG)
   - Support multiple trainable agents later?

6. **Scope**:
   - Is Phase 8 scope reasonable (~24 hours)?
   - Should we defer anything to Phase 9?

---

## Next Steps

After discussing this architecture:

1. **Finalize design** based on feedback
2. **Start implementation** (probably rewards first)
3. **Test incrementally** (unit tests for each component)
4. **Integrate** (wire everything together)
5. **Validate** (run full training loop end-to-end)
6. **Document** (update README, add examples)

Let me know your thoughts!
