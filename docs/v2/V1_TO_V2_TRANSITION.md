# V1 to V2 Transition Plan

## Decision: Keep V1 Observation Configuration

**Observation preset**: `gaplock` (738 dims) - **EXACT v1 match**
- LiDAR: 720 beams
- Ego pose: 4 dims
- Ego velocity: 3 dims
- Target pose: 4 dims
- Target velocity: 3 dims
- Relative pose: 4 dims
- **Total: 738 dims**

**No simplification** - Use proven v1 configuration.

---

## What V1 Has (7 Layers, 5,000+ Lines)

### Layer 1: CLI & Session Management
**Files**: `run.py`, `experiments/cli.py`, `experiments/session.py`
- **Lines**: ~500
- **Complexity**: Argument parsing, session state, multi-run orchestration

### Layer 2: Configuration System
**Files**: `config_models.py`, `config_schema.py`
- **Lines**: ~1,200
- **Complexity**: Pydantic models, nested validation, 100+ config fields

### Layer 3: Factory System (3 separate factories!)
**Files**: `builders.py`, `trainer_registry.py`, `agent_factory.py`
- **Lines**: ~1,586 (builders.py alone!)
- **Complexity**: Recursive builders, context objects, dependency injection

### Layer 4: Wrapper System
**Files**: `wrappers/observation.py`, `wrappers/reward.py`, `wrappers/action.py`
- **Lines**: ~1,000
- **Complexity**: Chained wrappers, dynamic composition

### Layer 5: Trainer Classes
**Files**: `trainer/on_policy.py`, `trainer/off_policy.py`
- **Lines**: ~800
- **Complexity**: Wrapper around agents, trajectory management

### Layer 6: Agent Implementations
**Files**: `agents/ppo/`, `agents/td3/`, etc.
- **Lines**: ~1,500
- **Complexity**: Algorithm implementations (KEEP THESE!)

### Layer 7: Runner & Training Loop
**Files**: `runner/train_runner.py`, `runner/eval_runner.py`
- **Lines**: ~1,000
- **Complexity**: Episode management, logging, checkpointing

**Total V1 Core**: ~5,000 lines (excluding agents)

---

## What V2 Has (3 Layers, 800 Lines)

### Layer 1: Configuration & Entry Point
**Files**: `v2/run.py`, `v2/core/scenario.py`, `v2/core/presets.py`
- **Lines**: ~300 (estimated)
- **What it does**: Load YAML → Parse → Create objects

### Layer 2: Factories
**Files**: `v2/core/agent_factory.py`, `v2/core/env_factory.py`
- **Lines**: ~200
- **What it does**: Simple registry pattern, create agents/envs

### Layer 3: Training Loop
**Files**: `v2/core/training.py`
- **Lines**: ~300
- **What it does**: env.step → agent.act → agent.update → log

**Total V2 Core**: ~800 lines (84% reduction)

**Plus**: Agents are reused from v1 (already ported)

---

## What's Being Removed/Simplified

### ❌ Removed Entirely

1. **Session Management** (`experiments/session.py`)
   - **Why**: Overly complex state management
   - **Replaced with**: Simple config loading

2. **CLI Framework** (`experiments/cli.py`)
   - **Why**: Too many options, complex arg parsing
   - **Replaced with**: Simple argparse in `v2/run.py`

3. **Pydantic Config Models** (`config_models.py`)
   - **Why**: Boilerplate, rigid schemas
   - **Replaced with**: Plain dicts with validation

4. **Complex Builders** (`builders.py` - 1,586 lines!)
   - **Why**: Recursive building, context objects, over-engineered
   - **Replaced with**: Simple factory functions

5. **Trainer Classes** (`trainer/on_policy.py`, `trainer/off_policy.py`)
   - **Why**: Unnecessary wrapper around agents
   - **Replaced with**: Direct agent usage in training loop

6. **Multiple Factory Systems** (3 different registries!)
   - **Why**: Confusing, redundant
   - **Replaced with**: 2 simple factories (AgentFactory, EnvFactory)

7. **Inheritance-based Agent Interface**
   - **Why**: Rigid, forces specific class structure
   - **Replaced with**: Protocol-based (duck typing)

### ✅ Simplified But Kept

1. **Observation Wrappers**
   - **V1**: Complex, but works well
   - **V2**: Keep ObsWrapper as-is (already ported)
   - **Simplification**: Presets instead of verbose config

2. **Reward System**
   - **V1**: 1,655 lines, 106 params
   - **V2**: ~400 lines, ~30 grouped params
   - **Simplification**: Component-based, presets

3. **Configuration**
   - **V1**: 297-line scenario files
   - **V2**: 10-30 line scenarios with presets
   - **Simplification**: Defaults + presets + overrides

4. **Training Loop**
   - **V1**: Complex runner with state management
   - **V2**: Simple loop with direct agent calls
   - **Simplification**: No wrappers, direct interface

### ✅ Kept Unchanged

1. **Agents** (PPO, TD3, SAC, DQN, Rainbow, FTG)
   - Already ported to v2
   - No changes needed

2. **Environment** (F110ParallelEnv)
   - Already works with v2
   - No changes needed

3. **Wrappers** (Observation, Frame stack, Action noise)
   - Already ported to v2
   - No changes needed

---

## What's Being Added (New in V2)

### 🆕 New Features

1. **Metrics Tracking**
   - Episode outcomes (6 types)
   - Rolling statistics
   - Success/failure tracking
   - **V1 equivalent**: Had basic logging, but not structured

2. **W&B Integration**
   - Auto-init from config
   - Per-episode logging
   - Hyperparameter tracking
   - **V1 equivalent**: Manual W&B setup

3. **Rich Terminal Output**
   - Progress bars
   - Live metrics table
   - **V1 equivalent**: Basic print statements

4. **Preset System**
   - Algorithm presets
   - Reward presets
   - Observation presets
   - **V1 equivalent**: None (always verbose config)

5. **Explicit Agent Roles**
   - `role: attacker` / `role: defender`
   - Auto-resolve target agents
   - **V1 equivalent**: Had roles, but less explicit

---

## Complete V2 Architecture Map

```
v2/
├── run.py                          # 🆕 CLI entry point (~150 lines)
│
├── core/
│   ├── agent_factory.py            # ✅ Exists (~100 lines)
│   ├── env_factory.py              # ✅ Exists (~100 lines)
│   ├── training.py                 # ✅ Exists, needs enhancement (~150 → 300 lines)
│   ├── scenario.py                 # 🆕 YAML parser (~150 lines)
│   ├── presets.py                  # 🆕 Preset definitions (~200 lines)
│   └── config.py                   # 🆕 Config utilities (~100 lines)
│
├── agents/                         # ✅ Ported from v1 (reuse as-is)
│   ├── ppo/
│   ├── td3/
│   ├── sac/
│   ├── dqn/
│   ├── rainbow/
│   └── ftg/
│
├── wrappers/                       # ✅ Ported from v1 (reuse as-is)
│   ├── observation.py
│   ├── frame_stack.py
│   ├── action_noise.py
│   └── recorder.py
│
├── rewards/                        # 🆕 NEW (simplified from v1)
│   ├── base.py                     # Protocols (~50 lines)
│   ├── composer.py                 # Composition (~50 lines)
│   ├── presets.py                  # Presets (~100 lines)
│   └── gaplock/
│       ├── gaplock.py              # Main class (~100 lines)
│       ├── terminal.py             # Terminal rewards (~50 lines)
│       ├── pressure.py             # Pressure rewards (~80 lines)
│       ├── distance.py             # Distance shaping (~60 lines)
│       ├── heading.py              # Heading alignment (~40 lines)
│       ├── speed.py                # Speed bonuses (~40 lines)
│       ├── forcing.py              # Forcing rewards (~150 lines)
│       └── penalties.py            # Penalties (~40 lines)
│
├── metrics/                        # 🆕 NEW
│   ├── outcomes.py                 # Outcome enum (~50 lines)
│   ├── tracker.py                  # Metrics tracking (~150 lines)
│   └── aggregator.py               # Statistics (~100 lines)
│
└── logging/                        # 🆕 NEW
    ├── wandb_logger.py             # W&B integration (~150 lines)
    └── console.py                  # Rich output (~150 lines)
```

**Total new code**: ~2,000 lines
**Total v2**: ~2,800 lines (agents + core + rewards + metrics + logging)
**vs V1**: ~7,000 lines (agents + core)

**Reduction**: 60% less code overall (84% reduction in core infrastructure)

---

## Implementation Breakdown

### Phase 8.1: Rewards (10 hrs)

**Goal**: Port v1 gaplock reward to v2 component-based system

#### What to implement:

1. **Base Infrastructure** (2 hrs)
   ```python
   # v2/rewards/base.py
   - RewardComponent protocol
   - RewardStrategy protocol

   # v2/rewards/composer.py
   - ComposedReward class (combines components)

   # v2/rewards/presets.py
   - GAPLOCK_PRESET (738-dim observation config)
   ```

2. **Terminal Rewards** (1 hr)
   ```python
   # v2/rewards/gaplock/terminal.py
   - TerminalReward class
   - Handle 6 outcome types:
     * TARGET_CRASH → +60.0
     * SELF_CRASH → -90.0
     * COLLISION → -90.0
     * TIMEOUT → -20.0
     * IDLE_STOP → -5.0
     * TARGET_FINISH → -20.0
   ```

3. **Dense Shaping** (4 hrs)
   ```python
   # v2/rewards/gaplock/pressure.py
   - PressureReward class
   - Distance threshold, bonus, streak tracking

   # v2/rewards/gaplock/distance.py
   - DistanceReward class
   - Gradient interpolation, near/far rewards

   # v2/rewards/gaplock/heading.py
   - HeadingReward class
   - Alignment reward

   # v2/rewards/gaplock/speed.py
   - SpeedReward class
   - Speed bonus with target

   # v2/rewards/gaplock/penalties.py
   - BehaviorPenalties class
   - Idle, reverse, brake penalties
   ```

4. **Forcing Rewards** (3 hrs)
   ```python
   # v2/rewards/gaplock/forcing.py
   - ForcingReward class
   - Pinch pocket Gaussians
   - Clearance reduction (LiDAR-based)
   - Turn shaping
   - Port from v1 lines 960-1212
   ```

**Deliverables**:
- ✅ All reward components implemented
- ✅ Unit tests for each component
- ✅ Integration test (full episode)
- ✅ Matches v1 reward values (verified)

---

### Phase 8.2: Metrics (2 hrs)

**Goal**: Track episode outcomes and statistics

#### What to implement:

1. **Outcome Definitions** (30 min)
   ```python
   # v2/metrics/outcomes.py
   - EpisodeOutcome enum (6 types)
   - determine_outcome(info) → outcome
   ```

2. **Metrics Tracking** (1 hr)
   ```python
   # v2/metrics/tracker.py
   - EpisodeMetrics dataclass
   - MetricsTracker class
   - Rolling statistics (success rate, avg reward, etc.)
   ```

3. **Tests** (30 min)
   ```python
   # tests/test_metrics.py
   - Test outcome determination
   - Test rolling stats
   ```

**Deliverables**:
- ✅ Outcome tracking working
- ✅ Rolling stats computed correctly
- ✅ Tests passing

---

### Phase 8.3: W&B Integration (2 hrs)

**Goal**: Log metrics to Weights & Biases

#### What to implement:

1. **W&B Logger** (1.5 hrs)
   ```python
   # v2/logging/wandb_logger.py
   - WandbLogger class
   - Auto-init from config
   - log_episode(metrics)
   - log_rolling_stats(episode, stats)
   - Flatten nested config for W&B
   ```

2. **Tests** (30 min)
   ```python
   # tests/test_wandb_logger.py
   - Test config flattening
   - Test logging (mocked)
   ```

**Deliverables**:
- ✅ W&B integration working
- ✅ Metrics logged correctly
- ✅ Tests passing

---

### Phase 8.4: Console Output (2 hrs)

**Goal**: Rich terminal output with progress bars and metrics

#### What to implement:

1. **Console Logger** (1.5 hrs)
   ```python
   # v2/logging/console.py
   - ConsoleLogger class
   - Progress bar (rich.progress)
   - Metrics table (rich.table)
   - Live updates
   ```

2. **Tests** (30 min)
   ```python
   # tests/test_console_logger.py
   - Test table generation
   - Test formatting
   ```

**Deliverables**:
- ✅ Rich progress bar working
- ✅ Metrics table displayed
- ✅ Clean, readable output

---

### Phase 8.5: Observation Integration (5 hrs)

**Goal**: Wire ObsWrapper into v2 pipeline

#### What to implement:

1. **Observation Presets** (1 hr)
   ```python
   # v2/core/presets.py (add to existing file)

   OBSERVATION_PRESETS = {
       'gaplock': {  # Exact v1 match (738 dims)
           'max_scan': 12.0,
           'components': [
               {'type': 'lidar', 'params': {'beams': 720, 'max_range': 12.0, 'normalize': True, 'clip': 1.0}},
               {'type': 'ego_pose', 'params': {'angle_mode': 'sin_cos'}},
               {'type': 'velocity', 'params': {'normalize': 2.0, 'include_speed': True}},
               {'type': 'target_pose', 'params': {'angle_mode': 'sin_cos'}},
               {'type': 'velocity', 'target': 'auto', 'params': {'normalize': 2.0, 'include_speed': True}},
               {'type': 'relative_pose', 'params': {'angle_mode': 'sin_cos'}},
           ],
           'normalize_running': True,
           'normalize_clip': 10.0,
       },
   }
   ```

2. **Config Schema** (1 hr)
   ```python
   # v2/core/config.py
   - add 'observation' field to agent config
   - add 'role' field to agent config
   - Support preset + overrides
   ```

3. **Auto-compute obs_dim** (1 hr)
   ```python
   # v2/core/scenario.py
   def compute_obs_dim(obs_config, env_config):
       # Create dummy env
       dummy_env = create_dummy_env(env_config)
       # Create obs wrapper
       obs_wrapper = create_obs_wrapper(obs_config)
       # Get sample
       dummy_obs = dummy_env.reset()
       sample = obs_wrapper(dummy_obs, 'car_0', 'car_1')
       return sample.shape[0]
   ```

4. **Wire into TrainingLoop** (1.5 hrs)
   ```python
   # v2/core/training.py (enhance existing)
   class TrainingLoop:
       def __init__(self, env, agents, obs_wrappers, ...):
           self.obs_wrappers = obs_wrappers  # Per-agent

       def _run_episode(self):
           raw_obs = env.reset()
           # Process observations
           for agent_id, agent in self.agents.items():
               if agent_id in self.obs_wrappers:
                   processed_obs = self.obs_wrappers[agent_id](raw_obs, agent_id, target_id)
                   action = agent.act(processed_obs)
               else:
                   # FTG baseline, no wrapper
                   action = agent.act(raw_obs[agent_id])
   ```

5. **Tests** (30 min)
   ```python
   # tests/test_observation_integration.py
   - Test preset loading
   - Test obs_dim computation
   - Test wrapper creation
   ```

**Deliverables**:
- ✅ Observation presets defined
- ✅ obs_dim auto-computed
- ✅ ObsWrapper integrated into training loop
- ✅ Running normalization only for trainable agents
- ✅ Tests passing

---

### Phase 8.6: Scenario System (4 hrs)

**Goal**: Parse YAML scenarios and create configuration

#### What to implement:

1. **Scenario Parser** (2 hrs)
   ```python
   # v2/core/scenario.py
   def load_scenario(path: str) -> dict:
       # Load YAML
       # Expand presets (algorithm, reward, observation)
       # Apply overrides
       # Validate
       # Return config dict
   ```

2. **Preset Expansion** (1 hr)
   ```python
   # v2/core/presets.py
   def expand_presets(config: dict) -> dict:
       # Expand algorithm presets
       # Expand reward presets
       # Expand observation presets
       # Merge overrides
   ```

3. **Validation** (30 min)
   ```python
   # v2/core/config.py
   def validate_config(config: dict):
       # Check required fields
       # Validate types
       # Check agent roles
       # Validate observation dims match agent params
   ```

4. **Tests** (30 min)
   ```python
   # tests/test_scenario_parser.py
   - Test YAML loading
   - Test preset expansion
   - Test validation
   ```

**Deliverables**:
- ✅ Scenario parser working
- ✅ Presets expanded correctly
- ✅ Validation catches errors
- ✅ Tests passing

---

### Phase 8.7: CLI (2 hrs)

**Goal**: Command-line interface for running experiments

#### What to implement:

1. **Main CLI** (1.5 hrs)
   ```python
   # v2/run.py
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument('--scenario', required=True)
       parser.add_argument('--render', action='store_true')
       parser.add_argument('--wandb', action='store_true')
       parser.add_argument('--seed', type=int)
       args = parser.parse_args()

       # Load scenario
       config = load_scenario(args.scenario)

       # Apply CLI overrides
       if args.wandb:
           config['wandb']['enabled'] = True
       # ...

       # Create components
       env = EnvironmentFactory.create(config['environment'])
       agents = create_agents(config['agents'])
       obs_wrappers = create_obs_wrappers(config['agents'])
       reward_strategy = create_reward_strategy(config['agents']['car_0']['reward'])

       # Create training loop
       training_loop = TrainingLoop(env, agents, obs_wrappers, reward_strategy, config)

       # Run!
       tracker = training_loop.run()
   ```

2. **Tests** (30 min)
   ```python
   # tests/test_cli.py
   - Test argument parsing
   - Test override merging
   ```

**Deliverables**:
- ✅ CLI working
- ✅ Can run: `python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb`
- ✅ Tests passing

---

### Phase 8.8: Training Loop Enhancement (3 hrs)

**Goal**: Integrate all components into training loop

#### What to implement:

1. **Enhanced TrainingLoop** (2 hrs)
   ```python
   # v2/core/training.py (major enhancement)
   class TrainingLoop:
       def __init__(self, env, agents, obs_wrappers, reward_strategy, config):
           self.env = env
           self.agents = agents
           self.obs_wrappers = obs_wrappers  # Per-agent
           self.reward_strategy = reward_strategy
           self.config = config

           # Metrics
           self.tracker = MetricsTracker(window=100)

           # Logging
           wandb_enabled = config.get('wandb', {}).get('enabled', False)
           self.wandb_logger = WandbLogger(config, enabled=wandb_enabled)
           self.console_logger = ConsoleLogger(config['experiment']['episodes'])

           # Roles
           self.attacker_id = self._find_role(config, 'attacker')
           self.defender_id = self._find_role(config, 'defender')

       def run(self):
           max_episodes = self.config['experiment']['episodes']

           for episode in range(max_episodes):
               # Run episode
               episode_reward, components, steps, info = self._run_episode()

               # Determine outcome
               outcome = determine_outcome(info)

               # Create metrics
               metrics = EpisodeMetrics(episode, outcome, steps, episode_reward, components)

               # Track
               self.tracker.add_episode(metrics)

               # Log
               self.wandb_logger.log_episode(metrics)
               if episode % 10 == 0:
                   self.wandb_logger.log_rolling_stats(episode, self._get_rolling_stats())
               self.console_logger.update(self.tracker, episode)

           self.wandb_logger.finish()
           return self.tracker

       def _run_episode(self):
           raw_obs = self.env.reset()
           self.reward_strategy.reset()

           episode_reward = 0.0
           episode_components = {}
           steps = 0
           done = False

           while not done:
               # Get actions
               actions = {}
               for agent_id, agent in self.agents.items():
                   if agent_id in self.obs_wrappers:
                       # Trainable agent with obs wrapper
                       wrapper = self.obs_wrappers[agent_id]
                       target_id = self.defender_id if agent_id == self.attacker_id else None
                       processed_obs = wrapper(raw_obs, agent_id, target_id)
                       actions[agent_id] = agent.act(processed_obs)
                   else:
                       # Baseline (FTG)
                       actions[agent_id] = agent.act(raw_obs[agent_id])

               # Step environment
               next_obs, env_rewards, dones, infos = self.env.step(actions)

               # Compute custom reward (for attacker only)
               step_info = {
                   'obs': raw_obs[self.attacker_id],
                   'target_obs': raw_obs.get(self.defender_id),
                   'done': dones[self.attacker_id],
                   'truncated': infos[self.attacker_id].get('truncated', False),
                   'info': infos[self.attacker_id],
                   'timestep': self.env.timestep,
               }
               reward, components = self.reward_strategy.compute(step_info)

               # Accumulate components
               for comp, value in components.items():
                   episode_components[comp] = episode_components.get(comp, 0.0) + value

               # Update agents
               for agent_id, agent in self.agents.items():
                   if hasattr(agent, 'update'):  # Trainable
                       agent_reward = reward if agent_id == self.attacker_id else env_rewards[agent_id]
                       if agent_id in self.obs_wrappers:
                           wrapper = self.obs_wrappers[agent_id]
                           target_id = self.defender_id if agent_id == self.attacker_id else None
                           processed_obs = wrapper(raw_obs, agent_id, target_id)
                           processed_next_obs = wrapper(next_obs, agent_id, target_id)
                           agent.update(processed_obs, actions[agent_id], agent_reward, processed_next_obs, dones[agent_id])
                       else:
                           agent.update(raw_obs[agent_id], actions[agent_id], agent_reward, next_obs[agent_id], dones[agent_id])

               episode_reward += reward
               steps += 1
               raw_obs = next_obs
               done = any(dones.values())

           return episode_reward, episode_components, steps, infos[self.attacker_id]
   ```

2. **Tests** (1 hr)
   ```python
   # tests/test_training_loop_integration.py
   - Test full episode
   - Test metrics tracking
   - Test logging
   - Test multi-agent handling
   ```

**Deliverables**:
- ✅ Training loop fully integrated
- ✅ All components working together
- ✅ Tests passing

---

### Phase 8.9: Example Scenarios (2 hrs)

**Goal**: Create example scenario files

#### What to implement:

1. **Gaplock PPO** (30 min)
   ```yaml
   # scenarios/v2/gaplock_ppo.yaml
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

2. **Gaplock TD3** (30 min)
   ```yaml
   # scenarios/v2/gaplock_td3.yaml
   # Same as PPO but algorithm: td3
   ```

3. **Gaplock SAC** (30 min)
   ```yaml
   # scenarios/v2/gaplock_sac.yaml
   # Same as PPO but algorithm: sac
   ```

4. **Ablation Studies** (30 min)
   ```yaml
   # scenarios/v2/ablation/no_forcing.yaml
   # Gaplock without forcing rewards

   # scenarios/v2/ablation/simple_rewards.yaml
   # Gaplock with minimal rewards (terminal + pressure only)
   ```

**Deliverables**:
- ✅ 3 main scenarios (PPO, TD3, SAC)
- ✅ 2 ablation scenarios
- ✅ All scenarios tested

---

### Phase 8.10: Documentation (2 hrs)

**Goal**: Document v2 usage

#### What to create:

1. **Update README.md** (30 min)
   - Add v2 quick start
   - Add example command
   - Link to scenario examples

2. **Create v2/README.md** (1 hr)
   - Architecture overview
   - Component descriptions
   - Usage examples
   - Preset documentation

3. **Create scenarios/v2/README.md** (30 min)
   - Scenario format
   - Available presets
   - Example configurations

**Deliverables**:
- ✅ Documentation complete
- ✅ Examples clear
- ✅ Easy for new users

---

## Total Phase 8 Effort

| Task | Hours |
|------|-------|
| 8.1 Rewards | 10 |
| 8.2 Metrics | 2 |
| 8.3 W&B | 2 |
| 8.4 Console | 2 |
| 8.5 Observations | 5 |
| 8.6 Scenarios | 4 |
| 8.7 CLI | 2 |
| 8.8 Training Loop | 3 |
| 8.9 Examples | 2 |
| 8.10 Docs | 2 |
| **Total** | **34 hours** |

**Realistic estimate**: 1-2 weeks of focused work

---

## V1 vs V2 Comparison Table

| Aspect | V1 | V2 | Change |
|--------|----|----|--------|
| **Core Code** | 5,000 lines | 800 lines | -84% |
| **Scenario File** | 297 lines | 10-30 lines | -90% |
| **Layers** | 7 layers | 3 layers | -57% |
| **Factories** | 3 systems | 2 simple | -33% |
| **Config Models** | Pydantic (rigid) | Plain dicts | Simpler |
| **Agents** | Inheritance-based | Protocol-based | Flexible |
| **Observation** | 738 dims | 738 dims | **Same** |
| **Rewards** | 106 params | ~30 grouped | Simpler |
| **Training Loop** | Wrapped | Direct | Simpler |
| **CLI** | Complex | Simple | Simpler |
| **Metrics** | Basic logging | Structured tracking | Better |
| **W&B** | Manual | Auto | Better |
| **Terminal** | Print | Rich | Better |
| **Presets** | None | Yes | New |
| **Tests** | 30 tests | 69 tests | +230% |

---

## What Stays EXACTLY the Same

1. **Observation**: 738 dims (v1 gaplock configuration)
2. **Agents**: All algorithm implementations
3. **Environment**: F110ParallelEnv
4. **ObsWrapper**: Component-based system
5. **LiDAR**: 720 beams, 12.0m max range
6. **Normalization**: Running normalization for trainable agents

**Goal**: Same observations, same agents, same environment → Same performance

**What changes**: Infrastructure around them (simpler, cleaner, more maintainable)

---

## Migration Path

### For Current V1 Users

1. **Keep using v1** until v2 is complete (Phase 8)
2. **Test v2** on a few scenarios (Phase 9)
3. **Migrate** when confident (Phase 10)
4. **Archive v1** when no longer needed

### For New Users

1. **Start with v2** immediately
2. **Use presets** for quick start
3. **Customize** when needed

---

## Success Criteria

Phase 8 is complete when:

✅ Can run: `python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb`
✅ Training progresses with rich terminal output
✅ Metrics logged to W&B
✅ Episode outcomes tracked correctly
✅ Reward components visible
✅ Observation dims match (738)
✅ Can compare algorithms (PPO vs TD3 vs SAC)
✅ All 69+ tests passing
✅ Documentation complete
✅ Performance matches v1 (same obs → same results)

---

## Next Steps

Ready to start implementation?

**Recommended order**:
1. Start with **Rewards** (core functionality)
2. Then **Metrics** (need for logging)
3. Then **Logging** (W&B + Console)
4. Then **Observations** (wire into pipeline)
5. Then **Scenario parser** (config loading)
6. Then **CLI** (entry point)
7. Then **Training loop** (integration)
8. Finally **Examples + Docs**

**Or**: Work in parallel on independent components (Rewards + Metrics + Logging simultaneously)

Let me know if you want to proceed with implementation!
