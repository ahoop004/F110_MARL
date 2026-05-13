# F110 MARL — Project Roadmap & Todo

## Goals
- Remove SB3 and Gymnasium entirely — pure PyTorch
- Clean config hierarchy: each layer owns exactly one concern
- Single `run.py` dispatches to the right training loop based on scenario algorithm
- MVP: single-agent PPO working end-to-end
- Then: all other RL algorithms (SAC, TD3, DQN)
- Future: MARL starting with MAPPO — architecture designed for this from day one

---

## Codebase Audit — What Goes Where

### Delete now
- `ros2/` — ROS2 deployment scripts (PPO.py, SAC.py, TD3.py, RainbowDQN.py, gaplock_utils.py)
- `agilex_ros2_ws/` — ROS2 workspace
- `src/agents/buffers/` — empty placeholder
- `src/agents/episodic/` — empty placeholder
- `src/agents/rainbow/` — empty placeholder (Rainbow DQN not in scope)
- `src/tasks/` — empty module, no planned use

### Gitignore (keep locally, never commit)
- `outputs/` — trained model checkpoints
- `warm_starts/` — pre-trained weights
- `sb3_models/` — SB3 format saves (incompatible with new agents)
- `wandb/` — W&B local logs
- `plots/` — generated plots

### Delete after Phase 2 (SB3 removal)
- `src/baselines/` — all 6 sb3_*.py files
- `src/agents/sb3_agents.py`
- `src/replay/sb3_distributed_buffer.py`
- `src/replay/prioritized_replay.py` — replaced by pure PyTorch version
- `run_sb3.py`, `run_sb3_offpolicy.py`, `run_marl_roles.py`, `run_v2.py`, `run_parallel_distributed.py`
- `configs/agents/` — heuristic defaults move to Python
- `configs/env/line2_gaplock.yaml`, `circle_6car.yaml`, `gaplock_multi.yaml`, `centerline_multi.yaml`, `marl_6car.yaml`
- Remove `stable-baselines3`, `sb3-contrib`, `gymnasium`, `pettingzoo` from requirements

### Defer — revisit post-MVP
- `eval.py`, `scenarios/eval/`
- `configs/evaluation/`, `configs/curriculum/`
- `src/curriculum/`, `src/core/evaluator.py`, `src/core/spawn_curriculum.py`

### Keep and update
- `sweeps/` — W&B sweep configs; update `program: run_sb3.py` → `program: run.py` when ready
- `tools/` — `average_phase_checkpoints.py`, `obs_probe.py`, `watch_best_model.py` — keep as-is
- `plots_viewer.ipynb` — keep for analysis

### Keep unchanged (solid, no SB3 dependency)
- `src/physics/` — vehicle dynamics simulation
- `src/env/` — env (modified to remove Gymnasium)
- `src/render/` — visualization
- `src/metrics/` — outcome tracking and aggregation
- `src/loggers/` — console, W&B, CSV loggers
- `src/utils/` — centerline, map_loader, torch_io, spawn_generator, reward_utils
- `src/replay/distributed_buffer.py` — pure Python distributed buffer
- `src/core/scenario.py`, `checkpoint_manager.py`, `run_id.py`, `run_metadata.py`, `protocol.py`
- `src/agents/ftg.py` — FTG heuristic agent
- `src/agents/waypoint.py` — Pure Pursuit, Stanley, Hybrid PP+FTG heuristic agents

---

## Final Source Structure

```
src/
  agents/
    ftg.py              # FTG heuristic policy (keep, no changes)
    waypoint.py         # Pure Pursuit / Stanley / Hybrid PP+FTG (keep, no changes)
    ppo/                # Pure PyTorch PPO (Phase 1)
    sac/                # Pure PyTorch SAC (Phase 2)
    td3/                # Pure PyTorch TD3 (Phase 2)
    dqn/                # Pure PyTorch DQN (Phase 2)
    common/             # Shared network building blocks (MLP, Actor, Critic)

  wrappers/
    observations/
      base.py           # ObservationComponent(ABC)
      composer.py       # ObservationComposer — assembles components from config
      lidar.py          # raw lidar from env (beams/range set by env config)
      ego_state.py      # velocity, pose, angular velocity
      target_state.py   # opponent vehicle state (requires target_id)
      relative_pose.py  # derived: distance + bearing to target
      progress.py       # derived: centerline progress, track position
      prev_action.py    # last action taken (RL agents only)
    rewards/
      base.py           # RewardComponent(ABC)
      composer.py       # RewardComposer — assembles components from config
      centerline.py     # progress along track
      collision.py      # crash penalty
      terminal.py       # success / timeout bonuses
      speed.py          # velocity incentives
      gaplock.py        # adversarial gap-based components
    action.py           # existing action scaling + discrete templating (keep)
    common.py           # existing shared utilities (keep)
    normalize.py        # existing normalizer (keep)

  training/             # NEW
    on_policy_trainer.py
    off_policy_trainer.py
    hooks.py            # ConsoleHook, WandbHook, CheckpointHook

  env/
    f110ParallelEnv.py  # modified: remove ParallelEnv + gymnasium.spaces
    spaces.py           # NEW: SpaceSpec dataclass — replaces gymnasium.spaces
    collision.py
    start_pose_state.py
    state_buffer.py

  core/
    scenario.py         # config loading (extend: maps: key, obs/reward file refs)
    setup.py            # training setup (extend: training_defaults, SpaceSpec)
    config.py           # AgentFactory (extend: register new PyTorch agents)
    checkpoint_manager.py
    run_id.py
    run_metadata.py
    protocol.py
    # DELETE: observations.py, obs_flatten.py → migrated to wrappers/observations/
    # DELETE: enhanced_training.py, training.py → replaced by src/training/
    # DELETE: evaluator.py, spawn_curriculum.py → deferred

  replay/
    distributed_buffer.py   # keep
    replay_buffer.py         # NEW Phase 2: pure PyTorch replay buffer
    # DELETE: sb3_distributed_buffer.py, prioritized_replay.py

  metrics/      # keep unchanged
  loggers/      # keep unchanged
  physics/      # keep unchanged
  render/       # keep unchanged
  utils/        # keep unchanged
```

### Migrations
- `src/wrappers/observation.py` (1020 lines, existing) → decompose into `src/wrappers/observations/` components
- `src/rewards/` (existing component system) → migrate to `src/wrappers/rewards/`
- `src/core/observations.py` + `src/core/obs_flatten.py` → absorbed into `src/wrappers/observations/`

---

## Wrappers: Base Classes and Composers

Both observation and reward systems use the same pattern: base class defines the component interface, individual component files implement it, composer assembles them from config.

```python
# src/wrappers/observations/base.py
class ObservationComponent(ABC):
    @abstractmethod
    def compute(self, raw_obs: dict, info: dict) -> np.ndarray: ...
    @property
    @abstractmethod
    def dim(self) -> int: ...

# src/wrappers/observations/composer.py
class ObservationComposer:
    """Assembles ObservationComponents → flat numpy array."""
    def __init__(self, components: List[ObservationComponent]): ...
    @property
    def obs_dim(self) -> int: ...                                       # sum of component dims
    def wrap(self, raw_obs: dict, info: dict = None) -> np.ndarray: ... # concatenate components
    @classmethod
    def from_config(cls, obs_config: dict, env_config: dict) -> 'ObservationComposer': ...

# src/wrappers/rewards/base.py
class RewardComponent(ABC):
    @abstractmethod
    def compute(self, step_info: dict) -> Dict[str, float]: ...         # named sub-rewards

# src/wrappers/rewards/composer.py
class RewardComposer:
    """Assembles RewardComponents → total scalar + breakdown dict."""
    def reset(self) -> None: ...
    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]: ...
    @classmethod
    def from_config(cls, reward_config: dict) -> 'RewardComposer': ...
```

---

## Config Layer Responsibilities

| Concern | Location |
|---|---|
| Timestep, lidar beams/range, render, centerline/walls autoload | `configs/env/default.yaml` |
| Map name(s), max_steps, action_repeat, map_cycle | Scenario `environment:` block |
| num_agents | Derived from agents block — never set explicitly |
| Spawn positions and spawn point names | `maps/<name>/<name>.yaml` annotations |
| Vehicle physics | `configs/vehicle/default.yaml` — scenario overrides individual values |
| PPO/A2C training hyperparams | `configs/training/on_policy.yaml` |
| SAC/TD3/DQN training hyperparams | `configs/training/off_policy.yaml` |
| Observation components per policy type | `configs/observations/<policy>.yaml` |
| Reward components + weights | `configs/reward/<name>.yaml` |
| W&B metric groups | `configs/wandb.yaml` |
| Evaluation | 🔲 deferred |
| Curriculum | 🔲 deferred |

**Rules:**
- Config files NEVER reference specific agent IDs
- Heuristic agent defaults live in Python `__init__` params — scenario sets overrides inline
- Scenario always wins via deep merge (includes load first, scenario layer overrides)
- `training_defaults:` block merged with each RL agent's `params:` at runtime — agent params win

### Final config directory structure
```
configs/
  env/
    default.yaml              # timestep, lidar_beams, lidar_range, render, centerline flags
  vehicle/
    default.yaml              # physics params — scenario overrides individual values
  training/
    on_policy.yaml            # PPO/A2C defaults + progress_unit: episodes
    off_policy.yaml           # SAC/TD3/DQN defaults + progress_unit: steps
  observations/
    rl_attacker.yaml          # RL agent chasing a target
    rl_racer.yaml             # RL agent racing (no target, uses progress)
    ftg.yaml                  # FTG: raw lidar
    pure_pursuit.yaml         # Pure Pursuit: centerline + ego pose
    hybrid_pp_ftg.yaml        # Hybrid: lidar + centerline + ego state
  reward/
    gaplock_attack.yaml
    centerline_racing.yaml
  wandb.yaml

# DELETE: configs/agents/, configs/evaluation/, configs/curriculum/
# DELETE: configs/env/*_gaplock.yaml, circle_6car.yaml, etc.
```

### Key config file contents

`configs/env/default.yaml`:
```yaml
environment:
  timestep: 0.01
  lidar_beams: 108
  lidar_range: 10.0
  render: false
  centerline_autoload: false
  walls_autoload: false
```

`configs/vehicle/default.yaml`:
```yaml
vehicle_params:
  mu: 1.0489
  C_Sf: 4.718
  C_Sr: 5.4562
  lf: 0.15875
  lr: 0.17145
  h: 0.074
  length: 0.32
  width: 0.225
  m: 3.74
  I: 0.04712
  s_min: -0.46
  s_max: 0.46
  sv_min: -3.2
  sv_max: 3.2
  v_switch: 7.319
  a_max: 9.51
  v_min: -5.0
  v_max: 10.0
```

`configs/training/on_policy.yaml`:
```yaml
training_defaults:
  progress_unit: episodes
  n_steps: 2048
  n_epochs: 10
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  batch_size: 64
  learning_rate: 3.0e-4
  gamma: 0.99
```

`configs/training/off_policy.yaml`:
```yaml
training_defaults:
  progress_unit: steps
  buffer_size: 1000000
  learning_starts: 10000
  train_freq: 1
  gradient_steps: 1
  tau: 0.005
  batch_size: 256
  learning_rate: 3.0e-4
  gamma: 0.99
```

---

## Map Bundle Design

Single `maps:` key — replaces both `map` (single) and `map_bundles` (multi):
```yaml
environment:
  maps: [line2]                    # single map, no cycling
  maps: [Budapest_map, circle_map] # explicit multi-map list
  maps: auto                       # discover all valid maps in maps/ directory
  map_cycle: per_episode           # ignored when maps has 1 entry
  map_pick: random
```
`maps: auto` reuses `_discover_map_bundles()` in `setup.py` (validates centerline + walls exist).

---

## Agent Definition in Scenarios

Algorithm name determines type:
- **RL**: `ppo | a2c | sac | td3 | ddpg | dqn | mappo` → training loop, reward, observation, params
- **Heuristic**: `ftg | hybrid_pp_ftg | pure_pursuit | stanley` → fixed policy, params overrides only

```yaml
agents:
  car_0:
    algorithm: ppo
    role: attacker
    target_id: car_1
    observation: ../configs/observations/rl_attacker.yaml
    reward: ../configs/reward/gaplock_attack.yaml
    params:                             # overrides training_defaults
      learning_rate: 0.0003
      hidden_dims: [256, 256]
    action_constraints:
      prevent_reverse: true
      speed_index: 1

  car_1:
    algorithm: ftg
    role: defender
    observation: ../configs/observations/ftg.yaml
    params:                             # overrides FTG Python defaults
      max_speed: 0.8
      bubble_radius: 3
```

### Policy observation requirements

| Policy | Observation components |
|---|---|
| RL attacker | lidar + ego_state + target_state + relative_pose + prev_action |
| RL racer | lidar + ego_state + progress + prev_action |
| FTG | lidar (normalize: false) |
| Pure Pursuit | centerline + ego_state (include_pose: true) |
| Hybrid PP+FTG | lidar + centerline + ego_state |

---

## Target `scenarios/ppo.yaml`

```yaml
includes:
  - ../configs/env/default.yaml
  - ../configs/vehicle/default.yaml
  - ../configs/training/on_policy.yaml
  - ../configs/wandb.yaml

experiment:
  name: ppo_gaplock
  episodes: 2000
  seed: 42

environment:
  maps: [line2]
  max_steps: 1500
  action_repeat: 2
  vehicle_params:           # override for slow gaplock task
    v_switch: 0.8
    a_max: 2.0
    v_max: 1.0

agents:
  car_0:
    algorithm: ppo
    role: attacker
    target_id: car_1
    observation: ../configs/observations/rl_attacker.yaml
    reward: ../configs/reward/gaplock_attack.yaml
    params:
      learning_rate: 0.0003
      hidden_dims: [256, 256]
    action_constraints:
      prevent_reverse: true
      speed_index: 1

  car_1:
    algorithm: ftg
    role: defender
    observation: ../configs/observations/ftg.yaml
    params:
      max_speed: 0.8
      bubble_radius: 3

wandb:
  enabled: false
  project: marl-f110
  entity: ahoop004-old-dominion-university
  group: algo_comparison
  job_type: ppo
  tags: [ppo, gaplock]
```

---

## Training Loop Design

`run.py` reads algorithm from the RL agent and dispatches:

| Algorithm | Trainer | Progress unit |
|---|---|---|
| `ppo`, `a2c` | `OnPolicyTrainer` | `experiment.episodes` |
| `sac`, `td3`, `ddpg`, `dqn` | `OffPolicyTrainer` | `experiment.total_steps` |
| `mappo` | `MAPPOTrainer` (Phase 3) | `experiment.episodes` |

### On-Policy Loop
```
for episode in 0..n_episodes:
    obs_dict = env.reset()
    obs = obs_composer.wrap(obs_dict[rl_agent_id])
    while not done:
        action, log_prob, value = rl_agent.act(obs)
        heuristic agents act via agent.act(obs_dict[their_id])
        obs_dict, rew_dict, term, trunc, info = env.step(all_actions)
        reward, breakdown = reward_composer.compute(step_info)
        buffer.add(obs, action, reward, log_prob, value, done)
        if buffer.full or done:
            next_val = rl_agent.value(next_obs) if not done else 0.0
            metrics = rl_agent.update(buffer, next_val)
            buffer.clear()
        obs = obs_composer.wrap(obs_dict[rl_agent_id])
    hooks.on_episode_end(episode, reward, info, metrics)
```

### Off-Policy Loop
```
obs_dict = env.reset()
obs = obs_composer.wrap(obs_dict[rl_agent_id])
for step in 0..total_steps:
    action = random() if step < learning_starts else rl_agent.act(obs)
    heuristic agents act via agent.act(obs_dict[their_id])
    next_obs_dict, rew_dict, term, trunc, info = env.step(all_actions)
    reward, breakdown = reward_composer.compute(step_info)
    next_obs = obs_composer.wrap(next_obs_dict[rl_agent_id])
    replay_buffer.add(obs, action, reward, next_obs, done)
    if step >= learning_starts and step % train_freq == 0:
        metrics = rl_agent.update(replay_buffer.sample(batch_size))
    if done:
        hooks.on_episode_end(...)
        obs_dict = env.reset()
    obs = obs_composer.wrap(obs_dict[rl_agent_id])
```

---

## MAPPO Compatibility Constraint

PPO `Actor` class is identical to MAPPO actor — reused directly.
Critic input size is the only difference:
- PPO: `Critic(input_dim=obs_dim)` — local obs
- MAPPO: `Critic(input_dim=n_agents * obs_dim)` — global state

Design `Critic(nn.Module, input_dim)` with explicit arg now — no rearchitecting later.

---

## Phase 1 — MVP: Single-Agent Pure PyTorch PPO

### Completed ✅
- ✅ Delete `ros2/`, `agilex_ros2_ws/`, `src/agents/{buffers,episodic,rainbow}/`, `src/tasks/`
- ✅ `src/env/spaces.py` — `SpaceSpec`, `DictSpaceSpec`
- ✅ `src/env/f110ParallelEnv.py` — `ParallelEnv` + `gymnasium.spaces` removed, uses `SpaceSpec`; `env/__init__.py` made lazy to fix circular import
- ✅ `src/agents/common/networks.py` — `MLP`, `Actor`, `Critic` (MAPPO-compatible `input_dim`)
- ✅ `src/agents/ppo/__init__.py` — `RolloutBuffer`, `PPOAgent` (GAE, clipping, entropy)
- ✅ `src/wrappers/observations/` — `base.py`, `composer.py`, `lidar.py`, `ego_state.py`, `target_state.py`, `relative_pose.py`, `progress.py`, `prev_action.py`
- ✅ `src/wrappers/rewards/` — `base.py`, `composer.py`, `centerline.py`, `collision.py`, `terminal.py`, `speed.py`, `gaplock.py`
- ✅ `src/training/on_policy_trainer.py`, `hooks.py`
- ✅ `run.py` — single entry point, PPO dispatch
- ✅ `configs/env/default.yaml`, `configs/vehicle/default.yaml`
- ✅ `configs/training/on_policy.yaml`, `configs/training/off_policy.yaml`
- ✅ `configs/observations/{rl_attacker,rl_racer,ftg,pure_pursuit,hybrid_pp_ftg}.yaml`
- ✅ `configs/reward/gaplock_attack.yaml`, `centerline_racing.yaml`
- ✅ `src/core/setup.py` — `SpaceSpec`-aware, drop `gymnasium.spaces`
- ✅ Unit verification: `obs_dim=121`, `PPOAgent act+update`, `RewardComposer 4 components`

### Phase 1 Complete ✅
- ✅ `scenarios/ppo.yaml` — refactored to new structure
- ✅ `src/core/config.py` — `ppo` → `PPOAgent` registered
- ✅ `src/core/scenario.py` — `maps:` key accepted in validation
- ✅ `src/core/setup.py` — `maps:` normalized; pure PyTorch RL agents skipped
- ✅ End-to-end verified: `python run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 3`

### Deferred to Phase 2 cleanup
- [ ] Decompose `src/wrappers/observation.py` (1020 lines legacy) into `src/wrappers/observations/` components (currently coexists)
- [ ] Migrate `src/rewards/` → `src/wrappers/rewards/` (currently coexists; new code uses wrappers path)

---

## Phase 2 — Off-Policy Algorithms

- [ ] `configs/training/off_policy.yaml`
- [ ] `src/replay/replay_buffer.py` — pure PyTorch
- [ ] `src/agents/sac/__init__.py`, `td3/__init__.py`, `dqn/__init__.py`
- [ ] `src/training/off_policy_trainer.py`
- [ ] `run.py` — add `OffPolicyTrainer` dispatch
- [ ] `scenarios/sac.yaml`, `td3.yaml`, `dqn.yaml` — refactor to new structure
- [ ] Update `sweeps/*.yaml` — `program: run_sb3.py` → `program: run.py`

---

## Phase 3 — MARL: MAPPO

- [ ] `src/agents/mappo/__init__.py` — shared actor + centralized critic
- [ ] `src/training/marl_trainer.py` — focal agent cycling, multi-agent step
- [ ] `configs/training/mappo.yaml`
- [ ] `scenarios/mappo_defender.yaml`, `scenarios/mappo_attacker.yaml`

---

## Post-MVP: Evaluation & Curriculum
- [ ] Design eval config — spawn point names reference map YAML annotations
- [ ] Design curriculum config — phase advancement gates

---

## Verification Milestones

**Phase 1:**
```bash
python run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 5
grep -r "stable_baselines3\|from gymnasium\|from pettingzoo" run.py src/agents/ppo/ src/training/ src/wrappers/
# → no matches
```

**Phase 2:**
```bash
python run.py --scenario scenarios/sac.yaml --no-wandb
python run.py --scenario scenarios/td3.yaml --no-wandb
```

**Phase 3:**
```bash
python run.py --scenario scenarios/mappo_defender.yaml --no-wandb --episodes 5
```

---

## Current Status
- Branch: `marl-rewiring`
- Config hierarchy cleanup: ✅ reward files no longer embed agent IDs
- marl_attacker car_2 reward: ✅ fixed
- marl_defender car_3 hybrid_pp_ftg params: ✅ fixed
- Grid pose mismatch crash: ✅ fixed (sb3_role_wrapper._build_grid_poses)
- Codebase audit complete: ✅
- Delete ros2/, agilex_ros2_ws/, empty placeholder dirs: ✅
- New config directory structure: ✅ env/default, vehicle/default, training/on_policy+off_policy, observations/*, reward/*
- src/env/spaces.py (SpaceSpec, DictSpaceSpec): ✅
- src/env/f110ParallelEnv.py — Gymnasium + PettingZoo removed: ✅
- src/wrappers/observations/ — base + composer + 6 components: ✅
- src/wrappers/rewards/ — base + composer + 5 components: ✅
- src/agents/common/networks.py — MLP, Actor, Critic: ✅
- src/agents/ppo/ — RolloutBuffer, PPOAgent: ✅
- src/training/on_policy_trainer.py: ✅
- src/training/hooks.py: ✅
- run.py — single entry point with PPO dispatch: ✅
- All Phase 1 components verified (obs_dim=121, PPO act+update): ✅
- End-to-end run.py test with real env: ✅ (3 episodes, reward ~-121 timeout+shaping)
- scenarios/ppo.yaml refactor to new structure: ✅
- **Phase 1 COMPLETE** — next: Phase 2 off-policy algorithms (SAC, TD3, DQN)
