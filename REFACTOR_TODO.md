# F110_MARL Refactoring TODO

**Goal:** Reduce code bloat, eliminate redundant abstractions, and simplify the training pipeline architecture.

**Timeline:** 4 weeks (aggressive) / 6 weeks (conservative)
**Target Reduction:** ~3,500 lines of code, 7 → 4 abstraction layers
**Strategy:** Surgical refactor keeping all good RL/physics/domain code

---

## 📋 Project Overview

### Current State
- **Total LOC:** 25,045 lines
- **Abstraction Layers:** 7 (too many!)
- **Key Issues:**
  - Trainer wrapper layer (unnecessary indirection)
  - 3 overlapping factory systems
  - 2,011-line train_runner.py
  - Duplicate train/eval logic

### Target State
- **Total LOC:** ~23,000 lines (-8% reduction)
- **Abstraction Layers:** 4 (clean, simple)
- **Architecture:** Direct agent interface, unified factory, shared rollout logic

---

## 🎯 Phases

- [ ] **Phase 0:** Preparation & Validation (3-5 days)
- [ ] **Phase 1:** Create v2 Structure (2-3 days)
- [ ] **Phase 2:** Eliminate Trainer Wrappers (3-4 days)
- [ ] **Phase 3:** Consolidate Factory Systems (3-4 days)
- [ ] **Phase 4:** Simplify Rollout Logic (4-5 days)
- [ ] **Phase 5:** Config System Cleanup (2-3 days)
- [ ] **Phase 6:** Testing & Validation (3-5 days)
- [ ] **Phase 7:** Migration & Cleanup (2-3 days)

---

## Phase 0: Preparation & Validation
**Estimated Time:** 3-5 days
**Goal:** Ensure safe refactoring with tests and baseline validation

### 0.1 Setup Testing Infrastructure
- [ ] Create `tests/` directory structure
  - [ ] `tests/unit/` - Unit tests for algorithms
  - [ ] `tests/integration/` - End-to-end training tests
  - [ ] `tests/fixtures/` - Test configurations
- [ ] Set up pytest configuration
- [ ] Create test requirements.txt (pytest, pytest-cov, pytest-timeout)

### 0.2 Create Baseline Tests
- [ ] **PPO baseline test**
  - [ ] Simple env + PPO agent
  - [ ] 10 episodes, verify loss decreases
  - [ ] Save checkpoint, load, verify consistency
- [ ] **TD3 baseline test**
  - [ ] Simple env + TD3 agent
  - [ ] 10 episodes with replay buffer
  - [ ] Verify target network updates
- [ ] **DQN baseline test**
  - [ ] Discrete action env + DQN
  - [ ] Verify epsilon decay
  - [ ] Verify target updates
- [ ] **Environment test**
  - [ ] Reset, step, render functionality
  - [ ] Collision detection
  - [ ] Multi-agent coordination

### 0.3 Document Current Behavior
- [ ] Run gaplock_ppo.yaml scenario → save metrics baseline
- [ ] Run gaplock_td3.yaml scenario → save metrics baseline
- [ ] Document checkpoint structure for each algorithm
- [ ] Create `BASELINE_METRICS.md` with expected performance

### 0.4 Create Backup Branch
- [ ] Create `backup/pre-refactor` branch
- [ ] Tag current state as `v1.0-pre-refactor`
- [ ] Document rollback procedure in `ROLLBACK.md`

**Success Criteria:**
✅ All baseline tests passing
✅ Metrics documented
✅ Backup created
✅ Can run at least one full training scenario end-to-end

---

## Phase 1: Create v2 Structure
**Estimated Time:** 2-3 days
**Goal:** Set up parallel v2 codebase with clean architecture

### 1.1 Create Directory Structure
- [ ] Create `v2/` directory at project root
- [ ] Set up package structure:
  ```
  v2/
  ├── __init__.py
  ├── agents/           # RL algorithms (copied from policies/)
  ├── env/              # Environment (copied from envs/)
  ├── physics/          # Physics simulation (copied)
  ├── tasks/            # Reward functions (copied)
  ├── wrappers/         # Obs/action wrappers (copied)
  ├── core/             # NEW: Core training infrastructure
  │   ├── __init__.py
  │   ├── training_loop.py
  │   ├── factory.py
  │   ├── config.py
  │   └── utils.py
  └── scenarios/        # Config files (copied)
  ```

### 1.2 Copy Good Parts (No Changes)
- [ ] Copy `src/f110x/policies/` → `v2/agents/`
  - [ ] PPO (ppo.py, base.py, net.py, rec_ppo.py)
  - [ ] TD3 (td3.py, net.py)
  - [ ] SAC (sac.py, net.py)
  - [ ] DQN (dqn.py, net.py)
  - [ ] Rainbow (r_dqn.py, r_dqn_net.py)
  - [ ] Common (common/, buffers/)
  - [ ] Gap Follow (gap_follow.py)
- [ ] Copy `src/f110x/envs/` → `v2/env/`
- [ ] Copy `src/f110x/physics/` → `v2/physics/`
- [ ] Copy `src/f110x/tasks/` → `v2/tasks/`
- [ ] Copy `src/f110x/wrappers/` → `v2/wrappers/`
- [ ] Copy `src/f110x/render/` → `v2/render/`
- [ ] Copy `scenarios/` → `v2/scenarios/`

### 1.3 Fix Imports in Copied Code
- [ ] Update all `from f110x.` → `from v2.` imports
- [ ] Run: `find v2/ -name "*.py" -exec sed -i 's/from f110x\./from v2\./g' {} \;`
- [ ] Verify no broken imports: `python -m py_compile v2/**/*.py`

### 1.4 Create Initial Core Infrastructure
- [ ] Create `v2/core/config.py` (minimal Pydantic models)
- [ ] Create `v2/core/utils.py` (logging, checkpointing helpers)

**Success Criteria:**
✅ v2/ directory exists with copied code
✅ All imports resolve correctly
✅ No syntax errors in copied files
✅ Can import v2 modules from Python REPL

---

## Phase 2: Eliminate Trainer Wrappers
**Estimated Time:** 3-4 days
**Goal:** Agents implement interface directly, no wrapper layer

### 2.1 Define Agent Protocol
- [ ] Create `v2/core/agent_protocol.py`
  ```python
  class AgentProtocol(Protocol):
      def act(self, obs, deterministic: bool = False) -> Any
      def observe(self, transition: Transition) -> None
      def update(self) -> Optional[Dict[str, float]]
      def save(self, path: str) -> None
      def load(self, path: str) -> None
  ```

### 2.2 Verify Agents Implement Protocol
- [ ] **PPO Agent**
  - [x] Has `act()` method (already exists)
  - [x] Has `update()` method (already exists)
  - [ ] Add `observe()` method (wrapper around `store()`)
  - [ ] Verify save/load methods
- [ ] **TD3 Agent**
  - [x] Has `act()` method
  - [ ] Add `observe()` wrapper around `store_transition()`
  - [x] Has `update()` method
  - [ ] Verify save/load methods
- [ ] **SAC Agent** (same as TD3)
- [ ] **DQN Agent** (same as TD3)
- [ ] **Rainbow DQN** (same as TD3)

### 2.3 Add Transition Dataclass
- [ ] Create `v2/core/transition.py`
  ```python
  @dataclass
  class Transition:
      obs: Any
      action: Any
      reward: float
      next_obs: Any
      terminated: bool
      truncated: bool = False
      info: Optional[Dict] = None
  ```

### 2.4 Update Agents to Use Transition
- [ ] PPO: Add `observe(transition: Transition)` that calls existing `store()`
- [ ] TD3/SAC/DQN: Add `observe(transition: Transition)` wrapper
- [ ] Test each agent individually with protocol

**Success Criteria:**
✅ All agents implement AgentProtocol
✅ Can use agents directly without trainer wrapper
✅ Tests pass with new interface

---

## Phase 3: Consolidate Factory Systems
**Estimated Time:** 3-4 days
**Goal:** Single factory.py replacing 3 systems

### 3.1 Create Unified Factory
- [ ] Create `v2/core/factory.py`
- [ ] Implement `create_agent(algo: str, config: dict) -> AgentProtocol`
  - [ ] PPO factory
  - [ ] Recurrent PPO factory
  - [ ] TD3 factory
  - [ ] SAC factory
  - [ ] DQN factory
  - [ ] Rainbow DQN factory
  - [ ] Gap Follow factory (heuristic)
- [ ] Implement `create_env(config: dict) -> F110ParallelEnv`
- [ ] Implement `create_wrappers(agent, config) -> wrapped_agent`
  - [ ] Observation wrappers
  - [ ] Action wrappers
  - [ ] Reward wrappers

### 3.2 Add Config Parsing
- [ ] Implement `parse_scenario_yaml(path: Path) -> dict`
- [ ] Implement `merge_overrides(base_config, overrides) -> dict`
- [ ] Add validation for required fields

### 3.3 Test Factory
- [ ] Unit test: `create_agent("ppo", {...})` returns PPOAgent
- [ ] Unit test: `create_env({...})` returns valid env
- [ ] Integration test: Create full training setup from YAML

**Success Criteria:**
✅ Single factory.py file (~200-300 lines)
✅ Can create all agents from config
✅ Tests pass for all agent types
✅ YAML configs load correctly

---

## Phase 4: Simplify Rollout Logic
**Estimated Time:** 4-5 days
**Goal:** Replace 2,011-line train_runner with ~300-line training_loop.py

### 4.1 Create Core Rollout Function
- [ ] Create `v2/core/rollout.py`
- [ ] Implement `rollout_episode()`:
  ```python
  def rollout_episode(
      env: F110ParallelEnv,
      agents: Dict[str, AgentProtocol],
      max_steps: int = 1000,
      render: bool = False,
      deterministic: bool = False
  ) -> EpisodeResult
  ```
- [ ] Handle:
  - [ ] Multi-agent step collection
  - [ ] Transition creation and agent observation
  - [ ] Episode termination (done/truncated)
  - [ ] Rendering (if enabled)
  - [ ] Return aggregated metrics

### 4.2 Create Training Loop
- [ ] Create `v2/core/training_loop.py`
- [ ] Implement `train()`:
  ```python
  def train(
      env: F110ParallelEnv,
      agents: Dict[str, AgentProtocol],
      config: TrainingConfig,
      logger: Optional[Logger] = None
  ) -> None
  ```
- [ ] Features:
  - [ ] Episode iteration loop
  - [ ] Agent update calls (on-policy vs off-policy handling)
  - [ ] Metrics logging
  - [ ] Checkpoint saving (every N episodes)
  - [ ] Evaluation episodes (every M episodes)
  - [ ] Early stopping (optional)
  - [ ] Progress bar (tqdm)

### 4.3 Create Evaluation Loop
- [ ] Implement `evaluate()` in `training_loop.py`:
  ```python
  def evaluate(
      env: F110ParallelEnv,
      agents: Dict[str, AgentProtocol],
      num_episodes: int = 10,
      deterministic: bool = True
  ) -> EvaluationMetrics
  ```
- [ ] Calculate:
  - [ ] Mean/std episode return
  - [ ] Success rate
  - [ ] Episode length stats
  - [ ] Per-agent metrics

### 4.4 Add Checkpointing
- [ ] Implement checkpoint utilities in `v2/core/utils.py`:
  - [ ] `save_checkpoint(agents, episode, path)`
  - [ ] `load_checkpoint(agents, path) -> episode_num`
  - [ ] Include: agent weights, optimizer states, episode number, RNG states

### 4.5 Add Logging
- [ ] Simple console logging
- [ ] Optional wandb integration
- [ ] CSV metrics export
- [ ] Tensorboard support (optional)

**Success Criteria:**
✅ Can run full training loop with PPO
✅ Checkpoints save/load correctly
✅ Metrics logged properly
✅ Code < 400 lines total (vs 2,011 before)

---

## Phase 5: Config System Cleanup
**Estimated Time:** 2-3 days
**Goal:** Simpler, Pydantic-based config system

### 5.1 Define Config Models
- [ ] Create `v2/core/config.py` with Pydantic models:
  - [ ] `EnvConfig` - Environment parameters
  - [ ] `AgentConfig` - Agent parameters (per-algo)
  - [ ] `TrainingConfig` - Training loop parameters
  - [ ] `LoggingConfig` - Wandb, CSV, tensorboard settings
  - [ ] `ScenarioConfig` - Top-level config

### 5.2 Implement Config Loading
- [ ] `load_scenario(path: Path) -> ScenarioConfig`
- [ ] Support YAML parsing
- [ ] Validate all fields
- [ ] Provide helpful error messages for missing/invalid fields

### 5.3 Add CLI
- [ ] Create `v2/cli.py` for command-line interface
- [ ] Arguments:
  - [ ] `--scenario` (path to YAML)
  - [ ] `--render` (enable rendering)
  - [ ] `--episodes` (override training episodes)
  - [ ] `--seed` (random seed)
  - [ ] `--checkpoint` (resume from checkpoint)

**Success Criteria:**
✅ YAML configs load and validate
✅ Can override config from CLI
✅ Helpful validation errors
✅ Backward compatible with existing YAML files

---

## Phase 6: Testing & Validation
**Estimated Time:** 3-5 days
**Goal:** Ensure v2 matches v1 behavior

### 6.1 Unit Tests
- [ ] Test each agent with v2 interface
- [ ] Test factory creates all agent types
- [ ] Test config loading/validation
- [ ] Test checkpoint save/load
- [ ] Test rollout_episode() logic

### 6.2 Integration Tests
- [ ] **PPO on gaplock scenario**
  - [ ] Run 100 episodes
  - [ ] Compare metrics to v1 baseline
  - [ ] Verify checkpoint compatibility
- [ ] **TD3 on gaplock scenario**
  - [ ] Run 100 episodes
  - [ ] Verify replay buffer behavior
  - [ ] Compare to v1 baseline
- [ ] **DQN on discrete action task**
  - [ ] Verify epsilon decay
  - [ ] Compare to v1 baseline

### 6.3 Performance Validation
- [ ] **Training speed**: v2 should be ≥ v1 (no slowdown)
- [ ] **Memory usage**: v2 should be ≤ v1
- [ ] **Final performance**: v2 agents should reach same reward as v1

### 6.4 Edge Case Testing
- [ ] Multi-agent scenarios (2, 3, 4+ agents)
- [ ] Checkpoint resume mid-training
- [ ] Config overrides from CLI
- [ ] Rendering enabled/disabled
- [ ] Different maps and scenarios

**Success Criteria:**
✅ All tests passing
✅ Performance matches v1 baselines (within 5%)
✅ No regressions in functionality
✅ Edge cases handled correctly

---

## Phase 7: Migration & Cleanup
**Estimated Time:** 2-3 days
**Goal:** Promote v2 to main, archive v1

### 7.1 Documentation
- [ ] Update README.md with new structure
- [ ] Create MIGRATION_GUIDE.md (v1 → v2 changes)
- [ ] Update architecture diagrams
- [ ] Document new API in docstrings
- [ ] Create examples/ directory with tutorials

### 7.2 Cleanup v2
- [ ] Remove any unused imports
- [ ] Run linter (ruff, black, mypy)
- [ ] Add type hints where missing
- [ ] Ensure consistent code style

### 7.3 Archive v1
- [ ] Move `src/f110x/` → `legacy/v1/`
- [ ] Move `v2/` → `src/f110x/`
- [ ] Update all imports in tests
- [ ] Update setup.py / pyproject.toml

### 7.4 Final Validation
- [ ] Run ALL tests on promoted v2
- [ ] Re-run baseline scenarios
- [ ] Verify wandb logging works
- [ ] Test on fresh clone of repo

### 7.5 Git Cleanup
- [ ] Create PR: v2-refactor → main
- [ ] Tag release: `v2.0.0`
- [ ] Update CHANGELOG.md
- [ ] Close refactoring GitHub issues

**Success Criteria:**
✅ v2 is now main codebase
✅ All documentation updated
✅ Tests passing in CI
✅ Clean git history

---

## 📊 Progress Tracking

### Phase Completion
- [ ] Phase 0: Preparation (0%)
- [ ] Phase 1: v2 Structure (0%)
- [ ] Phase 2: Agent Protocol (0%)
- [ ] Phase 3: Factory (0%)
- [ ] Phase 4: Training Loop (0%)
- [ ] Phase 5: Config (0%)
- [ ] Phase 6: Testing (0%)
- [ ] Phase 7: Migration (0%)

### Metrics
- **Lines Removed:** 0 / ~3,500 target
- **Files Removed:** 0 / ~10 target
- **Tests Added:** 0 / ~30 target
- **Abstraction Layers:** 7 / 4 target

---

## 🚨 Risks & Mitigations

### Risk: Breaking existing checkpoints
**Mitigation:**
- Keep checkpoint format identical
- Add compatibility layer if needed
- Test load/save extensively

### Risk: Performance regression
**Mitigation:**
- Profile before/after
- Keep v1 code for comparison
- Optimize hot paths

### Risk: Scope creep
**Mitigation:**
- Stick to refactor, no new features
- Mark "nice to haves" for later
- Time-box each phase

---

## 📝 Notes

### What NOT to Change
- RL algorithm logic (PPO, TD3, SAC, DQN internals)
- Physics simulation
- Environment core logic
- Reward functions (gaplock, etc.)

### What CAN Change
- Training loop orchestration
- Factory/builder systems
- Config parsing
- Logging/metrics collection
- File organization

---

## 🎯 Success Definition

**The refactor is successful when:**
1. ✅ All baseline tests pass
2. ✅ Training performance matches v1 (±5%)
3. ✅ Code reduced by ≥2,000 lines
4. ✅ Abstraction layers reduced from 7 → 4
5. ✅ New code is easier to understand and modify
6. ✅ No loss of functionality
7. ✅ Documentation is complete

---

## 🔄 Weekly Checkpoints

### Week 1 Goals
- [ ] Complete Phase 0 (Preparation)
- [ ] Complete Phase 1 (v2 Structure)
- [ ] Start Phase 2 (Agent Protocol)

### Week 2 Goals
- [ ] Complete Phase 2 (Agent Protocol)
- [ ] Complete Phase 3 (Factory)
- [ ] Start Phase 4 (Training Loop)

### Week 3 Goals
- [ ] Complete Phase 4 (Training Loop)
- [ ] Complete Phase 5 (Config)
- [ ] Start Phase 6 (Testing)

### Week 4 Goals
- [ ] Complete Phase 6 (Testing)
- [ ] Complete Phase 7 (Migration)
- [ ] Project complete! 🎉

---

**Last Updated:** 2025-12-25
**Status:** Not Started
**Current Phase:** Phase 0 (Preparation)
