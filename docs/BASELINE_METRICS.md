# Baseline Metrics - Pre-Refactor

**Date:** 2025-12-25
**Version:** v1.0-pre-refactor
**Branch:** backup/pre-refactor

This document records the baseline behavior and metrics of the codebase before refactoring.

---

## 📊 Codebase Metrics

### Code Statistics
- **Total Python Files:** 86
- **Total Lines of Code:** 25,045
- **Abstraction Layers:** 7
- **Factory Systems:** 3 (trainer/registry.py, utils/builders.py, engine/builder.py)

### Largest Files
| File | Lines | Category |
|------|-------|----------|
| `runner/train_runner.py` | 2,011 | Training pipeline |
| `utils/builders.py` | 1,586 | Factory code |
| `envs/f110ParallelEnv.py` | 1,818 | Environment |
| `tasks/reward/gaplock.py` | 1,654 | Reward function |
| `render/rendering.py` | 2,141 | Visualization |
| `runner/eval_runner.py` | 881 | Evaluation |
| `engine/rollout.py` | 798 | Trajectory collection |

---

## ✅ Test Results

### Unit Tests (Phase 0.2)
- **Total Tests:** 24
- **Passed:** 24 (100%)
- **Failed:** 0
- **Test Categories:**
  - PPO Agent: 7 tests
  - TD3 Agent: 9 tests
  - DQN Agent: 8 tests

### Test Coverage
- ✅ Agent creation and initialization
- ✅ Action selection (stochastic and deterministic)
- ✅ Transition storage (buffers and replay)
- ✅ Learning updates (loss computation)
- ✅ Checkpoint save/load
- ✅ Algorithm-specific features (GAE, twin critics, epsilon decay, etc.)

---

## 🔧 Algorithm Implementations

### RL Algorithms Status
| Algorithm | Implementation | Tests | Notes |
|-----------|---------------|-------|-------|
| **PPO** | ✅ Excellent | ✅ All passing | Clipped objective, GAE, tanh squashing |
| **Recurrent PPO** | ✅ Excellent | ⏭️ Not tested yet | LSTM/GRU support |
| **TD3** | ✅ Fixed | ✅ All passing | Twin critics, delayed updates, PER fix applied |
| **SAC** | ✅ Excellent | ⏭️ Not tested yet | Auto-temp tuning, entropy regularization |
| **DQN** | ✅ Fixed | ✅ All passing | Double DQN, reset_optimizers fix applied |
| **Rainbow DQN** | ✅ Excellent | ⏭️ Not tested yet | All 6 Rainbow components |

### Recent Fixes (2025-12-25)
1. **DQN `reset_optimizers()` Bug:**
   - Issue: Undefined `ckpt` variable causing NameError
   - Fix: Removed copy-pasted load() code, proper optimizer reset
   - Commit: 7589c10

2. **TD3 Critic Loss PER Weighting:**
   - Issue: Double-applied importance weights in PER
   - Fix: Apply weights separately to each critic loss
   - Commit: 7589c10

---

## 🎯 Expected Performance

### Training Scenarios
Below are expected metrics for common scenarios (to validate refactor doesn't regress):

#### Gaplock PPO Scenario
**Config:** `scenarios/gaplock_ppo.yaml`
- **Environment:** Line track, 3 agents (1 attacker + 2 defenders)
- **Expected Behavior:**
  - PPO agent should learn to approach defenders
  - Policy loss should decrease over first 100 episodes
  - Value loss should stabilize
  - No NaN values in losses
- **Baseline (approximate):**
  - Episodes to reasonable policy: ~200-300
  - Final average reward: TBD (run to establish)

#### Gaplock TD3 Scenario
**Config:** `scenarios/gaplock_td3.yaml`
- **Environment:** Line track, 3 agents
- **Expected Behavior:**
  - Replay buffer fills to 1M transitions
  - Critic loss decreases
  - Target networks update correctly
  - Exploration noise decays
- **Baseline (approximate):**
  - Episodes to stable Q-values: ~100-200
  - Buffer usage: Gradual fill to capacity

---

## 🏗️ Architecture Documentation

### Current Training Pipeline Flow

```
1. experiments/cli.py
   ↓ Parse args
2. experiments/session.py
   ↓ Load YAML config
3. engine/builder.py → build_runner_context()
   ↓ Create RunnerContext
4. utils/builders.py → build_agents()
   ↓ Instantiate agents
5. trainer/registry.py → create_trainer()
   ↓ Wrap in Trainer
6. trainer/on_policy.py or off_policy.py
   ↓ Delegation layer
7. policies/* (PPOAgent, TD3Agent, etc.)
   ↓ Actual RL implementation
```

### Trainer Wrapper Overhead
- `OnPolicyTrainer`: 76 lines of delegation
- `OffPolicyTrainer`: 153 lines of delegation
- **Total overhead:** ~230 lines for simple pass-through

---

## 🐛 Known Issues (Pre-Refactor)

### Architectural Issues
1. **7 abstraction layers** (too deep)
2. **3 overlapping factory systems**
3. **2,011-line train_runner.py** (needs simplification)
4. **Duplicate train/eval logic** (~40-50% overlap)
5. **TrajectoryBuffer** adds minimal value

### Code Smells
- Mega-file `builders.py` (1,586 lines)
- Unnecessary `Trainer` wrapper layer
- Complex config parsing with multiple models

### No Critical Bugs
- All RL algorithms tested and working
- Checkpointing functional
- Physics simulation stable

---

## 📝 Checkpoint Format

### PPO Checkpoint Structure
```python
{
    "actor": state_dict,
    "critic": state_dict,
    "actor_opt": optimizer_state,
    "critic_opt": optimizer_state,
}
```

### TD3 Checkpoint Structure
```python
{
    "actor": state_dict,
    "actor_target": state_dict,
    "critic1": state_dict,
    "critic2": state_dict,
    "critic_target1": state_dict,
    "critic_target2": state_dict,
    "actor_opt": optimizer_state,
    "critic_opt": optimizer_state,
    "total_it": int,
}
```

### DQN Checkpoint Structure
```python
{
    "q_net": state_dict,
    "target_q_net": state_dict,
    "optimizer": optimizer_state,
    "step_count": int,
    "updates": int,
    "episode_count": int,
    "epsilon_value": float,
    "action_set": ndarray,
    "obs_dim": int,
}
```

---

## ⚠️ Critical Preservation Requirements

**DO NOT CHANGE:**
1. RL algorithm update logic (PPO, TD3, SAC, DQN, Rainbow internals)
2. Physics simulation (vehicle dynamics, collision detection)
3. Environment core logic (F110ParallelEnv step/reset)
4. Reward function parameters (especially gaplock tuning)
5. Checkpoint save/load format (maintain compatibility)

**CAN CHANGE:**
- Training loop orchestration
- Factory/builder systems
- Config parsing
- Trainer wrapper layer
- Logging/metrics collection

---

## 🔄 Validation Checklist

After refactoring, verify:
- [ ] All 24 baseline tests still pass
- [ ] Training speed ≥ v1 baseline (no slowdown)
- [ ] Checkpoints load/save correctly
- [ ] Metrics match baseline ranges
- [ ] No new NaN/Inf values in training
- [ ] Memory usage ≤ v1 baseline
- [ ] Can run `gaplock_ppo.yaml` end-to-end
- [ ] Can run `gaplock_td3.yaml` end-to-end

---

## 📚 Additional Resources

- **Test Suite:** `tests/unit/` and `tests/integration/`
- **Config Examples:** `scenarios/*.yaml`
- **RL Algorithm Review:** See code review from 2025-12-25
- **Refactor Plan:** `REFACTOR_TODO.md`

---

**This baseline will be used to validate that refactoring preserves all functionality and performance.**
