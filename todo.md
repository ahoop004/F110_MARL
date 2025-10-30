# Federated MARL TD3 Averaging

## 1. TD3 Policy & Trainer Interfaces
- [x] Expose in-memory weight access helpers in `TD3Agent`
  - [x] Implement `state_dict(self, *, include_optim: bool = True)` returning actor/critic state
  - [x] Implement `load_state_dict(self, state, *, strict: bool = True, include_optim: bool = True)`
  - [x] Update `save`/`load` to reuse the new helpers
- [x] Extend `OffPolicyTrainer`
  - [x] Add passthrough `state_dict()` / `load_state_dict()` methods
  - [x] Guard with `hasattr` checks and raise helpful errors for unsupported agents
  - [ ] Ensure TD3 tests cover the new surface

## 2. Config Schema & Scenario Wiring
- [x] Extend `ExperimentConfig.main` schema to include `federated` block
  - [x] Fields: `enabled`, `interval`, `agents`, `root`, `mode`, `weights`, `timeout`
  - [x] Propagate defaults and validation (positive interval, non-empty agents)
- [x] Update `scenarios/convoy_lock_td3.yaml`
  - [x] Add `main.federated` block mirroring the schema
  - [x] Provide agent list (e.g. `["car_0"]` or both attack/defense)
- [x] Adjust `run.py`
  - [x] Inject `FED_CLIENT_ID` / `FED_TOTAL_CLIENTS` env vars per run
  - [x] Surface optional overrides like `FED_INTERVAL` from config

## 3. Federated Averager Component
- [x] Create `src/f110x/federated/averager.py`
  - [x] Define `FederatedAverager` class taking config + logger
  - [x] Implement `sync(trainer_map, episode_idx)`
    - [x] Serialize local trainer weights to `round_{k}/client_{id}.pt`
    - [x] Wait for peer checkpoints (respect timeout)
    - [x] Average tensors (weighted if configured)
    - [x] Reload averaged weights into participating trainers
    - [x] Return metrics (delta norms, duration)
  - [x] Handle filesystem race safety (atomic writes, partial cleanup)
  - [x] Add utility for weighted averaging (new module or helper function)
- [x] Unit tests for averager
  - [x] Temp dir with synthetic weights; verify averaging and reload
  - [x] Timeout behaviour when peers absent

## 4. Train Runner Integration
- [x] Update `TrainRunner`
  - [x] Instantiate `FederatedAverager` when `main.federated.enabled`
  - [x] After update cycle, trigger `sync` when `(episode + 1) % interval == 0`
  - [x] Merge returned metrics into logger events / W&B
  - [x] Ensure env resets and trainer states remain consistent post-sync
- [x] Optionally, coordinate with `BestReturnTracker`
  - [x] Save averaged checkpoints when new best mean return observed

## 5. Orchestration & Execution Flow
- [x] Document required environment variables (`FED_CLIENT_ID`, `FED_TOTAL_CLIENTS`, optional `FED_ROUND`) in README
- [x] Provide example command for spawning `n` parallel runs with shared averaging dir
- [x] Update `scenarios/convoy_lock_td3_sweep.yaml` or new manifest to point to federated config

## 6. Validation & Tooling
- [x] Write integration script to launch two local runs and confirm convergence
- [x] Capture metrics (delta norms, reward curves) for sanity check
- [x] Update CI/test harness if available to cover new modules

## 7. Stretch Goals / Follow-ups
- [x] Extend averaging support to SAC/DQN once TD3 path solid
- [ ] Add gossip or ring-allreduce mode for large client counts
- [x] Explore optimizer state averaging vs. reinitialisation strategies
