# TODO

## High Priority
- [ ] Extend trainer registry beyond PPO.
  - [ ] TD3 trainer (shared replay buffer + delayed actor updates).
    - [x] Add `policies/buffers/` module with replay buffer reusable by TD3/DQN.
    - [x] Implement TD3 networks (`policies/td3/net.py`) and agent core (`policies/td3/td3.py`).
    - [x] Create `TD3Trainer` adapter implementing select_action/observe/update/save-load.
    - [x] Register TD3 builder in `AGENT_BUILDERS`, read TD3 config block.
    - [x] Expose TD3 overrides in `configs/config.yaml` (lr, tau, noise, policy delay).
    - [x] Smoke test TD3 vs heuristic opponent, log metrics to wandb.
  - [ ] DQN (or Rainbow) trainer path for discrete agents.
    - [x] Define discrete action abstraction (e.g., throttle/steer bins) + config.
    - [x] Reuse/extend replay buffer utilities for discrete training needs.
    - [x] Implement DQN/Rainbow networks and agent core (`policies/dqn/`).
    - [x] Create `DQNTrainer` adapter, register builder, expose config block.
    - [x] Ensure evaluation wrappers support greedy vs epsilon-greedy modes.
    - [x] Run a short DQN smoke test vs heuristic agent, confirm logging.
  - [ ] Shared infrastructure updates.
    - [ ] Update docs/examples to highlight `AgentTeam` + trainer usage (PPO/TD3/DQN).
    - [ ] Write Trainer registry README snippet explaining config wiring.
    - [ ] Add unit tests (or sanity scripts) covering trainer selection + builder outputs.
- [ ] Update CLI/docs for new `--render`, `--episodes`, and trainer workflow.

## Medium Priority
- [ ] Build map_features utility for derived artefacts (centerline, walls, friction).
  - [ ] Generate centerline/waypoint data from MapData.
  - [ ] Extract wall/out-of-bounds masks supporting varied color schemes.
  - [ ] Define friction/track property map generation hooks.
- [ ] Prepare sweeps.yaml variants per algorithm once factories are in place.
  - [ ] Define sweeps for PPO (lr, ent_coef, clip).
  - [ ] Add SAC/TD3 sweeps once trainers exist.

## Completed
- [x] Refactor config handling using ExperimentConfig dataclasses.
- [x] Extract env/agent factory utilities so train/eval share a common build API (see ticket).
- [x] Wrap PPO logic in a generic Trainer interface; plan for plugging in other agents (SAC, TD3).
- [x] Standardize evaluation wrapper with deterministic actions and richer metrics (collision counts, lap stats).
- [x] Integrate structured logging (wandb/TensorBoard) for both training updates and eval runs.
