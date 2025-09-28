# TODO

## High Priority

- [ ] Trainer registry polish.
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
- [x] Adversarial gap-busting experiment (RL vs Follow-the-Gap).
  - [x] Finalize scenario configs (train/eval) with roster: RL agent + follow-gap opponent.
  - [x] Define adversarial reward shaping (drive opponent into wall, keep ego safe).
    - [x] Add explicit opponent-crash bonus / survival penalty to `RewardWrapper`.
    - [x] Tune wall-distance and forward-progress terms for attacker.
    - [x] Verify reward signals via short scripted rollouts (logging metrics).
  - [x] Implement algorithm-specific action wrappers (continuous normalizer, discrete templating).
  - [x] Extend evaluation metrics (opponent collision flag, time-to-crash, survival time).
    - [x] Log per-episode attacker/defender crash steps and survival horizon.
    - [x] Surface success-rate summaries in wandb/TensorBoard.
  - [x] Run baseline sweeps for PPO/TD3/DQN using the scenario; log to wandb.
    - [x] Prepare sweep configs (hyper grids) for each algorithm.
    - [x] Automate scenario reset + logging tags via env-driven config selection.
