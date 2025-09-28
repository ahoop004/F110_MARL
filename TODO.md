# TODO 


- [x] Refactor config handling (Hydra/gin/etc.) so algorithm/env/reward variants are declarative.
- [ ] Extract env/agent factory utilities so train/eval share a common build API.
- [ ] Wrap PPO logic in a generic Trainer interface; plan for plugging in other agents (SAC, TD3).
- [ ] Standardize evaluation wrapper with deterministic actions and richer metrics (collision counts, lap stats).
- [ ] Integrate structured logging (wandb/TensorBoard) for both training updates and eval runs.
- [ ]  Add unit smoke tests for reset_environment and start pose adjustments to catch regressions.
- [ ] Prepare sweeps.yaml variants per algorithm once factories are in place.
- [ ] [Ticket] Extract shared env/agent factory module (centralize map setup, start poses, policy creation).
  - [ ] Identify map-loading logic duplication in train.py/eval.py.
  - [ ] Design factory interface returning env, PPO agent, heuristic policy, wrappers.
  - [ ] Refactor train/eval/main to use factories without global state.
  - [ ] Add regression smoke tests for factory path.
