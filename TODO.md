# TODO Backlog


- [ ] Refactor config handling (Hydra/gin/etc.) so algorithm/env/reward variants are declarative.
- [ ] Extract env/agent factory utilities so train/eval share a common build API.
- [ ] Wrap PPO logic in a generic Trainer interface; plan for plugging in other agents (SAC, TD3).
- [ ] Standardize evaluation wrapper with deterministic actions and richer metrics (collision counts, lap stats).
- [ ] Integrate structured logging (wandb/TensorBoard) for both training updates and eval runs.
- [ ]  Add unit smoke tests for reset_environment and start pose adjustments to catch regressions.
- [ ] Prepare sweeps.yaml variants per algorithm once factories are in place.
