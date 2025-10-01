# TODO

## Multi-Agent Support

- [x] Draft parallel multi-agent experiment profile (new config file, keep current defaults untouched).
- [x] Introduce role-aware lookup helpers / shared replay scaffolds behind unused feature flags.
- [x] Add unit tests + mock fixtures for multi-attacker rosters (no change to live training path).

- [ ] Update experiment configs to roster two attackers plus one defender (n_agents, start pose options, roster entries).
- [ ] Refactor roster/team role handling so multiple attackers can share a role without collisions.
- [ ] Extend reward shaping to always target the defender when several attackers participate.
- [ ] Rework training and evaluation loops for per-attacker metrics, checkpointing, and success criteria.
- [ ] Land coordinated multi-agent trainers (e.g. MADDPG, MATD3) and hook them into the builder registry.
- [ ] Add integration tests covering multi-attacker rollouts and reward bookkeeping.
- [ ] Add per-role observation pipelines (target resolution, shared embeddings, central-state features).
- [ ] Centralise replay/logging utilities for shared critics and parameter-sharing policies.
- [ ] Stand up self-play scheduling and defender co-training (curriculum, cross-play evaluation).
- [ ] Build scenario randomisation hooks (start poses, map subsets, vehicle params) tuned for multi-agent runs.

## Algorithm Baselines

- [ ] Rainbow DQN attacker baseline (distributional targets, noisy nets, prioritized replay refresh).
- [ ] RNN-PPO attacker variant with LSTM core and sequence batching.
- [ ] DRQN / Deep Recurrent Q attacker for discrete throttle-steer grids.
- [ ] MAPPO-style multi-agent PPO head for coordinated attackers.
- [ ] Multi-agent SAC (MASAC/MATD3 hybrid) for continuous joint control.
- [ ] QMIX / VDN-style discrete attackers for cooperative pursuit behaviour.
- [ ] Parameter-sharing PPO/IMPALA baseline to compare against independent learners.
- [ ] Offline baselines (CQL/IQL) seeded from logged self-play rollouts.
- [ ] Behaviour cloning / DAgger attacker from expert defender trajectories as a warm-start.

## Experiment Ops

- [ ] Bundle run artifacts (config, git SHA, metrics) into a single manifest per training job.
- [ ] Define resource envelopes for large sweeps (GPU/CPU concurrency, storage layout, retention policy).
- [ ] Build a submission helper with retry/status tracking and basic monitoring hooks (W&B dashboards, log watchdog).
