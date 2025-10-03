# TODO

## Configuration Architecture

- [ ] Define modular config layering (shared base, task, policy, experiment overlays).
- [ ] Implement loader/manifest that composes task + policy + overrides into final configs.
- [ ] Split existing `configs/experiments.yaml` into the new structure with representative examples.
- [ ] Document the workflow (how to add a task/policy/experiment) in README.

## Reward Enhancements

- [ ] Reward mode curriculum that transitions from progress shaping to sparse gaplock bonuses.
  - [ ] Define curriculum schedule template and config knobs.
  - [ ] Wire the schedule into reward wrapper factory and add tests.
  - [ ] Document recommended usage in README / configs.
- [ ] Centerline projection diagnostics and tooling.
  - [ ] CLI or notebook to visualise centerline projection errors on recorded trajectories.
  - [ ] Sanity check script for waypoint spacing / continuity per map.
  - [ ] Optional live logging hook for lateral/heading error distributions during training.
- [ ] Evaluation metrics expansion.
  - [ ] Log per-episode progress, fastest lap time, and centerline error statistics.
  - [ ] Surface metrics in evaluation summaries and W&B reporting.

## Experiment Templates

- [ ] Create dedicated `gaplock_dqn_progress` experiment profile.
  - [ ] Tune progress reward weights and DQN hyperparams.
  - [ ] Add config documentation and comparison notes vs. gaplock baseline.
- [ ] Extend evaluation config to compare gaplock vs. progress runs in a single sweep.
- [ ] Map-based spawn point annotations.
  - [ ] Update MapLoader/env reset to consume `annotations.spawn_points` in order.
  - [ ] Allow experiments to request named spawns or random selection from annotated set.
  - [ ] Persist selection metadata to logs for reproducibility.
  - [ ] Document the YAML schema and configuration options.

## Map Asset Packaging

- [x] Restructure map assets into per-map folders bundling YAML, image (png/pgm), centerline CSV, and wall CSV files.
- [x] Update `MapLoader` resolution logic to support nested map directories and normalise relative asset paths.
- [x] Adjust configuration parsing/CLI plumbing so `map_yaml` / `map` entries and builder logic resolve foldered maps gracefully (with backwards-compatible fallbacks).
- [x] Refresh scripts/tests (e.g., `map_validator.py`, fixtures) to cover the new directory structure.
- [x] Document the folder convention and migration steps in README and user guides.

## Render Controls

- [x] Add renderer tracking toggles (follow attacker/defender/free camera).
- [x] Restore and expose zoom controls via keyboard/mouse bindings.
- [x] Implement viewport pan/“move view” shortcuts and document usage in README.

## Algorithm Baselines

- [ ] Rate-based discrete control head for DQN attacker.
  - [ ] Tune steering/brake rates and document defaults per track.
  - [ ] Add regression test covering replay index logging for rate mode.
  - [ ] Benchmark vs absolute action grid (W&B sweep template).
- [ ] Rainbow DQN attacker baseline (distributional targets, noisy nets, prioritized replay refresh).
- [ ] RNN-PPO attacker variant with LSTM core and sequence batching.
- [ ] DRQN / Deep Recurrent Q attacker for discrete throttle-steer grids.
- [ ] MAPPO-style multi-agent PPO head for coordinated attackers.
- [ ] Multi-agent SAC (MASAC/MATD3 hybrid) for continuous joint control.
- [ ] QMIX / VDN-style discrete attackers for cooperative pursuit behaviour.
- [ ] Parameter-sharing PPO/IMPALA baseline to compare against independent learners.
- [ ] Offline baselines (CQL/IQL) seeded from logged self-play rollouts.
- [ ] Behaviour cloning / DAgger attacker from expert defender trajectories as a warm-start.

## Multi-Agent Support

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

## Experiment Ops

- [ ] Bundle run artifacts (config, git SHA, metrics) into a single manifest per training job.
- [ ] Define resource envelopes for large sweeps (GPU/CPU concurrency, storage layout, retention policy).
- [ ] Build a submission helper with retry/status tracking and basic monitoring hooks (W&B dashboards, log watchdog).
