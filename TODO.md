# TODO

## High-Priority · Redundancy & Engine Cleanup

- [x] Retire the legacy simulator duplicate in `src/f110x/physics/base_classes.py` or collapse it into a compatibility shim.
- [x] Share a single `Integrator` enum between `src/f110x/physics/simulaton.py` and `src/f110x/physics/vehicle.py`.
- [x] Extract the repeated render-observer assembly block in `src/f110x/envs/f110ParallelEnv.py` into a shared helper.
- [x] Remove the `src/f110x/trainers` compatibility package after updating any lingering imports to `f110x.trainer`.
- [ ] Sweep for other redundant modules or shims left over from the refactor (e.g., deprecated runner helpers) and delete them once callers are migrated.

## High-Priority · Testing & Regression

- [ ] Add unit/integration coverage for `utils/config.load_config`, runner context creation, engine builder, rollout loop, and trainer registry resolution.
- [ ] Add smoke tests covering the new logger façade and runner wiring (train/eval).
- [ ] Run regression passes with baseline PPO and DQN scenarios to validate the extraction path.
- [ ] Capture before/after benchmarks so MARL-focused changes can be validated against current single-agent performance.

## High-Priority · MARL Enablement

### Reward Task Refactor

- [x] Extract task-specific reward orchestration into `src/f110x/tasks/reward/` (one module per scenario/task).
- [x] Remove global `reward.mode` branching in favor of task registry wiring.
- [x] Update `build_reward_wrapper` / runners to resolve rewards via the task registry, including config migration helpers.
- [x] Document the new task-centric reward pipeline and provide migration notes for existing configs.

### Multi-Agent Roster & Training

- [x] Update experiment configs to roster two attackers plus one defender (agent counts, start pose options, roster entries).
- [x] Refactor roster/team role handling so multiple attackers can share a role without collisions.
- [x] Extend reward shaping to always target the defender when several attackers participate.
- [ ] Rework training and evaluation loops for per-attacker metrics, checkpointing, and success criteria.
- [ ] Land coordinated multi-agent trainers (e.g., MADDPG, MATD3) and hook them into the builder registry.
- [ ] Add integration tests covering multi-attacker rollouts and reward bookkeeping.
- [ ] Add per-role observation pipelines (target resolution, shared embeddings, central-state features).
- [ ] Centralise replay/logging utilities for shared critics and parameter-sharing policies.
- [ ] Stand up self-play scheduling and defender co-training (curriculum, cross-play evaluation).
- [ ] Build scenario randomisation hooks (start poses, map subsets, vehicle params) tuned for multi-agent runs.
- [ ] Stress-test dual-attacker DQN training to confirm replay buffer handling and trainer updates under shared roles.
- [ ] Emit per-role training/eval telemetry (returns, collision counts, checkpoint ownership) so attacker progress is measurable.

### Async / Distributed Infrastructure

- [ ] Wrap `engine/rollout.run_episode()` in a rollout actor interface usable by subprocess or remote workers.
- [ ] Introduce a pluggable `ExperienceSource` (local iterator vs async queue) so trainers stay transport-agnostic.
- [ ] Expose lightweight policy sync hooks (`get_policy_snapshot`, `sync_policy`) on trainers/agents for stale weight refresh.
- [ ] Thread logger and checkpoint broadcasts through the coordinator so metric tags include actor/agent IDs.
- [ ] Build a minimal async regression test (single learner + >1 worker) before scaling to cluster deployments.

### Continuation & Resume

- [ ] Build checkpoint resume + experiment continuation workflow leveraging the new runner context and logger utilities.
- [ ] Simplify CLI entrypoints once legacy flags are retired (e.g., `python main.py train --algo ppo --episodes 1000`).

## Scenario Smoke Tests · Shanghai Fastlap

- [ ] Validate Shanghai map assets and the 108-beam lidar profile needed for the scenario.
- [ ] Author the `shanghai_dqn_fastlap` scenario YAML with spawn ordering, map selection, and runner wiring.
- [ ] Implement time-trial reward and termination hooks (progress bonus, collision retire, all-finished exit).
- [ ] Add an integration test that runs a short headless evaluation with four agents to confirm the scenario boots end-to-end.
- [ ] Capture a README/docs snippet describing configuration knobs and expected outputs for the fastlap test.

## Supporting Work · Metrics & Tooling

- [ ] Remove attacker/defender assumptions from roster metadata and reward plumbing so four symmetric agents load cleanly.
- [ ] Ensure evaluation/telemetry surfaces per-agent lap completion, lap time, and collision status for the fastlap run.
- [ ] Centralise replay/logging utilities so shared DQN cores and per-agent buffers behave consistently during the scenario test.
- [ ] Prune redundant scenario manifest overrides and align on `algo`/`config_ref` usage to avoid conflicts when loading Shanghai configs.
- [ ] Update scenario-loading fixtures and CLI helpers to include the new scenario (`tests/resources/test_env_config.yaml`, docs examples).
- [ ] Bundle fastlap test artifacts (config snapshot, git SHA, key metrics) into a lightweight manifest for reproducibility.

## Secondary · Configuration & Reward

- [ ] Polish scenario manifests (remove redundant overrides, ensure consistent `algo`/`config_ref` usage).
- [ ] Provide scenario-level reward task selectors once the registry lands; strip obsolete `reward.mode` entries.
- [ ] Audit manifests for redundant keys/wrappers (drop duplicate `env.map` aliases, remove unused heuristic pipelines).
- [ ] Update tests & tooling to consume scenarios (`tests/resources/test_env_config.yaml`, sweep configs, docs examples).
- [ ] Document the scenario workflow (adding maps/agents, override keys, migration tips).

### Reward Enhancements

- [ ] Reward mode curriculum that transitions from progress shaping to sparse gaplock bonuses.
  - [ ] Define curriculum schedule template and config knobs.
  - [ ] Wire the schedule into the reward wrapper factory and add tests.
  - [ ] Document recommended usage in README / configs.
- [ ] Centerline projection diagnostics and tooling.
  - [ ] CLI or notebook to visualise centerline projection errors on recorded trajectories.
  - [ ] Sanity check script for waypoint spacing / continuity per map.
  - [ ] Optional live logging hook for lateral/heading error distributions during training.
- [ ] Evaluation metrics expansion.
  - [ ] Log per-episode progress, fastest lap time, and centerline error statistics.
  - [ ] Surface metrics in evaluation summaries and W&B reporting.

## Secondary · Experiment Templates

- [ ] Create dedicated `gaplock_dqn_progress` experiment profile.
  - [ ] Tune progress reward weights and DQN hyperparams.
  - [ ] Add config documentation and comparison notes vs. gaplock baseline.
- [ ] Extend evaluation config to compare gaplock vs. progress runs in a single sweep.
- [ ] Map-based spawn point annotations.
  - [ ] Update MapLoader/env reset to consume `annotations.spawn_points` in order.
  - [ ] Allow experiments to request named spawns or random selection from annotated set.
  - [ ] Persist selection metadata to logs for reproducibility.
  - [ ] Document the YAML schema and configuration options.

## Secondary · Algorithm Baselines

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

## Secondary · Experiment Ops

- [ ] Bundle run artifacts (config, git SHA, metrics) into a single manifest per training job.
- [ ] Define resource envelopes for large sweeps (GPU/CPU concurrency, storage layout, retention policy).
- [ ] Build a submission helper with retry/status tracking and basic monitoring hooks (W&B dashboards, log watchdog).
