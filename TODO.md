# TODO


## High-Priority · Testing & Regression

- [ ] Add unit/integration coverage for `utils/config.load_config`, runner context creation, engine builder, rollout loop, and trainer registry resolution.
- [ ] Add smoke tests covering the new logger façade and runner wiring (train/eval).
- [ ] Run regression passes with baseline PPO and DQN scenarios to validate the extraction path.
- [ ] Capture before/after benchmarks so MARL-focused changes can be validated against current single-agent performance.

## High-Priority · Observation Pipeline & Telemetry

- [x] Introduce a component registry in `ObsWrapper` covering raw sensors (lidar, pose) and derived blocks (relative pose, distances).
- [x] Update wrapper builders/config schema to parse component lists and remove role-based targeting semantics.
- [ ] Ensure the environment exposes a consistent telemetry bundle (pose, velocity, lap, collision, time) for every agent and document availability.
- [ ] Migrate all scenarios/configs to the new component schema and provide conversion helpers for legacy configs.
  - [x] Update first-party scenario manifests (`gaplock_dqn`, `shanghai_dqn_fastlap`) to use component lists.
  - [ ] Ship a conversion helper or CLI to translate legacy wrapper definitions.
- [ ] Add validation helpers/tests that assert component vectors match expected dimensions for each agent.
- [ ] Update heuristics (e.g., follow-gap) and trainers that assume legacy observation layouts.

## High-Priority · Multi-Agent Training

- [x] Update experiment configs to roster two attackers plus one defender (agent counts, start pose options, roster entries).
- [x] Refactor roster handling so multiple attackers can share an identifier without role collisions.
- [x] Extend reward shaping to always target the defender when several attackers participate.
- [ ] Rework training/evaluation loops for per-agent metrics, checkpointing, and success criteria.
  - [x] Log per-agent returns, collision counts, and speeds in `TrainRunner`/`EvalRunner`.
  - [ ] Handle multi-attacker rosters when deriving success/failure signals and idle cutoffs.
  - [ ] Split checkpoint management so every trainable agent can emit "best" artefacts.
- [ ] Land coordinated multi-agent trainers (MADDPG, MATD3) and hook them into the builder registry.
- [ ] Add integration tests covering multi-agent rollouts, reward bookkeeping, and the new observation pipeline.
- [ ] Centralise replay/logging utilities for shared critics and parameter-sharing policies.
- [ ] Stand up self-play scheduling and defender co-training (curriculum, cross-play evaluation).
- [ ] Build scenario randomisation hooks (start poses, map subsets, vehicle params) tuned for multi-agent runs.
- [ ] Stress-test multi-agent DQN training to confirm replay buffer handling and trainer updates under shared observations.
- [ ] Emit per-agent training/eval telemetry (returns, collision counts, checkpoint ownership) for clear progress tracking.
  - [x] Forward per-agent metrics through logger sinks (`train/*`, `eval/*`).
  - [ ] Attach checkpoint ownership metadata and multi-agent progress summaries to logs.
- [ ] Document MARL DQN design options for upcoming experiments:
  - Shared parameter networks with agent-ID conditioning.
  - Team-wide replay buffers with agent-tagged samples.
  - Correlated exploration schedules and shared epsilon decay.
  - Centralised critics with mixers (VDN, QMIX, QTRAN/QPLEX).
  - Role-aware heads and curriculum conditioning (ROMA, IDQL).
  - Distributional/dueling heads for stabilising sparse reward runs.

## Scenario Smoke Tests · Shanghai Fastlap

- [x] Validate Shanghai map assets and the 108-beam lidar profile needed for the scenario.
- [x] Author the `shanghai_dqn_fastlap` scenario YAML with spawn ordering, map selection, and runner wiring.
  - [x] Configure four independent DQN agents using the observation component schema (per-agent lidar, ego pose, relative/target features).
  - [x] Reuse the Shanghai bundle spawn annotations as default start poses and confirm component outputs line up with trainer expectations.
- [ ] Implement time-trial reward and termination hooks (progress bonus, collision retire, all-finished exit).
  - [x] Provide `fastest_lap` reward strategy with lap/leaderboard bonuses.
  - [ ] Add termination wiring for all-finished exit and defender collision retire.
- [ ] Add an integration test that runs a short headless evaluation with four agents to confirm the scenario boots end-to-end.
- [ ] Capture a README/docs snippet describing configuration knobs and expected outputs for the fastlap test.

## Supporting Work · Metrics & Tooling

- [ ] Remove attacker/defender assumptions from roster metadata and reward plumbing so symmetric agents load cleanly.
  - [ ] Expand `AgentTeam.primary_role` to surface multiple members per role (or return lists).
  - [ ] Update runner success logic and reward helpers to tolerate absent role labels.
- [ ] Ensure evaluation/telemetry surfaces per-agent lap completion, lap time, and collision status for fast-lap runs.
  - [x] Export per-agent lap counts and collision summaries from `EvalRunner`.
  - [ ] Track lap durations/best-lap timings and expose them via metrics/logs.
- [ ] Centralise replay/logging utilities so shared DQN cores and per-agent buffers behave consistently during the scenario test.
- [ ] Prune redundant scenario manifest overrides and align on `algo`/`config_ref` usage to avoid conflicts when loading Shanghai configs.
- [ ] Update scenario-loading fixtures and CLI helpers to include the new Shanghai fast-lap scenario.
- [ ] Bundle fast-lap test artifacts (config snapshot, git SHA, key metrics) into a lightweight manifest for reproducibility.

## Reward Enhancements

- [ ] Reward mode curriculum that transitions from progress shaping to sparse gaplock bonuses.
  - [ ] Define curriculum schedule template and config knobs.
    - [ ] Document sample `reward_curriculum` blocks and defaults in configs/docs.
  - [ ] Wire the schedule into the reward wrapper factory and add tests.
    - [x] Pass `reward_curriculum` through `build_runner_context` and `build_reward_wrapper`.
    - [ ] Add unit tests covering curriculum stage transitions and fallbacks.
  - [ ] Document recommended usage in README/configs.
- [ ] Centerline projection diagnostics and tooling.
  - [ ] CLI or notebook to visualise projection errors on recorded trajectories.
  - [ ] Sanity-check script for waypoint spacing/continuity per map.
  - [ ] Optional live logging hook for lateral/heading error distributions during training.
- [ ] Evaluation metrics expansion.
  - [ ] Log per-episode progress, fastest lap time, and centerline error statistics.
  - [ ] Surface metrics in eval summaries and W&B reporting.

## Experiment Templates & Baselines

- [ ] Create dedicated `gaplock_dqn_progress` experiment profile (tuned progress rewards, DQN hyperparams, documentation).
- [ ] Extend evaluation config to compare gaplock vs. progress runs in a single sweep.
- [ ] Build additional baselines: Rainbow DQN, Rate-based discrete head, DRQN/LSTM PPO, multi-agent SAC, QMIX/VDN, parameter-sharing PPO/IMPALA, offline (CQL/IQL), and BC/DAgger warm-starts.

## Experiment Operations

- [ ] Bundle run artifacts (config, git SHA, metrics) into a single manifest per training job.
- [ ] Define resource envelopes for large sweeps (GPU/CPU concurrency, storage layout, retention policy).
- [ ] Build a submission helper with retry/status tracking and basic monitoring hooks (W&B dashboards, log watchdog).
