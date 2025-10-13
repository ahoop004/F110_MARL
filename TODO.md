# TODO

## Code Cleanup · Refactors

- [ ] Break `F110ParallelEnv.__init__` into dedicated configuration helpers (rendering, map IO, vehicle params) for clarity and reuse.
- [x] Let `MapLoader` feed preloaded metadata into `F110ParallelEnv` instead of re-reading map files.
- [ ] Source vehicle parameter defaults from `config_schema` so environment and schema stay in sync.
- [ ] Pull shared collision handling into a single simulator helper to remove duplicate buffer work.
- [ ] Make map loading idempotent by caching the `RaceCar.scan_simulator` map per asset.
- [ ] Centralise centerline renderer updates in a helper shared by `reset` and `set_centerline`.

##  Observation Pipeline & Telemetry

- [ ] Ensure the environment exposes a consistent telemetry bundle (pose, velocity, lap, collision, time) for every agent and document availability.


##  Multi-Agent Training

- [ ] Centralise replay/logging utilities for shared critics and parameter-sharing policies.
- [ ] Stand up self-play scheduling and defender co-training (curriculum, cross-play evaluation).
- [ ] Build scenario randomisation hooks (start poses, map subsets, vehicle params) tuned for multi-agent runs.


## Reward Enhancements

- [ ] Reward mode curriculum that transitions from progress shaping to sparse gaplock bonuses.
  - [ ] Define curriculum schedule template and config knobs.
    - [ ] Document sample `reward_curriculum` blocks and defaults in configs/docs.

- [ ] Centerline projection diagnostics and tooling.
  - [ ] CLI or notebook to visualise projection errors on recorded trajectories.
  - [ ] Sanity-check script for waypoint spacing/continuity per map.
  - [ ] Optional live logging hook for lateral/heading error distributions during training.


## Experiment Operations

- [ ] Bundle run artifacts (config, git SHA, metrics) into a single manifest per training job.
- [ ] Define resource envelopes for large sweeps (GPU/CPU concurrency, storage layout, retention policy).
- [ ] Build a submission helper with retry/status tracking and basic monitoring hooks (W&B dashboards, log watchdog).



## Recently Completed
