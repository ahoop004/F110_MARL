# 2v2 experiment metrics and behavior review TODO

## Goal and working instructions

Prepare the existing MAPPO 2v2 experiments for useful monitoring, fair run
comparisons, and reviewing action sequences as candidate skills for future
hierarchical learning. Implement the tasks below in order; extend existing
logging, recording, and replay code rather than building a separate platform.

- Prioritize working functionality. Do not add extra design documents, broad
  README rewrites, exhaustive tests, or unrelated refactors. Keep this checklist
  current and report the command needed to use each delivered feature.
- Use focused existing checks and short smoke runs. Add a small targeted check
  only when needed for metric correctness, temporal alignment, or data loss.
  Broad regression campaigns and extensive documentation can happen later.
- Inspect current code before implementing: some pieces below already exist.
- Keep W&B optional and local outputs sufficient for the entire workflow.
- Preserve physics, actor observations, rewards, opponent settings, and current
  checkpoint-selection semantics. Surface those semantics; changing the research
  objective is a separate decision. Do not launch full training sweeps here.
- Use existing dependencies where practical. Keep recording optional and avoid
  unnecessary worker-to-parent traffic when it is disabled.
- Initial behavior vocabulary: individual maneuvers with optional multi-agent
  pattern labels. This is a starting convention, not a fixed skill ontology.

The previous physics roadmap is deferred, not completed. Its full text remains
in git history; existing engineering history is in `done.md`. Hardware
calibration, architecture experiments, broad performance studies, and actual
hierarchical-policy training are outside this delivery.

## Starting points and known gaps

- Scenarios: `configs/scenarios/mappo_2v2_base.yaml`,
  `configs/scenarios/mappo_2v2_penalties_base.yaml`, and
  `scenarios/mappo_2v2_{penalties_scratch,first_place}.yaml`.
- Metrics: `src/metrics/racing_eval.py`, `src/metrics/race_penalties.py`.
- Training/evaluation: `src/training/marl_trainer.py`,
  `src/training/parallel_mappo.py`, `src/training/mappo_evaluator.py`.
- Output: `src/training/hooks.py`, `src/loggers/{console,csv_logger,wandb_logger}.py`.
- Recording: `src/env/types.py`, `src/replay/dataset_writer.py`, `run.py`.
- Review: `replay.py` already loads recordings and filters episodes/maps.

Known details to account for:

- The penalties configuration currently uses 400 environments / 100 workers.
  Default per-episode printing and recording every transition will be costly.
- `team_win_rate` can mean one learner beats one opponent. Use explicit
  first-place, sweep, and both-finished labels for headline results.
- All `team_*` checkpoint strategies currently prioritize both-finished rate,
  including `team_first_place`. Print the actual selection ordering.
- Penalties record terminal involvement, not fault or contact partner.
  Wall/teammate/opponent collision classifications are currently deferred.
- `TransitionRecord` carries reward components and `info`, but the dataset writer
  does not persist those dictionaries. Existing MAPPO transition rows are for
  active learners; global state carries other-car context. Do not assume complete
  opponent action streams or race-tail coverage already exist.
- Current datasets distinguish pre-decision global state from post-decision
  lifecycle fields. Preserve that distinction in new fields and replay.
- Base evaluation uses 8 episodes and final testing 20. Treat small evaluations
  as monitoring; expose counts and configurable larger matched evaluations.

## P0 - Consistent race metrics and run comparison data

- [ ] Reuse authoritative environment facts for a common training/evaluation
  metric contract. Keep team results distinct from per-car results and reward.
- [ ] Persist both-learners-finished, team first-place, sweep, and rank score.
  Explicitly distinguish at-least-one-finished from both-finished.
- [ ] Persist each car's finish status/position, collision DNF, boundary DNF,
  timeout, laps, and earned net progress. Preserve finish facts after completion;
  later contact with a stopped finisher must not overwrite its race result.
- [ ] Persist clean finish times and valid lap times with sample counts. Represent
  unavailable measurements as missing, not zero; do not let DNF-heavy policies
  appear faster by reporting only their few finishers without context.
- [ ] Keep own and opponent incident counts and penalty contributions separate.
  Save reward-component totals for diagnosing incentives. Do not infer blame,
  cooperation, or collision partners from terminal flags.
- [ ] Attach stable run/environment/episode IDs, map, seed, spawn configuration,
  policy/update identity, and environment-step count to comparison records.
  Reuse resolved config and physics/action/observation provenance already saved.
- [ ] Save optimizer diagnostics (available KL, entropy, clipping, losses/value
  diagnostics) and throughput. Distinguish race environment steps from learner
  transitions and physics substeps; label the denominator consistently.
- [ ] Make scalar output schemas stable across episodes and late-appearing events;
  ensure CSV does not silently drop metrics absent from the first episode.
- [ ] Provide local per-map and overall evaluation summaries, with numerators,
  denominators, and per-car results. Keep training and evaluation separate.
- [ ] Expose evaluation episode budget and record the evaluation protocol/seed
  set so runs can use matched races. Keep final held-out evaluation separate
  from checkpoint selection; aggregate training seeds separately when available.

Done when a short 2v2 run produces readable episode/per-car records and an
unambiguous evaluation summary without relying on reward as the race result.

## P1 - Compact terminal monitoring

- [ ] Replace noisy default multi-line per-episode reporting for parallel 2v2
  with configurable periodic summaries using a bounded recent-race window.
  State the actual number of completed races in that window.
- [ ] Show environment steps, update number, environment steps/s, both-finished,
  first-place, sweep, any-learner collision-DNF rate, per-learner finish rates,
  and mean training return. Define rates consistently with P0.
- [ ] Print evaluation separately with race count, key results, rank score,
  clean finish time plus finisher count, and checkpoint decision.
- [ ] Print checkpoint-selection priority and active experiment/map configuration
  once at startup. Preserve existing quiet behavior; make detail opt-in.
- [ ] Keep optimizer/reward details in saved logs, with a brief configurable
  periodic diagnostic line rather than every episode's full breakdown.

Done when a short parallel run can be monitored without scrolling through every
race and the terminal numbers agree with its saved records.

## P2 - Selective, synchronized behavior recording

- [ ] Add configurable recording of a reproducible representative sample of
  complete races across training. Record the sampling rule/probability and policy
  version; avoid selecting only successful or interesting races.
- [ ] Add separately identified event clips using bounded per-environment
  pre-event buffers and post-event capture. Include encounters, candidate passes,
  boundary departures, and terminal incidents; cap storage and merge overlaps.
  Persist why a clip was retained so event-heavy samples are not mistaken for
  unbiased behavior frequencies.
- [ ] Preserve synchronized state/lifecycle context for all four cars, learner
  observations, normalized actions, applied physical commands, reward components,
  and relevant structured event facts. Capture opponent commands where available;
  distinguish unavailable data from zero commands.
- [ ] Cover the race through the last relevant car's termination, even after both
  learners stop producing transitions. Prefer a shared race-frame/event stream
  where appropriate instead of duplicating every car's state in extra rows.
- [ ] Save enough context to derive relative longitudinal/lateral gaps and speeds,
  track curvature/width, and lap-aware progress. Separate measured simulator
  events from heuristic behavior labels. Keep unavailable collision pairing
  explicit; do not fabricate it from proximity alone.
- [ ] Make time alignment explicit: simulation time, decision/physics index,
  timestep/action repeat, stable environment/episode/agent IDs, and pre/post-step
  semantics. Do not stitch adjacent records from different workers together.
- [ ] Extend/version the dataset schema as needed; retain old-recording loading.
  Store only useful structured info fields instead of blindly serializing info.
- [ ] Integrate selection before expensive worker transfer/serialization where
  possible. Keep buffers bounded and recorded chunks incremental. Capture a short
  enabled/disabled throughput comparison, not a full benchmark campaign.

Done when sampled races and event clips survive load/replay with aligned cars,
actions, rewards, and terminal facts, including parallel environments.

## P3 - Local review and segment labeling

- [ ] Extend the existing replay path with an episode/clip index filterable by
  run, checkpoint, map, outcome, event, and agent. Read only needed chunks for
  selected clips rather than loading an entire long training run into memory.
- [ ] Add pause, time scrub/seek, playback speed, and clip looping. Use recorded
  timing metadata rather than silently applying replay's legacy timing defaults.
- [ ] Display the race alongside synchronized speed, steering, acceleration
  command, relative-gap, reward-component, and event views. Identify cars/teams
  consistently and distinguish observed state from commands.
- [ ] Let the reviewer set segment start/end, participating cars, behavior label,
  outcome, confidence, and notes. Save editable annotations separately from raw
  trajectories with stable references and label/detector versions.
- [ ] Start with free driving, following, pass attempt, yielding, avoidance, and
  recovery. Keep success/failure/aborted separate from behavior; allow uncertain,
  overlapping, and user-created labels.
- [ ] Support optional multi-car pattern labels, but do not equate observed
  positioning with intentional blocking, assistance, or causal team benefit.
- [ ] Provide a local run-comparison view or command using P0 outputs: per-map
  results, learning curves against environment steps, counts, and seed variation
  where multiple seeds exist. Link interesting results to recorded races/clips.

Done when a user can find a race, inspect an interaction, label a temporal
segment, close the tool, and reopen the same segment and annotation.

## P4 - Group segments for future hierarchical learning

- [ ] Generate candidate segments from configurable event boundaries with
  pre/post context. For passes, use lap-aware progress, active-car filtering,
  sustained order changes, and hysteresis to avoid seam/lapping/DNF false labels.
- [ ] Export per-segment state/action/outcome features, duration, start/end
  conditions, participants, and source references. Include gaps, relative speeds,
  track context, and control changes; do not cluster actions alone.
- [ ] Add a simple similarity/grouping pass over normalized segment features
  using existing tools where practical. Surface representative clips and let
  humans accept, rename, split, or reject groups; retain uncertain examples.
- [ ] Keep representative-race samples and event-selected clips distinguishable
  in exports. Group train/validation splits by source race/run rather than
  scattering overlapping clips across splits.
- [ ] Export candidate skill records with initiation context, observed behavior,
  termination condition, duration, and outcome. Mark fields derived from privileged
  simulator state separately from features available to the actor at deployment.
  These are candidate skills, not trained options or verified team strategies.

Done when reviewed segments can be exported and grouped with traceable source
clips for a later hierarchical-learning experiment.

## Minimal validation and next-agent handoff

- [ ] Run relevant existing metric/lifecycle checks for touched logic and one
  small headless 2v2 smoke run with reduced environments/workers and budget.
  Use a temporary override; do not weaken the actual experiment configs.
- [ ] Verify local metrics, sampled recording, and an annotation save/load cycle.
  Confirm recording disabled still works. Add targeted tests only for uncovered
  consequential failures such as ID mixing, terminal timing, or lost fields.
- [ ] Manually inspect the terminal summary and one replay/label workflow. If
  graphical review is unavailable, state which UI interaction remains unchecked.
- [ ] Mark completed tasks here and report concise launch/review commands plus
  real limitations. No additional documentation package or exhaustive test suite
  is required for this delivery.

Start with P0, then P1. Before long experiment runs, finish P2 so behavior data
is retained. P3 and P4 can then develop against actual recorded races.
