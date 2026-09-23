# 2v2 experiment metrics and behavior review TODO

## Goal and working instructions

Prepare the existing MAPPO 2v2 experiments for useful monitoring, fair run
comparisons, and identifying both individual tactics and combined team tactics
from action sequences as candidate skills for future hierarchical learning.
Implement the tasks below in order; extend existing
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
- Individual tactics and combined tactics are both required discovery targets.
  Individual tactics describe one car's behavior in traffic; combined tactics
  describe how teammates' actions relate over time, including simultaneous and
  sequential actions. Link combined segments to their constituent individual
  segments. The vocabulary should remain editable as new tactics emerge.

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
  Periodic summaries now replace default per-episode printing for parallel
  MAPPO. P2 now supports selective recording before worker transfer.
- `team_win_rate` can mean one learner beats one opponent. Use explicit
  first-place, sweep, and both-finished labels for headline results.
- All `team_*` checkpoint strategies currently prioritize both-finished rate,
  including `team_first_place`. Print the actual selection ordering.
- Penalties record terminal involvement, not fault or contact partner.
  Wall/teammate/opponent collision classifications are currently deferred.
- Legacy version-2 `TransitionRecord` datasets omit reward-component/info
  dictionaries and contain active-learner rows. Version 3 adds shared four-car
  physics frames, reward components, opponent commands, and finite-race tails.
- Legacy datasets distinguish pre-decision global state from post-decision
  lifecycle fields. Version 3 explicitly stores pre/post physics-step state.
- Continuous base training disables lap completion and ends after both learners
  crash; evaluation restores a finite 20-lap race. Do not compare training
  completion rates with finite-race results. P2 streams long sampled episodes
  incrementally and labels budget-cut recordings as partial.
- Base evaluation uses 8 episodes and final testing 20. Treat small evaluations
  as monitoring; expose counts and configurable larger matched evaluations.

## P0 - Consistent race metrics and run comparison data

- [x] Reuse authoritative environment facts for a common training/evaluation
  metric contract. Keep team results distinct from per-car results and reward.
- [x] Persist both-learners-finished, team first-place, sweep, and rank score.
  Explicitly distinguish at-least-one-finished from both-finished.
- [x] Persist each car's finish status/position, collision DNF, boundary DNF,
  timeout, laps, and earned net progress. Preserve finish facts after completion;
  later contact with a stopped finisher must not overwrite its race result.
- [x] Persist clean finish times and valid lap times with sample counts. Represent
  unavailable measurements as missing, not zero; do not let DNF-heavy policies
  appear faster by reporting only their few finishers without context.
- [x] Keep own and opponent incident counts and penalty contributions separate.
  Save reward-component totals for diagnosing incentives. Do not infer blame,
  cooperation, or collision partners from terminal flags.
- [x] Attach stable run/environment/episode IDs, map, seed, spawn configuration,
  policy/update identity, and environment-step count to comparison records.
  Reuse resolved config and physics/action/observation provenance already saved.
- [x] Save optimizer diagnostics (available KL, entropy, clipping, losses/value
  diagnostics) and throughput. Distinguish race environment steps from learner
  transitions and physics substeps; label the denominator consistently.
- [x] Make scalar output schemas stable across episodes and late-appearing events;
  ensure CSV does not silently drop metrics absent from the first episode.
- [x] Provide local per-map and overall evaluation summaries, with numerators,
  denominators, and per-car results. Keep training and evaluation separate.
- [x] Expose evaluation episode budget and record the evaluation protocol/seed
  set so runs can use matched races. Keep final held-out evaluation separate
  from checkpoint selection.
- [x] Aggregate results across training seeds separately when multiple runs exist
  (P3 notebook: explicit experiment groups, equal seed weights, sample SD and counts).

Done when a short 2v2 run produces readable episode/per-car records and an
unambiguous evaluation summary without relying on reward as the race result.

## P1 - Compact terminal monitoring

- [x] Replace noisy default multi-line per-episode reporting for parallel 2v2
  with configurable periodic summaries using a bounded recent-race window.
  State the actual number of completed races in that window.
- [x] Show environment steps, update number, environment steps/s, both-finished,
  first-place, sweep, any-learner collision-DNF rate, per-learner finish rates,
  and mean training return for finite races. Continuous training instead shows
  earned progress, laps, episode duration, and incident counts; finish-based
  results are unavailable. Define rates consistently with P0.
- [x] Print evaluation separately with race count, key results, rank score,
  clean finish time plus finisher count, and checkpoint decision.
- [x] Print checkpoint-selection priority and active experiment/map configuration
  once at startup. Preserve existing quiet behavior; make detail opt-in.
- [x] Keep optimizer/reward details in saved logs, with a brief configurable
  periodic diagnostic line rather than every episode's full breakdown.

Done when a short parallel run can be monitored without scrolling through every
race and the terminal numbers agree with its saved records.

## P2 - Selective, synchronized behavior recording

- [x] Add configurable recording of a reproducible representative sample of
  complete races across training. Record the sampling rule/probability and policy
  version; avoid selecting only successful or interesting races.
- [x] Add separately identified event clips using bounded per-environment
  pre-event buffers and post-event capture. Include encounters, candidate passes,
  boundary departures, and terminal incidents; cap storage and merge overlaps.
  Persist why a clip was retained so event-heavy samples are not mistaken for
  unbiased behavior frequencies.
- [x] Preserve synchronized state/lifecycle context for all four cars, learner
  observations, normalized actions, applied physical commands, reward components,
  and relevant structured event facts. Capture opponent commands where available;
  distinguish unavailable data from zero commands.
- [x] Retain enough shared pre/post context to review both teammates together,
  including sequential interactions where one teammate acts before the other.
  Do not crop all event clips independently around a single car's maneuver.
- [x] Cover the race through the last relevant car's termination, even after both
  learners stop producing transitions. Prefer a shared race-frame/event stream
  where appropriate instead of duplicating every car's state in extra rows.
- [x] Save enough context to derive relative longitudinal/lateral gaps and speeds,
  track curvature/width, and lap-aware progress. Separate measured simulator
  events from heuristic behavior labels. Keep unavailable collision pairing
  explicit; do not fabricate it from proximity alone.
- [x] Make time alignment explicit: simulation time, decision/physics index,
  timestep/action repeat, stable environment/episode/agent IDs, and pre/post-step
  semantics. Do not stitch adjacent records from different workers together.
- [x] Extend/version the dataset schema as needed; retain old-recording loading.
  Store only useful structured info fields instead of blindly serializing info.
- [x] Integrate selection before expensive worker transfer/serialization where
  possible. Keep buffers bounded and recorded chunks incremental. Capture a short
  enabled/disabled throughput comparison, not a full benchmark campaign.

Done when sampled races and event clips survive load/replay with aligned cars,
actions, rewards, and terminal facts, including parallel environments.

## P3 - Local review and segment labeling

Notebook delivery includes run comparison/plots, synchronized clip review,
and editable individual/combined annotations. Reuse Python analysis
helpers; store future annotations separately from notebook and raw trajectories.

- [x] Extend the existing replay path with an episode/clip index filterable by
  run, checkpoint, map, outcome, event, and agent. Read only needed chunks for
  selected clips rather than loading an entire long training run into memory.
- [x] Add pause, time scrub/seek, playback speed, and clip looping. Use recorded
  timing metadata rather than silently applying replay's legacy timing defaults.
- [x] Display the race alongside synchronized speed, steering, acceleration
  command, relative-gap, reward-component, and event views. Identify cars/teams
  consistently and distinguish observed state from commands.
- [x] Let the reviewer set segment start/end, individual/combined scope,
  participating cars, participant roles, target cars, tactic label, outcome,
  confidence, and notes. Save editable annotations separately from raw
  trajectories with stable references and label/detector versions.
- [x] Start individual tactic labels with following/pressure, pass attempt,
  position defense, yielding, avoidance, and recovery; retain free driving as
  context. Keep success/failure/aborted separate from tactic identity; allow
  uncertain, overlapping, and user-created labels.
- [x] Make combined tactics a first-class annotation type. Candidate labels
  include sequential teammate passes, one teammate yielding for the other,
  one teammate contesting an opponent while the other passes, and joint position
  defense. Treat these as review hypotheses, not behaviors guaranteed to emerge.
- [x] Link each combined segment to its individual segments and record temporal
  order/overlap, role changes, and individual plus team outcomes. Permit combined
  intervals to span several individual maneuvers. Distinguish an observed pattern
  from inferred coordination or intent; team benefit requires separate evidence.
- [x] Provide a local run-comparison view or command using P0 outputs: per-map
  results, learning curves against environment steps, counts, and seed variation
  where multiple seeds exist. Link interesting results to recorded races/clips.

Done when a user can inspect and label individual tactics and a combined tactic
with linked constituent segments, then reopen the same clips and annotations.

## Scenario recording integration — before P4

- [x] Extend selective shared-frame recording to serial and parallel two-team
  self-play, with stable episode IDs and both policies' update versions.
- [x] Retain both teams' rewards after either team becomes inactive; record
  clearance/removal and hide removed cars during replay.
- [x] Load self-play training metrics and update diagnostics in the notebook;
  show explicit team identities/outcomes and preserve them in annotations.
- [x] Verify enabled/disabled training consistency, shared worker storage caps,
  terminal tails, notebook execution, and a short recording throughput comparison.
- [x] Add self-play recording to evaluation races (selection and standalone), with paired
  checkpoint identity and evaluation protocol/seed references; adapt self-play
  evaluation outputs for notebook comparisons.
- [x] Add recording allocation across long runs with explicit training-step
  windows and separate reserved frame/byte budgets. Exhausting an early window
  pauses recording until a later window; it cannot consume later capacity.
- [x] Extend shared-frame evaluation recording to fixed-opponent MAPPO and PPO:
  selection and standalone paths, exact checkpoint references, notebook outcome
  joins, and explicit unavailable rewards/incomplete evaluator boundaries.

Training recording is opt-in; the production self-play scenario is unchanged:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_2v2_selfplay.yaml --record-races --no-wandb --output-dir outputs/selfplay_recorded
venv/bin/python replay.py outputs/selfplay_recorded/behavior --list
venv/bin/python -m jupyter lab notebooks/run_review.ipynb
```

Set notebook `RUN_NAMES = ["selfplay_recorded"]`. Self-play plots/table report
both teams separately; its completion/progress win is not a first-place metric.
Use `CLIP_FILTERS` with `team="team_a"` or `"team_b"` for team outcomes. The
existing annotation workflow retains explicit teams and paired policy versions.
Use top-level `recording` settings for sample probability and global frame/byte
caps; `--dataset-dir` can redirect shared-frame output for self-play.
Self-play evaluations now use the existing `--record-races` opt-in too. Selection
evaluations write to `evaluation_behavior` (or `--dataset-dir/evaluation`), with
fixed paired checkpoints under `evaluation_pairs/eval_NNNNNN`. Standalone
evaluation writes to `behavior` (or `--dataset-dir`) and references its loaded
pair. Clips, scalar records, and annotations retain phase, protocol, seed plan,
exact pair path/file hashes, and both policies' update versions. Training and
evaluation have separate frame/byte budgets; evaluation samples every race by
default. Override `evaluation.recording` with the existing recording settings,
or set `enabled: false` there to keep training-only recording. Checkpoint files
are additional disk usage outside frame/byte limits.

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_2v2_selfplay.yaml --eval --eval-protocol final --checkpoint outputs/selfplay_recorded/final_pair --record-races --no-wandb --output-dir outputs/selfplay_final_eval
```

Select that output folder in the notebook; use `CLIP_FILTERS` with
`phase="evaluation"`, `protocol="final"` (or `"selection"`/`"custom"`), and the
exact checkpoint path/hash if desired. Self-play evaluation tables and plots
show each team/map with counts, seeds, and separate checkpoint/protocol groups.
They remain separate from the original fixed-opponent comparison/seed tables.
Older logs retain missing checkpoint/protocol facts rather than inventing them.
Fixed-opponent/PPO evaluation recording is delivered in the handoff below.
Windowed allocation is also available below.

Validation: 44 focused tests passed across self-play trainers, recording,
analysis, and playback (including four new self-play recording checks). Serial
and spawned parallel enabled/disabled runs produced bit-identical actor and
critic weights for both teams. Targeted checks cover delayed rewards after a
team becomes inactive, clearance removal in playback, shared global frame caps,
policy-version references, and annotation save/load.

A separate 128-step CPU CLI comparison (2 environments / 1 worker, 100% sample
probability, small networks) retained 128 unique frames, four complete sampled
races, and four event clips; both policies again matched exactly. Collection
plus update throughput was 33.61 steps/s disabled and 28.97 enabled (about 14%
lower). This short noisy run is not a 400-environment benchmark or an estimate
of default sampling overhead. Smoke overrides were removed; production budgets
were not changed. Artifacts: `outputs/selfplay_recording_{disabled,enabled}`.

The notebook executed against the self-play recording, exported CSV/PNG/PDF
artifacts, and the team plots/playback image were visually inspected. The
executed example and validation measurements are under
`outputs/analysis/selfplay_recording_review`. Playback removal and annotation
controls passed focused checks; browser/IDE interaction was not repeated for
this increment (the earlier P3 browser check is documented below).

Evaluation increment validation: 38 focused trainer/recording/analysis/playback
checks passed, including serial and spawned parallel training with recorded
selection evaluations and a standalone final evaluation. Both teams' actor and
critic weights and evaluation outcomes matched recording-disabled runs exactly.
Checks cover independent recording caps, unique phase/round/race IDs, checkpoint
file hashes, policy versions, protocol seeds, clip filters, and annotation source
references. Custom episode budgets are labeled `custom`, not fixed selection.

CLI artifacts: `outputs/selfplay_evaluation_recording` (32 training steps,
32 evaluation frames across two paired snapshots) and
`outputs/selfplay_final_recording` (16 held-out evaluation frames). The full
notebook executed against both folders and exported per-team/per-map tables and
separate selection/final figures under `outputs/analysis/selfplay_evaluation_review`.
Smoke overrides were removed; production scenario budgets were unchanged.
Browser/IDE interaction was not repeated for this increment.

### Recording allocation across long runs

- Optional `recording.windows` is an ordered list of non-overlapping
  `{start_step, end_step, max_frames, max_bytes}` ranges. Ends are exclusive;
  reserved budgets must sum to no more than the existing dataset caps. Limits
  are enforced globally across workers, with bounded capture/transfer before the
  next barrier reports exhausted windows. Unused allocations are not borrowed.
- Serial progress counts completed joint decisions; parallel progress is the
  aggregate count at the last completed collection barrier, held fixed during
  that collection round. Recorded fields identify the clock explicitly. Windows
  shorter than the collection round can be skipped; choose ranges accordingly.
- Both fixed-opponent MAPPO and two-team self-play support serial/parallel
  windowed recording. Capture stops outside configured ranges or at a window
  cap and resumes in later ranges, even during a continuous race. Training,
  rewards, observations, policy updates, and race reset boundaries are preserved.
- Clips close at boundaries; event history is discarded across gaps. Sampling
  still uses the deterministic episode hash, with no training RNG consumption.
  A sample resumed mid-race is `representative_segment` with `complete=false`,
  not a complete race. Event-clip limits restart per episode/window. Later
  segments have unique IDs and policy versions at their actual capture start.
- Dataset metadata exposes configured windows and actual accepted frames/bytes,
  including unused windows. Notebook `recording_windows` tables/CSV exports,
  `CLIP_FILTERS["window_index"]`, the playback clock, and annotation sources
  expose the selected window. Existing recordings and `windows: []` still work.
- Evaluation keeps its independent, unwindowed budget by default. Explicit
  `evaluation.recording.windows` uses checkpoint training steps; a checkpoint
  with unknown steps cannot be assigned a window. Selection snapshot files are
  saved only when the corresponding evaluation recording allocation is available.

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_2v2_selfplay_recorded.yaml --no-wandb --output-dir outputs/selfplay_windowed
```

The new optional scenario includes `configs/recording/mappo_windowed.yaml`,
splitting the unchanged 100,000-frame / 2 GB precompression budget into three
stages of the existing 120M-step task. Original scenarios remain opt-in via their
existing flags. The preset reserves three stages, not continuous coverage or
complete races; event-heavy recording at 400 environments can still use each
stage's allocation quickly. Adjust windows, caps, sampling, or event capture for
the intended analysis. Fixed-opponent MAPPO scenarios can include the same file.

Windowed recording validation: 38 focused checks passed across the recorder,
self-play, fixed-opponent MAPPO, analysis, and playback. Checks cover frame/byte
reservations, gaps, event-buffer boundaries, mid-race resumption, shared caps
across spawned workers, config validation, and legacy unwindowed recording.
Both serial and parallel self-play produced bit-identical actor/critic weights
with recording disabled/enabled. No full training sweep was launched.

A separate 64-step CLI comparison retained exactly six training frames in each
of three windows, no frames in the intervening gap, and six evaluation frames
under the independent unwindowed default. Both teams' final weights matched
exactly. Artifacts: `outputs/window_recording_{disabled,enabled}`. The notebook
executed on this run with `window_index=2` selected and exported budget/clip
CSV tables and plots to `outputs/analysis/window_recording_review`; each late
sample reopened as a bounded partial segment. The temporary smoke scenario was
removed. Browser/IDE interaction was not repeated for this increment.

### Fixed-opponent MAPPO and PPO evaluation handoff

- Standalone evaluation now accepts `--record-races`; shared frames go to
  `OUTPUT_DIR/behavior` or `--dataset-dir`. Use the matching scenario/checkpoint
  with `--eval --eval-protocol final`, then select that output in the notebook.
- During training, `evaluation.recording.enabled: true` enables selection
  recording independently of training capture. MAPPO's global recording opt-in
  also enables it. Selection datasets live in `OUTPUT_DIR/evaluation_behavior`,
  with exact evaluated snapshots under `evaluation_checkpoints/evalNNNNNN.pt`.
  Frame/byte caps exclude checkpoint files. PPO shared-frame training capture
  remains unsupported; this increment adds its evaluation paths.
- Clip/race IDs, checkpoint hashes, training steps when known, policy versions,
  protocol, seed, and map references connect evaluation clips to notebook facts
  and annotations. Older best-model checkpoints recover their training step and
  policy version from embedded checkpoint-selection metadata.
- Selection evaluators do not compute rewards: frames explicitly mark them
  unavailable. Standalone evaluation retains existing computed components.
  MAPPO records opponent tails after learner exits. PPO selection retains its
  learner-exit stopping point and marks a still-running race incomplete with
  `end_reason: evaluator_boundary`. Action-repeat substeps keep physics timing.

Validation includes enabled/disabled actor/critic equality and identical
evaluation outcomes/checkpoint choices for both algorithms, exact checkpoint
hashes, replay/annotation source loading, held-action timing, opponent tails,
and incomplete PPO boundaries. Existing evaluator, recorder, analysis, and
widget tests also pass. Four tiny CPU CLI runs are saved as
`outputs/{ppo,mappo}_{evaluation,final}_recording`. The notebook executed against
all four and exported tables/PNG/PDF plots to
`outputs/analysis/fixed_evaluation_review`, including `executed_review.ipynb`.
Playback images were inspected; browser/IDE interaction was not repeated.
Production scenario files and training budgets were not changed.

Next: P4's bounded candidate-segment export with traceable source intervals,
before adding similarity grouping or learned skills.

## P4 - Group segments for future hierarchical learning

- [ ] Generate candidate segments from configurable event boundaries with
  pre/post context. For passes, use lap-aware progress, active-car filtering,
  sustained order changes, and hysteresis to avoid seam/lapping/DNF false labels.
- [ ] Export per-segment state/action/outcome features, duration, start/end
  conditions, participants, and source references. Include gaps, relative speeds,
  track context, and control changes; do not cluster actions alone.
- [ ] Build individual and combined tactic feature views. For combined segments,
  include both teammates' actions, relative geometry, opponent context, temporal
  order/overlap, and role changes. Group by participant roles where appropriate
  so swapping car IDs does not automatically create a different tactic.
- [ ] Add a simple similarity/grouping pass over normalized segment features
  using existing tools where practical. Surface representative clips and let
  humans accept, rename, split, or reject groups; retain uncertain examples.
- [ ] Keep representative-race samples and event-selected clips distinguishable
  in exports. Group train/validation splits by source race/run rather than
  scattering overlapping clips across splits.
- [ ] Export candidate skill records with initiation context, observed behavior,
  termination condition, duration, and outcome. Mark fields derived from privileged
  simulator state separately from features available to the actor at deployment.
  Include tactic scope, participant roles, and constituent-skill links so later
  work can explore individual options and their joint/sequential compositions.
  These are candidate skills, not trained options or verified team strategies.

Done when individual and combined tactics can both be exported and grouped with
traceable source clips and constituent links for later hierarchical learning.

## Minimal validation and next-agent handoff

- [x] Run relevant existing metric/lifecycle checks for touched logic and one
  small headless 2v2 smoke run with reduced environments/workers and budget.
  Use a temporary override; do not weaken the actual experiment configs.
- [x] Verify local metrics, sampled recording, and an annotation save/load cycle.
  Confirm recording disabled still works. Add targeted tests only for uncovered
  consequential failures such as ID mixing, terminal timing, or lost fields.
- [x] Manually inspect the terminal summary and one replay/label workflow. If
  graphical review is unavailable, state which UI interaction remains unchecked.
- [x] Mark completed tasks here and report concise launch/review commands plus
  real limitations. No additional documentation package or exhaustive test suite
  is required for this delivery.

P0/P1 metrics and monitoring, P2 selective recording, and P3 notebook review
and annotation are implemented for fixed-opponent MAPPO and two-team self-play.
PPO and MAPPO selection/standalone evaluations now support shared-frame capture.
The remaining planned increment is P4 candidate segments and feature exports;
selective shared-frame PPO training capture remains outside this delivery.

Validation: focused metric, lifecycle, CSV, checkpoint, and parallel-collector
checks passed. Two-environment headless runs covered continuous pretrained and
finite scratch training, plus selection and standalone evaluation. Verified
32 joint decisions / 64 learner transitions / 32 physics steps, four episode
records with sixteen car rows, and two optimizer updates. Existing replay loading
read all 64 recorded learner transitions across four distinct episodes. The
chosen pretrained checkpoint loaded successfully; smoke overrides were removed.

### Delivered metrics/monitoring handoff

- All active pretrained MAPPO arms (including inherited LoRA arms) and the
  pretrained PPO transfer entry point now use
  `outputs/L_map_pretrain/L_map_best_model.pt` by default.
- Local outputs: `race_metrics.jsonl` (four-car facts, IDs, spawn context, local
  and shared reward totals), `episode_metrics.csv`, `agent_metrics.csv`, and
  `update_metrics.csv`. Late scalar fields expand CSV headers without data loss.
  Reward-component dictionaries occupy JSON cells rather than changing columns.
- Per-environment `environment_decisions_total` is exact at episode end.
  `reported_at_environment_steps` is the aggregate collector count at the parent
  reporting barrier, not a fabricated global timestamp for race completion.
  Policy start/end versions identify episodes spanning multiple updates.
- Training and evaluation reuse race facts. Continuous-training finish-based
  values are missing; episode duration/incident summaries describe completed
  episodes and are not an unbiased survival estimate. Budget-cut episodes are
  excluded from completed-episode statistics.
- Selection history now includes four-car episode records, per-map summaries,
  explicit counts/denominators, seed protocols, and the unchanged checkpoint
  priority. Standalone 2v2 console headlines use explicit race results; legacy
  win/success aliases remain in JSON for backward compatibility. Standalone evaluation retains `evaluation_report.json`.
- Configure `experiment.terminal_recent_episodes`, `terminal_every_updates`,
  `terminal_diagnostic_every_updates` (0 disables diagnostics), and
  `terminal_episode_detail` in an override. Defaults are 100, 10, 100, false.
  `--quiet` suppresses informational monitoring lines.
- Legacy `--dataset-dir` recording remains the version-2 learner-transition
  format. Use `--record-races` for P2 shared frames and selective capture below.
  Annotation UI work remains in P3.

Bounded launch with temporary CLI overrides (the experiment YAML is unchanged):

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_2v2_base_pretrained.yaml --no-wandb --num-envs 2 --num-workers 1 --torch-threads 1 --total-steps 32 --max-steps 8 --output-dir outputs/metrics_smoke
```

Use `scenarios/mappo_2v2_penalties_pretrained.yaml` for the finite-race variant.
Read the CSVs directly; inspect structured records with
`python -m json.tool --json-lines outputs/metrics_smoke/race_metrics.jsonl`.
For a matched final evaluation of a normally trained checkpoint, run:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_2v2_base_pretrained.yaml --eval --checkpoint outputs/YOUR_RUN --eval-protocol final --no-wandb --output-dir outputs/YOUR_FINAL_EVAL
```

A bounded smoke checkpoint carries its override provenance; reuse matching
budget/environment overrides when evaluating it. Small smoke runs validate
plumbing, not policy quality. Evaluation budgets remain configurable through
`evaluation.episodes`, `evaluation.final_test.episodes`, or `--eval-episodes N`
for an explicitly custom evaluation protocol.

### Delivered P2 recording handoff

- `--record-races` enables MAPPO sampling and writes to `OUTPUT_DIR/behavior`,
  or to `--dataset-dir` when supplied. It works without W&B. Include
  `configs/recording/mappo_selective.yaml` in a scenario, or override its
  top-level `recording` settings, to enable/configure the same feature in YAML.
- Default representative selection is a 1% deterministic episode hash using
  the recording seed, environment ID, and episode number. It consumes no
  training RNG. Event clips are a separate, biased sample with retention reasons;
  do not estimate behavior frequencies from event clips or incomplete samples.
- Events include shared encounters, sustained candidate passes, measured track
  boundary departures, and terminal facts. Candidates are explicitly heuristic;
  collision partners and fault remain unknown. All four cars share each clip,
  so overlapping teammate events merge and can retain sequential interactions.
- Defaults: 60 physics steps before / 100 after events (3/5 seconds at 0.05 s),
  600-step maximum event clip, 16 event clips per episode, and 128 retained event
  descriptions per clip. `event_count` retains the full count when descriptions
  are capped. Complete-race samples stream incrementally without a clip-length cap.
- Dataset-wide limits default to 100,000 unique shared frames and 2 GB of
  serialized frame payload before compression. Small indices/provenance are extra.
  The parent enforces the shared limits and stops collectors at their next update
  barrier. Increase caps or reduce sampling for long experiments; a reached cap
  stops further recording, and affected captures are explicitly partial.
- Version 3 stores compressed `frames_*.jsonl.gz`, `frame_chunks.jsonl`, and
  `clips.jsonl`, plus metadata/provenance. Samples and event clips reference the
  same frames by episode and physics interval; frames are not duplicated.
  Index updates and chunk manifests are incremental. Interrupted datasets retain
  their written chunks and open/incomplete clip records.
- Each frame has separate pre/post four-car states, actual simulator input after
  clipping/terminal control, requested commands, active learner observations and
  normalized actions, per-physics-step reward components, policy version,
  decision/substep/physics indices, and simulation times. Missing commands remain
  null. Track/Frenet context, body-frame velocities, lifecycle facts, and initial
  spawn/physics context are retained. This is privileged review data.
- Finite `all_agents` races include opponent-only tails. Continuous training keeps
  its existing reset boundary: `episode_complete` and `all_cars_terminal` are
  separate. Surviving opponents are never labeled finished merely because the
  environment resets. Budget cuts, frame/byte caps, and clip-length caps have
  explicit end reasons; shortened post-event context is labeled too.
- Selection and bounded pre-event buffers live inside each collector. Physics
  provenance now travels once per MAPPO reset, so recording-disabled MAPPO does
  not transfer complete transitions merely to save provenance.
- Replay retains old datasets and uses recorded physics timing for version 3.
  Listing/filtering and shared-frame rendering work headlessly; interactive
  graphical review and annotation controls remain for P3.

```bash
# Normal training with selective recording (choose an appropriate output path).
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_2v2_base_pretrained.yaml --record-races --no-wandb --output-dir outputs/team_recorded
# Enumerate representative samples and event clips, including partial captures.
venv/bin/python replay.py outputs/team_recorded/behavior --list
# Replay a chosen clip with its recorded timing.
venv/bin/python replay.py outputs/team_recorded/behavior --clip CLIP_ID --speed 2
```

Python review can use `src.replay.race_reader.load_clips`, `clip_frames`, and
`iter_race_frames`; the latter two read only overlapping chunks for a selected
race/interval. Legacy `--dataset-dir` without `--record-races` retains the original
all-transition behavior.

Validation: the recorder suite passes 8 tests; parallel MAPPO and friction
provenance suites pass 40 tests. Terminal/continuous handling and legacy dataset
contract checks also passed. Tests cover spawned collectors, shared clip frames,
storage cuts, interrupted writers, and finite-race opponent-only tails.

A matched 256-step CPU smoke run (2 environments / 1 worker, 100% episode
sampling) retained 256 unique, aligned four-car frames, 8 complete environment
episodes, 1 budget-cut sample, and 17 event clips. One completed continuous
episode correctly retains surviving opponents. Actor and critic weights were
bit-identical with recording disabled/enabled. Aggregate collection+update
throughput was 13.53 versus 13.13 joint environment steps/s (about 3% lower).
This short, noisy comparison is not a 400-environment benchmark or an estimate
of default 1% sampling overhead. Output used about 1.27 MB on disk. Headless
clip rendering passed; desktop interaction was not manually exercised.
Artifacts: `outputs/p2_recording_verified_{disabled,enabled}`.

### Delivered P3 notebook analysis increment

- Open `notebooks/run_review.ipynb` with the repository's `venv/bin/python`
  kernel. It discovers saved runs and exposes `RUN_NAMES`, smoothing, optional
  external recording folders, clip filters, and export settings in editable cells.
- `src/analysis/run_review.py` loads P0 race records, selection history,
  standalone evaluation reports, update CSVs, and P2 clip indices. Folder paths
  distinguish runs even when their user-supplied IDs repeat. No checkpoints or
  trajectory chunks are loaded for analysis/index browsing.
- `src/analysis/plots.py` supplies learning curves, optimizer/throughput plots,
  evaluation curves, per-map outcomes with counts, clean finish distributions
  with completion counts, and cross-training-seed comparisons. Per-car and
  overall results are also available as tables. Continuous training, finite
  training, selection evaluation, final evaluation, and custom evaluation remain
  distinguishable. Missing measurements stay missing.
- Seed aggregation uses explicit experiment-group assignments and one selected
  snapshot per seed/group/protocol. It gives seeds equal weight and reports
  sample SD and counts; duplicate training seeds are rejected. Recorded protocol
  fingerprints separate differing evaluation conditions. Review provenance and
  opponent settings before treating runs as matched experiments.
- Clip tables retain sample/event kind, completeness, policy versions, outcomes
  where available, and source IDs, with commands to open existing replay.
  Checkpoint-aware interactive clip review and annotation save/load remain open.
- `EXPORT = True` writes CSV tables, PNG/PDF figures, and a selection/provenance
  manifest under `outputs/analysis/run_review`. The notebook is stored without
  execution outputs. Analysis dependencies are optional in `pyproject.toml` and
  have been installed into the current venv.

```bash
# Fresh environment only:
venv/bin/python -m pip install -e '.[analysis]'
venv/bin/python -m jupyter lab notebooks/run_review.ipynb
```

Validation: 6 focused tests pass, covering count/missing-value semantics,
finisher weighting, checkpoint separation, equal seed weights, duplicate IDs,
incomplete log tails, and plot exports. The full notebook executed against
three existing smoke output folders and an empty output directory. Six figures
were exported and representative plots visually inspected. The executed example
is `outputs/analysis/run_review/executed_review.ipynb`; browser/IDE interaction
was not manually exercised. No new training runs were required.

### P3 synchronized review and annotation handoff

- The notebook now includes a clip picker and explicit bounded interval loader
  (default maximum 2,000 physics frames). Only overlapping chunks are read;
  missing/discontinuous frames and mismatched source identities are rejected.
- The reviewer supports play/pause, boundary seeking, speed, looping, and event
  jumps using recorded interval durations. Long clips can be inspected in
  successive windows. Kernel/render latency can slow wall-clock playback.
- A shared map view and telemetry show four-car lifecycle state, observed speed
  and steering, applied references, recorded acceleration-reference rates,
  longitudinal/lateral gaps, per-interval reward components, and event facts.
  Wheel-reference acceleration uses rad/s², not vehicle acceleration in m/s².
- Individual labels support custom/uncertain/overlapping segments, actors,
  targets, roles, outcomes, confidence, and notes. Combined segments link at
  least two individual segments from the same recorded race, contain their
  intervals, and derive temporal order/overlap. Role changes and individual/team
  outcomes are retained. Inferred coordination or team-benefit claims require
  separate evidence; labels are reviewer hypotheses.
- Save/edit/reopen uses `RUN_FOLDER/review/annotations.json`, outside raw
  trajectories. Stable IDs, clip/race sources, policy/detector/label versions,
  half-open physics intervals, and simulation times are persisted. Saves are
  atomic and reject stale concurrent edits and changes that break linked segments.
- Exact checkpoint filters match only recorded checkpoint identities; they do
  not assign a final model to earlier trajectories. Policy-version interval
  filtering is available for training clips without a saved checkpoint identity.
- `ipywidgets` and `ipympl` are included in the optional analysis dependencies
  and installed in the current venv. Open `notebooks/run_review.ipynb`, select a
  recorded run, and use the synchronized review section. Restart an already
  running notebook kernel/server if newly installed widget extensions are absent.

Validation: 20 focused recording/analysis/reviewer tests passed (the six reviewer
checks were rerun after final layout/open-clip fixes). The full notebook executed
with 170 widget models. A live local JupyterLab session was exercised in headless
Chromium: play/pause, two individual segments, combined linking, save, reload,
and reopening the combined label all passed, with no browser page errors.
Screenshots were inspected; artifacts and validation-only labels are under
`outputs/analysis/p3_browser_validation`. These labels are UI fixtures, not
verified tactics. The installed widget stack emits a toolbar deprecation warning
in tests; current playback and editing work. The user's particular IDE frontend
was not exercised. Annotations are a local-file workflow on the current Linux
setup; no multi-user review service is included.
A final browser seek verified that clock, scene, and plot cursors agree at the
last physics boundary. Explicit nested-canvas sizing and rendered-frame updates
fix a stale/collapsed canvas found during visual checks; a pixel-change regression
check now guards seeking as well as annotation persistence.
