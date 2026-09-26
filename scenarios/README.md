# Scenario index

Interactive fixed-controller previews are in [`render/`](render/README.md):
MPC solo, MPC versus hybrid 2v2, and a passing demo on circle and Budapest.

The active entry points below use the shared MF6.1 vehicle profile. PPO keeps
400 environments; the explicit MAPPO base and penalty scratch/pretrained pairs
also use 400 environments across 100 workers, with two learners and two fixed
racing MPC opponents per race. Other MAPPO objectives retain their serial defaults.
The fixed-opponent `mappo_2v2_*.yaml` scenarios inherit the same opponent profile.
`mappo_2v2_selfplay.yaml` instead trains two independent teams (see below).
Historical scenarios and render comparisons retain
their original controllers. See [the training workflow](../docs/PPO_TO_MAPPO_PRETRAINING.md)
for commands, checkpoint compatibility, evaluation protocols, and remaining
physics calibration work.

| Entry point | Use |
|---|---|
| `ppo_lap_completion_pretrain.yaml` | Train the reusable 158-input LiDAR/driving actor on L_map |
| `ppo_lap_completion_transfer.yaml` | Circle fine-tuning or matched scratch training |
| `ppo_lap_completion_validate.yaml` | Evaluate driving on L_map, circle, and Budapest |
| `mappo_2v2_base_scratch.yaml` | Stage 1: continuous metre progress, random initialization |
| `mappo_2v2_base_pretrained.yaml` | Stage 1: continuous metre progress, PPO actor initialization |
| `mappo_2v2_penalties_scratch.yaml` | Stage 2: placement and incident penalties, random initialization |
| `mappo_2v2_penalties_pretrained.yaml` | Stage 2: placement and incident penalties, PPO actor initialization |
| `mappo_2v2_completion.yaml` | Shared completion reward for traffic adaptation |
| `mappo_2v2_combined.yaml` | Main combined finishing-position objective |
| `mappo_2v2_first_place.yaml` | First-place objective comparison |
| `mappo_2v2_sweep.yaml` | First-and-second-place objective comparison |
| `mappo_2v2_individual.yaml` | Individual-reward completion comparison |
| `mappo_2v2_asymmetric.yaml` | Two learner roles: lap reward / lap reward plus opponent crashes |
| `mappo_2v2_validate.yaml` | Held-out Silverstone/Spa evaluation |
| `mappo_2v2_selfplay.yaml` | Two trainable teams: three-lap completion and opponent crashes |

Both validation scenarios require `--eval --checkpoint PATH`. Training MAPPO
from a PPO source uses `--pretrained-actor PATH`; omitting it uses the scenario
default (scratch except for the explicitly pretrained base and penalty arms).
PPO and MAPPO share a 158-input driving prefix: 108 normalized LiDAR ranges
followed by 50 driving values. Fixed-opponent MAPPO adds three neighbor slots
with teammate flags (176 inputs); the asymmetric scenario also encodes vehicle
and ego IDs (192 inputs). MAPPO's
shared settings live in `configs/scenarios/mappo_2v2_base.yaml`; edit that fragment
when a setting should apply to the fixed-opponent team objectives.

## Two learner roles against racing MPC

```bash
PYGLET_HEADLESS=true python3 run.py \
  --scenario scenarios/mappo_2v2_asymmetric.yaml
```

`car_0` and `car_1` are trainable teammates; `car_2` and `car_3` use the fixed
advanced `racing_mpc` profile. Training defaults to circle with one environment
and 120 million aggregate joint decisions, matching the PPO pretraining budget.
There is no training lap or time limit: the surviving learner keeps driving
until both learners crash. Rollout boundaries update the policy without resetting
the episode. Checkpoints and evaluation use 4,096,000-step thresholds; evaluation
retains three-lap races with a 16,000-step cap. Use `--total-steps` for shorter
runs and `--num-envs` / `--num-workers` for parallel collection.
Add `--pretrained-actor outputs/PRETRAIN_RUN` to initialize from a compatible
lap-completion PPO actor; otherwise training starts fresh.

Both learners use the exact `lap_completion_pretraining.yaml` reward: signed
metre progress, replaced by -1 on a geometric boundary excursion. `car_1` also
gets +1 for each opposing car's collision terminal, at most +2 per race.
Change `bonus` in `configs/reward/tasks/lap_completion_opponent_crash.yaml`
to tune its weight. It counts either opponent, once each, including simultaneous
ego/opponent collisions. It does not establish collision causation. Finishes,
timeouts, and teammate crashes give no bonus. The learner collects this local
reward only while active, including its final transition; later crashes after
its own termination receive no credit.

Returns remain per agent, with an agent-conditioned centralized critic and a
shared actor conditioned on ego ID. The crash bonus is not averaged into the
racer's reward. Geometric boundary sensing supplies the pretraining penalty;
this race keeps physical wall collisions and collision/finish/time-limit
termination rather than the single-car pretraining boundary resets. Evaluation
continues to select checkpoints by team completion.

Each observation has 192 values: the unchanged 158-input PPO driving prefix,
three nearest-first slots of `[ds, dd, dvs, dvd, present, is_teammate, ID(4)]`,
and ego `ID(4)`. IDs are one-hot in the fixed order `car_0` through `car_3`.
All three other vehicles are visible through simulator-provided Frenet state;
IDs follow vehicles when distance ordering changes. Missing slots are zero.
The 34 added actor-input columns start at zero when loading PPO weights, so
initial actions match the pretrained driving policy. Existing 176-input MAPPO
checkpoints have a different observation contract.

## Two trainable teams

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_selfplay.yaml
```

This is simultaneous self-play: `car_0/car_1` form `team_a`, and `car_2/car_3`
form `team_b`. Each team has its own actor, centralized critic, optimizer, and
joint return. Both teammates share their team's reward. Both teams update from
the same race; weights stay fixed during each collection round. The default is
400 environments across 100 CPU worker processes, with both policies and critics
in the parent GPU process. Each worker advances four races and uses one compute
thread. Training starts from scratch; PPO actor transfer and training resume
are not yet supported.
Existing fixed-opponent scenarios retain their previous behavior.
Use `--num-envs N --num-workers W` to size the collector pool. `--num-envs 1`
selects the serial trainer. CLI overrides apply before validation.

Add `--record-races --output-dir outputs/selfplay_recorded` to save selective
shared-frame recordings under `outputs/selfplay_recorded/behavior` in either
serial or parallel training. The existing top-level `recording` settings control
sampling, event clips, and dataset-wide storage limits. Both teams' commands,
rewards, and policy versions are retained, including rewards after a team becomes
inactive and terminal-car clearance/removal. Open `notebooks/run_review.ipynb`
and select `selfplay_recorded` to plot team metrics, play clips, and save labels.
Its self-play win metric compares completion/progress; it is not first place.
With recording enabled, selection evaluations also write to `evaluation_behavior`
and save the evaluated pairs under `evaluation_pairs/eval_NNNNNN`. These fixed
pairs, their file hashes, protocol, and seeds are linked from clips and labels.
Standalone `--eval --checkpoint PATH --eval-protocol final --record-races` writes
to `behavior` and references the loaded pair. Both paths appear in the notebook's
separate self-play evaluation tables/plots; clip filters accept `phase`,
`protocol`, and the exact checkpoint path or hash.

Evaluation defaults to sampling every race, with an independent storage budget
inheriting the top-level recording caps. Override `evaluation.recording` with
the same settings to adjust its budget/sampling, enable only evaluation recording,
or set `enabled: false` to keep training-only recording. Paired checkpoint files
are additional disk usage outside the frame/byte caps. Recording stops when a
cap is reached; defaults do not guarantee coverage across the full 120M-step
budget. To reserve capacity across training stages, use:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_2v2_selfplay_recorded.yaml --no-wandb --output-dir outputs/selfplay_windowed
```

This opt-in scenario includes `configs/recording/mappo_windowed.yaml`: three
non-overlapping ranges (0–40M, 40–80M, 80–120M joint decisions) share the existing
100,000-frame / 2 GB total cap. Early data cannot borrow a later range's reserved
frames or bytes. Fixed-opponent MAPPO scenarios can include the same config.
Set `recording.windows: []` for the original single-budget behavior.

Each window requires `start_step`, `end_step` (exclusive), `max_frames`, and
`max_bytes`; allocations must fit within the dataset-wide caps. Exhausting a
window pauses capture until another window begins. Serial training uses the
current joint-decision count; parallel training uses the last completed
collection barrier. Narrow windows can be skipped if no barrier falls inside
them. Tune ranges when changing the total training budget or rollout size.
Unused capacity is not redistributed, and complete races are not guaranteed.

Clips close at window boundaries and pre-event buffers are cleared. Resuming
mid-race creates a `representative_segment`, explicitly marked incomplete;
event-clip limits restart for each window within the episode. The notebook shows
window budgets/usage and accepts `window_index` in `CLIP_FILTERS`. Annotation
sources retain window and progress-clock references. Evaluation does not inherit
training windows; explicit `evaluation.recording.windows` uses the evaluated
checkpoint's training-step count instead (unknown counts are not recorded).

Training uses **120,000,000 aggregate joint environment decisions**, matching the
continuous trainers' budget. One four-car race step counts once, with up to four
actor samples; evaluation steps are excluded. `--total-steps N` overrides the
budget. `--episodes N` explicitly switches back to an episode budget. A total
budget cutoff mid-race bootstraps the critic and records `budget_cut=1` without
inventing a race completion or timeout.

Each parallel round gathers up to 256 joint decisions per environment: 102,400
joint steps, up to 204,800 actor samples **per team** (409,600 total). Minibatches
contain 2,048 samples. Completed races reset independently during collection;
unfinished races carry on across rounds. Returns are computed separately for
each race, team, and fragment before pooling. The parent updates both teams
before workers resume. Inactive teams still contribute critic targets for later
opponent events, with no extra actor samples. Worker seeds are the base seed
plus environment index; an uneven budget is divided with exact remainders.

Startup is batched eight workers at a time. Failures/timeouts propagate to the
parent and shut down the collector group. All W&B and file writes happen in the
parent. These settings are a starting point for a 128-core allocation; benchmark
throughput and RAM on the target node before assuming 400 races is optimal.

Each car finishes after three laps or stops acting after its first collision.
The race continues until all cars finish/crash or the 16,000-step (800-second)
deadline expires, including when both cars of one team have crashed. Crashed
cars remain stationary during a 40-step clearance window. Finishers coast during
the same window. After clearance, both are excluded from car collisions, LiDAR
occlusions, and neighbor observations; their recorded terminal facts remain.
The existing renderer may still draw their terminal poses.

Team reward per step is the **sum** of signed progress (+1 per full lap per car),
+4 per new race finisher, -3 per own collision, +1 per opponent collision, and
+2 when both teammates have finished. Event rewards occur once, including
simultaneous crashes. A one-for-one crash trade is therefore -2 before progress.
There is no timeout penalty, per-step time cost, placement reward, or extra win
bonus. Opponent crashes are measured outcomes, not proof of causal responsibility.
Local weights live in `configs/reward/tasks/race_two_trainable_teams.yaml`;
team bonuses are under `two_team.events` in the scenario.

Actors use 178 inputs: the existing 158 driving/LiDAR inputs, three six-value
Frenet neighbor slots including teammate identity, remaining time fraction, and
own remaining lap fraction. Neighbor state is simulator-provided privileged
sensing. Each critic also receives the remaining time fraction. `gamma=0.9995`
gives an approximately 100-second discount horizon at 20 Hz. The race deadline
ends team returns; a rollout or total-step-budget cut bootstraps the critic.
Individual crashes/finishes do not end team credit or create dummy actor samples.

W&B is enabled in project **`marl-f110-selfplay`**, group
**`mappo-2v2-completion-crashes`**. Set `wandb.entity` to choose a team/workspace;
otherwise the logged-in account's default is used. `--no-wandb` keeps local logs.
Useful chart groups are:

- `selfplay/team_a/*`, `selfplay/team_b/*`: finish rates, both-finished events,
  collision counts, eliminated teams, timeouts, laps, signed progress, reward,
  and reward-component totals, plotted against joint environment steps.
- `selfplay/rolling100/*`: completion, crash, reward, win, and draw averages over
  completed races only; a partial total-budget race is excluded.
- `selfplay/car_0/*` through `selfplay/car_3/*`: individual finish/crash/lap facts.
- `train/team_a/*`, `train/team_b/*`: independent PPO optimizer metrics.
- `selfplay_eval/*`: deterministic current-pair evaluation on fixed seeds/maps.
- `perf/*`: collection/update time, joint steps per second, parent peak RSS,
  and the sum of worker peak RSS values in MiB (not instantaneous total memory).

Parallel episode rows identify their `environment_id`, `environment_episode`,
and seed. `selfplay/environment_steps` is the aggregate training counter at
report time; `selfplay/environment_local_steps` is that environment's counter.
The console prints collection/update throughput every ten updates. Set
`experiment.terminal_episode_detail: true` for every race's console summary.

Win/draw logging compares finisher count first, then summed signed progress in
laps (rounded to six decimals). It does not add learning reward. Evaluation
compares two changing policies, so these win rates are not an absolute strength
benchmark against a frozen opponent pool. Checkpoint selection favors the lower
of the two both-finished rates, then total finish rate, progress, and fewer crashes.

Runs save `team_metrics.jsonl`, `updates.jsonl`, `evaluation_metrics.jsonl`,
`evaluation_races.jsonl`, a resolved config snapshot, and `run_summary.json`.
`best_pair/`, `final_pair/`, and periodic `pair_stepNNNNNNNNNNNN/` directories each contain
both policies and `pair.json` with team membership, race/reward contracts, and
provenance. Evaluation and periodic checkpoints run every **1,024,000 training
steps**, at the next policy-update boundary, including during a race. Final
evaluation also runs at training completion unless that step was already
evaluated. Evaluation charts include the training step count as their x-axis.

```bash
# Evaluate both saved policies using the disjoint final-test seeds.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_selfplay.yaml \
  --num-envs 1 --eval --checkpoint outputs/RUN/best_pair --eval-protocol final

# Small parallel check, including a rollout boundary and an uneven remainder.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_selfplay.yaml \
  --num-envs 4 --num-workers 2 --total-steps 1043 --max-steps 16 --no-wandb

# Serial check using the same episode and reward code.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_selfplay.yaml \
  --num-envs 1 --total-steps 128 --max-steps 128 --no-wandb
```

## PPO and fixed-opponent MAPPO evaluation recordings

Standalone evaluations support the same shared-frame playback and annotations:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_transfer_3lap.yaml \
  --eval --eval-protocol final --checkpoint outputs/RUN/best_model.pt \
  --record-races --no-wandb --output-dir outputs/transfer_final_recorded
venv/bin/python -m jupyter lab notebooks/run_review.ipynb
```

Use the matching training scenario/checkpoint, or explicitly allow provenance
mismatches for an intentional transfer evaluation. Select the output folder in
the notebook. Frames go to `behavior` (`--dataset-dir` redirects it); clips retain
the loaded checkpoint path/hash, protocol, seeds, and computed reward components.

During training, enable only checkpoint-selection recording with:

```yaml
evaluation:
  recording:
    enabled: true
    max_frames: 100000
    max_bytes: 2000000000
```

Selection recordings go to `OUTPUT_DIR/evaluation_behavior`, with exact evaluated
snapshots in `evaluation_checkpoints/evalNNNNNN.pt`. Fixed-opponent MAPPO training
also enables evaluation capture with `--record-races`; PPO training uses the YAML
evaluation switch above because selective shared-frame training capture remains
MAPPO-only. Evaluation budgets are separate from training, default to sampling
every race, and do not inherit training windows. `evaluation.recording.enabled:
false` explicitly disables evaluation capture, including when the CLI flag is set.
Checkpoint files are additional disk usage outside frame/byte caps.

Selection evaluators measure outcomes without computing rewards; those recorded
reward channels remain unavailable. MAPPO retains opponent tails after learner
exits. PPO selection preserves its existing learner-exit boundary, marking such
clips incomplete (`evaluator_boundary`) if the environment is still running.
Notebook outcome and checkpoint filters cover both selection and standalone clips.

## Current-setup three-lap transfer

`ppo_lap_completion_transfer_3lap.yaml` loads
`outputs/L_map_pretrain/L_map_best_model.pt` and currently targets Budapest.
It inherits the current vehicle profile, fixed track observation scaling, and
400 workers. Episodes end after three laps or 16,000 steps (800 simulated seconds).
Training uses geometric boundary resets; evaluation also terminates on collisions.
Checkpoint selection uses completion and net progress. The destination budget is
4,096,000 decisions. Override the source with `--checkpoint PATH` for another run.

```bash
# Evaluate the new source before destination fine-tuning.
# The provenance override acknowledges the changed scenario/map, not changed physics.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_transfer_3lap.yaml \
  --eval --allow-provenance-mismatch --no-wandb \
  --output-dir outputs/budapest_3lap_before

# Fine-tune actor and critic with a fresh optimizer in a separate output directory.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_transfer_3lap.yaml
```

The older downloaded model and its compatibility configuration are retained as
historical artifacts. Active transfer scenarios no longer include that profile.
The generic `ppo_lap_completion_transfer.yaml` remains a scratch baseline unless
`--checkpoint PATH` is supplied.

## Experiment order

1. Run `mappo_2v2_base_scratch.yaml` and `mappo_2v2_base_pretrained.yaml`.
   Both use shared signed metre progress with an exclusive -1 collision cost,
   a 120M joint-step training budget, and 20-lap `team_completion` evaluation.
2. Run the penalty scratch/pretrained pair below using the same frozen PPO source,
   training seeds and parallel collection settings. This pair retains its
   three-lap races and 5,000-episode budget.
3. Compare `mappo_2v2_{base,penalties}_lora_shared.yaml` against
   `mappo_2v2_{base,penalties}_lora_per_agent.yaml`, with `lora_shared_r8` as the
   total-capacity control. See [LoRA implementation and commands](../docs/MAPPO_LORA.md).

All four current arms share `configs/training/mappo_parallel.yaml`. Both pretrained
arms initialize the full actor from PPO and train it normally; their critics and
optimizers start fresh. The base-to-penalty comparison changes rewards, training
budgets, race horizons, and selection objectives; it is not a penalty-only ablation.
The continuous base overrides live in `configs/scenarios/mappo_2v2_continuous_base.yaml`.

## Matched team penalty experiments

`mappo_2v2_penalties_scratch.yaml` and `mappo_2v2_penalties_pretrained.yaml`
compare random initialization with the new current-setup L-map actor under
identical physics, observations, opponents, and event-based penalties. The
pretrained arm defaults to `outputs/L_map_pretrain/L_map_best_model.pt`;
use `--pretrained-actor PATH` to select a different current-setup source.
Both penalty arms default to 400 environments across 100 CPU workers, with
256 decisions per environment per collection round. See
[Parallel MAPPO](../docs/PARALLEL_MAPPO.md) for HPC launch commands, bounded
smoke tests, and throughput metrics.
See [the experiment protocol](../docs/TEAM_RACING_EXPERIMENTS.md)
for pretraining, penalty definitions, scoring, and remaining comparison limits.

## Migration

| Previous entry point | Replacement |
|---|---|
| `ppo_lap_completion_pretrain_frenet.yaml` | `ppo_lap_completion_pretrain.yaml` |
| `ppo_lap_completion_pretrain_combined_slip.yaml` | `ppo_lap_completion_pretrain.yaml` |
| `mappo_2v2_frenet_ppo_pretrained.yaml` | `mappo_2v2_completion.yaml` |
| `mappo_2v2_frenet_ppo_pretrained_{combined,first_place,sweep}.yaml` | `mappo_2v2_{combined,first_place,sweep}.yaml` |
| Old `mappo_2v2_individual.yaml` | `legacy/mappo_2v2_individual.yaml` |
| `ppo_combined_slip_circle_stable.yaml` | `experiments/ppo_combined_slip_circle_stable.yaml` |
| Other old PPO / `complete_4*` / MAPPO scenarios | Same filename under `legacy/` |

The removed PPO aliases had different worker counts and hyperparameters. The
canonical replacement deliberately uses the retained 400-worker configuration;
it is not a bit-for-bit recreation of an old alias run. Current 158-input actors
cannot load old 50-input PPO or 65/68-input MAPPO checkpoints. Train a fresh
LiDAR-enabled PPO source; it transfers directly without adding input weights.

## Retained comparisons

- `experiments/`: distinct current-physics ablations. The circle convergence
  scenario explicitly selects circle for both training and evaluation.
- `legacy/`: fifteen historical experiments retained to reproduce old comparisons.
  Their older dynamics, observations, and actions are not compatible with current
  pretraining. Relative config paths have been adjusted for their new location.
- `calibration/`: historical fixed-controller calibration on legacy physics.
  These results do not validate opponents under the new MF6.1 setup.

Do not use held-out maps to select checkpoints or tune the current experiments.
The active MAPPO training/selection maps are Budapest and circle; Silverstone and
Spa are reserved for the team validation scenario.
