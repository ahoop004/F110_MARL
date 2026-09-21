# PPO pretraining and transfer

The current pipeline uses F1TENTH Gym chassis/steering defaults with synthetic
planar MF6.1 tires, 0.05 s decisions,
wheel-reference acceleration, and a 158-value LiDAR/driving observation. It follows the
paper's main time-trial recipe, with explicit changes for reuse across tracks.
It is not a calibrated reproduction of the authors' car.

## Pretrain on L_map

The canonical scenario retains **400 environments**, 1,024 decisions per worker,
409,600 pooled transitions per update, and 120 million total transitions.
Do not change worker count without also deciding whether to change the pooled
rollout size. The existing 400-worker HPC configuration is retained.

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml --seed 42 \
  --output-dir outputs/ppo_current_pretrain_s42
```

This output directory supplies the default source checkpoint for the current
three-lap transfer and pretrained team-penalty scenarios. Use a new output
directory for each independent run; override downstream checkpoint paths when
using another source. Freeze the selected checkpoint before matrix training.
The older downloaded `outputs/L_map_best_model.pt` is not the matrix source.

Use seeds 43 and 44 for independent training repeats. The optional W&B grid is
`sweeps/ppo_pretrain_seed_sweep.yaml`; schedule its runs within the allocated
resources rather than running three 400-worker jobs concurrently in one allocation.
The reported HPC allocation is 128 CPUs, one GPU, and 72 hours. The 400 processes
share those CPUs. This does not alter the algorithm's 400 environment streams.
120 million transitions require an end-to-end average above 463 transitions/s
to fit in 72 hours; leave additional time for startup and evaluation.

PPO epochs, GAE lambda, entropy coefficient, clipping, gradient norm, initial
standard deviation, and the absence of KL stopping are explicit in the scenario.
They remain implementation choices, not recovered paper hyperparameters. Monitor
KL, action saturation, exploration, and deterministic completion before tuning.

Curvature and width use fixed scales of **1 m^-1 and 1 m** across maps, with no
clipping. Thus a 2 m lane is distinguishable from a 1 m lane. Older observation
presets without `track_maxima` retain their per-map normalization. Changing these
scales changes the checkpoint observation contract even when its dimension stays
158. Start a fresh run; an old checkpoint cannot be relabeled as compatible.
Vehicle-state scales and N=20 remain provisional.

Training resets on geometric boundary violations; there is no lap or time-limit
termination. Reward is signed metre progress, replaced by -1 outside the track.
The current implementation still uses uncalibrated tire and actuator parameters.
The shared vehicle profile now uses a 0.58 x 0.31 m footprint, ±0.4189 rad steering
and ±3.2 rad/s steering-rate limits from a pinned F1TENTH Gym revision (see
[physics configuration and sources](PHYSICS_MODEL.md#configuration-and-calibration)).
This changes the physics contract relative to the earlier 0.32 x 0.225 m,
±0.5 rad profile. Existing processes retain their loaded configuration; new
checkpoints and downstream transfer must use the same revised physics contract.

Each PPO episode line includes `laps` (the environment's completed lap count)
and `lap_time` (the most recent timed lap, in simulation seconds). Episodes with
no timed lap show `lap_time=n/a`; this is not the episode duration. The same
values are saved as `lap_count` and `lap_time_s` in `episode_metrics.csv` and as
`episode/lap_count` and `episode/lap_time_s` in W&B. Missing lap times are blank
in CSV and omitted from W&B rather than recorded as zero.

## L-map lap-counter correction

The original generated `L_map.yaml` lacked a finish-line annotation. Forward
metre rewards continued to accumulate, but no lap tracker existed, so training
and evaluation reported zero laps and no lap times regardless of distance.
The map now defines a 1 m finish segment across centerline sample 51 (~2.55 m
along the CSV), oriented in the positive track direction. Geometry, physics,
reward weights, and continuous-training termination are unchanged.

Pretraining now requires a finish line and fails at environment setup or map
switch if it is missing. Preserve the annotation when regenerating this map with
the external map editor. With random spawning, the first forward crossing starts
the lap clock; each subsequent full circuit increments the completed-lap count.
Short episodes can therefore still correctly show zero laps.

Existing processes retain their loaded map; deploy the updated files and restart
the process to enable the fix. Old zero-lap metrics cannot establish whether laps
were completed, and old `best_model.pt` selection may not have ranked checkpoints
meaningfully. Re-evaluate saved periodic/final checkpoints with the corrected map
and matching physics/observation profiles before choosing a pretrained model.
The annotation changes map provenance; cross-version evaluation may require
`--allow-provenance-mismatch`, which does not bypass physics compatibility.

## Select and validate a model

Every 4,096,000 transitions, deterministic selection evaluates eight starting
seeds (10042–10049), each for 20 laps at nominal grip. Excursions are measured
without resets. The 800 s horizon bounds stalled policies. Selection ranks full
completion, valid-lap count, fastest valid lap, then off-track error.
`best_model.pt` stores the selected policy; `final_model.pt` stores the last update.
Twenty additional starting seeds (20042–20061) are reserved for the final protocol.
These starts are not independent training runs.

```bash
# Replace outputs/PRETRAIN_RUN with the actual run directory.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml --seed 42 \
  --eval --checkpoint outputs/PRETRAIN_RUN --eval-protocol final \
  --output-dir outputs/pretrain_final --no-wandb
```

Use the source training seed for provenance matching. Avoid using final results
to choose checkpoints or tune hyperparameters. Same-map final seeds establish
repeatability across starts, not map generalization.

`ppo_lap_completion_validate.yaml` provides a separate three-lap downstream test
on L_map, circle_map, and Budapest_map. It enables collision termination and
`evaluation.terminate_on_track_limit: true`. Its selection protocol uses four
starts per map and its final protocol uses ten. It never selects training
checkpoints (`evaluation.enabled: false`).

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_validate.yaml \
  --eval --checkpoint outputs/PRETRAIN_RUN --allow-provenance-mismatch \
  --eval-protocol selection --output-dir outputs/transfer_validation --no-wandb
```

The provenance override acknowledges changed maps and evaluation rules; it does
not bypass physics, action, or observation compatibility. Reports contain
`per_map` summaries and a `map_bundle` for every episode. Report completion,
collision/timeout rates, clean finish time, and off-track error by map. Circle and
Budapest are unseen only until used for training or tuning; reserve other maps
for a later independent generalization test if they become development tracks.

Proposed acceptance gates (experiment criteria, not existing results):

- All selection starts complete 20 L-map laps, with at least 99% valid measured
  laps, for each of three independently trained seeds.
- At least 95% clean three-lap finishes on each downstream final-test map. With
  ten starts per map, this requires all ten to finish cleanly; report the counts
  and uncertainty rather than claiming a precisely estimated population rate.
- Pretrained fine-tuning reaches the chosen completion threshold sooner than
  scratch training across seeds, at equal destination-transition budgets.

A smoke run or higher training reward does not meet these gates. If generalization
is weak, add diverse maps to a separate pretraining experiment and reserve new
held-out maps; keep the single-L-map reference for comparison.

## PPO transfer against a scratch baseline

`ppo_lap_completion_transfer.yaml` now targets **circle_map**, starts with
`experiment.checkpoint: null`, and inherits the same 400 workers and observation,
physics, and action contracts. Both arms use 4,096,000 destination transitions,
a constant learning rate of 1e-4, and evaluation every 409,600 transitions.
This initial budget gives ten pooled updates; extend both arms equally if needed.

```bash
# Zero-shot destination performance, before any destination training.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_transfer.yaml --eval \
  --checkpoint outputs/PRETRAIN_RUN --allow-provenance-mismatch \
  --eval-protocol selection --output-dir outputs/circle_before --no-wandb

# Fine-tuning: actor and critic loaded; optimizer and schedule start fresh.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_transfer.yaml \
  --checkpoint outputs/PRETRAIN_RUN --output-dir outputs/circle_finetune

# Matched scratch arm; checkpoint remains null in the YAML.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_transfer.yaml \
  --output-dir outputs/circle_scratch
```

Use the same destination seeds for paired comparisons (42, 43, 44), new output
directories, and equal transition budgets. Evaluate each chosen model with the
same final protocol. A checkpoint load is fine-tuning, not exact interrupted-run
resume. Older reduced-physics checkpoints are incompatible.

## Transfer the driving actor into MAPPO

All active `mappo_2v2_*.yaml` training scenarios share
`configs/scenarios/mappo_2v2_base.yaml`: the same MF6.1 dynamics, friction
protocol, 0.05 s decisions, wheel-acceleration actions, and two fixed racing MPC
opponents. The explicit base and penalty scratch/pretrained pairs use 400
environments across 100 workers. Other MAPPO objectives retain their serial
defaults. PPO retains 400 environments.

PPO and MAPPO use the same 158-input actor observation: 108 LiDAR ranges,
normalized by the 10 m sensor range and capped at one, followed by the same
50 vehicle/Frenet/track values. There are no explicit neighbor slots, relative
velocities, or teammate labels. LiDAR observes walls and visible vehicles as
obstacles; the driving block retains the same fixed scales and 20-point preview.
Single-car transfer and validation inherit this observation from pretraining.

Use `--pretrained-actor` with a compatible PPO checkpoint or run directory.
Omitting it uses the scenario default: the explicitly pretrained base and penalty
arms require `outputs/ppo_current_pretrain_s42/best_model.pt`; other training arms
start from scratch. `pretrained_actor_observation_extension` is null because
both actors have identical observation contracts. All actor weights and exploration
parameters are copied directly; the centralized critic and optimizer start fresh.
Old 50-input PPO and 65/68-input MAPPO checkpoints are incompatible. Train a fresh
LiDAR-enabled PPO source before running the pretrained arms.

| Scenario | Purpose |
|---|---|
| `mappo_2v2_completion.yaml` | Learn to finish together in traffic; shared completion reward |
| `mappo_2v2_combined.yaml` | Main team-racing experiment; combined finishing-position objective |
| `mappo_2v2_first_place.yaml` | First-place objective comparison |
| `mappo_2v2_sweep.yaml` | First-and-second-place objective comparison |
| `mappo_2v2_individual.yaml` | Matched completion baseline with individual rewards and agent-conditioned critic |
| `mappo_2v2_validate.yaml` | Evaluation only on held-out Silverstone and Spa |

Run the explicit base scratch/pretrained pair first, the penalty pair second,
then [shared/per-teammate LoRA](MAPPO_LORA.md); see the
[experiment matrix](TEAM_RACING_EXPERIMENTS.md). The standalone combined objective
below remains available as a separate comparison using the same PPO source. `--pretrained-actor` accepts PPO actors, not a MAPPO checkpoint;
completion-to-combined MAPPO continuation is not implemented. Compare scratch and
pretrained arms within each objective with the same seeds and destination budget.
The base pair uses 120M aggregate joint environment decisions, continuous training
without lap finishes, a metre-progress/collision reward, and 20-lap evaluation.
The penalty pair retains three-lap races and an episode budget. Record actual
learner samples and wall time as well as joint decisions when comparing runs;
equal episode counts are not equal sample budgets.

```bash
# Actor transfer into the main team objective.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_combined.yaml \
  --pretrained-actor outputs/PRETRAIN_RUN --output-dir outputs/team_transfer

# Matched scratch configuration.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_combined.yaml --output-dir outputs/team_scratch

# Final starts on the development maps after checkpoint selection.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_combined.yaml --eval \
  --checkpoint outputs/team_transfer --eval-protocol final \
  --output-dir outputs/team_final --no-wandb

# Held-out maps, once the model and settings are frozen.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_validate.yaml --eval \
  --checkpoint outputs/team_transfer --allow-provenance-mismatch \
  --eval-protocol final --output-dir outputs/team_heldout --no-wandb
```

Deterministic evaluation uses eight fixed starts across Budapest and circle.
The four explicit matrix arms evaluate every 1,024,000 aggregate environment
decisions after an update; the other team objectives retain their 100-episode cadence. It uses a separate environment and preserves training
random-number states. Checkpoint selection ranks both teammates finishing first,
then the configured objective, fewer learner collisions, and net progress
(or faster clean finishes once every start finishes).
`best_model.pt` records its evaluation in `checkpoint_selection`, and every
selection result is saved in `evaluation_history.jsonl`. Twenty separate starts
are reserved for final evaluation. Held-out validation uses separate seeds and
never participates in checkpoint selection. Neither validation entry point can
start training accidentally: both require `--eval`.

Fixed MPC opponents use `rolling_speed_to_wheel_v1`, a 3.5 m/s speed cap, and
vehicle dimensions from the environment. See the
[controller contract and initial completion results](RACING_MPC_OPPONENTS.md),
including its privileged traffic sensing. Broader starts, randomized grip, and
learned traffic still need qualification before interpreting team wins; keep
earlier hybrid-opponent results separate. The 800 s race
horizon is 16,000 steps and finish clearance remains 2 s. Team reward presets now
express time cost as -0.00025 per simulated second, preserving the prior -0.2
maximum over 800 s at 0.05 s steps.

## Scenario cleanup

The active entry points live directly under `scenarios/`. The two duplicate
PPO pretraining aliases were removed; use `ppo_lap_completion_pretrain.yaml`.
The former `mappo_2v2_frenet_ppo_pretrained*` names became the four completion/team
objective names above. Fifteen older experiments remain under `scenarios/legacy/`
for historical comparisons, including the old individual/team-shared pair.
Their physics and checkpoint contracts have not been migrated.

The circle convergence ablation lives under `scenarios/experiments/` and explicitly
uses circle for training and evaluation. Historical fixed-controller calibration
files remain under `scenarios/calibration/`; their results apply to legacy physics.
See [the scenario index](../scenarios/README.md) before creating another entry point.
