# PPO pretraining and transfer

The current pipeline uses synthetic planar MF6.1 physics, 0.05 s decisions,
wheel-reference acceleration, and a 50-value driving observation. It follows the
paper's main time-trial recipe, with explicit changes for reuse across tracks.
It is not a calibrated reproduction of the authors' car.

## Pretrain on L_map

The canonical scenario retains **400 environments**, 1,024 decisions per worker,
409,600 pooled transitions per update, and 120 million total transitions.
Do not change worker count without also deciding whether to change the pooled
rollout size. The existing 400-worker HPC configuration is retained.

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml --seed 42
```

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
50. Start a fresh run; an old checkpoint cannot be relabeled as compatible.
Vehicle-state scales and N=20 remain provisional.

Training resets on geometric boundary violations; there is no lap or time-limit
termination. Reward is signed metre progress, replaced by -1 outside the track.
The current implementation still uses uncalibrated tire and actuator parameters.

Each PPO episode line includes `laps` (the environment's completed lap count)
and `lap_time` (the most recent timed lap, in simulation seconds). Episodes with
no timed lap show `lap_time=n/a`; this is not the episode duration. The same
values are saved as `lap_count` and `lap_time_s` in `episode_metrics.csv` and as
`episode/lap_count` and `episode/lap_time_s` in W&B. Missing lap times are blank
in CSV and omitted from W&B rather than recorded as zero.

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

`mappo_2v2_frenet_ppo_pretrained.yaml` now uses the same MF6.1 dynamics, friction
protocol, 0.05 s decision interval, and wheel-acceleration action contract. Its
local actor observation has 65 inputs: the original 50 driving values followed by
three relative-neighbor slots (delta_s, delta_d, delta_vs, delta_vd, presence).
These are privileged simulator measurements, not inferred LiDAR detections.

Set `training_defaults.pretrained_actor_checkpoint` to a new compatible PPO
`best_model.pt`. With null it trains the matched scratch arm. The explicit
`pretrained_actor_observation_extension: frenet_neighbors` permits only this
appended block after an unchanged Frenet driving observation. The first-layer
weights for the 15 new inputs start at zero; all driving weights and exploration
parameters are copied. Initial actions match PPO even with neighbors present.
The new inputs receive gradients during MAPPO training. The centralized critic
and optimizer start fresh. Arbitrary input changes, scaling changes, and physics
mismatches still fail loading.

Fixed hybrid opponents retain their controller settings and use the explicit
`rolling_speed_to_wheel_v1` adapter. Their nominal 2.5 m/s setting is a starting
baseline, not evidence that they maintain this speed under MF6.1. The 800 s race
horizon is 16,000 steps; finish-clearance time remains 2 s. Reward presets retain
their per-decision coefficients, so the old 0.01 s time-cost rate is not preserved.

```bash
# After setting pretrained_actor_checkpoint in the YAML:
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_frenet_ppo_pretrained.yaml
```

The combined, first-place, and sweep reward variants inherit this contract.
Their team outcome rewards remain distinct experimental objectives. Automatic
checkpoint selection is PPO-only; select MAPPO checkpoints using its evaluation
selection seeds, then evaluate once on its final seeds. Keep actor transfer,
traffic adaptation, and unseen-map performance as separately reported results.
