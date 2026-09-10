# PPO actor pretraining for MAPPO

Train the compatible single-agent actor with:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --no-wandb
```

The selected checkpoint is written to the run output directory as
`best_model.pt`. Selection is based on deterministic evaluation in
this order: lap-completion rate, lower collision rate, mean lap progress, then
lower mean finish steps. `evaluation_history.jsonl` records every selection
decision.

The current scenario uses `circle_map` for both splits. A held-out generalization
experiment requires explicit disjoint training and evaluation maps.

For the observation comparison, use
`scenarios/ppo_lap_completion_pretrain_frenet.yaml`. It includes the base
pretraining scenario and overrides only the observation reference and run
labels; it explicitly repeats the same 20-point, 0.3 m preview settings.
Maps, spawn protocol, training/evaluation seeds, rewards, vehicle/action
parameters, PPO hyperparameters, and episode budget remain matched.

| Scenario | Actor observation | Dimension |
|---|---|---|
| `ppo_lap_completion_pretrain.yaml` | LiDAR, ego motion, progress/deviation, previous action | 115 |
| `ppo_lap_completion_pretrain_frenet.yaml` | LiDAR, normalized vehicle/Frenet state, curvature and width preview | 158 |

The Frenet variant reuses `rl_racer_vehicle_track_frenet.yaml`: 108 LiDAR
values, 10 vehicle/Frenet values, 20 curvatures, and 20 widths. It replaces the
baseline's seven non-LiDAR values; this compares complete observation
representations, not just the addition of track preview. Normalization and
clipping also differ. Hidden-layer widths match, but the larger input makes
the first actor/critic layers larger. Wheel-speed values are simulator proxies
derived from longitudinal speed and a configured wheel radius, not measured
wheel rotation. The direct target-speed action remains in use.

For a paired comparison, launch both scenarios with the same training seed:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml --seed 42 --no-wandb
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain_frenet.yaml --seed 42 --no-wandb
```

Repeat the pair with training seeds 43 and 44. Both scenarios retain evaluation
seed 10042 and the same eight evaluation episodes. Compare completion rate,
collision rate, progress, and mean finish steps in `evaluation_history.jsonl`;
finish steps times 0.01 s is the elapsed three-lap race time for successful
episodes, not a measured flying-lap time. Also report collected transitions
and wall time: equal episode budgets need not yield equal transition counts.
These are same-track results, and the checkpoint-selection episodes are not
an independent final test set. Short smoke tests do not measure performance.

Train the Frenet variant from scratch. Its output directory uses the distinct
experiment name. A receiving MAPPO actor needs the same 158-value observation
config and LeakyReLU architecture; the baseline's 115-input checkpoints cannot
initialize it. Changing the base scenario later also changes the included
variant, so retain both run snapshots and their source revision for comparisons.

The racer observation remains 115 values: 108 LiDAR ranges, body-frame
`[vx, vy, yaw_rate]`, normalized progress and cross-track distance, and two
previous-action values. The shared ego-state wrapper now reads yaw rate from
the environment's separate `angular_velocity` field in rad/s. Previously it
padded the two-value `velocity` field with zero, so yaw rate was unavailable to
the policy. This correction also affects other PPO/MAPPO observation configs
that enable ego velocity. The sensor scales and vector layout are unchanged.

Treat checkpoints and datasets produced before the yaw-rate correction as a
different observation contract, even though their dimensions match. Shape
checks alone cannot detect this semantic difference; use the original code
revision to reproduce old policies, and retrain for the corrected inputs.
Receiving MAPPO actors must use the same corrected observation implementation.

This scenario explicitly records the previously effective vehicle parameters
under `environment.vehicle_params`, including steering bounds ±0.4189 rad,
speed bounds ±20 m/s, `a_max: 2.0`, and `v_switch: 0.8`. Its unused top-level
vehicle include was removed. These declarations preserve the physical model
and action bounds; the yaw-rate correction is the behavioral change in this
stage. Reverse prevention still clips negative target-speed commands to zero.

The pretraining network and optimizer settings follow Section III-E of
[On learning racing policies with reinforcement learning, v2](https://arxiv.org/abs/2504.02420v2):
actor `[256, 256]`, critic `[512, 512]`, LeakyReLU with negative slope 0.2,
and minibatches of 1024. PPO's optional `lr_schedule: linear` interpolates
`learning_rate: 0.001` to `learning_rate_end: 0.0001` using globally completed
episodes divided by the episode budget. Updates use the rate set after the
latest episode-end event; the final episode-end event sets the endpoint.
The parent optimizer owns this schedule with parallel collectors, and
`train/learning_rate` records the rate actually used for each update.
Omitting the schedule retains a constant learning rate.

This is an episode-based adaptation, not the paper's 120-million-step budget;
the paper does not specify the decay curve. Collection remains four workers
with at most 2048 pooled transitions, so early episode ends can produce batches
smaller than 1024. PPO epochs, GAE lambda, clipping, and loss coefficients retain
the existing defaults. At the current 0.01-second decision interval,
`gamma = 0.99 ** (0.01 / 0.05)` matches the paper's physical discount horizon.
This does not also match GAE's trace horizon. Recompute gamma if the decision
interval changes. These joint changes form a new training condition; they do
not isolate the effect of any single hyperparameter.

New checkpoints require a receiving MAPPO actor with `pi_hidden_dims: [256, 256]`
and `activation: leaky_relu`. Existing tanh MAPPO scenarios and old checkpoint
references remain configured for their original actors; configure a matching
MAPPO experiment when transferring a newly trained actor. Critic size does not
need to match because the PPO critic is not transferred.

To initialize a MAPPO shared actor, set the same checkpoint parameter for every
trainable agent (their shared policy parameters must match). Relative paths are
resolved from the scenario file; for each trainable agent:

```yaml
agents:
  car_0:
    params:
      pretrained_actor_checkpoint: ../outputs/ppo_lap_completion_pretrain/<run-id>/best_model.pt
```

Only actor weights transfer. MAPPO retains a newly initialized centralized
critic and a fresh optimizer. Loading fails if the observation dimension,
action dimension or bounds, hidden layers, activation, or actor state shapes
do not match. The source path and SHA-256 digest are recorded in run
provenance.
