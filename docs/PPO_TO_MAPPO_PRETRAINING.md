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

Checkpoint selection uses eight fixed episodes with seeds 10042–10049. Final
testing uses a separate 20-episode set with seeds 20042–20061, configured under
`evaluation.final_test`. Both inherit `environment.max_steps`: currently 80000
physics steps, or 800 seconds at 0.01 s per step. The earlier selection limit
was 400 seconds; selection results across this change are different protocols.
An explicit `evaluation.max_steps` overrides the horizon for both fixed sets.
Overlapping selection/final seed ranges are rejected during scenario validation.

After choosing a checkpoint, run its final test explicitly:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain_frenet.yaml \
  --eval --eval-protocol final \
  --checkpoint outputs/ppo_lap_completion_pretrain_frenet/<run-id>/best_model.pt \
  --output-dir outputs/ppo_lap_completion_pretrain_frenet/<run-id>/final-test \
  --no-wandb
```

Use the same scenario and training overrides as the checkpoint's run (for
example `--seed 43` for a policy trained with seed 43). Named evaluation
protocols apply their own seeds to a runtime copy after retaining the training
configuration for the provenance comparison. `--eval-protocol selection`
replays the selection set. Omit `--eval-episodes` for either fixed protocol;
plain `--eval` retains the existing custom episode/seed behavior.

`evaluation_report.json` contains the checkpoint SHA-256, provenance, exact
seeds, horizon, summary, and per-episode results. Completion, collision, and
timeout rates are reported alongside `mean_clean_finish_time_s` and its sample
count. Time measures elapsed physics steps through a clean completion, including
the first step; no clean finishes produces `null`. This is full three-lap race
time from reset, not a flying-lap measurement. The existing finish-step ranking
is unchanged. For teams, clean-finish time pools individual trainable finishers;
it is not the time for the whole team to finish.

Final tests never update `best_model.pt` and are not run by the training hook.
Keep the final set out of tuning decisions; once used to choose configurations,
it is no longer an independent test set. These seed sets sample reset conditions,
not independent training runs. Older checkpoints may fail the strict scenario
hash check after this config change; use the original run configuration for
reproduction, or explicitly allow a provenance mismatch for a documented
cross-protocol evaluation. Action-contract checks still apply.

The current scenario uses `circle_map` for both splits. A held-out generalization
experiment requires explicit disjoint training and evaluation maps.

The enhanced Frenet scenario is `scenarios/ppo_lap_completion_pretrain_frenet.yaml`.
It includes the base pretraining scenario and adds vehicle/Frenet observations,
an integrated speed reference, and stronger vehicle acceleration limits.
Maps, spawn protocol, training/evaluation seeds, rewards, PPO hyperparameters,
and episode budget remain matched. This is now a combined observation/control/
dynamics experiment, not an observation-only ablation.

With `action_constraints.speed_control: acceleration`, the normalized second
action is scaled by `max_acceleration: 5.0` for positive inputs and
`max_deceleration: 5.0` for negative inputs, in m/s². Once per decision:
`v_ref = clip(v_ref + acceleration * timestep * action_repeat, 0, 20)`.
Zero holds the current reference; negative inputs brake. The stored reference
is clamped too, preventing windup at either limit. Steering retains its physical
angle interpretation. The environment still receives `[steering, target_speed]`.

Each collector and MAPPO agent owns its own integrator, reset to 0 m/s at every
episode, matching the environment's reset command reference. Training, checkpoint
evaluation, and CLI evaluation use the same decision interval and reset behavior.
Dataset normalized actions contain acceleration commands; physical actions
contain the resulting speed references. The transition schema is unchanged.
Checkpoints store the action contract, including rate limits and decision interval;
loading or PPO-to-MAPPO transfer rejects incompatible control semantics even when
tensor dimensions and environment bounds match. Legacy checkpoints without this
metadata are treated as direct-speed policies.

The Frenet vehicle uses `a_max: 5.0` and `v_switch: 20.0`, removing the earlier
`1.6/v` acceleration taper throughout its allowed forward speed range. This is
an acceleration cap, not a promise of 5 m/s² realized acceleration: the existing
speed controller and vehicle dynamics still determine the actual response.
The observed speed-reference rate is zero for a hold decision and preserved
across that decision's repeated physics steps.

| Scenario | Actor observation | Dimension |
|---|---|---|
| `ppo_lap_completion_pretrain.yaml` | LiDAR, ego motion, progress/deviation, previous action | 115 |
| `ppo_lap_completion_pretrain_frenet.yaml` | LiDAR, normalized vehicle/Frenet state, curvature and width preview | 158 |

The Frenet variant extends `rl_racer_vehicle_track_frenet.yaml` through
`rl_racer_vehicle_track_frenet_acceleration.yaml`: 108 LiDAR
values, 10 vehicle/Frenet values, 20 curvatures, and 20 widths. It replaces the
baseline's seven non-LiDAR values; this compares complete observation
representations, not just the addition of track preview. Normalization and
clipping also differ. Hidden-layer widths match, but the larger input makes
the first actor/critic layers larger. Wheel-speed values are simulator proxies
derived from longitudinal speed and a configured wheel radius, not measured
wheel rotation. Reference-rate normalization is now 100 rad/s², corresponding
to the 5 m/s² action limit divided by the configured 0.05 m wheel radius.
This changes observation scaling but retains the 158-value layout.

To compare the combined changes against the baseline, use the same training seed:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml --seed 42 --no-wandb
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain_frenet.yaml --seed 42 --no-wandb
```

Repeat the pair with training seeds 43 and 44. Both scenarios retain the same
selection and final-test sets. Compare completion rate, collision rate, timeout
rate, progress, and clean finish time in the final reports. Selection history
remains in `evaluation_history.jsonl`. Also report collected transitions
and wall time: equal episode budgets need not yield equal transition counts.
These are same-track results. Short smoke tests do not measure performance.

Train the Frenet variant from scratch. Its output directory uses the distinct
experiment name. A receiving MAPPO actor needs the same 158-value observation
config, LeakyReLU architecture, and acceleration action contract; earlier
direct-speed Frenet checkpoints and the baseline's 115-input checkpoints cannot
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
the paper does not specify the decay curve. Collector count is controlled by
`experiment.num_envs` (currently one), with at most 2048 transitions per rollout,
so early episode ends can produce batches
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

## Reverse-enabled Frenet transfer to a 2v2 team

`scenarios/mappo_2v2_frenet_ppo_pretrained.yaml` defines two trainable MAPPO
teammates (`car_0`, `car_1`) against the existing fixed hybrid PP+FTG opponents
(`car_2`, `car_3`). It uses a mean shared team reward and a shared centralized
critic. The 158-value local actor input remains unchanged; LiDAR sees traffic,
but no explicit teammate/opponent identity inputs are added.

This variant requires a PPO checkpoint trained with `prevent_reverse: false`.
Its acceleration integrators allow references from -20 to +20 m/s, with signed
reference rates of up to 5 m/s², at a 0.01 s decision interval. Negative action
first reduces a positive reference, then commands reverse after crossing zero.
The earlier forward-only contract described above is incompatible even though
the network dimensions match. The current reverse-enabled pretraining scenario
must be used to generate this initialization; changing only the receiving YAML
cannot convert a forward-only checkpoint.

The new scenario snapshots the vehicle dynamics, actor architecture and action
contract instead of including the entire mutable PPO experiment. Its training
maps are explicitly Budapest and circle, with circle evaluation, three full
laps, and an 80,000-step horizon. MAPPO uses one environment, a fresh
`[512, 512]` centralized critic, a fresh optimizer with constant learning rate
`0.0001`, and the physical discount `gamma = 0.9979919516614258`. The base
scenario now averages the exact local reward from Frenet PPO pretraining.
Reverse commands are permitted; signed progress penalizes travel backward
along the track. Fixed opponents remain unchanged.

The configured initialization path is the completion-selected `best_model.pt`
from run `ppo_lap_completion_pretrain_frenet_ppo_s42_1789067806_2f5d`. Wait for
that PPO run to finish before using its selected checkpoint. If using another
run, change `training_defaults.pretrained_actor_checkpoint` in the new scenario;
the path is relative to `scenarios/`, and its digest is captured in provenance.
The episode-zero checkpoint is useful for an integration smoke test, not as
evidence that the source actor has learned to finish laps. A missing selected
checkpoint fails explicitly; the scenario never falls back to random weights.

```bash
PYGLET_HEADLESS=true python3 run.py \
  --scenario scenarios/mappo_2v2_frenet_ppo_pretrained.yaml --no-wandb
```

MAPPO does not yet run the PPO automatic evaluation-selection hook. Evaluate
periodic MAPPO checkpoints explicitly on the fixed selection seeds, choose one,
and evaluate the chosen checkpoint once on the disjoint final seeds:

```bash
PYGLET_HEADLESS=true python3 run.py \
  --scenario scenarios/mappo_2v2_frenet_ppo_pretrained.yaml \
  --eval --eval-protocol selection --checkpoint <mappo-checkpoint.pt> --no-wandb
PYGLET_HEADLESS=true python3 run.py \
  --scenario scenarios/mappo_2v2_frenet_ppo_pretrained.yaml \
  --eval --eval-protocol final --checkpoint <selected-mappo-checkpoint.pt> --no-wandb
```

Keep any training CLI overrides identical for provenance checks. A MAPPO
`best_model.pt` chosen by training reward is not a completion-selected model.
The fixed opponents retain their 2.5 m/s pace while the learners can command
20 m/s: this is an initial transfer-to-traffic experiment, not a matched-speed
comparison. Evaluation on circle is not a held-out-map result. Before making
competitive claims, calibrate the opponents and compare against scratch MAPPO
with the same team reward, physics, maps, seeds, horizon, and training budget.

### Three team reward variants

The base `mappo_2v2_frenet_ppo_pretrained.yaml` is the mean-PPO-reward control.
Three thin scenario variants add competitive objectives while preserving maps,
seeds, actors, critics, control, opponents, and optimizer settings:

| Scenario suffix | New reward task | Shared additions |
|---|---|---|
| `_combined.yaml` | `race_team_2v2_combined.yaml` | +1 when both teammates finish cleanly, plus normalized finish-rank points |
| `_first_place.yaml` | `race_team_2v2_first_place.yaml` | +2 when a teammate takes first place |
| `_sweep.yaml` | `race_team_2v2_sweep.yaml` | +2 when teammates take first and second |

All three task files include `lap_completion_normalized_progress.yaml`, the
reward used by `ppo_lap_completion_pretrain_frenet.yaml`. The original PPO task
is unchanged: signed progress of approximately +1 per forward lap (clamped to
±0.025 per physics step), +1 clean race completion, -1 collision, -1 timeout,
and -0.0000125 per physics step. For two active teammates these local terms are
averaged; an inactive teammate contributes zero with the denominator fixed at
two. The shared additions are then added once, without dividing a late bonus
by two when only one teammate remains active.

Combined rank points are `((4-position_0)/3 + (4-position_1)/3)/2`; a DNF earns
zero points. Each finisher's contribution is paid as its position becomes known.
First+fourth and second+third tie at 0.5 points; a sweep earns 5/6 points plus
the +1 both-finished bonus. Opponent DNFs rank behind clean finishers. No finish
earns no first-place/sweep bonus, and no collision earns a competitive bonus
by itself. These are additive objectives, not guarantees that a win always
outweighs every possible difference in accumulated local rewards.

CLI reports `team_objective`, `team_result_means`, and `team_results_by_episode`.
For combined results, `success_rate` is both-finished rate, with rank score
reported separately. For the other variants it is first-place or sweep rate.
These explicit results should be used instead of the older generic
`team_win_rate`, which can award a progress-based win without a race finish.

The four Frenet team scenarios opt into `training_defaults.team_return_mode:
joint`. GAE is computed on the common reward/value timeline and gathered only
for real per-agent decisions. Finishing or crashing one car does not end the
team return: a later teammate result contributes to earlier decisions within
the rollout; across rollout cuts, continuation bootstraps through the shared
critic. There are no extra actor decisions for inactive cars. Team returns end
when no learner can act, including a race timeout: this finite-horizon timeout
is terminal for learning. Opponents may finish later for full-race reporting,
but the configured team bonuses are already decided at the last learner exit.
This mode requires shared rewards, a shared team critic, action repeat one,
and all-agent or all-trainable episode termination. Other scenarios retain the
legacy per-agent GAE and truncation behavior.

MAPPO checkpoints record the return mode and reject incompatible loads; PPO
actor-only initialization remains compatible. Dataset schema stays at 2.0 and
records one row per actual learner decision. Terminated/truncated flags remain
the individual car's factual flags, not the shared-return boundary. Metadata
records `team_return_mode` and the shared reward contract. To reconstruct joint
returns, group rows by episode/step, use the shared reward once per joint step,
and use trainable lifecycle masks for the team boundary; do not stop at the
first teammate's termination. Per-row component breakdowns separate local terms
from shared additions, so their sum need not equal the row's averaged reward.

For example, after the PPO checkpoint is available:

```bash
PYGLET_HEADLESS=true python3 run.py \
  --scenario scenarios/mappo_2v2_frenet_ppo_pretrained_combined.yaml --no-wandb
```

Use the same suffix when evaluating its checkpoints. Run each variant with the
same training seeds and compare the factual team result metrics, not raw reward
totals across objectives. Historical runs using the old 2v2 shaped reward or
per-agent return boundaries are a different experimental condition.
