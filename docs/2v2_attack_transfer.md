# 2v2 racer and transferred attacker

`scenarios/mappo_2v2_attack_transfer.yaml` trains two rank-4 LoRA adapters
against two fixed racing MPC opponents. `car_0` is the racer; `car_1` is the
attacker. Both use the exact frozen driving base embedded in a trained 1v1
LoRA checkpoint. The racer adapter starts with zero effect. The attacker
adapter and its exploration parameters are imported and continue learning.
The racer exploration parameters use `log_std_init`. The four-car centralized,
agent-conditioned critic and optimizer start fresh. Returns and rewards are
individual, with per-learner advantage normalization.

## Source checkpoint

Set the source to an actual trained `mappo_1v1_attack_lora` checkpoint:

```yaml
training_defaults:
  adapter_transfer:
    checkpoint: /absolute/path/to/1v1/best_model.pt
    source_agent: car_0
    target_agent: car_1
```

The checked-in relative path `../outputs/mappo_1v1_attack_lora/best_model.pt`
is a placeholder. Relative paths resolve from `scenarios/`. There is no
scratch fallback: a missing or incompatible checkpoint fails explicitly.
A separate PPO baseline file is unnecessary because the LoRA checkpoint
contains the frozen base. Physics, action semantics, architecture, rank,
alpha and the original observation scales must match.

The saved 2v2 checkpoint contains both adapters, their observation contracts,
the frozen base, exploration parameters, critic and optimizer. Standalone
evaluation does not need the original 1v1 checkpoint. The existing CLI supports
MAPPO checkpoint evaluation, but does not provide training resumption through
`--checkpoint`; adapter transfer starts a new training run.

## Observations and transfer

Both learners receive the original 158 driving inputs (108 LiDAR plus 50
vehicle/track values), followed by 34 neighbor inputs: three slots of relative
longitudinal/lateral positions and velocities, presence, team membership and
four-way vehicle identity, plus four-way ego identity. Neighbor slots follow
nearest absolute wrapped track distance; identity stays explicit when slots
reorder. Removed cars have no present neighbor slot.

The racer therefore has **192 inputs**. The attacker also receives the original
five target-relative inputs and a four-way selected-target identity, for
**201 inputs**. An absent target produces zeros for all nine target inputs.

Transfer explicitly maps the old attacker's driving and target columns to
their new positions. Added adapter input columns are zero initially, preserving
its outputs for matching original inputs. Additional frozen base input columns
are zero too. Each first-layer adapter has its learner's actual input width;
rollout storage and inference pad to 201 internally. Padding is checked and
never exposed as an observation feature. Inference groups each learner's rows
across environments, retaining batched collection and PPO updates.

## Targets, rewards and lifecycle

The attacker selects the nearest active opponent with positive signed wrapped
longitudinal separation. Finish-line wrapping uses the shortest signed track
distance. Teammates and cars behind are excluded; equal-distance ties use ID.
Selection updates every physics decision. The decision's selected target owns
its interaction credit even if the next observation selects another car.
Each opponent has separate recent-interaction and pending-survival history.
Switches and respawns do not earn shaping rewards.

| Learner | Reward |
|---|---|
| Racer | +1 per signed metre, +10 on completing five laps, exclusive −20 collision/boundary failure |
| Attacker | +10 per survival-qualified opponent crash, +0.01 per signed metre, exclusive −20 collision/boundary failure |

Attack qualification retains the 1v1 rule: moving forward above 0.5 m/s,
within 3 m of the selected opponent during the last second, then surviving
0.5 s after its collision or boundary exit. Unrelated opponent crashes do not
earn credit. Pending credit is cancelled by attacker failure and discarded
when its trajectory ends. The checkpoint imports behavior, not optimizer,
reward history or critic state.

Each learner finishes at **five laps**. Collision or boundary exit ends that
learner immediately, with one terminal failure penalty. Finished and failed
learners are removed from collisions, LiDAR, neighbor observations and rendering;
no further transitions or penalties are collected for them. The survivor keeps
driving. The episode ends when both learners are done, including mixed
finish/failure outcomes. There is no physics-step timeout.

Both opponents keep driving beyond five laps and respawn after boundary exits
or collisions at the nearest unoccupied centerline point, aligned with the
track and initially stationary. Their controllers reset; relocation earns no
lap or progress credit. Removed learners remain removed after opponent respawns.
Episode resets sample all four cars independently along the centerline with
3 m minimum physical separation. Training continues to `experiment.total_steps`.

Training and evaluation use these same rules. Without a timeout, evaluation
can continue indefinitely if a learner stalls without failing or finishing.
Checkpoint selection prioritizes racer completion, both learners finishing,
fewer learner failures, attack score, then racer finish time/progress. Attack
score is `(successes - 2 * attacker_failures) / scheduled_attacker_laps`, retaining
the full five-lap budget for early failures. Reports include both learners'
completion/failure facts and attack metrics even though the racer is focal.

## Run and benchmark

Local inspection (replace the checkpoint path):

```bash
venv/bin/python run.py --scenario scenarios/mappo_2v2_attack_transfer.yaml \
  --set 'training_defaults.adapter_transfer.checkpoint="/absolute/path/to/1v1/best_model.pt"' \
  --render --num-envs 1 --no-wandb
```

Headless collection on a 128-core allocation, with some CPU capacity left for
the parent process and evaluation:

```bash
PYGLET_HEADLESS=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python run.py --scenario scenarios/mappo_2v2_attack_transfer.yaml \
  --set 'training_defaults.adapter_transfer.checkpoint="/absolute/path/to/1v1/best_model.pt"' \
  --num-envs 128 --num-workers 112 --torch-threads 1 --no-render
```

The scenario requests CUDA; the normal device resolver handles unavailable
CUDA. Set both learners' `params.device` consistently for an explicit CPU run.
Worker count is a starting configuration, not a measured optimum for your HPC.
Compare layouts with the existing benchmark, keeping environment count and
rollout horizon fixed and disabling checkpoint evaluation for timing:

```bash
venv/bin/python scripts/benchmark_collectors.py \
  --scenario scenarios/mappo_2v2_attack_transfer.yaml \
  --set 'training_defaults.adapter_transfer.checkpoint="/absolute/path/to/1v1/best_model.pt"' \
  --set evaluation.enabled=false \
  --num-envs 16 --workers 8 16 --scheduling synchronous \
  --rollout-steps-per-env 128 --total-steps 8192
```

Evaluate a saved team checkpoint with `--eval --checkpoint /path/to/team/best_model.pt`
and `--eval-protocol final`. Evaluation needs the same scenario configuration
used for training; its original adapter source file may be absent.

## Local validation

A short CPU throughput check used two environments on `circle_map`, the full
256/256 actor and 512/512 critic, and the scenario's two full racing MPC
controllers. It ran 64 joint decisions, excluding the first collection/update
round from timing (48 measured decisions over three rounds):

| Workers | Collection decisions/s | Collection + update decisions/s |
|---|---:|---:|
| 1 | 16.0 | 14.6 |
| 2 | 24.4 | 21.3 |

This used the available pretrained driving base with a temporary untrained
adapter checkpoint to exercise the full transfer path. It measures execution,
not attack skill or expected HPC scaling. Checkpoint evaluation was disabled;
its cost is additional. Full 1v1 trained-adapter evaluation awaits the selected
source checkpoint.

The existing `test_team_support.py::test_scenarios_match_task_physics_and_training_protocol`
assertion has an unrelated pre-existing failure: the older full asymmetric
scenario specifies opponent MPC speed 20, while its LoRA counterpart specifies
5. Those files are untouched; this scenario specifies 5 for both opponents.
