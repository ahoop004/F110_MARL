# 2v2 experiment sequence: base, penalties, then LoRA

Run the base scratch/pretrained comparison first, then the penalty pair, then
LoRA combinations once adapter training is implemented. All four current arms use
the same 68-input observation, vehicle physics, MPC opponents, seed, and 400
environments across 100 workers. Collection, minibatch size, and evaluation
cadence come from `configs/training/mappo_parallel.yaml`. Within each pair only
actor initialization and experiment name differ. The critic and optimizer start
fresh, and all actor parameters are trained in both arms.

The base pair now trains continuously for 120M aggregate joint environment
steps and evaluates at 20 laps. The penalty pair retains the finite three-lap,
5,000-episode protocol until the next experiment phase.

| Stage | Scenario | Initialization | Objective |
|---|---|---|---|
| 1 | `scenarios/mappo_2v2_base_scratch.yaml` | Random actor and critic | Signed metre progress; exclusive collision cost |
| 1 | `scenarios/mappo_2v2_base_pretrained.yaml` | PPO actor; random critic | Signed metre progress; exclusive collision cost |
| 2 | `scenarios/mappo_2v2_penalties_scratch.yaml` | Random actor and critic | Completion, placement, incident penalties |
| 2 | `scenarios/mappo_2v2_penalties_pretrained.yaml` | PPO actor; random critic | Completion, placement, incident penalties |
| 3 | Planned LoRA combinations | To be specified when adapters are implemented | Matched base and penalty tasks |

The base task uses `race_team_continuous_progress.yaml`: signed metre progress
or an exclusive -1 collision-ending reward, averaged over the fixed two-learner
team. This preserves the PPO driving reward scale while using wall/car collisions
as the failure signal instead of single-car geometric boundaries. It has no
progress clipping, completion bonus, time cost, or timeout penalty. Training
resets when both learners crash; a surviving learner keeps driving. Lap count
does not end training, and there is no artificial training time limit.

The base pair inherits `configs/scenarios/mappo_2v2_continuous_base.yaml`. Its
20-lap evaluation restores lap-based finishing with a 120,000-step safety cap
(6,000 simulated seconds) and `team_completion` selection. The larger cap allows
more time than the former three-lap races. Selection still uses eight starts;
final testing still uses twenty separate starts.

The penalty task keeps normalized progress, completion/placement terms, incident
penalties, and its original finite-race horizons. These are now different task
protocols as well as different rewards; compare scratch/pretrained within each
pair, and do not attribute cross-stage differences solely to penalty shaping.

Train the new source with the unchanged 400-environment pretraining scenario:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml --seed 42 \
  --output-dir outputs/ppo_current_pretrain_s42
```

Both pretrained arms default to `outputs/ppo_current_pretrain_s42/best_model.pt`.
They fail if this file is missing. For a different run, pass
`--pretrained-actor outputs/YOUR_RUN/best_model.pt`. Validate the source using
the [pretraining workflow](PPO_TO_MAPPO_PRETRAINING.md#select-and-validate-a-model)
before starting the pretrained arms. Freeze the selected source checkpoint for all pretrained
arms; record its hash and pretraining cost. Do not compare arms initialized from
different updates of a still-running source job.

```bash
# Stage 1: continuous metre-progress driving.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_base_scratch.yaml

PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_base_pretrained.yaml

# Stage 2: placement and incident penalties.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_penalties_scratch.yaml

PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_penalties_pretrained.yaml
```

These scenarios inherit the current shared vehicle profile and fixed track
observation scaling. The first 50 inputs retain their source
semantics, and 18 neighbor/team inputs are appended with zero initial weights
when loading PPO. Both fixed racing MPC opponents use the same 3.5 m/s speed cap
and the environment's 0.58 m by 0.31 m vehicle geometry. Their traffic sensing
uses perfect current simulator states within 10 m; this is privileged sensing.
The downloaded `outputs/L_map_best_model.pt` is a historical artifact and is not
used by this matrix. Results from the previous older-physics A/B pair should be
kept separate; rerun A under this configuration for the new comparison.

## Penalty policy: terminal_incidents_v1

Penalties are factual events, independent of either controller's reward function.
The supported events are `collision_dnf` and `boundary_dnf`, identified from the
immutable terminal reason and terminal step. Each penalized vehicle receives one
positive penalty point. Persistent collision flags, repeated info payloads, later
contact with a finished vehicle, timeouts, negative progress, and ordinary time
costs do not create penalty points. Each reward composer resets its event ledger
between episodes.

This policy scores collision involvement; it does not assign blame or distinguish
wall, teammate, and opponent contacts. A collision terminating two vehicles creates
one penalty for each involved vehicle. That is two vehicle penalties, not repeated
payment for the same vehicle. Boundary DNFs count only when explicitly emitted:
the current 2v2 configuration does not enable geometric boundary detection.
Nonterminal off-track/contact incidents and racing-rule violations remain future
work and must not be inferred from reward signs.

The initial version fixes these proposed weights so training and evaluation share
one definition in `src/metrics/race_penalties.py`:

```
penalty_adjustment = -mean(our penalty points) + 0.25 * mean(opponent penalty points)
```

Means use configured team sizes, including cars that already finished or crashed.
For a two-car team, one own incident contributes -0.5; one opponent incident adds
+0.125. A one-for-one collision therefore has a -0.375 penalty adjustment. This
reduces a direct collision incentive but does not prove that the complete reward
cannot favor contact: placement changes and other terms can still affect the
tradeoff. Review race replays before treating a learned maneuver as desirable.

The new reward inherits signed lap progress, completion bonuses, a physical-time
cost, a timeout penalty, and the combined finishing-position objective. Its local
collision reward is disabled because the shared event ledger now charges that
incident. Reward breakdowns expose `race_penalties/own` and
`race_penalties/opponents`; opponent neural-network rewards are never required.

## Checkpoint selection and reports

The penalty pair uses `team_combined_penalties`, which ranks:

1. Fraction of starts where both learners finish.
2. Mean finish-rank score plus the penalty adjustment.
3. Fewer learner collisions.
4. Faster mean clean race finishes once all starts finish; net progress otherwise.

Thus time is currently a successful-race tie-breaker, plus a training time cost.
A weighted lap-time/placement/penalty competition score has not yet been defined.
Keep that distinction explicit when comparing results.

For all four arms, selection runs after an update every 1,024,000 aggregate
environment decisions using the same eight starting seeds on Budapest and circle. Final evaluation uses the existing twenty separate starts. Reports
include the versioned policy, per-vehicle penalty events (episode, agent, kind,
step, points), mean own/opponent penalty points, mean penalty adjustment, and
`team_rank_penalty_score`. Aggregate and per-map reports use the same calculations
as the training penalty component.

Equal episode counts are not equal sample budgets. MAPPO now supports an exact
aggregate environment-step budget, used by the base pair. Record joint environment
decisions, actual learner samples, and wall time separately; the penalty pair
still uses its original episode budget. Pretraining cost should be reported separately.
The new source must use the corrected L-map finish line and report valid timed
laps. The fixed MPC opponents have completed three-lap solo, traffic, and paired
checks on both training maps under the current physics at nominal grip.

Both A/B arms now include the same [racing MPC](RACING_MPC_OPPONENTS.md) opponents.
Broader seeds, randomized grip, the exact matrix spawn policy, and learned traffic
still need qualification. Keep previous hybrid-opponent results separate and
rerun both arms when comparing this opponent setup.

## Subsequent comparisons

Role-conditioned offensive/defensive observations and reward/value targets, LoRA
adapters, a common weighted race score, and richer incident attribution remain
separate planned changes. Compare each new reward under scratch and full pretrained
fine-tuning before comparing that same reward with LoRA. No role specialization or
LoRA is enabled by the four current scenarios. Adapter placement, rank, frozen
parameters, and any teammate-specific combinations must be specified for stage 3.
