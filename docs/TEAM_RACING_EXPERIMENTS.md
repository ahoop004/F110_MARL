# Initial 2v2 comparison: recorded race penalties

The first pair compares full MAPPO training from scratch with actor initialization
from `outputs/L_map_best_model.pt`. Both use the same 68-input observation, vehicle
physics, three-lap races, opponents, reward, seed, and 5,000-episode budget. The
critic and optimizer start fresh in both arms. Only the actor initialization and
experiment name differ.

| Scenario | Initialization |
|---|---|
| `scenarios/mappo_2v2_penalties_scratch.yaml` | Random actor and critic |
| `scenarios/mappo_2v2_penalties_pretrained.yaml` | Downloaded PPO actor; random centralized critic |

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_penalties_scratch.yaml

PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_penalties_pretrained.yaml
```

These scenarios intentionally use the downloaded checkpoint's older physics and
per-map driving-observation scaling. The first 50 inputs retain their source
semantics, and 18 neighbor/team inputs are appended with zero initial weights
when loading PPO. The fixed hybrid opponents use the matching 0.225 m width.
The active current-physics scenarios and the downloaded checkpoint are unchanged.

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

Both arms use `team_combined_penalties`, which ranks:

1. Fraction of starts where both learners finish.
2. Mean finish-rank score plus the penalty adjustment.
3. Fewer learner collisions.
4. Faster mean clean race finishes once all starts finish; net progress otherwise.

Thus time is currently a successful-race tie-breaker, plus a training time cost.
A weighted lap-time/placement/penalty competition score has not yet been defined.
Keep that distinction explicit when comparing results.

Selection runs every 100 episodes using the same eight starting seeds on Budapest
and circle. Final evaluation uses the existing twenty separate starts. Reports
include the versioned policy, per-vehicle penalty events (episode, agent, kind,
step, points), mean own/opponent penalty points, mean penalty adjustment, and
`team_rank_penalty_score`. Aggregate and per-map reports use the same calculations
as the training penalty component.

Equal episode counts are not equal sample budgets. Record environment decisions,
learner samples, and wall time; add a MAPPO transition-budget control before making
strict sample-efficiency claims. Pretraining cost should be reported separately.
The source checkpoint's saved evaluation did not demonstrate lap completion, and
fixed-opponent completion under these physics still needs benchmarking.

## Subsequent comparisons

Role-conditioned offensive/defensive observations and reward/value targets, LoRA
adapters, a common weighted race score, and richer incident attribution remain
separate planned changes. Compare each new reward under scratch and full pretrained
fine-tuning before comparing that same reward with LoRA. No role specialization or
LoRA is enabled by these two initial scenarios.
