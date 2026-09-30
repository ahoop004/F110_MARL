# Independent passing and defending adapters

`scenarios/mappo_1v1_pass_lora.yaml` and
`scenarios/mappo_1v1_defend_lora.yaml` each train one MAPPO learner against a fixed
racing MPC. They initialize independently from `outputs/pretrain/best_model2.pt`.
Base network weights, biases and the output head remain frozen; rank-4 LoRA,
exploration `log_std`, and a fresh centralized critic learn. The effective learning
rate is `agents.car_0.params.learning_rate: 0.0001`.

Both hidden linear layers receive LoRA. The first layer is necessary because the
five appended target-relative inputs have zero frozen base weights. The original
158 driving inputs, physics, and wheel-acceleration actions retain their pretrained
contracts. A zero adapter has the base policy's action distribution.

## Train

From the repository root:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_1v1_pass_lora.yaml --no-wandb
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_1v1_defend_lora.yaml --no-wandb
```

Use `--pretrained-actor outputs/YOUR_COMPATIBLE_BASE` to replace the source, and
`--output-dir outputs/YOUR_NEW_RUN` to choose the output directory. Each scenario
defaults to 20 million joint environment decisions. `--num-envs 8` enables parallel
collection; optionally add `--set experiment.num_workers=4`. Serial and parallel
rollout sizes retain the existing MAPPO conventions, so compare experiments at
matched collection settings as well as decision budgets.

## Tasks and rewards

There are no respawns or lap-completion terminals in the tactical episodes.
Ego collision or boundary exit is a failure. Opponent collision or boundary exit
ends the attempt without success credit. Failure takes priority over simultaneous
success. Opponent failure counts as an unsuccessful trial, including in curriculum
evaluation.

| Task | Start | Success | Horizon |
|---|---|---|---|
| Pass | Ego behind MPC | Lead by at least 1 m for a continuous second while moving forward above 0.5 m/s | 30 s / 600 steps |
| Defend | Ego ahead of MPC | No completed opponent pass; finish at least 1 m ahead with at least 40 m earned forward progress | 20 s / 400 steps |

For defense, the opponent completes a pass by leading by at least 1 m for one
continuous second. Brief order changes reset that timer. Pulling away is a valid
defense. Stopping earns no standing lead reward and cannot satisfy the progress
requirement.

Ordering uses the initial signed arc separation plus each car's signed earned
metre progress. It does not use shortest wrapped target distance after reset, so
finish-seam crossings, reverse motion, and passing the half-track separation do
not create false overtake events.

Both tasks give `0.1 × ego metre progress` and an exclusive `−20` ego-failure
penalty. Passing adds `0.5 × signed relative metre progress` and a one-time `+10`
success bonus. Defense adds `+0.2` per second ahead while moving forward, `+10`
on success, and `−10` for a confirmed lost lead. All coefficients live under
`agents.car_0.reward.reward.skill`.

Skill success, lost lead, failed defense, and opponent failure are true task
terminals. Pass timeout, or an explicitly shortened safety time limit, is a
truncation and retains critic bootstrapping. The defense decision at 20 seconds
is a task terminal, taking precedence over the matching time limit. Task terminals
use `task_complete` lifecycle status and `skill_success`/`skill_failure` reasons;
they award neither completed laps nor race finish positions.

## Automatic curriculum

`environment.skill_task` defines the skill, `ego_id`, `target_id`, `lead_margin`,
`confirmation_s`, `moving_speed`, `duration_s`, and `min_progress`.
`skill_curriculum.stages` defines the following reset distributions:

| Stage | Maps | Pass gap / MPC cap | Defense lead / MPC cap | Lateral offsets | Initial speeds |
|---|---|---|---|---|---|
| Basic | Circle | 2–4 m / 2 m/s | 3–5 m / 3 m/s | 0 m | 2 m/s |
| Close interaction | Circle | 2–6 m / 3–4 m/s | 1.5–3 m / 4–4.5 m/s | ±0.25 m | 2–3 m/s |
| Generalization | Eight maps | 2–8 m / 4–4.5 m/s | 1–4 m / 4.5–5 m/s | ±0.4 m | 2–4 m/s |

Maps are Budapest, circle, Melbourne, Montreal, Shanghai, Silverstone, Spa and
Spielberg. Track location is uniform by arc length; ranges are uniform draws per
episode. Lateral offsets and initial speeds are drawn independently for both
cars. Conservative circumscribed-footprint clearance checks reject unsafe draws;
256 unsuccessful attempts produce an explicit error. Explicit reset seeds
reproduce map selection, starts and opponent speed. Task spawning is authoritative;
legacy scalar spawning continues to work in other scenarios.

The parent advances after two consecutive evaluations with current-stage success
at least 80%, ego failures at most 10%, earlier-stage success at least 80%, and
passing solo retention. Stage updates reach parallel workers at update barriers
and take effect on their next reset. Existing episodes retain their stage. Each
reset rebuilds the MPC's physical limits and clears its planning memory after
sampling its speed cap. Training continues to its configured budget after the
final stage qualifies.

## Evaluation and artifacts

Every 409600 decisions, deterministic selection evaluates **all stages**, with
20 circle trials in each of the first two stages and five trials per map in the
third: 80 tactical trials. Stage success/failure rates have equal weight, regardless
of training stage. Checkpoints are ranked by success rate, fewer ego failures,
then faster successful passes or greater defense progress.

Solo retention adds five episodes per map: one full lap, up to 16000 steps, zero
target inputs, collision/boundary termination, and the same seeds for base and
adapter. Completion must drop by at most five percentage points in aggregate.
Among paired clean completions with measured full-lap times, the ratio of mean
adapter lap time to mean base lap time must be at most 1.10. No paired measured
clean laps fails this gate. Per-map and per-episode facts remain in the reports.

The frozen base is reconstructed from the checkpoint network with zero LoRA
residuals and evaluated deterministically. Base measurements are cached in memory
for the fixed protocol and written to `skill_base_selection.json`; subsequent
selection evaluations reuse them. Evaluation preserves training RNG state.

Only retention-qualified candidates create/update `best_model.pt`. Qualification
does not by itself assert tactical competence: inspect skill success and curriculum
completion too. `final_model.pt` and periodic `checkpoint_step*.pt` remain available
even when no candidate qualifies. These checkpoints contain the base, adapter,
critic, optimizer, source hash, and curriculum stage/streak/evaluation state; the
original PPO file is unnecessary for evaluation. The existing CLI does not provide
exact interrupted-run MAPPO resumption.

`evaluation_history.jsonl` records selection and curriculum decisions;
`race_metrics.jsonl` includes per-episode skill outcomes, progress, lead retention,
pass time, and stage. W&B exposes `episode/skill/*`, `eval/skill_*`,
`eval/retention_*`, and `eval/curriculum_*` under the usual logging controls.

Run isolated final tests for **each** trained adapter:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_1v1_pass_lora.yaml --eval --eval-protocol final \
  --checkpoint outputs/YOUR_PASS_RUN/best_model.pt --output-dir outputs/pass_final --no-wandb
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_1v1_defend_lora.yaml --eval --eval-protocol final \
  --checkpoint outputs/YOUR_DEFEND_RUN/best_model.pt --output-dir outputs/defend_final --no-wandb
```

Each final report evaluates that adapter and its frozen base on passing,
defending, and solo driving. Final testing doubles all episode counts, uses
separate seeds, and never selects checkpoints or advances curricula. The other
skill's benchmark comes from its sibling scenario, with the evaluated policy's
physics/observation/action contracts preserved. Together the two reports compare
both adapters and the base on both tasks. Base results appear in both reports for
paired comparisons. Reports are `evaluation_report.json` and `skill_base_final.json`.

Selection seed blocks start at `evaluation.seed` plus 0 (pass), 100000 (defend),
and 200000 (solo). Final blocks use the same offsets from `evaluation.final_test.seed`.
Configuration validation checks disjointness. `--eval-episodes` does not apply to
these stratified suites; configure each stage's `evaluation_episodes_per_map` and
retention's `episodes_per_map`, keeping the aggregate selection/final tactical
episode counts consistent. Such reduced protocols are execution checks, not final
performance measurements.

## Smoke checks

These commands use the actual configured pretrained actor, produce checkpoints,
and keep the expensive benchmark disabled:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_1v1_pass_lora.yaml \
  --total-steps 16 --output-dir /tmp/f110-pass-smoke --no-wandb \
  --set agents.car_0.params.device=cpu --set agents.car_0.params.n_steps=8 \
  --set agents.car_0.params.n_epochs=1 --set environment.max_steps=4 --set evaluation.enabled=false
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_1v1_defend_lora.yaml \
  --total-steps 16 --num-envs 2 --output-dir /tmp/f110-defend-smoke --no-wandb \
  --set agents.car_0.params.device=cpu --set agents.car_0.params.n_epochs=1 \
  --set training_defaults.rollout_steps_per_env=4 --set environment.max_steps=4 --set evaluation.enabled=false
PYGLET_HEADLESS=true venv/bin/python -m pytest tests/test_skill_adapters.py -q
```

The tests include short real training and standalone evaluation in serial and
parallel modes, checkpoint self-containment, frozen base tensors, target input
learning, termination semantics, curriculum boundaries and retention rejection.
Prior to this work, the focused LoRA/target-observation baseline was 32 passed and
two failures referencing the absent `ppo_1v1_racing_mpc_circle.yaml`; those missing
legacy fixtures are independent of these scenarios. Training effectiveness still
requires the full paired benchmark.
