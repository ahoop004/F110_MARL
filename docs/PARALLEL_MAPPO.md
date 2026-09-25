# Parallel MAPPO on the HPC

The matched `mappo_2v2_base_{scratch,pretrained}.yaml` and
`mappo_2v2_penalties_{scratch,pretrained}.yaml` pairs default to 400 independent races
across 100 spawned CPU workers. Each race has two trainable teammates and two
racing MPC opponents. One parent process owns the GPU actor, critic, and
optimizer. The two-trainable-team scenario also defaults to 400 races and 100
workers, with a separate actor, critic, and optimizer per team; see below.
Other MAPPO scenarios retain their serial defaults.

## Two trainable teams

```bash
python run.py --scenario scenarios/mappo_2v2_selfplay.yaml --no-render

# Bounded 4-environment / 2-worker check before using the full allocation.
python run.py --scenario scenarios/mappo_2v2_selfplay.yaml \
  --num-envs 4 --num-workers 2 --total-steps 1043 --max-steps 16 --no-wandb
```

Self-play uses the same 120M aggregate joint-step budget and 1,024,000-step
evaluation/checkpoint cadence as continuous driving. Its races still finish
after three laps or the finite deadline. Four cars are trainable, so each joint
step produces up to four actor samples. At the default 256-step horizon, a full
round produces up to 204,800 samples per team. Both teams are updated in the
parent while workers wait; workers never own GPU policies or log to W&B.

The self-play collector preserves joint team returns after crashes, including
critic-only targets once both teammates become inactive. Rollout cuts continue
the current race and bootstrap each team independently. Episode/budget counters,
seeds, and remainder allocation are per environment; the parent reports aggregate
steps, samples, throughput, and peak memory. Full details and metric names are in
[the scenario guide](../scenarios/README.md#two-trainable-teams).

The sections below describe the fixed-opponent scenarios.

Activate the project's Python environment and run from the repository root,
inside the existing Slurm GPU allocation. Do not launch one trainer per CPU.

```bash
python run.py --scenario scenarios/mappo_2v2_base_scratch.yaml \
  --no-render --run-id mappo_400_base_scratch_s42
```

For PPO actor transfer, use the matching pretrained scenario and supply the
checkpoint location on the HPC:

```bash
python run.py --scenario scenarios/mappo_2v2_base_pretrained.yaml \
  --pretrained-actor /path/to/ppo/best_model.pt \
  --no-render --run-id mappo_400_base_pretrained_s42
```

Run the base pair first, then substitute the penalty scenario names for the
second phase. Both pairs share the collection settings below, but retain different
reward and race-horizon protocols.

Add `--wandb` to enable W&B. Startup progress and collection/update timing are
printed even without W&B. Logs and checkpoints use the existing output layout.

## Bounded checks

First exercise a small process pool, or run the second command to test all 400
environments with a short aggregate step budget. These commands test collection
and shutdown without changing the continuous race termination rules.

```bash
python run.py --scenario scenarios/mappo_2v2_base_scratch.yaml \
  --num-envs 4 --num-workers 2 --total-steps 256 \
  --no-wandb --no-render --run-id mappo_parallel_smoke

python run.py --scenario scenarios/mappo_2v2_base_scratch.yaml \
  --total-steps 204800 --no-wandb --no-render \
  --run-id mappo_400_smoke
```

The base pair uses **120,000,000 aggregate joint environment decisions**, counting
one step of a race once regardless of whether one or two learners remain active.
Actor samples are recorded separately (up to 240M for that budget). The budget is
split across environment indices, including exact remainders, and continues
across resets. A budget ending mid-race bootstraps the critic and does not invent
a collision, truncation, or completed episode.

Base training has no lap finish or artificial time limit. It resets when both
learners have crashed; a surviving teammate continues, with crashed cars retaining
the existing stationary collision behavior. Rewards use signed metre progress
or an exclusive -1 on collision termination, shared as a fixed two-learner mean.
There are no finish bonuses, time costs, timeout penalties, or progress clipping.
Unlike PPO time trials, the multi-car setup uses physical collisions as its failure
signal; geometric track-limit time trials remain single-car only.

Base evaluation restores lap termination at **20 laps**, with a 120,000-step cap
(6,000 simulated seconds) to accommodate the longer races on Budapest and circle.
Checkpoint selection remains `team_completion`, and the separate 20 final-test
starts remain unchanged. Episode count and target lap count are different settings.

Penalty scenarios retain their original 5,000 aggregate episodes, three-lap races,
16,000-step race limit, and reward/selection objectives. MAPPO supports either
`experiment.total_steps` or an episode budget; `--total-steps N` overrides the step
budget, while `--episodes N` selects episode mode and removes a configured step
budget. These two CLI options are mutually exclusive.

## Collection and optimization

Settings are shared in `configs/training/mappo_parallel.yaml`:

- `experiment.num_envs: 400`: independent race states.
- `experiment.num_workers: 100`: four races per worker. Workers are capped at
  `num_envs`; uneven divisions are supported. Override with `--num-workers`.
- `training_defaults.rollout_steps_per_env: 256`: joint environment decisions per
  round, including opponent-only steps after the learners become inactive.
  This is separate from serial MAPPO's `n_steps` and PPO's pooled `n_steps`.
- `training_defaults.batch_size: 2048`: optimizer minibatch size, with the
  existing 10 epochs. At most 204,800 actor samples are gathered in a full round;
  inactive learners produce no actor samples.
- Evaluation and periodic checkpoints run every 1,024,000 aggregate environment
  decisions, at the next completed update. `final_model.pt` is saved on normal
  completion. Short smoke tests may not reach the first evaluation or produce
  `best_model.pt`.

The policy stays fixed while every worker collects its round. Episodes reset
independently; their fragments are bootstrapped before being pooled. Joint team
credit continues after an individual teammate finishes or crashes, following
the existing finite-race terminal semantics. Rollout cuts bootstrap the critic;
completed races do not leak returns into the next reset. Action samples retain
their raw pre-tanh actions for PPO probability ratios.

Workers own CPU buffers and controllers, with BLAS, OpenMP, NumPy-related thread
pools, PyTorch, and Numba limited to one thread where supported. Worker startup
is batched (eight processes at a time); failures and timeouts terminate the
collector group. Each environment uses base seed plus environment index.
The existing `worker_id` field in episode/transition hook metadata identifies
that environment index; the process index is `worker_id % num_workers`.
Sampling is repeatable for a fixed worker configuration; changing worker count
can change the order of stochastic policy draws.

## Reading performance

Each round prints `collect`, `update`, `samples`, and `env_steps/s`. W&B also
receives `perf/collection_env_steps_per_second`,
`perf/round_env_steps_per_second`, `train/environment_steps`, and
`train/agent_steps`. Round throughput includes collection and optimization;
evaluation/checkpoint hook time is excluded. Collection time includes any
episode logging performed during collection.

Compare 64, 100, and 128 workers using the same 400 environments, seed, rollout
horizon, and race limit. MPC opponent solves and simulation remain CPU work;
400 environments do not imply 400 simultaneously executing CPU cores. The full
400-environment throughput and memory use must be measured on the HPC.

## Asymmetric training and readiness scheduling

The asymmetric 2v2 scenario resets training when both learners terminate and
keeps full-race evaluation. Both fixed-opponent MAPPO and PPO can use
`--collector-scheduling ready` to serve ready workers without a global per-action
barrier; weights remain frozen until the rollout update barrier. This changes
stochastic draw ordering. See [collector performance](COLLECTOR_PERFORMANCE.md)
for settings, timing definitions, and the shared benchmark.
