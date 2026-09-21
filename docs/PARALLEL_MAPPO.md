# Parallel MAPPO on the HPC

The matched `mappo_2v2_base_{scratch,pretrained}.yaml` and
`mappo_2v2_penalties_{scratch,pretrained}.yaml` pairs default to 400 independent races
across 100 spawned CPU workers. Each race has two trainable teammates and two
racing MPC opponents. One parent process owns the GPU actor, critic, and
optimizer. Other MAPPO scenarios retain their serial defaults.

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
second phase. Both pairs share the collection settings below.

Add `--wandb` to enable W&B. Startup progress and collection/update timing are
printed even without W&B. Logs and checkpoints use the existing output layout.

## Bounded checks

First exercise a small process pool, or run the second command to test all 400
environments with a bounded race length. The short race limits change the task;
these commands are infrastructure smoke tests, not learning comparisons.

```bash
python run.py --scenario scenarios/mappo_2v2_base_scratch.yaml \
  --num-envs 4 --num-workers 2 --episodes 8 --max-steps 32 \
  --no-wandb --no-render --run-id mappo_parallel_smoke

python run.py --scenario scenarios/mappo_2v2_base_scratch.yaml \
  --episodes 400 --max-steps 512 --no-wandb --no-render \
  --run-id mappo_400_smoke
```

The full defaults use 5,000 aggregate episodes, divided across environments;
this is not 5,000 episodes per environment. MAPPO still uses an episode budget;
`experiment.total_steps` remains PPO-only. Environments stop after their assigned
episode quota, so the number collecting can decline near the end of training.

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
