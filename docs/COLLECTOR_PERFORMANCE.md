# PPO and asymmetric MAPPO collector performance

Use the local preset for single-car development. Its 1,024-step rollout avoids
waiting for the historical 409,600-transition pooled batch on one environment:

```bash
python3 run.py --scenario scenarios/ppo_lap_completion_pretrain_local.yaml --no-render
```

The original `ppo_lap_completion_pretrain.yaml` retains its existing configuration
for compatibility: one environment and a 409,600-transition batch. The local
preset deliberately changes optimization frequency. Existing transfer scenarios
also retain their configured batch; use `--rollout-steps-per-env 1024` to resize
one explicitly.

On a 128-core node, the HPC preset starts **400 environments in 100 CPU worker
processes**, with four environments per worker and one parent GPU learner:

```bash
python3 run.py --scenario scenarios/ppo_lap_completion_pretrain_hpc.yaml --no-render
```

It preserves 1,024 transitions per environment, a 409,600-transition PPO update,
1,024-sample minibatches, and 10 epochs. `--num-workers 80` instead assigns five
environments per process. Process count is capped at environment count. Both PPO
and MAPPO collectors limit native-library threads before spawning; no worker owns
a CUDA policy. Environment identity, RNG seed offsets, episode metadata, and
budget remainders are tied to environment index, independently of process count.

## Readiness-based inference

The default `synchronous` scheduler retains ordered inference batches. To remove
the global per-action barrier, add:

```bash
--collector-scheduling ready
```

In ready mode the parent batches requests from workers whose pipes are ready and
responds without waiting for all other workers. This is available for PPO and
fixed-opponent MAPPO. Multiple environments inside one worker still step in
sequence. The learner still waits at the **rollout update barrier**, and actor
and critic weights stay frozen throughout collection. There is no overlap of
training with stale-policy rollouts. Ready scheduling changes random-action draw
order and is not seed-trajectory deterministic across OS schedules. Scheduler
mode is recorded in provenance. Compare learning curves as well as throughput.

The two-trainable-team self-play collector retains its existing scheduler.
Worker-local inference, multi-node Ray execution, and asynchronous evaluation are
not part of this change.

## Asymmetric 2v2

`mappo_2v2_asymmetric.yaml` now ends training races when all trainable cars finish
or crash (`all_trainable`). A surviving learner continues racing. Once neither
learner is active, its per-agent terminal returns are already fixed and no new
learner transitions can be collected. Evaluation explicitly retains `all_agents`
through `evaluation.episode_termination_mode`, including standalone evaluation.
Full-race evaluation outcomes remain available; training episode outcomes and
update cadence change, so start a new experiment when comparing learning curves.

Example on the same node:

```bash
python3 run.py --scenario scenarios/mappo_2v2_asymmetric.yaml \
  --num-envs 400 --num-workers 100 --collector-scheduling ready \
  --rollout-steps-per-env 256 --no-render
```

This retains the asymmetric scenario's optimizer settings. A larger minibatch is
a separate optimization/learning experiment, not silently coupled to workers.
The new rollout CLI option sets PPO's pooled `n_steps = num_envs * horizon`;
for MAPPO it sets the parallel per-environment horizon and serial buffer horizon.

## Measure before increasing the allocation

Run this with the GPU and CPUs reserved exclusively for the benchmark:

```bash
python3 scripts/benchmark_collectors.py \
  --scenario scenarios/ppo_lap_completion_pretrain_hpc.yaml \
  --num-envs 400 --workers 64 80 100 112 \
  --scheduling synchronous ready --rollout-steps-per-env 1024 \
  --total-steps 1228800 --repetitions 3 --output-dir /tmp/ppo_collector_benchmark
```

Every arm uses the same environment count, seed, rollout horizon, optimizer
configuration, and aggregate transition budget. Each run has its own directory,
console log, config/provenance, CSVs, and final checkpoint. `results.json` is
flushed after each completed run. The first collection/update round is excluded
from steady-state rates; total subprocess time includes imports, startup, hooks,
and checkpoint writing. The harness refuses to overwrite existing run folders.
It does not claim identical trajectories across scheduling modes. Scenario
logging/evaluation cadence is retained, so choose the budget accordingly.

Parallel training writes `perf/startup_seconds`, `perf/collection_seconds`,
`perf/update_seconds`, collection and round environment steps/second, and
`perf/num_workers` / `perf/num_envs` to update metrics. The cumulative
`perf/end_to_end_env_steps_per_second` includes startup and prior hook/evaluation
time up to the metric sample; it is sampled before that update's hooks and final
checkpoint. Use subprocess wall time for complete job throughput. Collection
includes episode event processing; the round rate excludes update-hook time.
Evaluation history and W&B eval metrics record `evaluation_seconds` and its
cumulative total separately.

For MAPPO also monitor `train/agent_steps`: environment throughput alone can
count opponent-only steps in scenarios that intentionally retain `all_agents`.
Do not extrapolate local speedups to an HPC node without measuring there.

## Local validation (2026-09-25)

The [saved summary](benchmarks/collector_optimization_local.json) records a Ryzen
5 5600G / RTX 5060 Ti comparison with eight PPO environments, 4,096 transitions,
128 transitions per environment per round, and two repetitions. Medians:

| Workers | Scheduling | Steady environment steps/s | Whole process seconds |
| --- | --- | ---: | ---: |
| 4 | synchronous | 684 | 20.1 |
| 8 | synchronous | 672 | 23.2 |
| 4 | ready | 589 | 21.5 |
| 8 | ready | 509 | 26.3 |

Grouped synchronous workers achieved similar steady throughput using half as
many processes and reduced startup/total time in this small sample. Readiness
scheduling was slower for this light local PPO workload; it remains opt-in.
These measurements do not establish the best layout on the 128-core node.

A separate 512-decision asymmetric scratch-policy probe changed from 182 to 512
decisions with an active learner. Learner transitions increased from 308 to 815;
training-call time changed from 10.46 to 20.29 seconds, giving approximately
29 to 40 useful learner transitions per second. More active-car work, resets,
and three updates instead of one explain why raw environment steps/second is
not the objective here. This single short probe validates removal of the
opponent-only tail; different update timing also changes the subsequent policy,
so it is not a fixed-policy speed or learning-quality comparison.

Focused validation covered 206 tests across PPO, asymmetric MAPPO, grouped
MAPPO, curriculum, observations, and collector scheduling. Grouped PPO matched
ungrouped rollout tensors/returns across resets and a partial final budget.
