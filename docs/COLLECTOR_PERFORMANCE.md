# PPO and asymmetric MAPPO collector performance

Scenario settings are now inline. Select parameter choices in the canonical YAML
file or use `--set KEY=YAML`; there are no scenario inheritance files in `configs`.

Use the local preset for single-car development. Its 1,024-step rollout avoids
waiting for the historical 409,600-transition pooled batch on one environment:

```bash
python3 run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml --set agents.car_0.params.n_steps=1024 --no-render
```

The original `ppo_lap_completion_pretrain.yaml` retains its existing configuration
for compatibility: one environment and a 409,600-transition batch. The local
preset deliberately changes optimization frequency. Existing transfer scenarios
also retain their configured batch; use `--rollout-steps-per-env 1024` to resize
one explicitly.

On a 128-core node, the HPC preset starts **400 environments in 100 CPU worker
processes**, with four environments per worker and one parent GPU learner:

```bash
python3 run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml --set experiment.num_envs=400 --set experiment.num_workers=100 --no-render
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
not part of this change. Fixed-opponent MAPPO checkpoint evaluation can now use
`evaluation.num_workers` for concurrent CPU races with the policy in the parent;
training still pauses until those races finish. See the
[attack HPC guide](1v1_attack.md#hpc-training-and-checkpoint-evaluation).
LoRA skill curricula also support parallel evaluation, sharing one process pool
across stages, baseline comparisons, and solo retention. Their fixed trial counts
and score gates are preserved; see the
[skill HPC guide](skill_adapters.md#parallel-evaluation-and-hpc-resources).

## Asymmetric 2v2

`mappo_2v2_asymmetric.yaml` trains continuously for **120 million aggregate joint
environment decisions**, matching the PPO pretraining step budget. Training has
no lap finish or episode time limit (`lap_completion: false`, `max_steps: 0`).
A surviving learner keeps driving; the environment resets once both learners
crash (`all_trainable`). Each joint decision produces up to two learner samples.
Rollout boundaries trigger learning updates without ending the ongoing episode.

Checkpointing and evaluation run at update boundaries after each 4,096,000-step
threshold, matching pretraining. Evaluation retains three-lap races, a finite
16,000-step limit, and `all_agents`, including standalone evaluation. Use
`--total-steps` for shorter runs. These training termination changes affect
learning curves, so start a new experiment for comparisons.

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

## Distinguishing a slow rollout from a stalled HPC job

Parallel fixed-opponent MAPPO prints a status heartbeat every 15 seconds,
including during worker startup and policy updates. Configure
`experiment.collector_progress_interval_s` to change this interval. `--quiet`
suppresses console status. The asymmetric scenario prints every completed update.
With 400 environments and horizon 256, a full round collects up to 102,400 joint
decisions before producing losses; the MPC opponents can make that take time.

During training the heartbeat reports completed episodes, mean and latest episode
return, average episode length and learner laps, finish rate, crash/boundary-exit
rate, and timeout rate. Attack tasks also show confirmed successes and target
crashes per episode. These statistics use the most recent
`experiment.terminal_recent_episodes` completed episodes (default 100), and update
as workers report episodes, without waiting for an optimizer update. Finish rate
means all learners finished; failure and timeout rates mean any learner was
affected. Until the first episode completes, return is explicitly pending.
`env_steps` counts decisions incorporated into completed updates.
`collected` includes completed decisions reported during the current round;
`round_steps` counts just that round, and `collect_steps/s` measures its rate.
These counters can advance while no episodes finish. They count confirmed steps,
not actions merely dispatched to workers. The rate freezes during optimization
and evaluation, then resets when the next collection round starts.
`barrier_waiting=N/W` counts workers that submitted their full rollout and are
waiting for the update; it is not a count of CPUs in use.

Both 1v1 attack scenarios print every completed update. `collect_s`, `update_s`
and `round_steps/s` separate collection and learning from startup and evaluation.
`env_steps/s` remains the cumulative rate including startup and earlier hooks.
Longer completed episodes can reduce episodes/second without reducing decisions/second.

During checkpoint evaluation, the heartbeat instead shows the current evaluation
episode, map, physics steps/limit, simulated seconds, learner laps/targets,
terminal outcomes, and attack counts. Episode starts and completions print
immediately. `progress_age_s` measures time since the latest evaluator report;
reports refresh at most once per second between decisions. The selection evaluator
does not compute rewards, so `train_return_mean` remains explicitly labeled as the
recent training return. Training workers are paused during this evaluation.

Worker readiness is still shown during startup. Additional collector counters
(messages, dispatched actions and worker-message age) remain in `collector/*`
telemetry.
A stuck individual worker still triggers the configured response timeout.
These heartbeats cannot diagnose a process that the scheduler has suspended or killed.

Separate `collector/*` W&B metrics and `collector_progress.csv` are published by
the parent during collection and at phase boundaries, starting before workers
launch. Their x-axis is elapsed time; they do not increment learning updates or
trigger checkpoint/evaluation hooks. During a blocking operation the console
heartbeat continues, while W&B/CSV refresh when the parent resumes. Dispatched
actions are requests sent to environments, not confirmed completed transitions.
Episode reward/finish metrics still require completed episodes.

Use unbuffered output when submitting a batch job, for example `python3 -u run.py
...`. Check the scheduler's job state and CPU/GPU utilization alongside the log;
a flat GPU graph alone can mean CPU simulation is still collecting.

## Measure before increasing the allocation

Launching Python directly from an allocated interactive session or Open OnDemand
desktop is supported. Workers inherit the terminal's CPU affinity; `--num-workers`
sets the process count and does not request more CPUs. Fixed-opponent MAPPO prints
the allowed logical CPU count and, when Linux exposes the topology, the number
of physical cores those CPUs represent. It records them as
`perf/cpu_affinity_count` and `perf/cpu_affinity_core_count`. Missing Slurm variables
alone do not establish whether an interactive session has an allocation.

For example, 128 allowed logical CPUs may represent 64 physical cores with two
hardware threads each. One hundred worker processes then share those physical
cores. More workers can help hide waiting but can also increase contention;
benchmark the layout instead of assuming each logical CPU is a separate core.
Affinity describes accessible CPUs, not exclusive ownership or a container's
CPU-time quota. Multiple environments in a worker execute sequentially.

Run this with the GPU and CPUs reserved exclusively for the benchmark:

```bash
python3 scripts/benchmark_collectors.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml --set experiment.num_envs=400 --set experiment.num_workers=100 \
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

Fixed-opponent MAPPO also records these parent-side collection timings in
`update_metrics.csv`:

| Metric | What it measures |
| --- | --- |
| `perf/inference_seconds` | Request preparation, policy/value inference, and transfer of results back to CPU |
| `perf/worker_receive_seconds` | Receiving and decoding worker messages; also blocking on workers in synchronous mode |
| `perf/worker_wait_seconds` | Scheduler time, including waiting for ready workers in ready mode |
| `perf/action_send_seconds` | Grouping and sending policy replies to workers |
| `perf/inference_requests_per_batch` | Mean race requests per inference call, including value/bootstrap requests |

These are wall times in the parent, not summed worker CPU times or GPU kernel
times. Workers can simulate while the parent handles other workers. A large
`wait_s` suggests checking worker simulation and CPU utilization; large `infer_s`
requires examining batch sizes and host/device overhead as well as GPU use.
These timings do not by themselves establish that an additional GPU would help.

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

Before the continuous-training configuration change, a separate 512-decision
asymmetric scratch-policy probe changed from 182 to 512
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
