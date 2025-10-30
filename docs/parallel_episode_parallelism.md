# Episode-Parallel Collector Design

## Objectives
- Run multiple environment episodes concurrently within a single training process to improve sample throughput without launching separate `run.py` jobs.
- Keep the existing `TrainRunner` as the coordinator so existing callbacks (federated sync, evaluation, logging) continue to work.
- Minimise disruption to algorithm implementations: trainers should continue to see transition batches via replay buffers as they do today.

## Proposed Architecture

### 1. Collector Manager (`ParallelCollector`)
- Lives in a new module (e.g. `src/f110x/collector/parallel.py`).
- Accepts:
  - `env_factory`: callable returning a fresh `F110ParallelEnv` instance.
  - `team_builder`: callable producing the `AgentTeam` & observation pipelines for each worker.
  - `worker_count`: number of concurrent environments.
  - `queue_maxsize`: back pressure for replay insertion.
- Spawns worker processes (preferred) using `multiprocessing.Context("spawn")` to avoid GPU conflicts.
- Provides async APIs:
  - `start()`: launch workers, prefetch initial observations.
  - `request_episode(worker_id, global_episode_idx)`: push work orders.
  - `gather()` iterator yielding completed `EpisodeRollout` objects.
  - `stop()` / context manager cleanup.

### 2. Worker Lifecycle
- Each worker process owns its own env/team/trainer reference to avoid sharing PyTorch modules across processes.
- Episode loop mirrors `run_episode` logic:
  1. Reset env (optionally seeded).
  2. Produce `EpisodeRollout` + raw transitions.
  3. Return to manager via `multiprocessing.Queue`.
- Managers serialise `Transition` objects (simple dataclasses) or push them through shared memory buffers; start with queue of python objects for simplicity, optimise later.
- Pass minimal info (obs, action, reward, done, info) to reduce IPC overhead.

### 3. Integration with `TrainRunner`
- New config block, e.g. `main.collect_workers` (int, default 1). When set to >1:
  - Instantiate `ParallelCollector` with matching count.
  - Replace direct calls to `run_episode` with dispatch: issue N outstanding episodes, gather results as they finish, and feed transitions/metrics to existing codepaths.
- Replay buffer handling:
  - Workers send transitions; manager replays them through existing `_record_transition` / `trajectory_buffers` flows.
  - Stats aggregation (returns, collisions, etc.) is already carried by `EpisodeRollout`.
- Hooks (spawn curriculum, defender curriculum) run on manager side using metrics per episode.

### 4. Federated Sync Considerations
- Sync should trigger after the same episode count as before; manager maintains `global_episode_idx` and increments as episode results arrive (regardless of worker).
- Optimizer strategies remain unchanged—the trainer map is still owned by the parent process.

### 5. Configuration Surface
```yaml
main:
  collect_workers: 4
  collect_prefetch: 8   # optional queue depth
  collect_seed_stride: 17  # optional per-worker seed offset
```
- CLI additions (`run.py`): pass through `--collect-workers` etc.
- Scenario defaults keep `collect_workers: 1` to preserve sequential behaviour.

### 6. Logging & Metrics
- Extend metrics with `collector/throughput_eps_per_min`, `collector/worker_latency_mean` for visibility.
- Workers can `log_queue` heartbeat info every N episodes; manager aggregates.

### 7. Future Enhancements
- Consider thread pool mode for CPU-only runs to reduce process overhead.
- Plug in Ray or asyncio for cluster-scale collection if needed later.
- Add shared-memory ring buffers to avoid pickling costs once basic version is stable.

## Next Steps
1. Implement `ParallelCollector` skeleton with queue-based worker processes.
2. Add integration switches to `TrainRunner` and extend config dataclasses.
3. Write smoke tests (e.g. fake environment generating deterministic returns) to validate multi-worker scheduling.
4. Benchmark sample throughput versus baseline to tune defaults.
