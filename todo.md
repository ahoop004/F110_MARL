# F110 MARL Roadmap

## General Prerequisites (Do First)
- [x] Strengthen configuration schema & CLI plumbing for new features
- [x] Improve telemetry/metrics (collector throughput, aggregator diagnostics)
- [x] Build synthetic test harnesses for parallel collectors and aggregators
- [x] Expand documentation (README, design docs, demos) for emerging features

## Shared Replay / Centralised Buffers
- [ ] Implement replay service manager (shared memory / IPC queues)
  - [ ] Define replay RPC interface (enqueue transitions, fetch batches, stats)
  - [ ] Build manager process with shared-memory ring buffers + priority queues
  - [ ] Add graceful shutdown & failure recovery
- [ ] Add per-agent prioritised sampling across clients
  - [ ] Track client-specific priorities and weights
  - [ ] Support dynamic reweighting (e.g., based on recent performance)
- [ ] Integrate back-pressure and buffer metrics
  - [ ] Enforce capacity limits with producer throttling
  - [ ] Expose telemetry (buffer fill, wait times) via logger/W&B
- [ ] Update trainers/tests to handle centralised sampling
  - [ ] Refactor `TrajectoryBuffer` consumers to read from replay service
  - [ ] Create synthetic multi-client tests to validate consistent sampling

## Advanced Averaging Strategies
- [ ] Design pluggable `FederatedAggregator` interface
  - [ ] Define aggregator base class with hooks for `accumulate`, `finalise`
  - [ ] Refactor existing mean averaging to use the interface
- [ ] Implement FedProx / adaptive weighting support
  - [ ] Capture client-specific metrics (sample counts, reward deltas)
  - [ ] Apply proximal regularisation when merging weights
- [ ] Add robust aggregators (median, trimmed mean)
  - [ ] Support coordinate-wise operations on tensors and optimizer state
  - [ ] Handle fallbacks when fewer than N clients respond
- [ ] Track per-round metadata to weight contributions
  - [ ] Persist history of client participation & weight factors
  - [ ] Surface diagnostics (variance, client dropouts) in logs/W&B
- [ ] Validate compatibility with optimizer strategies (average/reset)
  - [ ] Ensure aggregated optimizer state stays numerically stable
  - [ ] Add regression tests for TD3/SAC/DQN with new aggregators

## Expanded MARL / Federated Approaches
- [ ] Prototype gossip / ring-allreduce communication
  - [ ] Define peer-to-peer update protocol (ring schedule, gossip interval)
  - [ ] Implement fault handling (peer dropout, retry)
- [ ] Add policy distillation / ensemble aggregation option
  - [ ] Create evaluator to compare/merge policies via KL minimisation
- [ ] Explore meta-learning / continual-learning integration
  - [ ] Investigate MAML/Reptile style inner/outer loops for federated agents
- [ ] Implement adversarial update detection & mitigation
  - [ ] Add outlier detection on weight deltas (norm checks, cosine similarity)
- [ ] Investigate DP / secure aggregation hooks
  - [ ] Research libraries (Opacus, PySyft) for privacy-aware training
  - [ ] Scope encrypted/secure sum protocols

## Intra-run Episode Parallelism
- [ ] Implement rollout workers feeding a shared replay buffer
  - [ ] Create `ParallelCollector` class (manager queues, worker pool, lifecycle)
  - [ ] Build worker process entrypoint (env/team bootstrap, run loop, exception handling)
  - [ ] Define serialisable transition payload and reconnect with existing `TrajectoryBuffer`
- [ ] Integrate worker scheduling into `TrainRunner` (config flags, hooks)
  - [ ] Extend config schema (`main.collect_workers`, `collect_prefetch`, etc.)
  - [ ] Add CLI flags in `run.py` and propagate through sweep specs
  - [ ] Replace direct `run_episode` loop with collector dispatch + gather logic
- [ ] Ensure federated syncing remains correct with batched episodes
  - [ ] Track global episode indices independent of worker completion order
  - [ ] Trigger `FederatedAverager.sync` based on total episodes instead of per-worker counts
  - [ ] Verify optimizer strategies (`average`, `reset`) behave with aggregated transitions
- [ ] Instrument metrics/logging for per-worker throughput
  - [ ] Emit collector metrics (`collector/eps_per_min`, latency stats)
  - [ ] Surface worker health/heartbeat logs
  - [ ] Update smoke/integration scripts to validate collector metrics
