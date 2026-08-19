# F110 MARL - Todo

Completed work lives in `done.md`.

## Current Status

**P0-P7 historical baseline:** 393 tests green. **P8 is now planned and not yet
complete.** The current P8 terminal-handling foundation has 9 focused tests green.

**Working:**

```bash
python3 run.py --scenario scenarios/ppo.yaml          --no-wandb --episodes N
python3 run.py --scenario scenarios/sac.yaml          --no-wandb --total-steps N
python3 run.py --scenario scenarios/td3.yaml          --no-wandb --total-steps N
python3 run.py --scenario scenarios/dqn.yaml          --no-wandb --total-steps N
python3 run.py --scenario scenarios/mappo_gaplock.yaml --no-wandb --episodes N
python3 run.py --scenario scenarios/ppo_curriculum.yaml --no-wandb --episodes N
python3 run.py --scenario scenarios/ppo.yaml          --no-wandb --episodes 1 --dataset-dir /tmp/ds
```

**Invariants:**

- Zero SB3 / Gymnasium / PettingZoo in the training path.
- Single `run.py` entry point dispatched from `algorithm:` in scenario YAML.
- Env contract (`reset`, `step`, `get_global_state`, `get_agent_state`) stable.

---

## P8 - Multi-Agent Race Completion and Terminal Vehicle Lifecycle

**Priority:** blocking reliable MAPPO lap-completion experiments.

**Goal:** support configurable multi-lap races in which each agent independently
finishes, crashes, or times out; the episode ends only after every agent has left
the active decision set; and terminal vehicles remain in the simulator as
collidable physical objects.

### Locked Decisions

- [x] Race distance is configured with `environment.target_laps`.
- [x] Crossing the lap line before `target_laps` is a non-terminal event.
- [x] Reaching `target_laps` is a terminal event for that agent.
- [x] A collision is a terminal event for the active agent that crashed.
- [x] `episode_termination.mode` for this race is `all_agents`.
- [x] Finished and crashed vehicles remain physically present and collidable.
- [x] Terminal agents stop producing policy decisions and learning transitions.
- [x] A time limit truncates only agents that are still active.
- [x] A terminal result is immutable: a later collision cannot turn a finisher
  into a crash/DNF result.

### Terminology and Required State

Use these names consistently across environment, rewards, datasets, and metrics:

```text
Agent lifecycle status:
  active       still racing and requires an action
  finished     reached target_laps
  crashed      reached a terminal collision
  truncated    still active when max_steps was reached

Per-step facts:
  lap_crossed              forward lap-line crossing on this step
  lap_count                completed laps after this step
  target_laps              configured race distance
  race_completed           lap_count >= target_laps
  target_lap_count         configured target agent's lap count
  target_race_completed    configured target agent finished
  terminal_reason          race_complete | collision | time_limit | null
  terminal_step            first step at which status left active
  finish_position          one-based finishing position, or null
```

Keep two distinct agent collections:

```text
decision_agents / env.agents    active agents that require actions
physical_agents                all possible_agents still simulated for collisions
```

### Current Foundation

- [x] Add configurable `any_agent`, `all_agents`, and `all_trainable` episode
  termination policies in `src/env/collision_state.py`.
- [x] Make `MARLTrainer` use `env.agents` rather than observation-dict membership
  to select decision agents.
- [x] Stop `action_repeat` when active-agent membership changes.
- [x] Store MAPPO `terminated` and `truncated` flags separately.
- [x] Bootstrap time-limit truncations while stopping GAE recursion at episode
  boundaries.
- [x] Emit one MAPPO dataset record per active trainable-agent decision.
- [ ] Revalidate this foundation after the lifecycle work below; the current
  terminal map does not yet preserve lap-completion causes.

---

### P8.1 - Establish the Lifecycle Contract

Primary files:

```text
src/env/types.py
src/env/f110ParallelEnv.py
src/env/collision_state.py
src/env/state_views.py
```

- [ ] Add an `AgentRaceStatus` enum or equivalent typed constants for `active`,
  `finished`, `crashed`, and `truncated`.
- [ ] Add a per-agent lifecycle record containing status, terminal reason,
  terminal step, finish position, and lap facts.
- [ ] Initialize one lifecycle record per `possible_agent` during `reset`.
- [ ] Make all lifecycle transitions monotonic; no terminal agent may become
  active again before the next reset.
- [ ] Preserve the first terminal reason and terminal step permanently.
- [ ] Define `env.episode_done` as `all(status != active)`.
- [ ] Keep lifecycle code independent of MAPPO so fixed-policy and single-agent
  evaluation use the same contract.

Exit criteria:

- [ ] A pure unit test can drive lifecycle transitions without creating the
  simulator.
- [ ] Invalid transitions, such as `finished -> crashed`, leave the original
  status and result unchanged.

---

### P8.2 - Create One Authoritative Lap Tracker

Primary files:

```text
src/env/start_pose_state.py
src/env/centerline_state.py
src/env/f110ParallelEnv.py
src/env/types.py
maps/*/*.yaml
```

- [ ] Replace the current competing start-pose and explicit-finish terminal
  paths with one `LapTracker` contract.
- [ ] Use a shared start/finish line for every car in a race; do not treat each
  randomized grid pose as a separate race distance.
- [ ] Define the line explicitly in map metadata whenever practical.
- [ ] Audit every map used by `mappo_2v2_vs_hybrid_pp_ftg.yaml` and add or verify
  a reproducible finish-line annotation.
- [ ] Validate finish-line geometry at load time: two distinct points, usable
  direction, and placement close to the centerline/grid.
- [ ] Detect only forward crossings.
- [ ] Add hysteresis/debouncing so oscillation around the line cannot increment
  multiple laps.
- [ ] Emit `lap_crossed=True` only on the crossing step.
- [ ] Increment `lap_count` exactly once per accepted crossing.
- [ ] Set `race_completed=True` only when `lap_count >= target_laps`.
- [ ] Validate `target_laps` as a positive integer during scenario/environment
  construction.
- [ ] Do not terminate on lap 1 of a multi-lap race.
- [ ] Ensure randomized spawning cannot begin on the completed side of the line
  and immediately count a lap.

Exit criteria:

- [ ] With `target_laps: 3`, a deterministic synthetic trajectory produces
  three crossing events, lap counts `1, 2, 3`, and one completion event on lap 3.
- [ ] Reverse crossings and stationary line jitter produce no lap increments.
- [ ] The same trajectory produces identical counts for every supported map.

---

### P8.3 - Separate Terminal Causes from Episode Policy

Primary files:

```text
src/env/collision.py
src/env/collision_state.py
src/env/f110ParallelEnv.py
src/env/info_builder.py
```

- [ ] Compute raw per-agent events before applying an episode policy:

  ```text
  collision_event
  lap_crossed
  race_completed
  time_limit
  ```

- [ ] Transition an active agent to `finished` only from its own
  `race_completed` event.
- [ ] Transition an active agent to `crashed` only from its own terminal
  collision event.
- [ ] Stop propagating one agent's `terminated=True` flag to unrelated agents
  when mode is `all_agents`.
- [ ] Keep termination dictionaries sticky or otherwise unambiguous for agents
  that already left the decision set.
- [ ] At `max_steps`, mark only currently active agents as truncated.
- [ ] Set `episode_done` only after all agents are finished, crashed, or
  truncated.
- [ ] Ensure a collision involving a terminal vehicle can still crash an active
  vehicle without changing the terminal vehicle's recorded status.

Exit criteria:

- [ ] One finisher in a four-car race leaves three decision agents active.
- [ ] One crash in a four-car race leaves three decision agents active.
- [ ] The fourth terminal transition ends the episode.
- [ ] A time limit ends the episode and truncates only remaining active agents.

---

### P8.4 - Keep Terminal Vehicles Physical and Collidable

Primary files:

```text
src/env/f110ParallelEnv.py
src/env/state_buffer.py
src/env/collision_state.py
src/physics/
configs/env/default.yaml
```

Add a configuration contract:

```yaml
environment:
  terminal_agents:
    remain_collidable: true
    crashed_behavior: stationary
    finished_behavior: coast_then_stop
    finish_clearance_steps: 200
```

- [ ] Keep every `possible_agent` in the simulator's physical state after its
  learning trajectory terminates.
- [ ] Cache the last physical action and vehicle state when an agent becomes
  terminal.
- [ ] For crashed vehicles, command zero speed and deterministically suppress
  residual self-propulsion while retaining collision geometry.
- [ ] Decide whether collision impulses may move a crashed vehicle; encode and
  test the chosen behavior rather than relying on simulator side effects.
- [ ] For finished vehicles, implement a deterministic `coast_then_stop`
  controller owned by the environment, not the learned policy.
- [ ] Define the coast schedule precisely: steering behavior, speed decay,
  clearance duration, and final stationary state.
- [ ] Ensure the post-finish controller does not create actor log-probabilities,
  rewards, rollout entries, or dataset transitions.
- [ ] Continue passing terminal vehicles through collision detection.
- [ ] Do not send implicit `[0, 0]` actions without documenting their physical
  meaning; that can stop a winner directly on the finish line.
- [ ] Expose terminal-vehicle behavior in run metadata for reproducibility.

Exit criteria:

- [ ] A finished vehicle remains visible and collidable for the rest of the
  episode.
- [ ] An active vehicle can collide with it and become crashed.
- [ ] The finished vehicle retains its finish position and completion outcome.
- [ ] No post-terminal transition is written for the finished vehicle.

---

### P8.5 - Extend Public Info and Centralized State

Primary files:

```text
src/env/info_builder.py
src/env/state_views.py
src/env/types.py
src/env/f110ParallelEnv.py
src/agents/mappo/__init__.py
```

- [ ] Add the lap/lifecycle facts listed above to training/debug info payloads.
- [ ] Select and document the subset guaranteed in minimal info mode.
- [ ] Add immutable terminal facts to `AgentState` or its metadata.
- [ ] Add lifecycle masks to `GlobalState`:

  ```text
  active_mask
  finished_mask
  crashed_mask
  truncated_mask
  ```

- [ ] Add normalized lap progress to the centralized critic input.
- [ ] Preserve local observations for decentralized actor execution.
- [ ] Decide whether local racer observations need `lap_count`, `target_laps`,
  or normalized race progress; update observation dimensions explicitly if so.
- [ ] Record the global-state dimension change in checkpoint/run metadata.
- [ ] Treat existing MAPPO checkpoints as incompatible unless a deliberate
  migration path is implemented.

Exit criteria:

- [ ] The centralized critic can distinguish active, finished, and crashed
  physical vehicles.
- [ ] Actor inputs still contain no centralized/global-only fields.
- [ ] Observation and global-state dimensions are covered by contract tests.

---

### P8.6 - Update MAPPO Collection and GAE

Primary files:

```text
src/training/marl_trainer.py
src/agents/mappo/__init__.py
run.py
```

- [ ] Select trainable actions only for agents active at decision start.
- [ ] Build fixed-policy actions only for active fixed agents.
- [ ] Let the environment generate physical post-terminal actions internally.
- [ ] Stop an action-repeat block whenever an acted agent changes lifecycle
  status, then resume the outer loop with the reduced decision set.
- [ ] Store the completing/crashing transition exactly once.
- [ ] Never store transitions for a terminal agent on later joint steps.
- [ ] Continue collecting from other trainable agents until they terminate.
- [ ] Preserve separate per-agent rollout lengths.
- [ ] Block bootstrap on true finish/crash terminations.
- [ ] Bootstrap time-limit truncations from the final global state.
- [ ] Ensure pooled PPO updates accept one-step buffers without silently dropping
  short terminal trajectories.
- [ ] Keep per-agent last info/outcome snapshots immutable after termination.

Exit criteria:

- [ ] A two-trainable-agent test can finish one agent early, continue the other,
  and verify exact buffer lengths and terminal masks.
- [ ] GAE unit tests cover finish, crash, truncation, and buffer-full boundaries.

---

### P8.7 - Make Rewards Cause-Based

Primary files:

```text
src/wrappers/rewards/lap_completion.py
src/wrappers/rewards/target_finish.py
src/wrappers/rewards/track_completion.py
src/wrappers/rewards/collision.py
src/wrappers/rewards/timeout.py
configs/reward/components/
configs/reward/tasks/race_team_2v2_completion.yaml
```

- [ ] Stop inferring lap completion from generic `terminated`.
- [ ] Define a per-lap bonus triggered only by `lap_crossed`.
- [ ] Define a final completion bonus triggered only by `race_completed`.
- [ ] Trigger finish-ahead rewards from immutable finish order or explicit race
  completion facts.
- [ ] Trigger target-finish penalties from `target_race_completed`.
- [ ] Keep collision and timeout rewards tied to explicit terminal causes.
- [ ] Emit no rewards after the agent becomes terminal.
- [ ] Decide and document whether team rewards continue for an already-finished
  teammate; default to ending that agent's reward stream with its trajectory.
- [ ] Confirm reward components do not grant every agent completion credit when
  only one agent finishes.

Exit criteria:

- [ ] Reward unit tests cover intermediate laps, final lap, target finish,
  collision, timeout, and post-terminal joint steps.
- [ ] Reward totals remain independently attributable per trainable agent.

---

### P8.8 - Correct Outcomes, Standings, and Evaluation

Primary files:

```text
src/metrics/outcomes.py
src/metrics/racing_eval.py
src/metrics/tracker.py
src/training/hooks.py
run.py
```

- [ ] Replace attacker-specific finish assumptions where necessary with general
  race outcome facts.
- [ ] Record completion step/time and one-based finish position exactly once.
- [ ] Record final lap count for every agent.
- [ ] Distinguish finished, crashed/DNF, and truncated/DNF results.
- [ ] Keep a finisher's result unchanged after later physical collisions.
- [ ] Define 2v2 team placement metrics before using them for claims:
  completion rate, mean position, best finisher, both-finished rate, and DNF rate.
- [ ] Log per-agent and team terminal reasons to console, CSV, and W&B.
- [ ] Ensure evaluation continues until all agents are terminal, not merely until
  the focal agent finishes.

Exit criteria:

- [ ] A deterministic four-car result produces stable ordered standings.
- [ ] Evaluation and training classify the same terminal sequence identically.

---

### P8.9 - Migrate the Offline Dataset Schema Deliberately

Primary files:

```text
src/env/types.py
src/replay/dataset_writer.py
src/training/marl_trainer.py
```

- [ ] Decide the next dataset schema version; do not silently alter `1.0`.
- [ ] Persist at least:

  ```text
  lap_crossed
  lap_count
  target_laps
  race_completed
  terminal_reason
  lifecycle status/masks
  finish_position when available
  ```

- [ ] Preserve one record per actual agent decision.
- [ ] Keep normalized action, physical action, global state, map ID, spawn ID,
  episode ID, joint step, agent ID, termination, and truncation fields.
- [ ] Ensure no synthetic post-terminal controller action enters the dataset.
- [ ] Add metadata describing episode termination policy and terminal-vehicle
  behavior.
- [ ] Add a reader/validation test for old and new schema detection.

Exit criteria:

- [ ] A mixed finish/crash/timeout episode can be reconstructed from dataset
  records without guessing terminal causes.

---

### P8.10 - Update the 2v2 Scenario and Map Protocol

Primary files:

```text
scenarios/mappo_2v2_vs_hybrid_pp_ftg.yaml
configs/reward/tasks/race_team_2v2_completion.yaml
configs/env/default.yaml
maps/<active-map>/<active-map>.yaml
```

- [ ] Set the intended race distance explicitly with `target_laps`.
- [ ] Set `episode_termination.mode: all_agents` explicitly in the scenario even
  if it matches a shared default.
- [ ] Add the explicit terminal-agent physical behavior block.
- [ ] Switch both MAPPO racers to the verified completion-focused reward config.
- [ ] Replace `max_steps: 500000` with a measured limit that allows the intended
  lap count but still bounds failed episodes.
- [ ] Make train and evaluation map sets explicit; do not rely on an implicit
  shuffled 80/20 split for published comparisons.
- [ ] Correct stale comments and W&B notes that currently describe `line2` and a
  different step limit.
- [ ] Record finish-line annotation versions or map hashes in run metadata.
- [ ] Do not silently move existing centerlines, spawn points, or finish lines;
  review map metadata changes independently.

Proposed scenario shape:

```yaml
environment:
  target_laps: <chosen race distance>
  max_steps: <measured bound>
  episode_termination:
    mode: all_agents
  terminal_agents:
    remain_collidable: true
    crashed_behavior: stationary
    finished_behavior: coast_then_stop
    finish_clearance_steps: 200
```

Exit criteria:

- [ ] Scenario expansion reports the intended maps, lap count, terminal policy,
  reward config, and terminal physical behavior.
- [ ] A scenario smoke run cannot award race completion to a non-finisher.

---

### P8.11 - Required Test Matrix

Add focused tests under `tests/` before broad simulator runs.

Lap tracking:

- [ ] Lap 1 of 3 increments count and remains active.
- [ ] Lap 2 of 3 increments count and remains active.
- [ ] Lap 3 of 3 changes only that agent to finished.
- [ ] Reverse crossing, jitter, and stationary overlap do not count.
- [ ] Reset clears all lap and lifecycle state.

Multi-agent lifecycle:

- [ ] First finisher does not end a four-agent episode.
- [ ] First crash does not end a four-agent episode.
- [ ] Mixed finished/crashed agents remain outside the decision set.
- [ ] Final active agent finishing or crashing ends the episode.
- [ ] Time limit truncates only remaining active agents.

Physical terminal vehicles:

- [ ] Finished and crashed vehicles remain in physical/global state.
- [ ] They remain collidable.
- [ ] Collision with a terminal vehicle can crash an active vehicle.
- [ ] Later collision does not overwrite a finished result.
- [ ] Coast/stop behavior is deterministic under a fixed seed.

Training and data:

- [ ] MAPPO buffer lengths match each agent's active decision count.
- [ ] Final transitions contain the correct cause and flags.
- [ ] No post-terminal policy or dataset records are emitted.
- [ ] GAE bootstraps truncation but not finish/crash.
- [ ] Dataset round-trip preserves lifecycle facts.

Rewards and evaluation:

- [ ] Only the crossing agent receives a per-lap reward.
- [ ] Only the actual finisher receives completion credit.
- [ ] Target finish and finish-ahead terms identify the correct pair.
- [ ] Standings remain immutable after all later collisions.
- [ ] Training and evaluation agree on finish order and DNF status.

---

### P8.12 - Validation Gates

Run the smallest gate first and stop on unexplained failures.

Gate 1 - static and unit checks:

```bash
venv/bin/python -m compileall -q run.py src tests
venv/bin/python -m pytest tests/test_terminal_conditions.py -q
venv/bin/python -m pytest tests/test_lap_tracking.py -q
venv/bin/python -m pytest tests/test_mappo_terminal_handling.py -q
```

Gate 2 - full regression suite:

```bash
venv/bin/python -m pytest tests/ -q
rg "stable_baselines3|from gymnasium|from pettingzoo" run.py src configs scenarios
```

Gate 3 - deterministic controller validation:

- [ ] Run a one-lap fixed-controller race and verify one crossing/completion.
- [ ] Run a three-lap fixed-controller race and verify three crossings but only
  one completion per finisher.
- [ ] Run a scripted early crash and verify remaining agents continue.
- [ ] Run a scripted collision with a finished vehicle.

Gate 4 - MAPPO integration:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_vs_hybrid_pp_ftg.yaml \
  --no-wandb --episodes 1 --quiet
```

- [ ] Verify all four agents reach finished/crashed/truncated status.
- [ ] Verify per-agent buffer sizes and terminal reasons.
- [ ] Verify terminal vehicles remain rendered/physical and collidable.
- [ ] Repeat with dataset output and inspect record counts per agent.

Gate 5 - research readiness:

- [ ] Run multiple fixed seeds.
- [ ] Confirm map split, finish geometry, spawn assignments, and terminal behavior
  are recorded in metadata.
- [ ] Compare completion rates and episode lengths across maps.
- [ ] Set the final `max_steps` from observed race-duration distributions.
- [ ] Start long MAPPO training only after all earlier gates pass.

### Research Validity Notes

- Changing lap detection, reward attribution, global-state dimensions, or
  truncation targets creates a new experiment version.
- Do not compare new learning curves directly with checkpoints trained under the
  old joint-terminal/lap-reward behavior.
- Finished vehicles remaining collidable materially changes later agents'
  trajectories. The post-finish controller and clearance duration are part of
  the environment definition and must be held constant across comparisons.
- Shared finish geometry and explicit train/eval map lists are required for fair
  completion-time and placement comparisons.

### Suggested Delivery Slices

1. [ ] Lifecycle types and pure transition tests.
2. [ ] Shared lap tracker and map finish-line validation.
3. [ ] `all_agents` environment stepping and terminal physical controllers.
4. [ ] Public info/global state plus MAPPO collection updates.
5. [ ] Cause-based rewards, outcomes, and standings.
6. [ ] Dataset schema migration and reconstruction tests.
7. [ ] Scenario/map updates and deterministic validation.
8. [ ] MAPPO smoke tests and research-readiness audit.

### P8 Definition of Done

- [ ] `target_laps` is the only race-distance authority.
- [ ] Intermediate laps never terminate an agent.
- [ ] Every agent independently finishes, crashes, or truncates.
- [ ] The episode ends only when no active agents remain.
- [ ] Terminal vehicles remain physical and collidable.
- [ ] Terminal results cannot be overwritten.
- [ ] MAPPO stores no post-terminal experience and uses correct GAE masks.
- [ ] Rewards, metrics, and datasets identify the actual finisher and cause.
- [ ] One-lap and multi-lap deterministic tests pass on every configured map.
- [ ] The 2v2 scenario passes a headless MAPPO smoke run with auditable metadata.

---

## Deferred / Polish

Small items parked from earlier phases — not blocking anything.

- [ ] `--verbose` / `--debug` CLI flags for log-level control *(deferred from P3)*
- [ ] Normalize scenario-level render config into a single contract *(deferred from P2/P3)*

---

## Potential Next Work

Ideas for future sessions — not prioritised.

- **Real training run** — spin up PPO or MAPPO for real and look at reward curves / outcomes.
- **Map with named spawn points** — add `annotations.spawn_points` to a map YAML so the curriculum scenario (`ppo_curriculum.yaml`) actually uses phase-controlled spawning.
- **Eval harness** — `run.py --eval` mode: load checkpoint, run fixed-seed episodes, report success-rate table.
- **New algorithms** — MADDPG, QMIX, CTDE-SAC.
- **`--verbose` / `--debug`** — `logging.basicConfig(level=...)` wired to a CLI flag (the deferred P3 item).

---

## Key Source Layout

```text
run.py                                  # single training entry point

# --- Env core ---
src/env/f110ParallelEnv.py              # physics env coordinator
src/env/types.py                        # AgentState, SpawnPlan, TransitionRecord, StepFacts …
src/env/spawn.py                        # SpawnRequest/Result, resolve_reset_spawn
src/env/spawn_manager.py               # SpawnManager — per-env spawn config + episode state
src/env/centerline_state.py            # CenterlineRuntimeState, CenterlineProgressTracker
src/env/map_config.py                  # MapRuntimeConfig
src/env/map_schedule.py                # MapScheduler — bundle cycling + cache
src/env/info_builder.py                # build_step_facts, filter_info_payloads
src/env/state_views.py                 # build_agent_state, build_global_state, masks

# --- Core ---
src/core/env_builder.py                # build_env_kwargs, create_environment
src/core/agent_builder.py             # trainable/fixed role helpers
src/core/scenario.py                  # load_and_expand_scenario, validate_scenario
src/core/spawn_config.py              # normalize_spawn_config
src/core/protocol.py                  # HeuristicPolicy, OnPolicyAgent, OffPolicyAgent

# --- Agents ---
src/agents/ppo/__init__.py              # RolloutBuffer, PPOAgent
src/agents/mappo/__init__.py            # MAPPORolloutBuffer, MAPPOAgent (shared actor + centralised critic)
src/agents/sac/__init__.py             # SACAgent
src/agents/td3/__init__.py             # TD3Agent
src/agents/dqn/__init__.py             # DQNAgent
src/agents/ftg.py                      # FTGAgent (heuristic)
src/agents/waypoint.py                 # PurePursuitAgent, StanleyAgent, HybridPPFTGAgent
src/replay/replay_buffer.py            # CPU-stored PyTorch replay buffer
src/replay/dataset_writer.py           # DatasetWriter + DatasetHook (offline RL)

# --- Wrappers ---
src/wrappers/observations/base.py      # ObservationComponent (compute / compute_into)
src/wrappers/observations/composer.py  # ObservationComposer (pre-allocated buffer)
src/wrappers/rewards/composer.py       # RewardComposer
src/wrappers/actions/composer.py       # ActionComposer

# --- Training ---
src/training/on_policy_trainer.py      # episode-based PPO/A2C loop
src/training/off_policy_trainer.py     # step-based SAC/TD3/DQN loop
src/training/marl_trainer.py           # multi-agent MAPPO loop
src/training/curriculum.py             # CurriculumPhase, CurriculumManager
src/training/hooks.py                  # ConsoleHook, WandbHook, CheckpointHook,
                                       #   CurriculumHook, DatasetHook

# --- Config ---
configs/env/default.yaml
configs/vehicle/default.yaml
configs/training/on_policy.yaml
configs/training/off_policy.yaml
configs/training/mappo.yaml
configs/observations/
configs/reward/
scenarios/
```

## Verification

```bash
python3 -m compileall -q run.py src tests
pytest tests/ -q                          # 393 tests

# Smoke matrix
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/ppo.yaml   --no-wandb --episodes 1 --quiet
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/sac.yaml   --no-wandb --total-steps 10 --quiet
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/td3.yaml   --no-wandb --total-steps 10 --quiet
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/dqn.yaml   --no-wandb --total-steps 10 --quiet
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/mappo_gaplock.yaml --no-wandb --episodes 1 --quiet

# Confirm no SB3 / Gymnasium / PettingZoo in training path
rg "stable_baselines3|from gymnasium|from pettingzoo" run.py src configs scenarios
```
