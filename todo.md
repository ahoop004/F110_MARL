# F110 MARL - Todo

Completed work lives in `done.md`.

## Current Status

**393 tests green.** All planned phases (P0–P7) complete.

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
