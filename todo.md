# F110 MARL — Project Todo

## Project Goals
- Pure PyTorch training — no SB3, no Gymnasium, no PettingZoo in the training path
- Single `run.py` entry point; algorithm dispatched from scenario `algorithm:` field
- Clean config hierarchy: env / vehicle / training / observation / reward all separate concerns
- MVP: single-agent PPO ✅ → Phase 2: SAC, TD3, DQN → Phase 3: MAPPO

---

## Current State (Branch: `marl-rewiring`)

**Working:**
- `python run.py --scenario scenarios/ppo.yaml --no-wandb --episodes N` — fully functional
- `--render` flag works
- No SB3 / Gymnasium / PettingZoo imports in the training path
- Verified: 20+ episodes stable, reward ~-121 (timeout) or ~-220 (collision)

**Key source layout:**
```
run.py                                  # single entry point
src/env/spaces.py                       # SpaceSpec / DictSpaceSpec (replaces gymnasium.spaces)
src/env/f110ParallelEnv.py              # physics env — ParallelEnv base removed
src/agents/common/networks.py           # MLP, Actor, Critic (shared; Critic is MAPPO-ready)
src/agents/ppo/__init__.py              # RolloutBuffer, PPOAgent
src/wrappers/observations/              # ObservationComponent + ObservationComposer
src/wrappers/rewards/                   # RewardComponent + RewardComposer
src/training/on_policy_trainer.py       # episode-based PPO loop
src/training/hooks.py                   # ConsoleHook, WandbHook, CheckpointHook
configs/env/default.yaml                # timestep, lidar, render defaults
configs/vehicle/default.yaml            # vehicle physics params
configs/training/on_policy.yaml         # PPO/A2C training_defaults
configs/training/off_policy.yaml        # SAC/TD3/DQN training_defaults (stub)
configs/observations/                   # per-policy obs configs (rl_attacker, ftg, etc.)
configs/reward/                         # per-task reward configs
scenarios/ppo.yaml                      # refactored; uses new config hierarchy
```

**Legacy code still present (no SB3/gym dependency, low priority):**
- `src/rewards/` — old reward system (coexists; new code uses `src/wrappers/rewards/`)
- `src/wrappers/observation.py` — legacy obs wrapper (env still imports sector/radial helpers from it)

---

## Known Issues

No known issues.

---

## Phase 2 — Off-Policy Algorithms

### New files to create
- [ ] `src/replay/replay_buffer.py` — pure PyTorch ring buffer; stores `(obs, action, reward, next_obs, done)`; `add()`, `sample(batch_size)`, `__len__()`
- [ ] `src/agents/sac/__init__.py` — `SACAgent`: twin Q-critics, auto-alpha entropy tuning, target networks (polyak), continuous actions
- [ ] `src/agents/td3/__init__.py` — `TD3Agent`: twin critics, delayed actor updates, target policy smoothing noise
- [ ] `src/agents/dqn/__init__.py` — `DQNAgent`: ε-greedy exploration, target network (hard or soft update), discrete actions
- [ ] `src/training/off_policy_trainer.py` — step-based loop: interact → replay buffer → update every `train_freq` steps; episode hooks at episode boundaries

### Files to modify
- [ ] `run.py` — add `OffPolicyTrainer` dispatch for `sac|td3|ddpg|dqn` algorithms; read `experiment.total_steps` instead of `experiment.episodes`
- [ ] `src/core/config.py` — register `sac`, `td3`, `dqn` → new PyTorch agent classes
- [ ] `scenarios/sac.yaml`, `td3.yaml`, `dqn.yaml` — refactor to new structure: `maps:` key, `observation:` and `reward:` as config file refs, `includes: training/off_policy.yaml`

### Training loop design (reference)
```
obs = env.reset()
for step in 0..total_steps:
    action = random() if step < learning_starts else rl_agent.act(obs)
    heuristic agents act via agent.act(obs_dict[their_id])
    next_obs, reward, done, info = env.step(all_actions)
    replay_buffer.add(obs, action, reward, next_obs, done)
    if step >= learning_starts and step % train_freq == 0:
        metrics = rl_agent.update(replay_buffer.sample(batch_size))
    if done:
        hooks.on_episode_end(...)
        obs = env.reset()
    else:
        obs = next_obs
```

---

## Phase 3 — MARL: MAPPO

> Design constraint: `Actor` in `src/agents/common/networks.py` is already MAPPO-ready.
> Only the critic differs: `Critic(input_dim=n_agents * obs_dim)` for MAPPO vs `Critic(input_dim=obs_dim)` for PPO.

- [ ] `src/agents/mappo/__init__.py` — shared actor + centralized critic; focal agent cycling; per-agent rollout buffers
- [ ] `src/training/marl_trainer.py` — multi-agent episode loop; build global state for centralized critic
- [ ] `configs/training/mappo.yaml` — MAPPO-specific training defaults
- [ ] `scenarios/mappo_defender.yaml`, `scenarios/mappo_attacker.yaml`

---

## Cleanup (Phase 2 completion)

SB3/Gymnasium/PettingZoo fully removed ✅ (10,452 lines deleted)
- [ ] Remove `stable-baselines3`, `sb3-contrib`, `gymnasium`, `pettingzoo` from `requirements.txt`

Remaining migrations:
- [x] Extract sector/radial helpers from `src/wrappers/observation.py` → `src/wrappers/common.py`; deleted `observation.py`
- [x] Deleted `src/rewards/` — consolidated into `src/wrappers/rewards/`
- [ ] Update `sweeps/*.yaml` — `program: run_sb3.py` → `program: run.py`

---

## Post-MVP: Evaluation & Curriculum
> Deferred. Design when Phase 2 is complete.
- [ ] Eval config: spawn point names reference map YAML `annotations.spawn_points`
- [ ] Curriculum config: phase advancement gates tied to eval success rate

---

## Verification

```bash
# Phase 1 (done)
python run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 5
grep -r "stable_baselines3\|from gymnasium\|from pettingzoo" run.py src/agents/ppo/ src/training/ src/wrappers/
# → no matches

# Phase 2
python run.py --scenario scenarios/sac.yaml --no-wandb
python run.py --scenario scenarios/td3.yaml --no-wandb

# Phase 3
python run.py --scenario scenarios/mappo_defender.yaml --no-wandb --episodes 5
```
