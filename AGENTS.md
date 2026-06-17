# AGENTS.md

## Project purpose

This repository is for F1TENTH-style autonomous racing experiments using reinforcement learning, multi-agent reinforcement learning, classical fixed-policy controllers, and future MPC-based opponents.

The current branch is focused on a pure PyTorch training stack for adversarial and cooperative racing experiments.

Primary research goals:

* Compare RL and MARL agents against fixed-policy racing controllers.
* Support controlled experiments with repeatable scenarios, seeds, maps, rewards, and opponents.
* Use strong non-learning baselines before adding more complex MARL or MPC methods.
* Preserve reproducibility across training, evaluation, offline datasets, and curriculum runs.

## Active architecture

Use the current architecture unless the user explicitly asks for a migration.

Current active path:

```text
run.py
  -> core.scenario.load_and_expand_scenario()
  -> core.setup.create_training_setup()
  -> wrappers.observations.ObservationComposer
  -> wrappers.rewards.RewardComposer
  -> wrappers.actions.ActionComposer
  -> training.on_policy_trainer.OnPolicyTrainer
     or training.off_policy_trainer.OffPolicyTrainer
     or training.marl_trainer.MARLTrainer
```

Important directories:

```text
src/agents/                  Pure PyTorch RL agents and fixed-policy agents
src/agents/common/           Shared neural network modules
src/env/                     F110 parallel environment and state contracts
src/replay/                  Replay buffer and offline dataset writer
src/training/                On-policy, off-policy, MARL, hooks, curriculum
src/wrappers/actions/        Action processing components
src/wrappers/observations/   Observation components and composer
src/wrappers/rewards/        Reward components and composer
configs/                     Shared env, vehicle, training, observation, reward configs
scenarios/                   Full experiment definitions
sweeps/                      W&B sweep definitions
tests/                       Regression and contract tests
```

## Current invariants

Preserve these unless the user explicitly asks otherwise:

* `run.py` is the single training entry point.
* Scenarios are YAML files under `scenarios/`.
* Shared config fragments live under `configs/`.
* Trainable RL/MARL agents are instantiated from `run.py`.
* Fixed-policy opponents are registered through `AgentFactory`.
* Observations, rewards, and actions are composed through wrapper components.
* Environment contracts must remain stable:

  * `reset`
  * `step`
  * `get_global_state`
  * `get_agent_state`
* Do not reintroduce Stable-Baselines3, Gymnasium, or PettingZoo into the active training path.

## Agent and controller conventions

Trainable algorithms currently include names such as:

```text
ppo
a2c
sac
td3
ddpg
dqn
mappo
```

Fixed-policy controllers currently include names such as:

```text
ftg
pure_pursuit
stanley
hybrid_pp_ftg
```

When adding a new fixed-policy controller, prefer the existing fixed-agent pattern:

```text
1. Add the implementation under src/agents/ or a subpackage of src/agents/.
2. Add a small AgentFactory-compatible adapter class.
3. Register it in src/core/config.py.
4. Add the algorithm name to HEURISTIC_ALGOS in src/core/agent_builder.py.
5. Add or update a scenario YAML that uses the new controller.
6. Add tests for construction, role detection, and action output shape.
```

For future MPC work, prefer this pattern first:

```text
src/agents/mpc/
  __init__.py
  kinematic.py
  costs.py
  rollout.py
  constraints.py
```

Then expose MPC variants as fixed-policy agents, for example:

```text
kinematic_mpc
obstacle_mpc
defensive_mpc
```

Do not create a separate controller framework unless the task explicitly asks for a migration plan.

## Coding rules

* Inspect relevant files before editing.
* Prefer minimal, testable changes.
* Keep training, evaluation, environment, wrappers, agents, and logging concerns separate.
* Avoid broad rewrites unless explicitly requested.
* Preserve existing scenario compatibility.
* Do not silently change observation dimensions.
* Do not silently change action bounds.
* Do not silently change reward semantics.
* Do not silently change trainable/fixed agent role resolution.
* Do not add dependencies without asking first.
* Keep random seeds explicit.
* Keep magic numbers in config files or named constants where practical.
* Use type hints when they improve readability.
* Add comments for math-heavy sections, especially vehicle models, reward logic, MPC costs, and MAPPO critic logic.

## Research-code rules

For RL/MARL changes, check for:

* train/eval mismatch
* reward leakage
* hidden randomness
* hard-coded map assumptions
* hard-coded agent IDs
* unfair baseline comparisons
* observation/action shape mismatch
* missing seed/config/run metadata
* map-specific overfitting
* opponent behavior accidentally changing across experiments

When changing rewards, observations, actions, or environment stepping, explain the research implication in the final summary.

## Scenario rules

Scenarios should remain readable experiment definitions.

Prefer adding a new scenario over mutating a known-good one.

Good scenario naming examples:

```text
ppo_vs_ftg.yaml
ppo_vs_pure_pursuit.yaml
ppo_vs_stanley.yaml
ppo_vs_hybrid_pp_ftg.yaml
mappo_gaplock.yaml
mappo_2v2.yaml
```

When adding or editing a scenario, preserve these ideas:

* `experiment.name` should be unique and descriptive.
* `environment.maps` or `environment.map` should be explicit.
* Trainable agents should have `observation` and `reward` configs.
* Fixed-policy agents should use `algorithm` plus `params`.
* `target_id` should be explicit for adversarial roles.
* `trainable: true/false` may be used when inference is ambiguous.

## MAPPO rules

MAPPO uses:

* shared actor
* centralized critic
* per-agent rollout buffers
* pooled PPO update
* local observations for decentralized execution
* global state for centralized training

When editing MAPPO:

* Preserve CTDE behavior.
* Do not let the actor consume global state unless explicitly changing the algorithm.
* Keep per-agent reward composers independent.
* Keep trainable agents excluded from `other_agents`.
* Verify that multiple trainable agents are handled consistently.
* Check whether hooks need per-agent events, especially dataset logging.

## Dataset and offline RL rules

The dataset path uses `TransitionRecord` and `DatasetWriter`.

When changing dataset logging:

* Preserve the schema version unless intentionally migrating it.
* Record one transition per agent decision.
* Include normalized action and physical action.
* Include reward, next observation, termination/truncation flags, map ID, spawn ID, episode ID, step index, agent ID, and global state when available.
* Do not record incomplete transitions silently.
* If MAPPO dataset logging differs from single-agent logging, call that out.

## Curriculum rules

Curriculum should operate through public reset options and spawn plans.

Do not mutate environment internals directly unless there is no public route.

Preferred pattern:

```text
trainer -> spawn_plan_fn -> env.reset(options={"spawn_plan": plan})
```

Curriculum phase logic should remain testable without running full training.

## Logging and output rules

* Prefer `ConsoleLogger` and training hooks over raw `print`.
* Keep W&B logging optional.
* Respect `--no-wandb`.
* Keep headless smoke tests working with `PYGLET_HEADLESS=true`.
* Do not add noisy per-step logs to the default training path.

## Validation expectations

After most changes, run the smallest relevant validation first.

Baseline checks:

```bash
python3 -m compileall -q run.py src tests
pytest tests/ -q
```

Smoke tests:

```bash
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/sac.yaml --no-wandb --total-steps 10 --quiet
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/td3.yaml --no-wandb --total-steps 10 --quiet
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/dqn.yaml --no-wandb --total-steps 10 --quiet
PYGLET_HEADLESS=true python3 run.py --scenario scenarios/mappo_gaplock.yaml --no-wandb --episodes 1 --quiet
```

Dependency guard:

```bash
rg "stable_baselines3|from gymnasium|from pettingzoo" run.py src configs scenarios
```

If a command cannot be run, state why.

If a command fails, report:

* command run
* error summary
* likely cause
* whether the failure appears related to the change

## Preferred development order

For this branch, prefer work in this order:

```text
1. Stabilize the current pure PyTorch training path.
2. Add or improve evaluation mode.
3. Add baseline comparison scenarios.
4. Verify MAPPO dataset logging.
5. Add stronger fixed-policy opponents.
6. Add kinematic MPC as a fixed-policy opponent.
7. Add obstacle-aware or defensive MPC.
8. Add MPCC or CBF-MPC only after evaluation is reliable.
```

## Good first tasks

High-value tasks for this branch:

```text
Add run.py --eval mode.
Add checkpoint loading support for evaluation.
Add baseline comparison scenarios.
Add docs/EXPERIMENTS.md.
Verify MAPPO per-agent dataset logging.
Add kinematic_mpc as a fixed-policy AgentFactory controller.
Add tests for fixed-policy role detection and controller action shape.
```

## Do not do

* Do not reintroduce old SB3 training paths.
* Do not add Gymnasium or PettingZoo wrappers unless explicitly requested.
* Do not create a second training entry point.
* Do not move the whole repo into a new architecture.
* Do not delete scenarios to simplify tests.
* Do not overwrite known-good configs when adding experiments.
* Do not mix training and evaluation logic in ways that make results hard to reproduce.
* Do not silently change map YAMLs, centerlines, or spawn semantics.
* Do not treat all agents as trainable unless the scenario explicitly says so.
* Do not optimize for elegance at the expense of reproducibility.

## Final response expectations

When completing a task, summarize:

```text
1. What changed.
2. Why it changed.
3. Files touched.
4. Validation commands run.
5. Any failures or limitations.
6. Suggested next step.
```

For research-relevant changes, also mention possible effects on experiment validity.
