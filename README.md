# F110_MARL

Pure PyTorch PPO and MAPPO experiments for F1TENTH racing. The repository supports
single-agent learning against fixed controllers, multi-agent learning with a
shared actor and centralized critic, checkpoint evaluation, and offline datasets.

## Run an experiment

Use the project virtual environment when available. Dependencies are listed in
`requirements.txt`; development checks use pytest. Headless examples:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_gaplock.yaml --no-wandb --episodes 1
```

Use `--seed` for repeatability, `--output-dir` for a specific output location,
`--dataset-dir` for transition recording, and `--render` for local visualization.
PPO and MAPPO use `--episodes`; the retired `--total-steps` option is unsupported.
W&B is optional and `--no-wandb` overrides scenario logging settings.

Evaluate a checkpoint with the same scenario and experiment overrides used for
training (the resolved configuration is checked against checkpoint provenance):

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --eval --checkpoint outputs/example/checkpoint_ep000000.pt --eval-episodes 5
```

Replace the example checkpoint path with the checkpoint produced by your run.
`--allow-provenance-mismatch` is available for intentional cross-scenario evaluation.
For fixed-controller-only scenarios, use the ordinary episode command without
`--eval` or a checkpoint.

## Experiment catalog

Scenario files retain their existing paths and experiment settings. Inspect each
YAML for its maps, seeds, rewards, vehicle limits, and episode budget before a run.

| Experiment | Scenario files under `scenarios/` |
|---|---|
| PPO attacker against FTG | `ppo.yaml`, `ppo_curriculum.yaml` |
| PPO centerline racing / time trial | `ppo_centerline.yaml`, `ppo_time_trial.yaml` |
| PPO against waypoint controllers | `ppo_vs_pure_pursuit.yaml`, `ppo_vs_stanley.yaml`, `ppo_vs_hybrid_pp_ftg.yaml` |
| PPO actor pretraining | `ppo_lap_completion_pretrain.yaml` |
| PPO pretraining with Frenet/track preview and acceleration actions | `ppo_lap_completion_pretrain_frenet.yaml` |
| PPO defender against hybrid controller | `marl_defender.yaml` |
| MAPPO gaplock | `mappo_gaplock.yaml` |
| MAPPO four-car reward/critic comparison | `complete_4_individual.yaml`, `complete_4_team_shared.yaml` |
| MAPPO initialized from a PPO checkpoint | `mappo_4car_1lap_circle_ppo_pretrained.yaml` (requires the configured local checkpoint) |
| Reverse-enabled Frenet PPO transfer to a MAPPO 2v2 team | `mappo_2v2_frenet_ppo_pretrained.yaml` (requires the configured Frenet `best_model.pt`) |
| MAPPO observation variants | `complete_4.yaml`, `complete_4_frenet.yaml`, `complete_4_frenet_neighbors.yaml` |
| MAPPO against two hybrid opponents | `mappo_2v2.yaml`, `mappo_2v2_vs_hybrid_pp_ftg.yaml`, `mappo_2v2_individual.yaml`, `mappo_2v2_team_shared.yaml` |
| Fixed-controller baselines | `nrl_1car.yaml` through `nrl_4car.yaml` |
| Race-duration calibration | `calibration/*.yaml` |

The historical planning templates `circle_attacker.yaml`, `circle_defender.yaml`,
and `marl_attacker.yaml` declare multiple PPO learners and are intentionally
rejected by validation. They require an explicit experiment redesign before use.
A supported training scenario has one PPO learner or a homogeneous MAPPO team.
Frozen PPO/MAPPO opponents and mixed trainable algorithms are not implemented.
Scenario curriculum is currently supported for PPO only.

MAPPO's individual arm uses per-agent rewards and an agent-conditioned critic,
`V_i(s)`. The team arm uses a configured team reward reduction and shared team
critic, `V(s)`. Actors use local observations in both modes. Raw shaped rewards
across these arms are not interchangeable evaluation scores.

The pretraining scenario currently uses `circle_map` for both train and evaluation;
it does not measure generalization to held-out maps. The `complete_4` experiment
has explicit disjoint splits. See [actor transfer](docs/PPO_TO_MAPPO_PRETRAINING.md),
[duration calibration](docs/RACE_DURATION_CALIBRATION.md), and
[performance checks](docs/PERFORMANCE.md) for their respective workflows.
Seed and opponent comparison sweeps are described in [sweeps/README.md](sweeps/README.md).

## Architecture and extension points

```text
run.py
  -> core.scenario.load_and_expand_scenario()
  -> core.setup.create_training_setup()
  -> observation / reward / action composers
  -> training.on_policy_trainer.OnPolicyTrainer
     or training.marl_trainer.MARLTrainer
```

| Location | Responsibility |
|---|---|
| `src/agents/ppo/`, `src/agents/mappo/` | Policy updates and rollout buffers |
| `src/agents/common/` | Shared actor, critic, and MLP modules |
| `src/agents/ftg.py`, `src/agents/waypoint.py`, `src/agents/mpc/` | Fixed-policy opponents |
| `src/env/` | Simulation coordination, lifecycle, and public state contracts |
| `src/wrappers/` | Observation, reward, and continuous action composition |
| `src/training/` | Collection loops, evaluation, hooks, and curriculum |
| `src/replay/dataset_writer.py` | Transition datasets used by PPO and MAPPO |
| `configs/`, `scenarios/`, `sweeps/` | Shared fragments, experiments, and sweep definitions |

For headless PPO on multiple CPU cores, use `--num-envs 8 --torch-threads 1`
with one GPU. Single-environment training remains the default. See
[HPC collection and reproducibility](docs/PERFORMANCE.md#headless-ppo-on-hpc)
for allocation, rollout-size, seed, and evaluation details.

PPO and MAPPO share advantage calculation, minibatch loss/optimizer steps, and
metric reduction in `src/agents/common/__init__.py`. Each agent retains its own
rollout storage, minibatch selection, and critic inputs.
Curriculum uses `src/training/curriculum.py`; logging uses training hooks,
`ConsoleLogger`, and the CSV/W&B loggers. Unused alternate curriculum, metrics,
console, and checkpoint utilities were retired from `src/` and remain available
at Git revision `cfd19fb20cf4762dccaa1613bcb3ec7487ee461c`.

Keep changes small and prefer reusing existing files. `run.py` stays the single
training entry point. To implement another trainable algorithm later:

1. Implement it under `src/agents/<algorithm>/`, reusing `agents/common` where appropriate.
2. Declare its supported role in `src/core/agent_builder.py` and validation in
   `src/core/scenario.py`; unsupported names must fail instead of changing roles.
3. Construct it from `run.py`. Reuse a trainer only when its collection/update
   contract fits; add replay or discrete-action support when the implementation needs it.
4. Integrate deterministic evaluation, checkpoint metadata, explicit seeds, and
   complete per-agent dataset transitions. Add a scenario and focused contract tests.

For another fixed opponent, use the existing `AgentFactory` adapter pattern in
`src/core/config.py` and `HEURISTIC_ALGOS` in `src/core/agent_builder.py`.
Preserve environment contracts, action bounds, observation dimensions, reward
semantics, and MAPPO's decentralized actors. No additional controller framework,
plugin system, or placeholder algorithm infrastructure is needed.

Related wrapper classes share modules: rewards use `motion.py`, `completion.py`,
`events.py`, and `interaction.py`; observations use `ego.py`, `track.py`, and
`neighbors.py`. Action components and their composer live in
`src/wrappers/actions/composer.py`. Add related components to these modules while
keeping each component's configuration key, observation slice, and reset behavior
explicit. Reward and observation composers retain their existing import paths.

Reward settings live in complete presets under `configs/reward/` and task
definitions under `configs/reward/tasks/`. Task files may include one complete
shared preset; task-specific settings are written directly in their `reward:`
block. Avoid separate files for individual penalty or bonus values. Existing
scenario and reward-task paths remain stable. The 30 former component fragments
and the previous Python module layout are available in Git history; external
scripts importing moved component classes must use the grouped modules above.

Scenario, reward, and environment-feature configuration use the same recursive
YAML loader, `core.scenario.load_yaml_config`, with later includes and local
settings taking precedence. `load_and_expand_scenario` remains the entry point;
it validates the configuration and resolves targets without obsolete preset
expansion passes. Configure evaluation in scenario YAMLs. The unused legacy
files under `configs/evaluation/`, protocol interfaces, factory test runner,
logger methods, and output-directory helpers are retained in Git history.
Factory checks now run with the regular readiness tests. Current logging uses
training hooks, `CSVLogger.log_training_episode`, and `WandbLogger.log_metrics`.

## Historical algorithms

SAC, TD3, DQN, and the A2C/DDPG/QR-DQN/TQC compatibility configurations were retired
from the active tree. Their source, scenarios, sweeps, and obsolete distributed
replay documentation are preserved at Git revision
`7c13266d109a797b202f1f22d5606ecb0f7f9851`.
Use a separate historical checkout to inspect or reproduce those runs:

```bash
git worktree add --detach ../F110_MARL-legacy 7c13266d109a797b202f1f22d5606ecb0f7f9851
```

Those names did not all identify distinct implementations: A2C used PPO, DDPG and
TQC configurations used SAC, and QR-DQN configurations used DQN. Historical
algorithm labels should be interpreted accordingly.

## Validation and remaining correctness work

```bash
venv/bin/python -m compileall -q run.py src tests
PYGLET_HEADLESS=true venv/bin/python -m pytest tests/ -q
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo.yaml --no-wandb --episodes 1 --quiet
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/mappo_gaplock.yaml --no-wandb --episodes 1 --quiet
rg "stable_baselines3|from gymnasium|from pettingzoo" run.py src configs scenarios
```

The dependency guard should produce no matches. Contract tests also check retained
scenario resources, invalid algorithm/role combinations, and sweep CLI arguments.
Before cleanup the suite had 147 passes and one scan-isolation failure caused by a
scenario-dependent map fixture; that fixture now declares its own distinct maps.

The follow-up correctness fixes reset fixed opponents in PPO/MAPPO training,
including FTG steering and cutback history and the hybrid controller's FTG state.
Checkpoint-selection evaluation now ends an action repeat when any acting agent
terminates, matching training and CLI evaluation.

An explicit `env.reset(seed=...)` restarts the map scheduler from its configured
order and reseeds its RNG. Unseeded resets continue the existing cycle. Evaluation
uses `reset(seed=base_seed + episode, options={"map_episode_index": episode})`:
the index advances that seeded schedule before selecting the map, preserving
multi-map coverage without depending on sizing resets or earlier evaluations.
The index must be nonnegative and requires an explicit seed.

Dataset directories must be absent or empty. The writer reserves the directory
immediately with `metadata.json` (`complete: false`) and sets `complete: true`
only after successful close. Existing datasets cannot be resumed or overwritten;
use a new directory for each run. Chunk files are also created exclusively.
The array schema remains `2.0`. New `run.py` datasets record
`transition_contract.version: "1.0"`, with `global_state: pre_decision` and
`lifecycle_fields: post_decision`. Both PPO and MAPPO now pair `obs_t` with `s_t`.
Older datasets without this metadata retain their historical semantics: PPO
recorded `s_{t+1}`, while MAPPO recorded `s_t`; do not mix them without an explicit
conversion. A first state cannot generally be recovered from old PPO recordings.

Run provenance now includes `behavior_contracts` for opponent resets, seeded map
scheduling, and evaluation action-repeat boundaries. Checkpoints with older
provenance require `--allow-provenance-mismatch` for intentional reevaluation
under the corrected behavior. These fixes can change trajectories, baseline
scores, and selected checkpoints, so regenerate matched baseline/evaluation runs
before comparing new results with historical experiments. Observation dimensions,
action bounds, reward definitions, and decentralized actor inputs are unchanged.
