# F110_MARL

Pure PyTorch PPO and MAPPO experiments for F1TENTH racing. The repository supports
single-agent learning against fixed controllers, multi-agent learning with a
shared actor and centralized critic, checkpoint evaluation, and offline datasets.

## Run an experiment

Use the project virtual environment when available. Dependencies are listed in
`requirements.txt`; development checks use pytest. Headless examples:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/legacy/ppo.yaml --no-wandb --episodes 1
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/legacy/mappo_gaplock.yaml --no-wandb --episodes 1
```

Use `--seed` for repeatability, `--output-dir` for a specific output location,
`--dataset-dir` for transition recording, and `--render` for local visualization.
PPO and MAPPO support `experiment.total_steps` or `--total-steps N` for an
aggregate environment-decision budget, and `--episodes N` for an episode budget.
An explicit `--episodes` removes the configured step budget; the two CLI budget
options are mutually exclusive. For MAPPO, one joint race decision counts once,
while individual learner samples are tracked separately.
W&B is optional and `--no-wandb` overrides scenario logging settings.

Evaluate a checkpoint with the same scenario and experiment overrides used for
training (the resolved configuration is checked against checkpoint provenance):

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/legacy/ppo.yaml --no-wandb --episodes 1 --eval --checkpoint outputs/example/checkpoint_ep000000.pt --eval-episodes 5
```

Replace the example checkpoint path with the checkpoint produced by your run.
`--allow-provenance-mismatch` is available for intentional cross-scenario evaluation.
For fixed-controller-only scenarios, use the ordinary episode command without
`--eval` or a checkpoint.

### Load the best PPO model on another track

`--checkpoint` accepts a checkpoint file or a source run directory containing
`best_model.pt`. With PPO training, it loads the actor and critic and starts a
new optimizer, learning-rate schedule, and episode budget using the destination
scenario. This is fine-tuning, not an exact interrupted-run resume. The source
path and SHA-256 are recorded in run/checkpoint provenance. Use a new output
directory; reusing the source checkpoint directory is rejected.

You can also set the path in the scenario YAML for training or `--eval`:

```yaml
experiment:
  checkpoint: ../outputs/ppo_lap_completion_pretrain/YOUR_RUN/best_model.pt
```

YAML paths are relative to the scenario directory; absolute paths and run
directories also work. `--checkpoint` overrides this field and remains relative
to the working directory. Leave `checkpoint: null` to disable YAML loading.

`scenarios/ppo_lap_completion_transfer.yaml` inherits the pretraining observation,
reward, action, and network settings and selects circle_map. It uses a constant
1e-4 learning rate and a matched 4,096,000-transition destination budget. To use another track,
copy it, change its experiment name, and update all three map bundle lists.
First evaluate the original policy, then fine-tune, then evaluate the new policy:

```bash
# Replace outputs/PRETRAIN_RUN with the directory containing your best_model.pt.
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo_lap_completion_transfer.yaml --eval --checkpoint outputs/PRETRAIN_RUN --allow-provenance-mismatch --eval-protocol final --output-dir outputs/circle_before --no-wandb
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo_lap_completion_transfer.yaml --checkpoint outputs/PRETRAIN_RUN --output-dir outputs/circle_finetune --no-wandb
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo_lap_completion_transfer.yaml --eval --checkpoint outputs/circle_finetune --eval-protocol final --output-dir outputs/circle_after --no-wandb
```

Compare `evaluation_report.json` in the before/after directories for completion,
collisions, progress, and finish times under the same final-test seeds and horizon.
The initial evaluation measures transfer without further learning; after fine-tuning,
the destination track is part of training. Keep final-test results out of checkpoint
selection. Omit `--checkpoint` and leave `experiment.checkpoint: null` for a matched
from-scratch baseline. Both fine-tuning and scratch arms use the same destination learning rate and budget.
Network dimensions, activation, and physical action contracts must match the source;
keep observation component meanings/order compatible as well as their dimensions.

## Experiment catalog

The active scenarios share the current MF6.1 physics and driving observation.
See [the scenario index](scenarios/README.md) for the retained historical files.

| Experiment | Scenario files under `scenarios/` |
|---|---|
| PPO pretraining (400 environments) | `ppo_lap_completion_pretrain.yaml` |
| PPO transfer versus scratch | `ppo_lap_completion_transfer.yaml` |
| PPO cross-map evaluation | `ppo_lap_completion_validate.yaml` |
| MAPPO traffic adaptation | `mappo_2v2_completion.yaml` |
| MAPPO main team objective | `mappo_2v2_combined.yaml` |
| MAPPO objective comparisons | `mappo_2v2_first_place.yaml`, `mappo_2v2_sweep.yaml` |
| MAPPO individual completion baseline | `mappo_2v2_individual.yaml` |
| MAPPO held-out evaluation | `mappo_2v2_validate.yaml` |
| Circle convergence ablation | `experiments/ppo_combined_slip_circle_stable.yaml` |
| Historical experiments / controller calibration | `legacy/*.yaml`, `calibration/*.yaml` |

Use `--pretrained-actor outputs/PRETRAIN_RUN` for PPO-to-MAPPO initialization;
omit it for scratch. Active team scenarios select checkpoints with deterministic
evaluation every 100 episodes. The `*_validate.yaml` entry points require `--eval`.

MAPPO's individual arm uses per-agent rewards and an agent-conditioned critic,
`V_i(s)`. The team arm uses a configured team reward reduction and shared team
critic, `V(s)`. Actors use local observations in both modes. Raw shaped rewards
across these arms are not interchangeable evaluation scores.

The pretraining scenario currently uses `L_map` for both train and evaluation;
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
| `src/physics/` | Vehicle integration, tire dynamics, sensing, and collisions |
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

The existing nonlinear physics now uses `combined_slip_st` version 2 with a
planar MF6.1 force kernel, load-dependent combined slip, simultaneous load
transfer, and independent first-order wheel/steering actuators. The former
reduced friction-circle implementation has been replaced. Legacy physics remains
available for existing legacy scenarios. Parameters are synthetic and explicitly
uncalibrated; see [physics details and limitations](docs/PHYSICS_MODEL.md).

The canonical `scenarios/ppo_lap_completion_pretrain.yaml` configures the paper's
120-million-transition budget, 400 environments, and 1,024 transitions per worker
per rollout. It uses 50 vehicle/Frenet/track values with fixed track scales across maps, wheel-reference
acceleration, 0.05 s decisions, and episode friction randomization (relative
standard deviation 0.02). Reward is signed distance progress in metres, replaced
by -1 on a geometric boundary violation. Training resets only at that boundary;
lap crossings do not end an episode. Rollouts continue across resets.

Learning-rate decay, periodic checkpoints, and evaluation use transition counts.
Evaluation measures 20 laps with nominal grip and permits excursions to measure
off-track error. Selection uses eight starts and the final protocol uses 20 independent starts
on the same map. `final_model.pt` contains the final update; `best_model.pt` uses
the documented lap-time selection rule. Duplicate `_frenet` and
`_combined_slip` pretraining entry points have been removed.
For a smaller parallel experiment, set pooled `params.n_steps = num_envs * 1024`.

Vehicle parameters belong in `configs/vehicle/combined_slip.yaml`, under
`environment.vehicle_params`. They remain uncalibrated, and observation maxima
and N=20 remain provisional. `L_map` matches the reported dimensions but is an
approximation. See [physics and protocol details](docs/PHYSICS_MODEL.md) for the
remaining reproduction limits. No result on this map establishes sim-to-real
performance. Configure all three map bundle lists when changing tracks.
The separate `ppo_lap_completion_validate.yaml` tests collision and boundary
termination on L_map, circle_map, and Budapest_map; evaluation reports contain
per-map summaries. Fixed track scaling changes observation semantics and requires
fresh pretraining. See [the current workflow](docs/PPO_TO_MAPPO_PRETRAINING.md)
for matched scratch comparisons, independent seeds, and acceptance criteria.

For L-first training that automatically adds maps after five clean laps and
checks retention across the full bundle, see [the map curriculum workflow](docs/PPO_MAP_CURRICULUM.md).
Its entry point is `scenarios/ppo_lap_completion_curriculum.yaml`.

Use the independent final evaluation seeds after selecting a checkpoint:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml --eval --checkpoint outputs/YOUR_PRETRAIN_RUN --eval-protocol final --output-dir outputs/mf61_final --no-wandb
```

Replace the run directory with one produced by the same scenario and physics
contract. Final seeds use the configured evaluation track and do not establish
sim-to-real performance. The migrated `mappo_2v2_completion.yaml` supports the new actor
with 18 appended neighbor/team inputs initialized to zero weight. Other legacy MAPPO
scenarios still need compatible physics/observation/action configurations.

For the circle convergence experiment, use
`scenarios/experiments/ppo_combined_slip_circle_stable.yaml` with the same `run.py` command.
It preserves the three-lap task, observations, physical actions, spawn protocol,
and tires. It reduces the time cost from 0.1 to 0.01 per simulated second,
lowers the initial learning rate to 1e-4, uses four epochs and 256-decision
minibatches, and stops further minibatches when sampled KL exceeds 0.01.
Discount and GAE horizons are approximately 20 s and 5 s. Initial Gaussian
log standard deviation is -0.5 and the entropy coefficient is 0.001.
These are experimental settings, not evidence of convergence. Compare
deterministic completion/progress and finish time at matched decision budgets;
raw rewards are not comparable to the original reward configuration.

PPO's optional `min_rollout_steps: 2048` pools independently bootstrapped
fragments across short episodes and parallel collector rounds while the actor
stays frozen. Episode boundaries still stop GAE. The last partial pool is
updated at training shutdown, so the final update may be smaller. Without this
option the historical immediate fragment-update behavior remains available.
The threshold is checked after each fragment/collector round, so a pooled
update may exceed `min_rollout_steps` by the size of the final collected group.
Metrics include rollout size, optimizer steps, KL early stopping, Gaussian
standard deviation, and action saturation. For circle fine-tuning, pass a
compatible combined-slip checkpoint with `--checkpoint` to this circle scenario;
the optimizer/schedule restart, while the checkpoint's learned standard deviation
overrides `log_std_init`. The transfer YAML targets circle_map.

PPO and MAPPO now retain pre-tanh action samples internally to score saturated
actions correctly and use the entropy of the squashed Gaussian (32-point
quadrature). The Gaussian KL estimate uses `mean(exp(log_ratio)-1-log_ratio)`.
Observation/action dimensions, deterministic inference, checkpoint parameter
layouts, and dataset schemas are unchanged. These training behavior changes
are recorded in provenance; older evaluation checkpoints require the existing
explicit provenance-mismatch override. Historical training curves must be
identified by code revision, even when using the original scenario YAML.

Lap counting cancels backward passages through the finite finish segment before
granting another forward lap. Local loops and reversing across the line cannot
earn extra laps; `count_initial_crossing_as_lap` still controls the first valid
forward crossing. Finish segments must span the drivable track width. With
`completion_progress` selection, earned progress breaks ties while some races
remain incomplete. Once every evaluation race finishes, safety and finish time
determine the best checkpoint, ignoring finish-line overshoot. These behavior
changes are recorded in run provenance. Re-evaluate older checkpoints before
comparing completion rates or selected-model lap times across this change;
`--eval --allow-provenance-mismatch` explicitly acknowledges the changed behavior
when loading an older checkpoint for evaluation.

For parallel PPO in this repository, `n_steps` is the **pooled collection-round** limit:
`steps_per_worker = n_steps / num_envs`. With episode budgets, episode ends flush
shorter fragments, so without `min_rollout_steps` the actual update size can be
smaller. Transition budgets keep collecting across resets. At a 0.01 s decision interval,
`n_steps=2048` gives at most 20.48 s per worker with one environment, 5.12 s with
four, and 2.56 s with eight. More workers at fixed `n_steps` therefore shorten
the sampled GAE trajectories and increase dependence on critic bootstrapping.
To preserve the fragment horizon, scale `n_steps` proportionally to `num_envs`;
this also increases the pooled batch and changes update frequency. Compare runs
at matched environment-decision budgets as well as wall time: episode budgets
and episode-based learning-rate decay do not guarantee equal samples.

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
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/legacy/ppo.yaml --no-wandb --episodes 1 --quiet
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/legacy/mappo_gaplock.yaml --no-wandb --episodes 1 --quiet
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
