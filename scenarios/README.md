# Scenario index

Each scenario is a complete, standalone experiment. Environment, vehicle,
observation, reward, training and logging settings are inline. Optional recording
settings and coordinated alternatives are documented in the same YAML file.
There are no scenario includes and no `configs/scenarios` directory. Component
configs remain available for library consumers and focused component checks.

| Scenario under `scenarios/` | Workflow and choices |
|---|---|
| [ppo_lap_completion_pretrain.yaml](ppo_lap_completion_pretrain.yaml) | Single-car PPO; local/HPC and held-out evaluation choices |
| [ppo_lap_completion_transfer.yaml](ppo_lap_completion_transfer.yaml) | PPO fine-tuning or matched scratch; continuous/three-lap transfer |
| [ppo_lap_completion_curriculum.yaml](ppo_lap_completion_curriculum.yaml) | Evaluation-driven progression through maps |
| [ppo_1v1_racing_mpc_circle.yaml](ppo_1v1_racing_mpc_circle.yaml) | PPO pursuit versus MPC; optional extra traffic |
| [mappo_2v2_continuous.yaml](mappo_2v2_continuous.yaml) | Continuous two-learner driving; scratch/pretrained/LoRA |
| [mappo_2v2_race.yaml](mappo_2v2_race.yaml) | Finite team racing; combined/completion/first-place/sweep/penalty or individual rewards |
| [mappo_2v2_asymmetric.yaml](mappo_2v2_asymmetric.yaml) | Different racing/crash-pressure rewards for the two learners |
| [mappo_2v2_selfplay.yaml](mappo_2v2_selfplay.yaml) | Two trainable teams; optional selective/windowed recording |
| [experiments/ppo_combined_slip_circle_stable.yaml](experiments/ppo_combined_slip_circle_stable.yaml) | Circle convergence ablation with distinct reward/optimization settings |
| [render/racing_mpc.yaml](render/racing_mpc.yaml) | Fixed MPC solo, 2v2 or passing visualization |
| [calibration/controller.yaml](calibration/controller.yaml) | Legacy controller calibration; hybrid/pure-pursuit and lap choices |
| [legacy/ppo.yaml](legacy/ppo.yaml) | Historical attacker/defender PPO |
| [legacy/ppo_time_trial.yaml](legacy/ppo_time_trial.yaml) | Historical single-car PPO time trial |
| [legacy/ppo_racing.yaml](legacy/ppo_racing.yaml) | Historical PPO versus pure pursuit, Stanley or hybrid |
| [legacy/mappo_gaplock.yaml](legacy/mappo_gaplock.yaml) | Historical adversarial MAPPO |
| [legacy/mappo_2v2.yaml](legacy/mappo_2v2.yaml) | Historical team racing and fixed-opponent choices |
| [legacy/complete_4.yaml](legacy/complete_4.yaml) | Historical four-learner race; observation/reward comparisons |

## Choosing parameters

Edit the active values in the YAML. Comments next to settings explain their
meaning and available options. The commented recipes at the end list coordinated
changes for the former variants; start from the active configuration before
applying a recipe. Recipes do not load other files.

You can also pass repeatable `--set KEY=YAML` arguments. Values are parsed as YAML:
booleans, numbers, lists and mappings retain their types. A mapping replaces the
selected subtree; `--set 'optional.key=!delete'` removes a key. A YAML/JSON list of
assignment strings can group choices for a sweep. Dedicated flags such as
`--seed`, `--episodes`, `--num-envs` and `--rollout-steps-per-env` take precedence.
Overrides are applied before validation and are included in resolved run provenance.

```bash
# Local PPO: one environment, 1024 transitions per update.
python3 run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --num-envs 1 --num-workers 1 --rollout-steps-per-env 1024 --no-render

# HPC PPO: 400 environments on 100 CPU workers, 409600 pooled transitions.
python3 run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --num-envs 400 --num-workers 100 --rollout-steps-per-env 1024 --no-render

# Continuous team driving with a pretrained actor and per-agent rank-4 LoRA.
python3 run.py --scenario scenarios/mappo_2v2_continuous.yaml \
  --pretrained-actor outputs/PRETRAIN_RUN \
  --set 'training_defaults.lora={mode: per_agent, rank: 4, alpha: 4.0, train_log_std: true}'
```

The PPO base preserves its historical one-environment, 409600-transition rollout;
the local command deliberately changes optimization frequency. MAPPO continuous
training defaults to 400 environments and 100 workers. The finite team-race base
preserves the former combined-objective scenario's serial defaults. Its penalty
recipe also sets the former penalty arm's worker, rollout and evaluation cadence.

Scratch uses a null checkpoint. MAPPO actor initialization uses
`training_defaults.pretrained_actor_checkpoint` or `--pretrained-actor`; PPO
transfer uses `experiment.checkpoint` or `--checkpoint`. LoRA is null for full
actor training, or a mapping with `mode: shared|per_agent`, positive `rank`,
`alpha` and `train_log_std`. LoRA requires a matching pretrained source.

## Evaluation and transfer

Use the same scenario and parameter choices as the trained checkpoint, then add
`--eval --checkpoint PATH --eval-protocol final`. The held-out evaluation recipes
set separate maps/seeds and `experiment.evaluation_only: true`; intentional
cross-scenario evaluation also requires `--allow-provenance-mismatch`.
`--set experiment.evaluation_only=true` makes any canonical scenario reject training.
Flattening or renaming a scenario changes its source/configuration hashes. When
evaluating an existing checkpoint with equivalent settings, an explicit
`--allow-provenance-mismatch` may therefore be needed; physical, action and
observation compatibility checks still apply.
Final-test seeds remain separate from checkpoint selection. Keep physics, action
units, observation ordering/scales and network dimensions compatible with the source.

## Two trainable teams

`mappo_2v2_selfplay.yaml` trains separate team policies and critics. This is a
different training contract from fixed-opponent MAPPO; it remains a separate
workflow. Pretrained actor initialization and LoRA are not supported on this path.
See [parallel MAPPO](../docs/PARALLEL_MAPPO.md) for scheduling and return semantics.

Use `--record-races` to enable the commented recording defaults. For windows,
apply the windowed-recording recipe at the end of the self-play YAML. Training
and evaluation recording have independent caps. Review output with
`notebooks/run_review.ipynb`.

## Historical comparisons and fixed controllers

The `legacy/` files retain legacy physics and distinct training topologies.
Their former observation, opponent and reward variants are parameter recipes in
the corresponding file. The complete-four benchmark selects observation arms
with `--observation baseline|frenet|frenet_neighbors` on one scenario.

[Render instructions](render/README.md) cover the fixed-controller options.
Historical benchmark summaries retain their original scenario identifiers;
those identifiers describe the measured configuration, not current file paths.
