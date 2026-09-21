# Scenario index

Interactive fixed-controller previews are in [`render/`](render/README.md):
MPC solo, MPC versus hybrid 2v2, and a passing demo on circle and Budapest.

The active entry points below use the shared MF6.1 vehicle profile. PPO keeps
400 environments; the explicit MAPPO base and penalty scratch/pretrained pairs
also use 400 environments across 100 workers, with two learners and two fixed
racing MPC opponents per race. Other MAPPO objectives retain their serial defaults.
All active `mappo_2v2_*.yaml` scenarios inherit the same opponent profile. Historical scenarios and render comparisons retain
their original controllers. See [the training workflow](../docs/PPO_TO_MAPPO_PRETRAINING.md)
for commands, checkpoint compatibility, evaluation protocols, and remaining
physics calibration work.

| Entry point | Use |
|---|---|
| `ppo_lap_completion_pretrain.yaml` | Train the reusable 50-input driving actor on L_map |
| `ppo_lap_completion_transfer.yaml` | Circle fine-tuning or matched scratch training |
| `ppo_lap_completion_validate.yaml` | Evaluate driving on L_map, circle, and Budapest |
| `mappo_2v2_base_scratch.yaml` | Stage 1: base completion reward, random initialization |
| `mappo_2v2_base_pretrained.yaml` | Stage 1: base completion reward, PPO actor initialization |
| `mappo_2v2_penalties_scratch.yaml` | Stage 2: placement and incident penalties, random initialization |
| `mappo_2v2_penalties_pretrained.yaml` | Stage 2: placement and incident penalties, PPO actor initialization |
| `mappo_2v2_completion.yaml` | Shared completion reward for traffic adaptation |
| `mappo_2v2_combined.yaml` | Main combined finishing-position objective |
| `mappo_2v2_first_place.yaml` | First-place objective comparison |
| `mappo_2v2_sweep.yaml` | First-and-second-place objective comparison |
| `mappo_2v2_individual.yaml` | Individual-reward completion comparison |
| `mappo_2v2_validate.yaml` | Held-out Silverstone/Spa evaluation |

Both validation scenarios require `--eval --checkpoint PATH`. Training MAPPO
from a PPO source uses `--pretrained-actor PATH`; omitting it uses the scenario
default (scratch except for the explicitly pretrained base and penalty arms).
MAPPO's observation has 68 inputs, including explicit teammate identity. Its
shared settings live in `configs/scenarios/mappo_2v2_base.yaml`; edit that fragment
when a setting should apply to every team objective.

## Current-setup three-lap transfer

`ppo_lap_completion_transfer_3lap.yaml` loads
`outputs/ppo_current_pretrain_s42/best_model.pt` and currently targets Budapest.
It inherits the current vehicle profile, fixed track observation scaling, and
400 workers. Episodes end after three laps or 16,000 steps (800 simulated seconds).
Training uses geometric boundary resets; evaluation also terminates on collisions.
Checkpoint selection uses completion and net progress. The destination budget is
4,096,000 decisions. Override the source with `--checkpoint PATH` for another run.

```bash
# Evaluate the new source before destination fine-tuning.
# The provenance override acknowledges the changed scenario/map, not changed physics.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_transfer_3lap.yaml \
  --eval --allow-provenance-mismatch --no-wandb \
  --output-dir outputs/budapest_3lap_before

# Fine-tune actor and critic with a fresh optimizer in a separate output directory.
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/ppo_lap_completion_transfer_3lap.yaml
```

The older downloaded model and its compatibility configuration are retained as
historical artifacts. Active transfer scenarios no longer include that profile.
The generic `ppo_lap_completion_transfer.yaml` remains a scratch baseline unless
`--checkpoint PATH` is supplied.

## Experiment order

1. Run `mappo_2v2_base_scratch.yaml` and `mappo_2v2_base_pretrained.yaml`.
   Both use the existing shared completion reward and `team_completion` selection.
2. Run the penalty scratch/pretrained pair below using the same frozen PPO source,
   training seeds, race settings, and parallel collection settings.
3. Compare LoRA combinations under the same base and penalty tasks after adapter
   training is implemented. These are planned experiments, not runnable scenarios.

All four current arms share `configs/training/mappo_parallel.yaml`. Both pretrained
arms initialize the full actor from PPO and train it normally; their critics and
optimizers start fresh. The base-to-penalty comparison changes the task's reward
and selection objective, so it is not an isolated penalty-only ablation.

## Matched team penalty experiments

`mappo_2v2_penalties_scratch.yaml` and `mappo_2v2_penalties_pretrained.yaml`
compare random initialization with the new current-setup L-map actor under
identical physics, observations, opponents, and event-based penalties. The
pretrained arm defaults to `outputs/ppo_current_pretrain_s42/best_model.pt`;
use `--pretrained-actor PATH` to select a different current-setup source.
Both penalty arms default to 400 environments across 100 CPU workers, with
256 decisions per environment per collection round. See
[Parallel MAPPO](../docs/PARALLEL_MAPPO.md) for HPC launch commands, bounded
smoke tests, and throughput metrics.
See [the experiment protocol](../docs/TEAM_RACING_EXPERIMENTS.md)
for pretraining, penalty definitions, scoring, and remaining comparison limits.

## Migration

| Previous entry point | Replacement |
|---|---|
| `ppo_lap_completion_pretrain_frenet.yaml` | `ppo_lap_completion_pretrain.yaml` |
| `ppo_lap_completion_pretrain_combined_slip.yaml` | `ppo_lap_completion_pretrain.yaml` |
| `mappo_2v2_frenet_ppo_pretrained.yaml` | `mappo_2v2_base_scratch.yaml` | Stage 1: base completion reward, random initialization |
| `mappo_2v2_base_pretrained.yaml` | Stage 1: base completion reward, PPO actor initialization |
| `mappo_2v2_penalties_scratch.yaml` | Stage 2: placement and incident penalties, random initialization |
| `mappo_2v2_penalties_pretrained.yaml` | Stage 2: placement and incident penalties, PPO actor initialization |
| `mappo_2v2_completion.yaml` |
| `mappo_2v2_frenet_ppo_pretrained_{combined,first_place,sweep}.yaml` | `mappo_2v2_{combined,first_place,sweep}.yaml` |
| Old `mappo_2v2_individual.yaml` | `legacy/mappo_2v2_individual.yaml` |
| `ppo_combined_slip_circle_stable.yaml` | `experiments/ppo_combined_slip_circle_stable.yaml` |
| Other old PPO / `complete_4*` / MAPPO scenarios | Same filename under `legacy/` |

The removed PPO aliases had different worker counts and hyperparameters. The
canonical replacement deliberately uses the retained 400-worker configuration;
it is not a bit-for-bit recreation of an old alias run. New 68-input team actors
cannot load old 65-input MAPPO checkpoints. Compatible 50-input PPO actors still
transfer with the 18 additional inputs initialized to zero weight.

## Retained comparisons

- `experiments/`: distinct current-physics ablations. The circle convergence
  scenario explicitly selects circle for both training and evaluation.
- `legacy/`: fifteen historical experiments retained to reproduce old comparisons.
  Their older dynamics, observations, and actions are not compatible with current
  pretraining. Relative config paths have been adjusted for their new location.
- `calibration/`: historical fixed-controller calibration on legacy physics.
  These results do not validate opponents under the new MF6.1 setup.

Do not use held-out maps to select checkpoints or tune the current experiments.
The active MAPPO training/selection maps are Budapest and circle; Silverstone and
Spa are reserved for the team validation scenario.
