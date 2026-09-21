# Scenario index

The active entry points below use the shared MF6.1 vehicle profile. PPO keeps
400 environments; MAPPO currently runs one environment with two learners and two
fixed opponents. See [the training workflow](../docs/PPO_TO_MAPPO_PRETRAINING.md)
for commands, checkpoint compatibility, evaluation protocols, and remaining
physics calibration work.

| Entry point | Use |
|---|---|
| `ppo_lap_completion_pretrain.yaml` | Train the reusable 50-input driving actor on L_map |
| `ppo_lap_completion_transfer.yaml` | Circle fine-tuning or matched scratch training |
| `ppo_lap_completion_validate.yaml` | Evaluate driving on L_map, circle, and Budapest |
| `mappo_2v2_completion.yaml` | Shared completion reward for traffic adaptation |
| `mappo_2v2_combined.yaml` | Main combined finishing-position objective |
| `mappo_2v2_first_place.yaml` | First-place objective comparison |
| `mappo_2v2_sweep.yaml` | First-and-second-place objective comparison |
| `mappo_2v2_individual.yaml` | Individual-reward completion comparison |
| `mappo_2v2_validate.yaml` | Held-out Silverstone/Spa evaluation |

Both validation scenarios require `--eval --checkpoint PATH`. Training MAPPO
from a PPO source uses `--pretrained-actor PATH`; omitting it trains from scratch.
MAPPO's observation has 68 inputs, including explicit teammate identity. Its
shared settings live in `configs/scenarios/mappo_2v2_base.yaml`; edit that fragment
when a setting should apply to every team objective.

## Migration

| Previous entry point | Replacement |
|---|---|
| `ppo_lap_completion_pretrain_frenet.yaml` | `ppo_lap_completion_pretrain.yaml` |
| `ppo_lap_completion_pretrain_combined_slip.yaml` | `ppo_lap_completion_pretrain.yaml` |
| `mappo_2v2_frenet_ppo_pretrained.yaml` | `mappo_2v2_completion.yaml` |
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
