# MAPPO LoRA experiments

Scenario settings are now inline. Select parameter choices in the canonical YAML
file or use `--set KEY=YAML`; there are no scenario inheritance files in `configs`.

LoRA adapts a frozen LiDAR-enabled PPO driving actor with small trainable
residuals. The actor observation stays `[108 LiDAR, 50 driving]`; rewards,
termination, collection, and evaluation stay with the chosen continuous or
finite-race parameter configuration. The centralized critic starts fresh and
trains normally.

## Experiment matrix

Each task has the existing scratch and full-fine-tuning controls plus:

| Parameter choice | Adapter routing | Rank | Trainable actor parameters |
|---|---|---|---:|
| `lora_shared` | One adapter used by both teammates | 4 | 3,706 |
| `lora_per_agent` | One adapter for each teammate | 4 each | 7,410 |
| `lora_shared_r8` | One shared adapter; total-capacity control | 8 | 7,410 |

Use `scenarios/mappo_2v2_continuous.yaml` or the penalty recipe in
`scenarios/mappo_2v2_race.yaml`, and set `training_defaults.lora` inline or with `--set`. Counts include the two shared
trainable `log_std` parameters and exclude the critic. The original
158→256→256→2 actor has 107,012 parameters.

Compare shared rank 4 with per-agent rank 4 for equal capacity per car; compare
shared rank 8 with per-agent rank 4 for equal total trainable actor parameters.
These controls do not make the policy classes identical. Use the same frozen PPO
source, seeds, and task protocol across arms. Base and penalty tasks still have
different horizons and budgets, so comparisons between them are not reward-only
ablations.

## Network and routing

Both hidden linear layers use `W(x) + (alpha / rank) * B(A(x))` before the
activation. Original weights, biases, and the action-mean output head are
frozen. `A` starts random and `B` starts at zero, preserving the source policy
at initialization. There is no adapter dropout or automatic merging of weights
when switching between training and evaluation.

The default `alpha / rank` is one. Exploration remains one shared `log_std`
vector for both routing modes; set `train_log_std: false` to freeze it too.
Only trainable actor parameters and critic parameters enter Adam and gradient
clipping. Frozen layers still propagate gradients into earlier adapters.

Per-agent routing uses the ordered trainable IDs: `car_0` gets adapter 0 and
`car_1` gets adapter 1. Routing is retained through parallel environment batching,
rollout fragments, shuffled PPO minibatches, and deterministic evaluation.
An inactive car contributes no actor samples. Its adapter receives no gradient
or Adam momentum update when absent from a minibatch. Both policies retain the
same shared team reward and centralized team critic. These are separate teammate
policies; attacker/defender objectives or dynamic role switching are not enabled.

## Run

First train a compatible 158-input PPO source. Older 50-input PPO checkpoints
cannot initialize these configurations. The commented pretrained recipes use
`outputs/L_map_pretrain/L_map_best_model.pt`; a missing source fails explicitly.

```bash
PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_continuous.yaml --set 'training_defaults.pretrained_actor_checkpoint="../outputs/L_map_pretrain/L_map_best_model.pt"' --set 'training_defaults.lora={"mode":"shared","rank":4,"alpha":4.0,"train_log_std":true}' --set 'experiment.name="mappo_2v2_base_lora_shared"' \
  --pretrained-actor outputs/YOUR_PPO_RUN/best_model.pt \
  --output-dir outputs/base_lora_shared_s42

PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_continuous.yaml --set 'training_defaults.pretrained_actor_checkpoint="../outputs/L_map_pretrain/L_map_best_model.pt"' --set 'training_defaults.lora={"mode":"per_agent","rank":4,"alpha":4.0,"train_log_std":true}' --set 'experiment.name="mappo_2v2_base_lora_per_agent"' \
  --pretrained-actor outputs/YOUR_PPO_RUN/best_model.pt \
  --output-dir outputs/base_lora_per_agent_s42

PYGLET_HEADLESS=true venv/bin/python run.py \
  --scenario scenarios/mappo_2v2_continuous.yaml --set 'training_defaults.pretrained_actor_checkpoint="../outputs/L_map_pretrain/L_map_best_model.pt"' --set 'training_defaults.lora={"mode":"shared","rank":4,"alpha":4.0,"train_log_std":true}' --set 'experiment.name="mappo_2v2_base_lora_shared"' \
  --eval --checkpoint outputs/base_lora_shared_s42/best_model.pt \
  --eval-protocol final --output-dir outputs/base_lora_shared_eval
```

The adapter settings live in `configs/training/mappo_lora.yaml` under
`training_defaults.lora`. Use `null` for ordinary full fine-tuning. A mapping
accepts only `mode` (`shared` or `per_agent`), positive integer `rank`, positive
finite `alpha`, and boolean `train_log_std`. All hidden linear layers are adapted;
rank cannot exceed their input or output dimensions.

Checkpoints contain the entire frozen base, adapters, critic, optimizer, source
path/hash, and the adapter contract including ID routing. Evaluation needs the
matching LoRA scenario but does not need the original PPO file. Mode, rank,
alpha, exploration setting, architecture, and routing mismatches are rejected.
`MAPPOAgent.load` restores optimizer state; the training CLI retains its existing
restriction that `--checkpoint` training is PPO-only, so this does not add exact
MAPPO run resumption. Adapter-only exports and weight merging are not included.

Frozen base weights do not guarantee unchanged driving behavior once adapters
learn. Measure completion, collisions, placement, and driving retention across
seeds; no claim about training speed or racing performance follows from the
parameter counts alone.
