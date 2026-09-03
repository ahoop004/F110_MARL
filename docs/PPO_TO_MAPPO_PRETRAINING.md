# PPO actor pretraining for MAPPO

Train the compatible single-agent actor with:

```bash
PYGLET_HEADLESS=true python3 run.py \
  --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --no-wandb
```

The selected checkpoint is written to the run output directory as
`best_model.pt`. Selection is based on deterministic held-out evaluation in
this order: lap-completion rate, lower collision rate, mean lap progress, then
lower mean finish steps. `evaluation_history.jsonl` records every selection
decision.

To initialize a MAPPO shared actor, add the checkpoint to the focal MAPPO
agent's parameters. Relative paths are resolved from the scenario file:

```yaml
agents:
  car_0:
    params:
      pretrained_actor_checkpoint: ../outputs/ppo_lap_completion_pretrain/<run-id>/best_model.pt
```

Only actor weights transfer. MAPPO retains a newly initialized centralized
critic and a fresh optimizer. Loading fails if the observation dimension,
action dimension or bounds, hidden layers, activation, or actor state shapes
do not match. The source path and SHA-256 digest are recorded in run
provenance.
