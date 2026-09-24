# PPO versus racing MPC on circle_map

Run:

```bash
venv/bin/python run.py --scenario scenarios/ppo_1v1_racing_mpc_circle.yaml
```

The learner loads actor and critic from `outputs/L_map_pretrain/best_model.pt`,
preserves the original 158 driving inputs and appends five target-opponent inputs
(163 total). The actor and critic receive five zero-initialized input columns,
preserving their initial outputs, and training starts with a fresh optimizer.
The fixed racing MPC uses the validated 3.5 m/s profile. The learner starts
approximately 3 m behind it. Training uses one environment, 1024-step PPO
rollouts, a constant 0.0001 learning rate and a 120M-transition budget.

The reward retains signed metre progress and the exclusive -1 boundary cost.
`configs/reward/tasks/race_1v1_pursuit.yaml` adds -0.01 per decision while behind
and +1 for each isolated opponent crash. Ordering tracks unwrapped progress
across the finish seam; respawn displacement earns no progress. There is no
extra finish bonus. Ego collision termination adds a -1 penalty to that step's
progress and trailing reward.

Ego boundary violations and vehicle collisions terminate training episodes.
Lap counts never terminate episodes. There is no training timeout. An isolated MPC boundary/wall crash
resets only that vehicle to rest at a nearby unoccupied centerline point,
preserving completed laps and episode time. A simultaneous ego crash suppresses
the respawn bonus.

Evaluation also has no lap limit and retains a 120,000-step safety cap. Selection evaluates eight seeded episodes every 409,600 transitions;
final evaluation uses twenty episodes. `--checkpoint` overrides initialization;
`--total-steps` overrides the training budget.

The target slot follows the scenario's `target_id` and contains signed wrapped
track distance / 30 m, lateral separation / 5 m, along-track relative speed /
20 m/s, lateral relative speed / 20 m/s, and a presence mask. Missing targets
produce five zeros. Sensing uses simulator state at any distance; values are
not clipped. Wrapped distance describes relative location, not a lap-count lead.

`pretrained_observation_extension: target_frenet` explicitly enables 158-to-163
input expansion for training initialization. The original observation prefix,
physics, action bounds, and network architecture must otherwise match. Saved
163-input checkpoints subsequently load normally, including for evaluation.
