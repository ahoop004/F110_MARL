# PPO versus racing MPC on circle_map

Run:

```bash
venv/bin/python run.py --scenario scenarios/ppo_1v1_racing_mpc_circle.yaml
```

The learner loads actor and critic from `outputs/L_map_pretrain/best_model.pt`,
keeps the original 158-input observation, and starts with a fresh optimizer.
The fixed racing MPC uses the validated 3.5 m/s profile. The learner starts
approximately 3 m behind it. Training uses one environment, 1024-step PPO
rollouts, a constant 0.0001 learning rate and a 120M-transition budget.

The reward retains signed metre progress and the exclusive -1 boundary cost.
`configs/reward/tasks/race_1v1_pursuit.yaml` adds -0.01 per decision while behind
and +1 for each isolated opponent crash. Ordering tracks unwrapped progress
across the finish seam; respawn displacement earns no progress. There is no
extra finish bonus or collision penalty beyond the inherited reward.

Ego boundary violations, ego collisions, vehicle-to-vehicle collisions, or the
MPC completing five laps terminate training episodes. Ego lap completion does
not terminate. There is no training timeout. An isolated MPC boundary/wall crash
resets only that vehicle to rest at a nearby unoccupied centerline point,
preserving completed laps and episode time. A simultaneous ego crash suppresses
the respawn bonus.

Evaluation uses the same rules with twenty MPC laps and a 120,000-step safety
cap. Selection evaluates eight seeded episodes every 409,600 transitions;
final evaluation uses twenty episodes. `--checkpoint` overrides initialization;
`--total-steps` overrides the training budget.
