# Circle race with randomized MPC traffic

Train:

```bash
venv/bin/python run.py --scenario scenarios/ppo_1v1_mpc_traffic_circle.yaml
```

Add `--render` to view training. The PPO learner is blue, its designated racing
MPC opponent (`car_1`) is red, and five background traffic cars are gray:
kinematic MPC, obstacle-aware MPC, defensive MPC, CBF MPC, and MPCC.
Their controller types and speed ranges are configured per car in the scenario.

Every episode independently samples all seven cars' positions around the entire
circle, including the opponent. Sampling is uniform by centerline arc length,
with at least 2 m between car centres. A fixed reset seed reproduces the grid;
evaluation also uses randomized grids determined by its evaluation seeds.
The traffic count and controller mix are fixed; the positions and ordering vary.

The pretrained checkpoint, observation, reward, and learning settings inherit from
`ppo_1v1_racing_mpc_circle.yaml`. Training and evaluation have no lap limit;
evaluation retains its 120,000-step safety cap. Only `car_1` is the reward target. Background
traffic laps and crashes do not produce finish events or crash bonuses.

Any ego collision or boundary violation ends the episode. Fixed cars recover
from wall, boundary, or fixed-car collisions on the centerline while preserving
lap counts and episode time. The target's recovery still earns the original
+1 bonus, unless ego also crashes that step. Traffic-only recoveries earn no
bonus. Ego sees traffic through its existing LiDAR observation.
