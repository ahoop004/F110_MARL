# Circle race with randomized MPC traffic

Scenario settings are now inline. Select parameter choices in the canonical YAML
file or use `--set KEY=YAML`; there are no scenario inheritance files in `configs`.

Train:

```bash
venv/bin/python run.py --scenario scenarios/ppo_1v1_racing_mpc_circle.yaml \
  --set 'environment.spawn.policy="centerline_random"' \
  --set environment.spawn.centerline.min_distance=2.0 \
  --set 'environment.respawn_agents=["car_1","car_2","car_3","car_4","car_5","car_6"]' \
  --set environment.respawn_on_vehicle_collision=true \
  --set 'environment.rendering={"vehicle_colors":{"car_0":"#3288ff","car_1":"#ff2020","car_2":"#a0a0a0","car_3":"#a0a0a0","car_4":"#a0a0a0","car_5":"#a0a0a0","car_6":"#a0a0a0"}}' \
  --set 'experiment.name="ppo_1v1_mpc_traffic_circle"' \
  --set 'agents.car_2={"algorithm":"kinematic_mpc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":1.5,"min_speed":0.5,"max_speed":2.0}}' \
  --set 'agents.car_3={"algorithm":"obstacle_aware_mpc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":2.0,"min_speed":0.5,"max_speed":2.5}}' \
  --set 'agents.car_4={"algorithm":"defensive_mpc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":2.0,"min_speed":0.5,"max_speed":2.5}}' \
  --set 'agents.car_5={"algorithm":"cbf_mpc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":2.5,"min_speed":0.5,"max_speed":3.0}}' \
  --set 'agents.car_6={"algorithm":"mpcc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":2.5,"min_speed":0.5,"max_speed":3.0}}'
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

The pretrained checkpoint, observation, reward, and learning settings are configured in
`ppo_1v1_racing_mpc_circle.yaml`. Training and evaluation have no lap limit;
evaluation retains its 120,000-step safety cap. Only `car_1` is the reward target. Background
traffic laps and crashes do not produce finish events or crash bonuses.

Any ego collision or boundary violation ends the episode. Fixed cars recover
from wall, boundary, or fixed-car collisions on the centerline while preserving
lap counts and episode time. The target's recovery still earns the original
+1 bonus, unless ego also crashes that step. Traffic-only recoveries earn no
bonus. Ego sees traffic through its existing LiDAR observation. Five extra inputs
always describe its configured target (`car_1`), regardless of which traffic car
is closest. The observation has 163 inputs; pretrained actor/critic input layers
are expanded with zero new columns, preserving initial behavior.
