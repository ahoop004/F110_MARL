# PPO map curriculum

Run the L-first curriculum with:

```bash
PYGLET_HEADLESS=true venv/bin/python run.py --scenario scenarios/ppo_lap_completion_curriculum.yaml
```

This standalone scenario contains the pretraining physics, observations, action contract,
reward and learning-rate schedule. It starts fresh unless `--checkpoint` is
provided. It defaults to **400 parallel environments**, each collecting 1,024
policy decisions per PPO update (409,600 transitions pooled). CPU workers collect
experience; the parent performs batched policy inference and PPO updates on the
configured device. The original pretrain and transfer scenarios retain their behavior.

Collectors pause at the update barrier during evaluation. The parent broadcasts
map-pool changes with the update response, and each worker applies the new pool
at its next natural episode reset. Worker offsets stagger the weighted map
schedule. Unchanged pools do not restart map rotation. Passing the full bundle
broadcasts a stop to every collector before any more actions are sampled; final
validation starts after collector shutdown.

For fewer workers, override `--num-envs` and set `agents.car_0.params.n_steps` to
`num_envs * 1024` in the scenario to retain 1,024 decisions per worker. Keep
`evaluation.every_steps` and `checkpoint_every_steps` equal to that pooled size
for evaluation/checkpointing after every update. The pooled size must divide
evenly across workers. Single-environment training remains supported.
Startup batches/timeouts are explicitly configured in this scenario.

Training begins on L_map. After every 409,600 collected transitions and PPO
update, deterministic evaluation runs ten fixed-seed starts on each of the nine
maps. A successful start completes **five laps without a collision or track-limit
violation**, within 16,000 physics steps (800 seconds). Initial finish-line
crossings do not count as completed laps.

At least nine of ten starts must succeed on every active training map in two
consecutive evaluations before another map is added. The controller chooses the
untrained failing map closest to passing; configuration order breaks ties.
Already passing maps do not need to be trained. A regression on an active map
resets the stage streak. Each stage resets its streak when a map is added.

Half the scheduled training episodes use the newest map; the other half cycle
through previous maps. These are episode proportions, not transition proportions.
Changes take effect at the next natural reset. Five-lap completion or the finite
training horizon ensures that successful policies also rotate maps.

All maps are evaluated at every stage. Two consecutive evaluations with at least
90% clean completion **on every map** stop PPO updates, even if training has never
left L_map. The 120-million-transition budget is an upper bound, not a minimum.
Evaluation can be expensive as policies survive longer; `every_steps` can be
increased, but larger intervals can miss short-lived generalization windows.

Artifacts in the run directory:

- `best_model.pt`: ranked by number of passing maps, weakest-map success rate,
  mean success rate, then mean clean finish time.
- `evaluation_history.jsonl`: per-map results, active pool, rates, streaks and
  advancement decisions at every evaluation.
- `curriculum_state.json`: latest pool, streaks, step count and completion flag.
- `curriculum_passed.pt`: the frozen policy at the second whole-bundle pass.
- `curriculum_final_evaluation.json`: automatic final validation of that frozen
  policy, using twenty fresh starts per map, twenty clean laps and a 64,000-step
  horizon. `all_maps_passed` requires at least 90% clean completion on every map.

Final validation never changes weights or selects checkpoints. If it fails,
training remains stopped and the report records the failure. If the training
budget expires before the curriculum gate passes, no automatic final test is run.
The bundle is used for curriculum validation; these are not unseen test maps.
Two evaluations reuse the same selection starts to check successive policy
versions. Only final validation uses the separate seed range.

`--checkpoint` initializes actor/critic weights with a fresh optimizer and starts
the curriculum again from L; it does not restore curriculum progress. The state
JSON is an audit artifact, not a full training-resume checkpoint.
