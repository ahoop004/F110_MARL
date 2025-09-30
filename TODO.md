# TODO

- [x] Refactor RewardWrapper: extract Position-2 math + normalization into helper module; centralize coefficient definitions.
- [x] Consolidate observation/action wrapper utilities (shared LiDAR downsample, to_numpy helpers, observation pipeline abstraction).
- [x] Split F110ParallelEnv into submodules (LiDAR, collision state machine, start-poses) for readability/testing.
- [x] Break up F110ParallelEnv and physics helpers with clearer module boundaries or shape/type docstrings, improving readability around lidar/collision pipelines.
- [x] Introduce typed config schema (pydantic or custom) so env/agent/reward defaults live in one place.


- [ ] Create shared CLI utilities (yaml loading, logging, manifest helpers) for scripts/run.py/map_validator.py.

- [ ] Relocate checkpoints/eval artifacts to an ignored outputs/ directory and add helper script to bundle config + git SHA per run.
    - [ ] Add manifest writer (CSV/JSON) capturing run_id, algo, map, seed, output_dir, git_sha.
    - [ ] Update `.gitignore` for outputs/ and retrofit existing scripts to use the new directory.

    
- [ ] Let the map validator whitelist known-good assets or tune thresholds so sweep dashboards aren't flooded with wall-band warnings.
- [ ] Consider wiring the `lidar_beams` option into PPO/TD3 wrappers (or document why they keep full scans) for consistent obs shapes.


- [ ] Resource planning for massive sweeps.
    - [ ] Draft compute concurrency plan (runs per GPU/CPU, reserved debug slot).
    - [ ] Specify storage budget + retention policy (outputs/<algo>/<map>/<seed>/ layout).
    - [ ] Add submission helper (scripts/launch_array.py or similar) with retry/status tracking.
    - [ ] Set up monitoring/alerting (W&B dashboards, log watchdog).
