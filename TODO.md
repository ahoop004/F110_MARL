# TODO

- [ ] High-impact: Batch physics stepping/collision handling in simulator (`src/f110x/physics/simulaton.py`).
    - [ ] Preallocate reusable buffers for `prev_states`, `verts`, `scans`, and collision flags to cut per-step object churn.
    - [ ] Move agent/environment collision merging into an `@njit` helper so the hot path avoids Python loops.
    - [ ] Re-run LiDAR sampling after collision rewinds so `agent.scan` matches the restored pose (fixes TODO at line 171).

- [ ] High-impact: Eliminate per-step LiDAR occlusion stacking (`src/f110x/physics/simulaton.py`, `src/f110x/physics/vehicle.py`).
    - [ ] Pass opponent vertex views instead of rebuilding `np.stack([...])` for every agent each step.
    - [ ] Refactor `RaceCar.ray_cast_agents` to consume contiguous float32 buffers without copying/converting per opponent.

- [ ] Medium-impact: Trim repeated allocations in `RaceCar.update_pose` (`src/f110x/physics/vehicle.py`).
    - [ ] Cache frequently used params/arrays (`[sv, accl]`, `scan_pose`) instead of recreating them four times per integration step.
    - [ ] Hoist dict lookups for dynamics parameters into structured arrays for faster access inside numba kernels.

- [ ] Medium-impact: Reduce training-loop Python bookkeeping (`experiments/train.py`).
    - [ ] Replace per-agent dicts (`totals`, `speed_sums`, `collision_counts`, etc.) with preallocated numpy arrays keyed by agent index.
    - [ ] Vectorize idle-speed detection and reward aggregation to avoid repeated `np.asarray`/`np.linalg.norm` per agent.
    - [ ] Consider a lightweight trajectory buffer so trainers consume batched transitions instead of per-step dict assembly.

- [ ] Lower-impact: Cache map metadata/image loads in `MapLoader` (`src/f110x/utils/map_loader.py`).
    - [ ] Add an in-memory cache keyed by `(map_dir, map_yaml)` and share `track_mask`/image size across env builds.
    - [ ] Provide explicit cache invalidation when map files change on disk.

- [ ] Create shared CLI utilities (yaml loading, logging, manifest helpers) for scripts/run.py/map_validator.py.

- [ ] Relocate checkpoints/eval artifacts to an ignored outputs/ directory and add helper script to bundle config + git SHA per run.
    - [ ] Add manifest writer (CSV/JSON) capturing run_id, algo, map, seed, output_dir, git_sha.
    - [x] Update `.gitignore` for outputs/ and retrofit existing scripts to use the new directory.


- [ ] Resource planning for massive sweeps.
    - [ ] Draft compute concurrency plan (runs per GPU/CPU, reserved debug slot).
    - [ ] Specify storage budget + retention policy (outputs/<algo>/<map>/<seed>/ layout).
    - [ ] Add submission helper (scripts/launch_array.py or similar) with retry/status tracking.
    - [ ] Set up monitoring/alerting (W&B dashboards, log watchdog).
