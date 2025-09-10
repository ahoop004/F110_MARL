# Now (MVP path)

- [ ] Physics core in src/physics/ lifted & cleaned (no RL deps)
    - [ ] dynamic_models.py

    - [ ] collision_models.py
        - [ ] GJK iteration cap: reduce from 1e3 to ~32–64; avoid stalls on degeneracies while preserving convergence. (GJK should converge fast for 2D quads; temporal coherence helps.)
        - [ ] Robust eps checks: replace exact zero checks with small epsilon (e.g., 1e-9 to 1e-12) around direction norms/simplex transitions.
        - [ ] Temporal coherence (optional): seed each pair’s next-frame search direction with the previous separating axis to cut iterations.
        - [ ] Broad phase (future, if N grows): add sweep-and-prune (or grid hashing) to prune O(N²) pairs before GJK; incremental sort leverages coherence.
        - [ ] Edge/point touching policy: define whether grazing contact counts as collision and test it (edge–edge, point–edge).
        - [ ] Penetration depth (later): if needed for penalties or response, add EPA or SAT fallback to retrieve depth/normal.
        - [ ] Stress & margin tests: (a) tiny gap vs tiny overlap; (b) rotation sweep; (c) randomized convex quads; assert average iterations < threshold and no NaNs.

    - [ ] laser_models.py
    - [ ] base_classes.py
        - [ ] vehicle.py
        - [ ] simulation.py
    - [ ] Simulator.reset(spawn_poses) → obs_dict

    - [ ]  Simulator.step(actions_env) → obs_dict

 - [ ] Collision flags per-step (not cumulative)

 - [ ] ObservationBuilder in src/wrappers/ (fixed shape; LiDAR↓sample, ego, relatives)

 - [ ] ActionWrapper in src/wrappers/ (Box[-1,1]^2  <--> (steer, vel) bounds from configs/config.yaml)

 - [ ] Roles & scripted policies in src/wrappers/ or src/utils/ (target, benign, idle, gap_follow)

 - [ ] PettingZoo ParallelEnv in src/envs/f110_MARL_env.py

 - [ ] reset() → (obs, infos) per agent

 - [ ] step(actions) → (obs, rewards, terminations, truncations, infos)

 - [ ] Rewards module (role-aware; big bonus if any target crashes; safety shaping)

# Next (dev UX & quality)

 - [ ] Renderer (optional) in src/render/ with overlays; import-guarded

 - [ ] Config loader/validator; single configs/config.yaml (maps, lidar, action bounds, roles)

- [ ] Tests

    - [ ] Physics tick determinism & forced crash case

    - [ ] Obs shape/finite checks for N=1..4

    - [ ] PZ API conformance (spaces stable; dict keys)

    - [ ] Reward & termination semantics (per-agent term; time-limit trunc)

 - [ ] Example rollouts (examples/) + minimal README usage

# Later (polish & tooling)

 - [ ] Optional Gym single-agent shim for SB3

 - [ ] CI (lint + unit tests)

 - [ ] Benchmarks (steps/sec with/without rendering)

 - [ ] Docs: architecture diagram, interfaces, config reference

 - [ ] Issue templates (bug/feature/chore) & PR template

# Admin links (fill in)

Project board: <add GitHub Projects view>

Labels: type:feat, type:fix, type:chore, area:physics, area:env, prio:p0/p1/p2

Milestones: v0.1 MVP, v0.2 Perf, v0.3 Docs