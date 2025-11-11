## Evaluation prep

- [ ] Baseline FTG-only runs per track (solo defender) logging return, lap completion %, collisions.
- [ ] FTG vs FTG cooperative baseline to quantify natural interference (same metrics as above).
- [ ] FTG vs malicious TD3 agent experiments capturing success/failure, truncations, and qualitative heatmaps.
- [ ] Attack Success Rate (ASR) sweeps: vary map, spawn offsets, and log ASR over training time.
- [ ] Ablation studies for attacker reward shaping (disable pressure/relative terms) to attribute ASR gains.
- [ ] Cross-play robustness: evaluate attacker against FTG parameter variants (sensor noise, bubble radius, etc.).
- [ ] Resource/latency profiling comparing FTG vs TD3 inference cost for real-time viability claims.
- [ ] Align success metrics for FTG baselines with finish-line completion (load finish line from map YAML on corridor tracks).
- [ ] Fix notebooks so trajectory overlays use the correct map coordinate frame for all map bundles.
- [ ] Build clean FTG-only baselines on line2, Budapest, and Shanghai (straight, S-curve, hairpin) with consistent logging.
- [ ] Repeat the same three-map suite for FTG vs FTG to capture natural interference.
- [ ] Run RL attacker vs FTG defender experiments on the three benchmark maps and gather plots/metrics.
- [ ] Final sweeps on line2 (corridor) targeting success rate vs. curriculum tiers / hyperparameter variations.
- [ ] Refresh CLI output to reflect scenario type (solo FTG, FTG vs FTG, FTG vs TD3) and remove stale idle warnings.
- [ ] Untangle training vs eval wandb logging; add per-episode metrics (success, cause, reward breakdown) alongside step-level stats.

## Success metrics & logging

- [ ] Corridor maps (line2, etc.): load finish-line annotations from map YAML and trigger success when FTG crosses the segment.
- [ ] Looped tracks: double-check lap-completion logic and align success conditions (defender lap or attacker crash) with logging.
- [ ] Update CLI output to report scenario-specific success (finish-line hit, lap completion, crash, timeout) without idle noise.
- [ ] wandb: add per-episode success metrics (type, time to finish, distance closed) and separate train vs eval logs.
- [ ] Ensure the new map overlay notebook highlights success points so qualitative plots match the metrics.
## Code cleanup

- [ ] **Target pattern:** move toward hexagonal architecture with registries. Composition root builds env/team/trainers/rollout from a frozen `ExperimentConfig`. Domain = env + rollout engine. Adapters = policies, trainers, wrappers, maps, IO.
- [ ] **Phase 1 – Groundwork**
  - Create `docs/architecture.md` describing the target diagram/lifecycle.
  - Add a minimal registry utility with tests.
  - Add a config validator that freezes section views (`env_cfg`, `agents_cfg`, etc.) and enforces invariants; tests ensure immutability.
- [ ] **Phase 2 – Composition root**
  - New module assembles environment, roster, trainers, rollout engine, and returns bundle metadata (obs_dim, act_dim, bounds, device, seeds).
  - One entrypoint must call only the root; builders stop importing each other.
- [ ] **Phase 3 – Registries everywhere**
  - Standard registries for policies, wrappers, reward shapers, maps (trainers already have one).
  - Builders resolve via registry calls; swapping algorithms becomes config-only.
- [ ] **Phase 4 – Rollout refactor**
  - Split concerns: `StepEngine` (env stepping), `RewardApplier`, `Instrumentation`.
  - Trainers own buffers; engine is trainer-agnostic; add a no-trainer eval path.
- [ ] **Phase 5 – Scenario API**
  - Promote scenario config (map, spawn rules, laps/timeouts) into a `Scenario` object to derive deterministic starts from seeds.
- [ ] **Phase 6 – Observation contracts**
  - Define component interface with `out_shape/bounds/describe`, normalize units centrally, and record final shape/bounds in bundle metadata; property tests ensure per-map consistency.
- [ ] **Phase 7 – Reproducibility surface**
  - Emit a manifest per run (registry choices, versions, seeds, shapes, bounds, scenario hash). Optional trajectory snapshot toggle for eval.
- [ ] **Phase 8 – Public entrypoints**
  - Thin modules (`train.py`, `eval.py`, `rollout.py`) call the composition root only. Update docs/examples accordingly.
- [ ] **Phase 9 – CI & quality gates**
  - Add pre-commit (formatter/linter/type-check). CI runs unit + smoke tests on CPU; GPU optional.
- [ ] **Phase 10 – Docs & migration**
  - Update README + new architecture doc; add migration notes so users know about registries/entrypoints.
- [ ] Definition of done: single composition root, registries for adapters, frozen config, separated rollout concerns, manifests per run, public entrypoints only, CI enforcing tests/style.

## FTG improvements

- [x] Implement velocity-aware steering clamps so FTG can’t request impossible curvature at high speed.
- [x] Gate the centering bias by proximity (disable when min_scan is small so hairpins can hug the inside).
- [x] Adapt LiDAR smoothing/bubble sizes based on obstacle distance to improve responsiveness in tight turns.
- [x] Replace the tiered speed schedule with a spread/variance-aware rule so FTG slows before sharp bends.
- [x] Add a lightweight lookahead (projected pose one step ahead) to pick the safest gap when straights suddenly end.

## FTG next-gen ideas

- [x] Kinematic preview FTG
  - simulate short horizon for multiple steering candidates
  - pick steering whose projected path maintains best LiDAR clearance
- [x] Dynamic Window FTG
  - sample attainable speed/steer pairs given actuator limits
  - score each candidate by clearance + heading change and pick best
- [x] Enhanced gap scoring
  - cluster gaps and rank by curvature, distance-to-collision, historical risk
  - bias selection toward safer gaps without map data
- [x] LiDAR U-shape crawl heuristic
  - detect U-shaped patterns (both sides closing) via range derivatives
  - force crawl mode whenever the pattern indicates a switchback

## Reward scheduling

- [ ] Episode/step-based reward ramps for gaplock components
  - allow YAML schedules to scale reward terms by episode progress (0→max train episodes)
  - add per-episode step ramps using env.max_steps to taper intra-episode bonuses/penalties
