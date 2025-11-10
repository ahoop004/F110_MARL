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

- [ ] Audit overall modularity: revisit layered separation between env wrappers, policies, rewards, and logging to ensure responsibilities are clear.
- [ ] Prune the legacy idle-guard plumbing (duplicate thresholds, unused config fields) so scenarios only define the active path.
- [ ] Consolidate reward parameters (top-level vs `params`) and document the precedence to avoid double-defining truncation penalties.
- [ ] Normalize spawn/curriculum metadata by extracting shared snippets and removing commented scaffolding in scenarios.
- [ ] Document and refactor the new path logging so the hooks live in a dedicated module instead of being scattered through `TrainRunner`.
- [ ] Extract common FTG parameter presets/curricula into shared configs to reduce duplication across scenarios.
- [ ] Review observation/reward/task wrappers for redundancy and consider collapsing or relocating them into clearer modules.
- [ ] Standardize parameter handling/passing (env, agent, reward) and clean up the `f110x` core so config plumbing is obvious.
- [ ] Propose a tidy file layout/categorization (policies, runners, utilities) and document creation/structural patterns for contributors.

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
