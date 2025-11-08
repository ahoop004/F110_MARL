## Evaluation prep

- [ ] Baseline FTG-only runs per track (solo defender) logging return, lap completion %, collisions.
- [ ] FTG vs FTG cooperative baseline to quantify natural interference (same metrics as above).
- [ ] FTG vs malicious TD3 agent experiments capturing success/failure, truncations, and qualitative heatmaps.
- [ ] Attack Success Rate (ASR) sweeps: vary map, spawn offsets, and log ASR over training time.
- [ ] Ablation studies for attacker reward shaping (disable pressure/relative terms) to attribute ASR gains.
- [ ] Cross-play robustness: evaluate attacker against FTG parameter variants (sensor noise, bubble radius, etc.).
- [ ] Resource/latency profiling comparing FTG vs TD3 inference cost for real-time viability claims.

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
- [ ] Dynamic Window FTG
  - sample attainable speed/steer pairs given actuator limits
  - score each candidate by clearance + heading change and pick best
- [ ] Enhanced gap scoring
  - cluster gaps and rank by curvature, distance-to-collision, historical risk
  - bias selection toward safer gaps without map data
- [ ] LiDAR U-shape crawl heuristic
  - detect U-shaped patterns (both sides closing) via range derivatives
  - force crawl mode whenever the pattern indicates a switchback
