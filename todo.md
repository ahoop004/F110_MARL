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
- [ ] Add a lightweight lookahead (projected pose one step ahead) to pick the safest gap when straights suddenly end.
