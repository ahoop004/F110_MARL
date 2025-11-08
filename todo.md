## Evaluation prep

- [ ] Baseline FTG-only runs per track (solo defender) logging return, lap completion %, collisions.
- [ ] FTG vs FTG cooperative baseline to quantify natural interference (same metrics as above).
- [ ] FTG vs malicious TD3 agent experiments capturing success/failure, truncations, and qualitative heatmaps.
- [ ] Attack Success Rate (ASR) sweeps: vary map, spawn offsets, and log ASR over training time.
- [ ] Ablation studies for attacker reward shaping (disable pressure/relative terms) to attribute ASR gains.
- [ ] Cross-play robustness: evaluate attacker against FTG parameter variants (sensor noise, bubble radius, etc.).
- [ ] Resource/latency profiling comparing FTG vs TD3 inference cost for real-time viability claims.
