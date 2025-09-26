# TODO Backlog

Ordered by source location (top of file downward).

## src/f110x/envs/f110ParallelEnv.py

1. L40-57 — Make renderer buffers (`renderer`, `current_obs`, `render_callbacks`) per-instance.
2. L58-66 — Reset render callback storage per environment instance.
3. L71 — Normalize map identifiers so callers can pass bare stems (e.g. `"levine"`).
4. L79-101 — Wire vehicle parameters to `vehicle_params` config instead of default dict.
5. L109 — Allow per-agent termination overrides from scenario metadata.
6. L125-133 — Precompute proper start rotation matrices instead of identity, populate cached start poses from `start_poses`.
7. L134 — Expose `self.target_laps` via config/scenario wiring for lap counting.
8. L243 — Supply `self.target_laps` when computing terminations.
9. L279 — Remove redundant physics step during reset; use observation returned by `sim.reset`.
10. L287 & L313 — Re-index render/state bookkeeping by agent id so surviving agents keep consistent data.
11. L302 — Map actions by agent id to avoid misalignment after crashes.
12. L319 — Advance `self.current_time`, lap counters, and `_check_done()` each step before rewards.
13. L331 — Merge `_check_done()` outputs into termination logic.
14. L452 — Attach promised lap timing/count data in infos/observations.

## src/f110x/physics/simulaton.py

15. L145 — Re-run LiDAR after penetration rollback so scans match reverted pose.
16. L165 — Encode environment collisions distinctly in `collision_idx`.
17. L195 — Expose real lateral velocity instead of zero placeholder.

## src/f110x/physics/laser_models.py

18. L394 — Decide on image preprocessing policy (flip, grayscale expectations) and document/enforce it.
19. L395 — Validate map image specification and raise helpful errors on mismatch.

## main.py

20. Map scenario metadata (policies, lap goals, termination settings) onto environment/policy selection.

## configs/config.yaml

21. Provide explicit `map_image` path once asset layout is fixed to enable cache priming.

## Stable-Baselines Integration (new work)

22. Build SB3-compatible wrappers for observation, reward, and action shaping.
23. Create policy directory with controllers such as `gap_follow` and `waypoint_follow`.

