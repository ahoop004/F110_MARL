# TODO Backlog

Ordered by source location (top of file downward).

## src/f110x/envs/f110ParallelEnv.py

1. L432 & L458 — Re-index `render_obs` by agent id so surviving cars keep their own lap telemetry after drop-outs.
2. L447 — Map joint actions using `possible_agents` indices so controls stay aligned with simulator rows after removals.
3. L463 — Advance `self.current_time`, lap counters, and `_check_done()` bookkeeping before computing rewards.
4. L475 — Use `_check_done()` outputs (including lap targets) inside termination logic.
5. L595 — Include lap timing/count data in `_split_obs` outputs to match the declared observation space.

## src/f110x/physics/simulaton.py

6. L145 — Re-run LiDAR after penetration rollback so scans align with the restored pose.
7. L165 — Encode environment collisions distinctly in `collision_idx`.
8. L195 — Expose real lateral velocity instead of the zero placeholder.

## src/f110x/physics/laser_models.py

9. L394 — Decide and document preprocessing expectations (flip, grayscale) for map images.
10. L395 — Validate map image specification and raise descriptive errors on mismatch.

## main.py

11. L55 — Map scenario metadata (policies, lap goals, termination settings) into environment wiring and policy selection.

## configs/config.yaml

12. L13 — Include explicit `map_image` path once asset layout is finalized to enable renderer cache priming.

## Stable-Baselines Integration (new work)

13. Build SB3-compatible wrappers for observation, reward, and action shaping.
14. Create policy directory with controllers such as `gap_follow` and `waypoint_follow`.
