# TODO Backlog

Ordered by source location (top of file downward).

## src/f110x/envs/f110ParallelEnv.py


5. L595 — Include lap timing/count data in `_split_obs` outputs to match the declared observation space.

## src/f110x/physics/simulaton.py

6. L145 — Re-run LiDAR after penetration rollback so scans align with the restored pose.
7. L165 — Encode environment collisions distinctly in `collision_idx`.
8. L195 — Expose real lateral velocity instead of the zero placeholder.


## main.py

11. L55 — Map scenario metadata (policies, lap goals, termination settings) into environment wiring and policy selection.

## configs/config.yaml

12. L13 — Include explicit `map_image` path once asset layout is finalized to enable renderer cache priming.

## Stable-Baselines Integration (new work)

13. Build SB3-compatible wrappers for observation, reward, and action shaping.
14. Create policy directory with controllers such as `gap_follow` and `waypoint_follow`.
