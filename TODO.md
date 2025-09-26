# TODO Backlog

Ordered by source location (top of file downward).

## src/f110x/envs/f110ParallelEnv.py

<!-- 1. L345 — Expose MARLlib env registration helpers (env_info, policy mapping) so trainers can auto-configure agents. -->
2. L454 — Emit a centralized state tensor alongside per-agent obs for MARLlib centralized training pipelines.
3. L602 — Attach lap_counts, lap_times, and other scoreboard data promised in `observation_space`.

## src/f110x/physics/simulaton.py

4. L165 — Encode environment collisions distinctly in `collision_idx`.
5. L195 — Expose real lateral velocity instead of the zero placeholder.

## src/f110x/wrappers/__init__.py

6. L1 — Implement MARLlib-compatible wrappers (flattened obs, action normalization, env registration).

## main.py

7. L55 — Map the scenario metadata (policies, lap goals, termination settings) onto the environment and policy selection.
8. L58 — Wire this script into MARLlib training entrypoints (env creator, experiment config, baseline runners).

## configs/config.yaml

9. L13 — Include explicit `map_image` path once asset layout is finalized to enable renderer cache priming.

## MARLlib Baseline Prep

10. Provide reference MARLlib trainer configs (Ray Tune experiment definitions) targeting common baselines.
11. Document policy/obs conversion rules so external users can plug the environment into MARLlib without reading source.
