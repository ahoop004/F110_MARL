# TODO: Gaplock Refactor (Centerline-Aligned)

## 0) Locked decisions
- [x] Use map bundles (auto-discover only folders with YAML + image + `_centerline.csv` + `_walls.csv`).
- [x] Normalize actions to `[-1, 1]` in SB3 wrapper for gaplock too.
- [x] No curriculum for now.

## 1) Scenario + env config alignment
- [x] Create `configs/env/gaplock_multi.yaml` mirroring `centerline_multi.yaml` (map bundles auto, per-episode cycling).
- [x] Update `scenarios/ppo.yaml` to include the new gaplock multi env config and remove curriculum includes.
- [x] Create/adjust eval config for gaplock multi-track (if needed) and keep `car_1` FTG active.
- [x] Keep single-map `configs/env/line2_gaplock.yaml` as legacy (no curriculum).

## 2) Observation alignment
- [x] Ensure gaplock uses `flatten_observation()` with `preset: gaplock` and target `central_state` injection.
- [x] Verify `src/core/observations.py` gaplock preset dims match actual flattening output.
- [x] Confirm gaplock frame stacking uses the same path as centerline.

## 3) Reward refactor (gaplock vs centerline)
- [x] Implement new gaplock reward shape:
  - Reward when target is further from centerline and closer to a wall.
  - Large reward for success; large penalties for crashes/collisions.
- [x] Decide where to compute target-centerline deviation and wall distance:
  - Prefer env info when available; otherwise compute from target pose + centerline/walls.
- [x] Add new gaplock reward preset + YAML config (type/preset/overrides).
- [x] Keep old gaplock reward available for comparison (optional).

## 4) Agent constraints (maxv / reverse)
- [x] Ensure vehicle params are defined in the multi-track env configs (match line2 defaults).
- [x] Add an agent config file for action constraints (max_v, prevent_reverse, throttle index).
- [x] Apply these constraints in the SB3 action path (normalized action mapping).

## 5) Validation
- [ ] Run gaplock multi-track scenario (no curriculum) and confirm map cycling + spawn + reward.
- [ ] Log target wall distance + centerline deviation to verify reward signal.
