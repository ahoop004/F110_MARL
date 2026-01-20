# TODO: Centerline PPO Scenario

## 0) Scenario entry point (centerline + target vehicle)
- [x] Create `scenarios/ppo_centerline.yaml`.
- [x] Mirror `scenarios/ppo.yaml` includes list, but use centerline configs:
  - `../configs/env/centerline_multi.yaml` (reusable multi-track env config)
  - `../configs/reward/centerline_racing.yaml`
  - `../configs/evaluation/centerline_multi_eval.yaml`
  - `../configs/wandb.yaml` (reuse)
- [x] Set `environment.max_steps` appropriate for laps.
- [x] Set agent config (2-agent, target included in eval):
  - `agents.car_0.algorithm: sb3_ppo`
  - `agents.car_0.observation.preset: centerline`
  - `agents.car_0.frame_stack: 4`
  - `agents.car_0.target_id: car_1`
  - `agents.car_1.algorithm: ftg`
  - `environment.num_agents: 2`
- [x] PPO params: `learning_rate`, `n_steps`, `batch_size`, `gamma`, `activation: tanh`, plus `pi_hidden_dims`/`vf_hidden_dims`.

## 1) Observation + Action Interface (centerline)
- [x] Edit `src/core/obs_flatten.py`.
- [x] Add `flatten_centerline_obs(obs_dict, scales=None) -> np.ndarray`.
- [x] Implement features:
  - LiDAR: `obs_dict["scans"]`, clip to 10m, normalize `clip(scan, 0, 10) / 10`.
  - Speed magnitude: from `obs_dict["velocity"]`, `sqrt(vx^2 + vy^2)`, normalize by `scales["speed"]` when present, clip to `[0, 1]`.
  - Previous action: `obs_dict.get("prev_action")` (shape `(2,)`), clip to `[-1, 1]`, else zeros.
- [x] Ignore `target_id`/`central_state` for centerline preset (keep dims fixed with target present).
- [x] Update `flatten_observation()` dispatcher:
  - Add `elif preset == "centerline": return flatten_centerline_obs(...)`.
  - Update error message to “Supported: gaplock, centerline”.
- [x] Update `__all__`.
- [x] Edit `src/baselines/sb3_wrapper.py`.
- [x] Add member state `self._prev_action_norm = np.zeros(2, dtype=np.float32)` in `__init__`.
- [x] Reset it in `reset()` before returning first obs.
- [x] In `step()` before mapping to env bounds, store normalized policy output:
  - `self._prev_action_norm = np.clip(action, -1, 1)` (continuous case).
- [x] Before flattening, attach to agent obs:
  - `obs_with_prev = dict(obs_dict[self.agent_id])`
  - `obs_with_prev["prev_action"] = self._prev_action_norm`
  - Pass `obs_with_prev` to `_flatten_obs(...)`.
- [x] Keep frame stacking behavior unchanged.
- [x] Set continuous action space to normalized bounds:
  - `Box(low=[-1, -1], high=[1, 1])`.
- [x] Map normalized action to env bounds using wrapper params `action_low`/`action_high`:
  - `a_env = low + (a_norm + 1) / 2 * (high - low)`.
- [x] Store `prev_action` as `a_norm` (not `a_env`).
- [x] Keep discrete `action_set` behavior unchanged (normalize only continuous actions).
- [ ] Review: decide whether LiDAR should be hard-clipped to 10m (per spec) or use `scales["lidar_range"]` for normalization.
- [ ] Review: decide whether to record `prev_action` for discrete `action_set` cases (currently only continuous updates it).

## 2) Reward: implement centerline reward strategy type
- [x] Create `src/rewards/centerline.py` with `CenterlineReward` (implements `RewardStrategy`).
- [x] Define `reset()` and `compute(step_info) -> (total, components)`.
- [x] Expect `step_info` keys: `obs`, `next_obs`, `info`, `done`, `timestep`.
- [x] Use centerline metrics from `info` when present, else compute from pose + env centerline.
- [x] Keep reward independent of `target_id` (single-agent centerline objective).
- [x] Implement core terms (paper-aligned): `vs`, `vd`, `d`, `steer`, collision penalty.
- [x] Edit `src/rewards/builder.py`:
  - Import `CenterlineReward`.
  - Add `reward_type == "centerline"` branch.
- [x] Edit `src/rewards/presets.py`:
  - Add `CENTERLINE_RACING` preset and register in `PRESETS`.
- [x] (Optional) Update `src/rewards/__init__.py`.
- [x] Create reward config: `configs/reward/centerline_racing.yaml` using `agents -> car_0 -> reward` shape.
- [ ] Review: verify centerline points + pose/velocity are always available at reward time; otherwise reward collapses to zero.

## 3) Multi-Track Environment + Spawning (centerline + walls)
- [x] Create `configs/env/centerline_multi.yaml` (reusable multi-track env config).
- [x] Set LiDAR range and beams:
  - `lidar_range: 10.0`
  - `lidar_beams: <paper or sim beams>`
- [x] Enable centerline support (flags used in `src/core/setup.py`):
  - `centerline_autoload`, `centerline_csv`, `centerline_render`, `centerline_features`.
- [x] Include map bundle + split + cycle settings for multi-track training.
- [x] Ensure map bundles include wall data (`*_walls.csv`) for centerline-based spawn offsets.
- [x] Add `environment.map_bundles` with the bundle folders to split.
- [x] Add `environment.map_split` config:
  - `train_ratio: 0.8` (default)
  - `seed: <int>`
  - `shuffle: true|false`
  - `train_pick: first|random` (used only if not per-episode cycling)
  - `eval_pick: first|random` (used only if not per-episode cycling)
- [x] Update scenario validation to accept `map_bundles` instead of requiring `map`.
- [x] Apply split + resolve selected bundle when building the env (train vs eval).
- [x] Pass map path metadata (`map_dir`, `map_yaml`, `map_ext`) to the env when using bundles.
- [x] Ensure eval path uses `mode="eval"` when creating the env.
- [x] Add scenario knobs:
  - `environment.map_cycle: per_episode`
  - `environment.map_pick: random|round_robin` (and optional `epoch_shuffle`)
- [x] Update setup logic: when `map_cycle: per_episode`, keep bundle lists on the env (don’t collapse to a single bundle at setup time).
- [x] Implement map switching on reset (env-side):
  - Resolve next bundle (train or eval list based on mode).
  - Load map yaml/image/centerline via `MapLoader`.
  - Refresh `map_dir/map_yaml/map_path`, `map_meta/map_image_path`, `map_image_size`, `track_mask`.
  - Call `sim.set_map(...)` and update renderer + finish line tracking.
- [x] Add episode logging for `map_bundle_active` (training/eval logs + info dict).
- [x] Ensure per-episode switching also refreshes wall data for spawn logic.
- [x] Avoid reseeding RNG on map switch (keep RNG continuity across episodes).
- [x] Add scenario knobs (env-level):
  - `spawn_policy: centerline_relative`
  - `spawn_centerline: {mode: random|round_robin, avoid_finish: true, min_progress: <float>, max_progress: <float>}`
  - `spawn_offsets: {s_offset: <float>, d_offset: <float>, d_max: <float>}`
  - `spawn_target: {enabled: true, speed: <float>}`
  - `spawn_ego: {speed: <float>}`
- [x] Implement wall CSV loading in map metadata (use `*_walls.csv` when present).
- [x] Build centerline pose helper:
  - Compute tangent heading from centerline segment.
  - Clamp `d_offset` to stay inside walls (fallback to `d_max` when walls missing).
  - Avoid finish line window for spawn selection.
- [x] Apply policy in env reset:
  - Spawn target on centerline (`d=0`) at chosen `s`.
  - Spawn ego relative to target using `s_offset/d_offset`.
- [x] Eval mode: deterministic selection for `s`/offsets (fixed indices or seeded).
- [x] Record selected `s`, `d`, and bundle name in `info` for debugging.
- [x] Review: ensure `create_training_setup()` passes map cycling + spawn policy config into `F110ParallelEnv` (map bundles, map_cycle/map_pick, map_split_mode, spawn_*).
- [x] Review: honor `spawn_target.enabled` (currently `car_1` always spawned on centerline if present).

## 4) Evaluation config for centerline success metrics
- [x] Create `configs/evaluation/centerline_multi_eval.yaml`.
- [x] Define success (lap completion / finish line) and failure (collision).
- [x] Set:
  - `observation_presets: {car_0: centerline}`
  - `frame_stacks: {car_0: 4}`
- [x] Ensure eval uses the holdout map split and keeps `car_1` (FTG) active.
- [x] Review: preserve explicit empty `spawn_points`/`spawn_speeds` in `EvaluationConfig` (avoid falling back to pinch spawns so spawn_policy can take over cleanly).

## 5) PPO policy/value network split and tanh activation
- [x] Edit `run_sb3.py`.
- [x] Extend `build_policy_kwargs(params, model_name)`:
  - For PPO/A2C, accept `pi_hidden_dims` and `vf_hidden_dims`.
  - Build `net_arch = {"pi": pi_hidden_dims, "vf": vf_hidden_dims}`.
  - Ensure `activation: tanh` maps to `nn.Tanh` (already supported).
- [x] In `scenarios/ppo_centerline.yaml` set:
  - `activation: tanh`
  - `pi_hidden_dims: [32, 32]`
  - `vf_hidden_dims: [64, 64]`
  - Keep/adjust `hidden_dims` as fallback.
- [x] Mirror these PPO defaults in the new reusable env/scenario config if we add other algorithms later.

## 6) Optional: Observation presets registry (documentation/consistency)
- [x] Edit `src/core/observations.py`.
- [x] Add `CENTERLINE_OBS` entry and register it.
- [x] Update dimension logic to include speed scalar (1) + prev_action (2).
- [x] Keep preset dimensions aligned with `flatten_centerline_obs` (no target/central_state extras).

## Files to create
- [x] `scenarios/ppo_centerline.yaml`
- [x] `configs/env/centerline_multi.yaml`
- [x] `configs/reward/centerline_racing.yaml`
- [x] `configs/evaluation/centerline_multi_eval.yaml`
- [x] `src/rewards/centerline.py`

## Files to modify
- [x] `src/core/obs_flatten.py`
- [x] `src/baselines/sb3_wrapper.py`
- [x] `run_sb3.py`
- [x] `src/rewards/builder.py`
- [x] `src/rewards/presets.py`
- [ ] `src/core/observations.py` (optional)
- [x] `src/core/setup.py`
- [x] `src/core/scenario.py`
- [x] `eval.py`
- [x] `src/env/f110ParallelEnv.py`
- [x] `src/utils/map_loader.py`
- [x] `src/utils/centerline.py`
