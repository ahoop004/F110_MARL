# Unused Code Candidates

Generated Fri Oct  3 2025 via `vulture 2.14` (scanned `src experiments scripts run.py tests`) plus manual validation. Items below survive today’s cleanup and look intentionally retained despite lacking direct in-tree callers.

- `src/f110x/envs/f110ParallelEnv.py:525` – `observation_spaces` cache sticks around to satisfy PettingZoo callers that inspect the dict directly.
- `src/f110x/physics/base_classes.py:104` / `:194` / `:248` and `src/f110x/physics/vehicle.py:79` / `:234` – `accel` / `steer_angle_vel` slots store the last commanded inputs for downstream integrators or debug hooks even though the core sim now reads buffers instead.
- `src/f110x/render/rendering.py` – GUI hooks remain unused by repo code but needed for pyglet event wiring: `on_mouse_drag` (214), `on_mouse_scroll` (220), `set_camera_target` (433), `make_centerline_callback` (471), `make_waypoints_callback` (487).
- `src/f110x/utils/config_models.py:337-344` – dataclass fields (`rec_ppo`, `td3`, `sac`, `dqn`, `ppo_agent_idx`) map YAML sections even though today’s runtime looks up sections dynamically via `get_section`.
- `src/f110x/utils/config_schema.py:177-306` – schema defaults (e.g., `start_thresh`, recurrent PPO RNN knobs, TD3/DQN hyperparams) are exercised only through `to_dict()` when configs merge YAML; no direct Python references yet.

Everything else reported earlier has been culled.
