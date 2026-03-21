"""Stable-Baselines3 wrapper for F110 multi-agent environment.

Converts PettingZoo ParallelEnv to single-agent Gymnasium environment
for training with SB3 algorithms (SAC, TD3, PPO).
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.core.obs_flatten import flatten_observation
from src.core.adaptive_controller import AdaptiveScalarController
from src.rewards.base import RewardStrategy
from src.metrics.outcomes import determine_outcome


class SB3SingleAgentWrapper(gym.Env):
    """Wrapper to convert multi-agent F110 env to single-agent for SB3.

    Wraps one agent (attacker) while other agents (e.g., FTG defender)
    run their own policies.

    Args:
        env: PettingZoo ParallelEnv (F110 environment)
        agent_id: ID of agent to control (default: 'car_0')
        obs_dim: Observation dimension (default: 126 for gaplock)
        action_low: Action space lower bounds (default: [-0.46, -1.0])
        action_high: Action space upper bounds (default: [0.46, 1.0])

    Example:
        >>> from core.setup import create_training_setup
        >>> env, agents, _ = create_training_setup(scenario)
        >>>
        >>> # Wrap for SB3
        >>> wrapped_env = SB3SingleAgentWrapper(
        ...     env,
        ...     agent_id='car_0',
        ...     obs_dim=126
        ... )
        >>>
        >>> # Train with SB3
        >>> from stable_baselines3 import SAC
        >>> model = SAC("MlpPolicy", wrapped_env)
        >>> model.learn(total_timesteps=1000000)
    """

    def __init__(
        self,
        env,
        agent_id: str = 'car_0',
        obs_dim: int = 126,
        action_low: np.ndarray = np.array([-0.46, -1.0]),
        action_high: np.ndarray = np.array([0.46, 1.0]),
        observation_preset: Optional[str] = None,
        target_id: Optional[str] = None,
        reward_strategy: Optional[RewardStrategy] = None,
        action_set: Optional[np.ndarray] = None,
        spawn_curriculum: Optional[Any] = None,
        frame_stack: int = 1,
        action_stack: int = 0,
        action_repeat: int = 1,
        action_constraints: Optional[Dict[str, Any]] = None,
        spawn_x_curriculum: Optional[Dict[str, Any]] = None,
        spawn_y_curriculum: Optional[Dict[str, Any]] = None,
        mirrored_y_curriculum: Optional[Dict[str, Any]] = None,
        target_spawn_x_curriculum: Optional[Dict[str, Any]] = None,
        target_spawn_y_curriculum: Optional[Dict[str, Any]] = None,
        ftg_curriculum: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__()

        self.env = env
        self.render_mode = getattr(env, "render_mode", None)
        self.metadata = getattr(env, "metadata", {"render_modes": []})
        self.agent_id = agent_id
        self.observation_preset = observation_preset
        self.target_id = target_id
        self.reward_strategy = reward_strategy
        self.spawn_curriculum = spawn_curriculum
        self._episode_count = 0
        self._last_spawn_info: Optional[Dict[str, Any]] = None
        try:
            frame_stack = int(frame_stack)
        except (TypeError, ValueError):
            frame_stack = 1
        self.frame_stack = max(1, frame_stack)
        try:
            action_stack = int(action_stack)
        except (TypeError, ValueError):
            action_stack = 0
        self.action_stack = max(0, action_stack)
        try:
            action_repeat = int(action_repeat)
        except (TypeError, ValueError):
            action_repeat = 1
        self.action_repeat = max(1, action_repeat)
        self._frame_buffer: Optional[deque] = None
        self.spawn_x_curriculum = (
            dict(spawn_x_curriculum)
            if isinstance(spawn_x_curriculum, dict)
            else None
        )
        self.spawn_y_curriculum = (
            dict(spawn_y_curriculum)
            if isinstance(spawn_y_curriculum, dict)
            else None
        )
        self.mirrored_y_curriculum = (
            dict(mirrored_y_curriculum)
            if isinstance(mirrored_y_curriculum, dict)
            else None
        )
        self.target_spawn_x_curriculum = (
            dict(target_spawn_x_curriculum)
            if isinstance(target_spawn_x_curriculum, dict)
            else None
        )
        self.target_spawn_y_curriculum = (
            dict(target_spawn_y_curriculum)
            if isinstance(target_spawn_y_curriculum, dict)
            else None
        )
        self._local_rng = None
        self._spawn_x_controller: Optional[AdaptiveScalarController] = None
        self._spawn_x_context: Dict[str, float] = {}
        self._spawn_y_controller: Optional[AdaptiveScalarController] = None
        self._spawn_y_context: Dict[str, float] = {}
        self._target_spawn_x_controller: Optional[AdaptiveScalarController] = None
        self._target_spawn_x_context: Dict[str, float] = {}
        self._target_spawn_y_controller: Optional[AdaptiveScalarController] = None
        self._target_spawn_y_context: Dict[str, float] = {}
        self._spawn_x_stage_mode = (
            "active"
            if self.spawn_x_curriculum and bool(self.spawn_x_curriculum.get("enabled", False))
            else "off"
        )
        self._spawn_y_stage_mode = (
            "active"
            if self.spawn_y_curriculum and bool(self.spawn_y_curriculum.get("enabled", False))
            else "off"
        )
        self._mirrored_y_stage_mode = (
            "active"
            if self.mirrored_y_curriculum and bool(self.mirrored_y_curriculum.get("enabled", False))
            else "off"
        )
        self._mirrored_y_controller: Optional[AdaptiveScalarController] = (
            self._make_mirrored_y_controller(self.mirrored_y_curriculum)
            if self.mirrored_y_curriculum
            else None
        )
        self._target_spawn_x_stage_mode = (
            "active"
            if self.target_spawn_x_curriculum
            and bool(self.target_spawn_x_curriculum.get("enabled", False))
            else "off"
        )
        self._target_spawn_y_stage_mode = (
            "active"
            if self.target_spawn_y_curriculum
            and bool(self.target_spawn_y_curriculum.get("enabled", False))
            else "off"
        )
        self.ftg_curriculum = (
            {
                str(agent): dict(cfg)
                for agent, cfg in ftg_curriculum.items()
                if isinstance(cfg, dict)
            }
            if isinstance(ftg_curriculum, dict)
            else {}
        )
        self._ftg_stage_mode = (
            "active"
            if any(bool(cfg.get("enabled", False)) for cfg in self.ftg_curriculum.values())
            else "off"
        )
        self._ftg_curriculum_state: Dict[str, Dict[str, Any]] = {}

        # Define observation space (Box for continuous observations)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Define action space (Discrete or Box)
        self.action_set = action_set
        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)
        self._action_constraints = action_constraints or {}
        try:
            self._max_v = self._action_constraints.get("max_v")
            if self._max_v is not None:
                self._max_v = float(self._max_v)
        except (TypeError, ValueError):
            self._max_v = None
        self._prevent_reverse = bool(self._action_constraints.get("prevent_reverse", False))
        try:
            self._speed_index = int(self._action_constraints.get("speed_index", 1))
        except (TypeError, ValueError):
            self._speed_index = 1
        self._prev_action_norm = np.zeros_like(self._action_low, dtype=np.float32)
        self._action_history: Optional[deque] = None
        if action_set is not None:
            # Discrete action space for DQN/QR-DQN
            action_set_array = np.asarray(action_set, dtype=np.float32)
            if action_set_array.ndim != 2:
                raise ValueError("action_set must be 2D array of shape (n_actions, action_dim)")
            self.action_set = action_set_array
            self._prev_action_norm = np.zeros(action_set_array.shape[1], dtype=np.float32)
            self.action_space = spaces.Discrete(len(action_set_array))
        else:
            # Continuous action space for SAC/TD3/PPO
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self._action_low.shape,
                dtype=np.float32
            )

        # Store agents dict for non-SB3 agents
        self.other_agents = None

        # Resolve observation scales from environment
        self.obs_scales = self._resolve_obs_scales()

        # Track episode state for reward computation
        self.current_obs_dict = None
        self.episode_steps = 0

        if self.frame_stack > 1:
            self._reset_frame_buffer()
        self._reset_action_history()

    def _reset_frame_buffer(self) -> None:
        if self.frame_stack > 1:
            self._frame_buffer = deque(maxlen=self.frame_stack)
        else:
            self._frame_buffer = None

    def _reset_action_history(self) -> None:
        if self.action_stack <= 0:
            self._action_history = None
            return

        history = deque(maxlen=self.action_stack)
        zero_action = np.zeros_like(self._prev_action_norm, dtype=np.float32)
        for _ in range(self.action_stack):
            history.append(zero_action.copy())
        self._action_history = history

    def _append_action_stack(self, obs: np.ndarray) -> np.ndarray:
        if self.action_stack <= 0:
            return obs
        if self._action_history is None:
            self._reset_action_history()
        if self._action_history is None:
            return obs
        action_hist = np.concatenate(list(self._action_history), axis=0).astype(np.float32)
        return np.concatenate([obs.astype(np.float32), action_hist], axis=0)

    def _stack_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if self.frame_stack <= 1:
            return obs
        if self._frame_buffer is None:
            self._reset_frame_buffer()
        buffer = self._frame_buffer
        if buffer is None:
            return obs
        if len(buffer) == 0:
            frames = [obs] * self.frame_stack
            if update:
                buffer.extend(frames)
            return np.concatenate(frames, axis=0)
        if update:
            buffer.append(obs)
            frames = list(buffer)
        else:
            frames = list(buffer) + [obs]
            if len(frames) > self.frame_stack:
                frames = frames[-self.frame_stack:]
        return np.concatenate(frames, axis=0)

    def _resolve_obs_scales(self) -> Dict[str, float]:
        """Resolve fixed observation scales from the environment."""
        scales: Dict[str, float] = {}

        lidar_range = getattr(self.env, "lidar_range", None)
        if lidar_range is not None:
            try:
                scales["lidar_range"] = float(lidar_range)
            except (TypeError, ValueError):
                pass

        params = getattr(self.env, "params", None)
        if isinstance(params, dict):
            candidates = []
            for value in (params.get("v_max"), params.get("v_min")):
                if value is None:
                    continue
                try:
                    candidates.append(abs(float(value)))
                except (TypeError, ValueError):
                    continue
            if candidates:
                speed_scale = max(candidates)
                if speed_scale > 0.0:
                    scales["speed"] = speed_scale

        return scales

    def _flatten_obs(self, obs: Any, all_obs: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Flatten observation if it's a dict, otherwise return as-is.

        Args:
            obs: Raw observation (can be dict or numpy array)
            all_obs: Optional dict of all agent observations (for extracting target state)

        Returns:
            Flattened observation as numpy array
        """
        # If already a numpy array, just return it
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)

        # If it's a dict and we have a preset, flatten it
        if isinstance(obs, dict) and self.observation_preset:
            # If target_id specified and we have all observations, add target state to obs
            if self.target_id and all_obs and self.target_id in all_obs:
                # Create combined observation dict with central_state
                combined_obs = dict(obs)  # Copy agent's own observation
                combined_obs['central_state'] = all_obs[self.target_id]  # Add target as central_state
                return flatten_observation(
                    combined_obs,
                    preset=self.observation_preset,
                    target_id=self.target_id,
                    scales=self.obs_scales,
                )
            else:
                return flatten_observation(
                    obs,
                    preset=self.observation_preset,
                    target_id=self.target_id,
                    scales=self.obs_scales,
                )

        # Otherwise, try to convert to array
        try:
            return np.asarray(obs, dtype=np.float32)
        except Exception:
            raise ValueError(f"Cannot convert observation of type {type(obs)} to numpy array")

    def set_other_agents(self, agents: Dict[str, Any]):
        """Set other agents (e.g., FTG defender) that act in environment.

        Args:
            agents: Dict mapping agent_id to agent object
        """
        self.other_agents = {
            aid: agent for aid, agent in agents.items()
            if aid != self.agent_id
        }
        self._initialize_ftg_curriculum()

    def _attach_spawn_metadata_to_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(info, dict):
            info = {}
        merged = dict(info)
        if not self._last_spawn_info:
            self._attach_ftg_curriculum_metadata(merged)
            return merged

        spawn_mapping = self._last_spawn_info.get("spawn_points", {})
        if isinstance(spawn_mapping, dict):
            spawn_point = spawn_mapping.get(self.agent_id)
            if spawn_point:
                merged["spawn_point"] = spawn_point
            if self.target_id:
                target_spawn_point = spawn_mapping.get(self.target_id)
                if target_spawn_point:
                    merged["target_spawn_point"] = target_spawn_point

        y_curriculum = self._last_spawn_info.get("spawn_y_curriculum")
        if isinstance(y_curriculum, dict):
            merged.update(y_curriculum)
        x_curriculum = self._last_spawn_info.get("spawn_x_curriculum")
        if isinstance(x_curriculum, dict):
            merged.update(x_curriculum)
        target_y_curriculum = self._last_spawn_info.get("target_spawn_y_curriculum")
        if isinstance(target_y_curriculum, dict):
            merged.update(target_y_curriculum)
        target_x_curriculum = self._last_spawn_info.get("target_spawn_x_curriculum")
        if isinstance(target_x_curriculum, dict):
            merged.update(target_x_curriculum)
        self._attach_ftg_curriculum_metadata(merged)
        return merged

    @staticmethod
    def _normalize_stage_mode(mode: str) -> str:
        mode_key = str(mode).strip().lower()
        if mode_key in {"active", "on"}:
            return "active"
        if mode_key in {"frozen", "freeze", "hold", "locked"}:
            return "frozen"
        return "off"

    @staticmethod
    def _normalize_stage_key(stage: str) -> str:
        key = str(stage).strip().lower()
        aliases = {
            "spawn_y": "y",
            "y_spawn": "y",
            "spawn_x": "x",
            "x_spawn": "x",
            "target_spawn_y": "ty",
            "spawn_target_y": "ty",
            "target_y": "ty",
            "y_target": "ty",
            "ty": "ty",
            "target_spawn_x": "tx",
            "spawn_target_x": "tx",
            "target_x": "tx",
            "x_target": "tx",
            "tx": "tx",
            "ftg_curriculum": "ftg",
            "mirrored_y": "my",
            "mirrored_y_curriculum": "my",
            "mirror_y": "my",
            "my": "my",
        }
        return aliases.get(key, key)

    def _target_stage_available(self) -> bool:
        if not self.target_id:
            return False
        possible_agents = list(getattr(self.env, "possible_agents", []))
        if not possible_agents:
            return False
        return self.target_id in possible_agents

    def has_curriculum_stage(self, stage: str) -> bool:
        def _enabled(cfg: Optional[Dict[str, Any]]) -> bool:
            if not isinstance(cfg, dict):
                return False
            return bool(cfg.get("enabled", True))

        key = self._normalize_stage_key(stage)
        if key == "x":
            return _enabled(self.spawn_x_curriculum)
        if key == "y":
            return _enabled(self.spawn_y_curriculum)
        if key == "my":
            return _enabled(self.mirrored_y_curriculum)
        if key == "tx":
            return _enabled(self.target_spawn_x_curriculum) and self._target_stage_available()
        if key == "ty":
            return _enabled(self.target_spawn_y_curriculum) and self._target_stage_available()
        if key == "ftg":
            return bool(self._ftg_curriculum_state)
        return False

    def get_curriculum_stage_progress(self, stage: str) -> Optional[float]:
        key = self._normalize_stage_key(stage)
        if key == "x":
            if self._spawn_x_controller is None:
                return 0.0 if self.has_curriculum_stage("x") else None
            return self._spawn_x_controller.progress()
        if key == "y":
            if self._spawn_y_controller is None:
                return 0.0 if self.has_curriculum_stage("y") else None
            return self._spawn_y_controller.progress()
        if key == "my":
            if not self.has_curriculum_stage("my"):
                return None
            if self._mirrored_y_controller is None:
                return 0.0
            return self._mirrored_y_controller.progress()
        if key == "tx":
            if self._target_spawn_x_controller is None:
                return 0.0 if self.has_curriculum_stage("tx") else None
            return self._target_spawn_x_controller.progress()
        if key == "ty":
            if self._target_spawn_y_controller is None:
                return 0.0 if self.has_curriculum_stage("ty") else None
            return self._target_spawn_y_controller.progress()
        if key == "ftg":
            if not self._ftg_curriculum_state:
                return None
            for state in self._ftg_curriculum_state.values():
                progress = self._compute_ftg_progress(state)
                if progress is not None:
                    return progress
            return None
        return None

    def set_curriculum_stage_mode(self, stage: str, mode: str) -> None:
        key = self._normalize_stage_key(stage)
        normalized_mode = self._normalize_stage_mode(mode)
        if key == "x":
            self._spawn_x_stage_mode = normalized_mode
            if normalized_mode == "off" and self._spawn_x_controller is not None:
                self._spawn_x_controller.reset()
            return
        if key == "y":
            self._spawn_y_stage_mode = normalized_mode
            if normalized_mode == "off" and self._spawn_y_controller is not None:
                self._spawn_y_controller.reset()
            return
        if key == "my":
            self._mirrored_y_stage_mode = normalized_mode
            if normalized_mode == "off" and self._mirrored_y_controller is not None:
                self._mirrored_y_controller.reset()
            return
        if key == "tx":
            self._target_spawn_x_stage_mode = normalized_mode
            if normalized_mode == "off" and self._target_spawn_x_controller is not None:
                self._target_spawn_x_controller.reset()
            return
        if key == "ty":
            self._target_spawn_y_stage_mode = normalized_mode
            if normalized_mode == "off" and self._target_spawn_y_controller is not None:
                self._target_spawn_y_controller.reset()
            return
        if key != "ftg":
            return

        self._ftg_stage_mode = normalized_mode
        for agent_id, state in self._ftg_curriculum_state.items():
            params_state = state.get("params", {})
            if not isinstance(params_state, dict):
                continue
            updates: Dict[str, float] = {}
            for param_name, controller in params_state.items():
                if not isinstance(controller, AdaptiveScalarController):
                    continue
                if normalized_mode == "off":
                    controller.reset()
                updates[param_name] = float(controller.value)
            agent = self.other_agents.get(agent_id) if self.other_agents else None
            if agent is not None and updates:
                self._apply_agent_params(agent, updates)

    def _apply_agent_params(self, agent: Any, params: Dict[str, float]) -> None:
        if not params:
            return
        if hasattr(agent, "apply_config"):
            agent.apply_config(params)
            return
        for key, value in params.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

    def _initialize_ftg_curriculum(self) -> None:
        self._ftg_curriculum_state = {}
        if not self.ftg_curriculum or not self.other_agents:
            return

        for agent_id, cfg in self.ftg_curriculum.items():
            if not bool(cfg.get("enabled", False)):
                continue
            mode = str(cfg.get("mode", "success_adaptive")).strip().lower()
            if mode != "success_adaptive":
                continue

            agent = self.other_agents.get(agent_id)
            if agent is None:
                continue

            raw_params = cfg.get("params", {})
            if not isinstance(raw_params, dict) or not raw_params:
                continue

            try:
                window = max(1, int(cfg.get("window_episodes", 100)))
                update_every = max(1, int(cfg.get("update_every_episodes", window)))
                sr_low = float(cfg.get("sr_lower_bound", 0.85))
                sr_high = float(cfg.get("sr_upper_bound", 0.95))
            except (TypeError, ValueError):
                continue

            regress_enabled = bool(cfg.get("regression_enabled", False))
            regress_trigger = float(cfg.get("regression_sr_trigger", max(0.0, sr_low - 0.20)))
            regress_floor = float(cfg.get("regression_sr_floor", max(0.0, regress_trigger - 0.20)))
            regress_patience = max(1, int(cfg.get("regression_patience_updates", 1)))
            regress_cooldown = max(0, int(cfg.get("regression_cooldown_updates", 0)))

            params_state: Dict[str, AdaptiveScalarController] = {}
            start_params: Dict[str, float] = {}
            for param_name, param_cfg in raw_params.items():
                if not isinstance(param_cfg, dict):
                    continue
                if not hasattr(agent, param_name):
                    continue
                try:
                    current_value = float(getattr(agent, param_name))
                    start = float(param_cfg.get("start", current_value))
                    final = float(param_cfg.get("final", current_value))
                except (TypeError, ValueError):
                    continue

                delta = final - start
                span = abs(delta)
                direction = 1.0 if delta > 0 else -1.0
                default_inc_min = max(span * 0.05, 1e-6)
                default_inc_max = max(span * 0.20, default_inc_min)

                try:
                    inc_min_raw = float(
                        param_cfg.get("increment_min", cfg.get("increment_min", default_inc_min))
                    )
                except (TypeError, ValueError):
                    inc_min_raw = default_inc_min
                try:
                    inc_max_raw = float(
                        param_cfg.get("increment_max", cfg.get("increment_max", default_inc_max))
                    )
                except (TypeError, ValueError):
                    inc_max_raw = default_inc_max

                inc_small_mag = abs(inc_min_raw)
                inc_large_mag = abs(inc_max_raw)
                if inc_large_mag < inc_small_mag:
                    inc_small_mag, inc_large_mag = inc_large_mag, inc_small_mag

                try:
                    regress_inc_min_raw = float(
                        param_cfg.get(
                            "regression_increment_min",
                            cfg.get("regression_increment_min", abs(inc_min_raw)),
                        )
                    )
                except (TypeError, ValueError):
                    regress_inc_min_raw = abs(inc_min_raw)
                try:
                    regress_inc_max_raw = float(
                        param_cfg.get(
                            "regression_increment_max",
                            cfg.get("regression_increment_max", abs(inc_max_raw)),
                        )
                    )
                except (TypeError, ValueError):
                    regress_inc_max_raw = abs(inc_max_raw)
                regress_small_mag = abs(regress_inc_min_raw)
                regress_large_mag = abs(regress_inc_max_raw)
                if regress_large_mag < regress_small_mag:
                    regress_small_mag, regress_large_mag = regress_large_mag, regress_small_mag

                controller = AdaptiveScalarController(
                    initial_value=start,
                    value_min=min(start, final),
                    value_max=max(start, final),
                    window_size=window,
                    update_every=update_every,
                    advance_sr_low=sr_low,
                    advance_sr_high=sr_high,
                    advance_step_min=direction * inc_small_mag,
                    advance_step_max=direction * inc_large_mag,
                    regression_enabled=regress_enabled,
                    regression_sr_trigger=regress_trigger,
                    regression_sr_floor=regress_floor,
                    regression_step_min=-direction * regress_small_mag,
                    regression_step_max=-direction * regress_large_mag,
                    regression_patience_updates=regress_patience,
                    regression_cooldown_updates=regress_cooldown,
                    progress_start=start,
                    progress_end=final,
                )
                params_state[str(param_name)] = controller
                start_params[str(param_name)] = start

            if not params_state:
                continue

            self._apply_agent_params(agent, start_params)

            self._ftg_curriculum_state[agent_id] = {
                "mode": mode,
                "window": window,
                "update_every": update_every,
                "sr_low": sr_low,
                "sr_high": sr_high,
                "regress_enabled": regress_enabled,
                "regress_trigger": regress_trigger,
                "regress_floor": regress_floor,
                "regress_patience": regress_patience,
                "regress_cooldown": regress_cooldown,
                "completed_episodes": 0,
                "last_window_sr": None,
                "last_increment": {name: 0.0 for name in params_state.keys()},
                "last_adjustment": "hold",
                "params": params_state,
            }

    def _compute_ftg_progress(self, state: Dict[str, Any]) -> Optional[float]:
        params_state = state.get("params")
        if not isinstance(params_state, dict) or not params_state:
            return None

        progress_values: List[float] = []
        for controller in params_state.values():
            if not isinstance(controller, AdaptiveScalarController):
                continue
            progress = controller.progress()
            if progress is None:
                continue
            progress_values.append(float(np.clip(progress, 0.0, 1.0)))
        if not progress_values:
            return None
        return float(sum(progress_values) / len(progress_values))

    def _attach_ftg_curriculum_metadata(self, info: Dict[str, Any]) -> None:
        if self._ftg_stage_mode == "off":
            return
        if not self._ftg_curriculum_state:
            return
        for agent_id in sorted(self._ftg_curriculum_state.keys()):
            state = self._ftg_curriculum_state.get(agent_id)
            if not isinstance(state, dict):
                continue
            params_state = state.get("params", {})
            if not isinstance(params_state, dict) or not params_state:
                continue
            first_param = next(iter(params_state.values()))
            if not isinstance(first_param, AdaptiveScalarController):
                continue
            progress = self._compute_ftg_progress(state)
            info["ftg_curriculum_agent"] = agent_id
            info["ftg_curriculum_mode"] = state.get("mode")
            info["ftg_curriculum_progress"] = progress
            info["ftg_curriculum_window_sr"] = first_param.last_window_sr
            info["ftg_curriculum_last_adjustment"] = state.get("last_adjustment")
            info["ftg_curriculum_regress_low_streak"] = int(first_param.regression_low_streak)
            info["ftg_curriculum_regression_enabled"] = bool(first_param.regression_enabled)
            for param_name, controller in params_state.items():
                if not isinstance(controller, AdaptiveScalarController):
                    continue
                try:
                    info[f"ftg_{param_name}"] = float(controller.value)
                except (TypeError, ValueError):
                    continue
            break

    def _update_success_adaptive_ftg_curriculum(self, success: bool) -> None:
        if self._ftg_stage_mode != "active":
            return
        if not self._ftg_curriculum_state:
            return

        for agent_id, state in self._ftg_curriculum_state.items():
            if state.get("mode") != "success_adaptive":
                continue
            params_state = state.get("params", {})
            if not isinstance(params_state, dict) or not params_state:
                continue
            updated_cycle = False
            updates: Dict[str, float] = {}
            any_change = False
            representative: Optional[AdaptiveScalarController] = None
            for param_name, controller in params_state.items():
                if not isinstance(controller, AdaptiveScalarController):
                    continue
                if representative is None:
                    representative = controller
                cycle = controller.observe(bool(success))
                updated_cycle = updated_cycle or cycle
                if cycle:
                    state["last_increment"][param_name] = float(controller.last_delta)
                updates[param_name] = float(controller.value)
                if cycle and abs(float(controller.last_delta)) > 1e-12:
                    any_change = True

            if representative is not None:
                state["completed_episodes"] = int(representative.completed_events)
                state["last_window_sr"] = representative.last_window_sr
                if updated_cycle:
                    state["last_adjustment"] = (
                        str(representative.last_adjustment) if any_change else "hold"
                    )

            if updated_cycle and self.other_agents:
                agent = self.other_agents.get(agent_id)
                if agent is not None and updates:
                    self._apply_agent_params(agent, updates)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            obs: Initial observation for controlled agent
            info: Info dict
        """
        # Apply simple spawn-x/y curricula when configured.
        if options is None and (
            self.spawn_x_curriculum
            or self.spawn_y_curriculum
            or self.target_spawn_x_curriculum
            or self.target_spawn_y_curriculum
        ):
            sampled = self._sample_spawn_curricula(episode=self._episode_count)
            if sampled is not None:
                self._last_spawn_info = sampled
                options = sampled["options"]
            else:
                self._last_spawn_info = None
        # Otherwise apply spawn curriculum if configured and no explicit options provided.
        elif options is None and self.spawn_curriculum is not None:
            spawn_info = self.spawn_curriculum.sample_spawn(episode=self._episode_count)
            self._last_spawn_info = spawn_info
            options = {
                'poses': spawn_info['poses'],
                'velocities': spawn_info['velocities'],
                'lock_speed_steps': spawn_info['lock_speed_steps'],
            }
        else:
            self._last_spawn_info = None

        # Reset underlying environment
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)

        # Store current observations for reward computation
        self.current_obs_dict = obs_dict
        self.episode_steps = 0
        self._reset_frame_buffer()
        self._prev_action_norm = np.zeros(2, dtype=np.float32)
        self._reset_action_history()

        # Extract observation for controlled agent
        obs_with_prev = dict(obs_dict[self.agent_id])
        obs_with_prev["prev_action"] = self._prev_action_norm
        obs = self._flatten_obs(obs_with_prev, all_obs=obs_dict)
        obs = self._append_action_stack(obs)
        obs = self._stack_obs(obs, update=True)
        info = self._attach_spawn_metadata_to_info(info_dict.get(self.agent_id, {}))
        self._episode_count += 1

        return obs, info

    def _sample_spawn_curricula(self, episode: int) -> Optional[Dict[str, Any]]:
        sampled: Optional[Dict[str, Any]] = None
        if self.spawn_y_curriculum:
            sampled = self._sample_spawn_y_curriculum(episode=episode, base=sampled)
        if self.spawn_x_curriculum:
            sampled = self._sample_spawn_x_curriculum(episode=episode, base=sampled)
        if self.target_spawn_y_curriculum:
            sampled = self._sample_target_spawn_y_curriculum(episode=episode, base=sampled)
        if self.target_spawn_x_curriculum:
            sampled = self._sample_target_spawn_x_curriculum(episode=episode, base=sampled)

        # Mirrored-y stage: deterministically reflect attacker y about target y.
        # Active when the orchestrator has advanced to/past the "my" stage.
        if (
            sampled is not None
            and self._mirrored_y_stage_mode in {"active", "frozen"}
            and self.target_id
        ):
            options = sampled.get("options")
            if isinstance(options, dict):
                poses = options.get("poses")
                if poses is not None:
                    poses = np.asarray(poses, dtype=np.float32)
                    possible_agents = list(getattr(self.env, "possible_agents", []))
                    try:
                        agent_idx = possible_agents.index(self.agent_id)
                        target_idx = possible_agents.index(self.target_id)
                    except ValueError:
                        agent_idx = target_idx = -1
                    if (
                        agent_idx >= 0
                        and target_idx >= 0
                        and poses.ndim == 2
                        and poses.shape[0] > max(agent_idx, target_idx)
                    ):
                        target_y = float(poses[target_idx, 1])
                        poses[agent_idx, 1] = 2.0 * target_y - poses[agent_idx, 1]
                        options["poses"] = poses
                        y_meta = sampled.get("spawn_y_curriculum")
                        if isinstance(y_meta, dict):
                            y_meta["spawn_y"] = float(poses[agent_idx, 1])
                            y_meta["spawn_y_mirrored"] = True

        return sampled

    def _sample_spawn_y_curriculum(
        self,
        episode: int,
        base: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.spawn_y_curriculum
        if self._spawn_y_stage_mode == "off":
            return base
        if not cfg:
            return base

        poses = None
        if isinstance(base, dict):
            options = base.get("options")
            if isinstance(options, dict):
                existing_poses = options.get("poses")
                if existing_poses is not None:
                    poses = np.asarray(existing_poses, dtype=np.float32)

        if poses is None:
            base_poses = getattr(self.env, "start_poses", None)
            if base_poses is None:
                return base
            poses = np.asarray(base_poses, dtype=np.float32).copy()
        if poses.ndim != 2 or poses.shape[1] < 3:
            return base

        possible_agents = list(getattr(self.env, "possible_agents", []))
        if not possible_agents:
            return base
        try:
            agent_idx = possible_agents.index(self.agent_id)
        except ValueError:
            return base
        if agent_idx >= poses.shape[0]:
            return base

        try:
            mode = str(cfg.get("mode", "linear")).strip().lower()
        except (TypeError, ValueError):
            return base
        if mode == "success_adaptive":
            low, high, progress, extra_meta = self._resolve_success_adaptive_bounds(
                cfg=cfg,
                poses=poses,
                agent_idx=agent_idx,
            )
        else:
            try:
                y_start = float(cfg.get("y_start", poses[agent_idx, 1]))
                y_final = float(cfg.get("y_final", -1.0))
                ramp_episodes = int(cfg.get("ramp_episodes", 1000))
            except (TypeError, ValueError):
                return base

            if ramp_episodes <= 0:
                progress = 1.0
            else:
                progress = min(max(float(episode) / float(ramp_episodes), 0.0), 1.0)

            y_min = y_start + (y_final - y_start) * progress
            low = min(y_start, y_min)
            high = max(y_start, y_min)
            extra_meta = {}

        rng = getattr(self.env, "rng", None)
        if rng is None:
            if self._local_rng is None:
                self._local_rng = np.random.default_rng()
            rng = self._local_rng
        sampled_y = float(rng.uniform(low, high))

        poses[agent_idx, 1] = sampled_y

        sampled = dict(base) if isinstance(base, dict) else {}
        spawn_name = str(cfg.get("name", "spawn_y_curriculum"))
        spawn_points = dict(sampled.get("spawn_points", {}))
        existing_spawn = str(spawn_points.get(self.agent_id, "")).strip()
        if existing_spawn and spawn_name not in existing_spawn.split("+"):
            spawn_points[self.agent_id] = f"{existing_spawn}+{spawn_name}"
        else:
            spawn_points[self.agent_id] = spawn_name

        sampled["options"] = {"poses": poses}
        sampled["spawn_points"] = spawn_points
        sampled["spawn_y_curriculum"] = {
            "spawn_y_mode": mode,
            "spawn_y": sampled_y,
            "spawn_y_min": float(low),
            "spawn_y_max": float(high),
            "spawn_y_progress": float(progress),
            **extra_meta,
        }
        return sampled

    def _sample_spawn_x_curriculum(
        self,
        episode: int,
        base: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.spawn_x_curriculum
        if self._spawn_x_stage_mode == "off":
            return base
        if not cfg:
            return base

        poses = None
        if isinstance(base, dict):
            options = base.get("options")
            if isinstance(options, dict):
                existing_poses = options.get("poses")
                if existing_poses is not None:
                    poses = np.asarray(existing_poses, dtype=np.float32)

        if poses is None:
            base_poses = getattr(self.env, "start_poses", None)
            if base_poses is None:
                return base
            poses = np.asarray(base_poses, dtype=np.float32).copy()
        if poses.ndim != 2 or poses.shape[1] < 3:
            return base

        possible_agents = list(getattr(self.env, "possible_agents", []))
        if not possible_agents:
            return base
        try:
            agent_idx = possible_agents.index(self.agent_id)
        except ValueError:
            return base
        if agent_idx >= poses.shape[0]:
            return base

        try:
            mode = str(cfg.get("mode", "linear")).strip().lower()
        except (TypeError, ValueError):
            return base
        if mode == "success_adaptive":
            low, high, progress, extra_meta = self._resolve_success_adaptive_x_bounds(
                cfg=cfg,
                poses=poses,
                agent_idx=agent_idx,
            )
        else:
            try:
                x_start = float(cfg.get("x_start", poses[agent_idx, 0]))
                x_final = float(cfg.get("x_final", -1.0))
                ramp_episodes = int(cfg.get("ramp_episodes", 1000))
            except (TypeError, ValueError):
                return base

            if ramp_episodes <= 0:
                progress = 1.0
            else:
                progress = min(max(float(episode) / float(ramp_episodes), 0.0), 1.0)

            x_min = x_start + (x_final - x_start) * progress
            low = min(x_start, x_min)
            high = max(x_start, x_min)
            extra_meta = {}

        rng = getattr(self.env, "rng", None)
        if rng is None:
            if self._local_rng is None:
                self._local_rng = np.random.default_rng()
            rng = self._local_rng
        sampled_x = float(rng.uniform(low, high))

        poses[agent_idx, 0] = sampled_x

        sampled = dict(base) if isinstance(base, dict) else {}
        spawn_name = str(cfg.get("name", "spawn_x_curriculum"))
        spawn_points = dict(sampled.get("spawn_points", {}))
        existing_spawn = str(spawn_points.get(self.agent_id, "")).strip()
        if existing_spawn and spawn_name not in existing_spawn.split("+"):
            spawn_points[self.agent_id] = f"{existing_spawn}+{spawn_name}"
        else:
            spawn_points[self.agent_id] = spawn_name

        sampled["options"] = {"poses": poses}
        sampled["spawn_points"] = spawn_points
        sampled["spawn_x_curriculum"] = {
            "spawn_x_mode": mode,
            "spawn_x": sampled_x,
            "spawn_x_min": float(low),
            "spawn_x_max": float(high),
            "spawn_x_progress": float(progress),
            **extra_meta,
        }
        return sampled

    def _sample_target_spawn_y_curriculum(
        self,
        episode: int,
        base: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.target_spawn_y_curriculum
        if self._target_spawn_y_stage_mode == "off":
            return base
        if not cfg or not self.target_id:
            return base

        poses = None
        if isinstance(base, dict):
            options = base.get("options")
            if isinstance(options, dict):
                existing_poses = options.get("poses")
                if existing_poses is not None:
                    poses = np.asarray(existing_poses, dtype=np.float32)

        if poses is None:
            base_poses = getattr(self.env, "start_poses", None)
            if base_poses is None:
                return base
            poses = np.asarray(base_poses, dtype=np.float32).copy()
        if poses.ndim != 2 or poses.shape[1] < 3:
            return base

        possible_agents = list(getattr(self.env, "possible_agents", []))
        if not possible_agents:
            return base
        try:
            target_idx = possible_agents.index(self.target_id)
        except ValueError:
            return base
        if target_idx >= poses.shape[0]:
            return base

        try:
            mode = str(cfg.get("mode", "linear")).strip().lower()
        except (TypeError, ValueError):
            return base
        if mode == "success_adaptive":
            low, high, progress, extra_meta = self._resolve_success_adaptive_target_y_bounds(
                cfg=cfg,
                poses=poses,
                target_idx=target_idx,
            )
        else:
            try:
                y_start = float(cfg.get("y_start", poses[target_idx, 1]))
                y_final = float(cfg.get("y_final", y_start))
                ramp_episodes = int(cfg.get("ramp_episodes", 1000))
            except (TypeError, ValueError):
                return base

            if ramp_episodes <= 0:
                progress = 1.0
            else:
                progress = min(max(float(episode) / float(ramp_episodes), 0.0), 1.0)

            y_min = y_start + (y_final - y_start) * progress
            low = min(y_start, y_min)
            high = max(y_start, y_min)
            extra_meta = {}

        rng = getattr(self.env, "rng", None)
        if rng is None:
            if self._local_rng is None:
                self._local_rng = np.random.default_rng()
            rng = self._local_rng
        sampled_y = float(rng.uniform(low, high))

        poses[target_idx, 1] = sampled_y

        sampled = dict(base) if isinstance(base, dict) else {}
        spawn_name = str(cfg.get("name", "target_spawn_y_curriculum"))
        spawn_points = dict(sampled.get("spawn_points", {}))
        existing_spawn = str(spawn_points.get(self.target_id, "")).strip()
        if existing_spawn and spawn_name not in existing_spawn.split("+"):
            spawn_points[self.target_id] = f"{existing_spawn}+{spawn_name}"
        else:
            spawn_points[self.target_id] = spawn_name

        sampled["options"] = {"poses": poses}
        sampled["spawn_points"] = spawn_points
        sampled["target_spawn_y_curriculum"] = {
            "target_spawn_y_mode": mode,
            "target_spawn_y": sampled_y,
            "target_spawn_y_min": float(low),
            "target_spawn_y_max": float(high),
            "target_spawn_y_progress": float(progress),
            **extra_meta,
        }
        return sampled

    def _sample_target_spawn_x_curriculum(
        self,
        episode: int,
        base: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.target_spawn_x_curriculum
        if self._target_spawn_x_stage_mode == "off":
            return base
        if not cfg or not self.target_id:
            return base

        poses = None
        if isinstance(base, dict):
            options = base.get("options")
            if isinstance(options, dict):
                existing_poses = options.get("poses")
                if existing_poses is not None:
                    poses = np.asarray(existing_poses, dtype=np.float32)

        if poses is None:
            base_poses = getattr(self.env, "start_poses", None)
            if base_poses is None:
                return base
            poses = np.asarray(base_poses, dtype=np.float32).copy()
        if poses.ndim != 2 or poses.shape[1] < 3:
            return base

        possible_agents = list(getattr(self.env, "possible_agents", []))
        if not possible_agents:
            return base
        try:
            target_idx = possible_agents.index(self.target_id)
        except ValueError:
            return base
        if target_idx >= poses.shape[0]:
            return base

        try:
            mode = str(cfg.get("mode", "linear")).strip().lower()
        except (TypeError, ValueError):
            return base
        if mode == "success_adaptive":
            low, high, progress, extra_meta = self._resolve_success_adaptive_target_x_bounds(
                cfg=cfg,
                poses=poses,
                target_idx=target_idx,
            )
        else:
            try:
                x_start = float(cfg.get("x_start", poses[target_idx, 0]))
                x_final = float(cfg.get("x_final", x_start))
                ramp_episodes = int(cfg.get("ramp_episodes", 1000))
            except (TypeError, ValueError):
                return base

            if ramp_episodes <= 0:
                progress = 1.0
            else:
                progress = min(max(float(episode) / float(ramp_episodes), 0.0), 1.0)

            x_min = x_start + (x_final - x_start) * progress
            low = min(x_start, x_min)
            high = max(x_start, x_min)
            extra_meta = {}

        rng = getattr(self.env, "rng", None)
        if rng is None:
            if self._local_rng is None:
                self._local_rng = np.random.default_rng()
            rng = self._local_rng
        sampled_x = float(rng.uniform(low, high))

        poses[target_idx, 0] = sampled_x

        sampled = dict(base) if isinstance(base, dict) else {}
        spawn_name = str(cfg.get("name", "target_spawn_x_curriculum"))
        spawn_points = dict(sampled.get("spawn_points", {}))
        existing_spawn = str(spawn_points.get(self.target_id, "")).strip()
        if existing_spawn and spawn_name not in existing_spawn.split("+"):
            spawn_points[self.target_id] = f"{existing_spawn}+{spawn_name}"
        else:
            spawn_points[self.target_id] = spawn_name

        sampled["options"] = {"poses": poses}
        sampled["spawn_points"] = spawn_points
        sampled["target_spawn_x_curriculum"] = {
            "target_spawn_x_mode": mode,
            "target_spawn_x": sampled_x,
            "target_spawn_x_min": float(low),
            "target_spawn_x_max": float(high),
            "target_spawn_x_progress": float(progress),
            **extra_meta,
        }
        return sampled

    def _resolve_success_adaptive_bounds(
        self,
        cfg: Dict[str, Any],
        poses: np.ndarray,
        agent_idx: int,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        controller = self._spawn_y_controller
        if controller is None:
            y_ref = float(poses[agent_idx, 1])
            y_upper_cfg = float(cfg.get("y_upper", cfg.get("y_start", y_ref)))
            y_lower_start_cfg = float(cfg.get("y_lower_start", cfg.get("y_start", y_ref)))
            y_lower_bound_cfg = float(cfg.get("y_lower_bound", cfg.get("y_final", -1.0)))

            y_upper = max(y_upper_cfg, y_lower_start_cfg)
            y_lower_start = min(y_upper_cfg, y_lower_start_cfg)
            y_lower_bound = min(y_lower_bound_cfg, y_upper)

            window = max(1, int(cfg.get("window_episodes", 100)))
            update_every = max(1, int(cfg.get("update_every_episodes", window)))
            sr_low = float(cfg.get("sr_lower_bound", 0.85))
            sr_high = float(cfg.get("sr_upper_bound", 0.95))

            regress_enabled = bool(cfg.get("regression_enabled", False))
            regress_trigger = float(cfg.get("regression_sr_trigger", max(0.0, sr_low - 0.20)))
            regress_floor = float(cfg.get("regression_sr_floor", max(0.0, regress_trigger - 0.20)))
            regress_patience = max(1, int(cfg.get("regression_patience_updates", 1)))
            regress_cooldown = max(0, int(cfg.get("regression_cooldown_updates", 0)))

            inc_min_raw = float(cfg.get("increment_min", -0.001))
            inc_max_raw = float(cfg.get("increment_max", -0.01))
            inc_small = -abs(inc_min_raw)
            inc_large = -abs(inc_max_raw)
            if abs(inc_large) < abs(inc_small):
                inc_small, inc_large = inc_large, inc_small

            regress_inc_min_raw = float(cfg.get("regression_increment_min", abs(inc_min_raw)))
            regress_inc_max_raw = float(cfg.get("regression_increment_max", abs(inc_max_raw)))
            regress_inc_small = abs(regress_inc_min_raw)
            regress_inc_large = abs(regress_inc_max_raw)
            if regress_inc_large < regress_inc_small:
                regress_inc_small, regress_inc_large = regress_inc_large, regress_inc_small

            controller = AdaptiveScalarController(
                initial_value=y_lower_start,
                value_min=y_lower_bound,
                value_max=y_lower_start,
                window_size=window,
                update_every=update_every,
                advance_sr_low=sr_low,
                advance_sr_high=sr_high,
                advance_step_min=inc_small,
                advance_step_max=inc_large,
                regression_enabled=regress_enabled,
                regression_sr_trigger=regress_trigger,
                regression_sr_floor=regress_floor,
                regression_step_min=regress_inc_small,
                regression_step_max=regress_inc_large,
                regression_patience_updates=regress_patience,
                regression_cooldown_updates=regress_cooldown,
                progress_start=y_lower_start,
                progress_end=y_lower_bound,
            )
            self._spawn_y_controller = controller
            self._spawn_y_context = {
                "y_upper": float(y_upper),
                "y_lower_start": float(y_lower_start),
                "y_lower_bound": float(y_lower_bound),
            }

        y_upper = float(self._spawn_y_context.get("y_upper", poses[agent_idx, 1]))
        y_lower = float(controller.value)
        progress = controller.progress()
        if progress is None:
            progress = 0.0

        extra_meta = {
            "spawn_y_window_sr": controller.last_window_sr,
            "spawn_y_last_increment": float(controller.last_delta),
            "spawn_y_completed_episodes": int(controller.completed_events),
            "spawn_y_last_adjustment": str(controller.last_adjustment),
            "spawn_y_regress_low_streak": int(controller.regression_low_streak),
            "spawn_y_regression_enabled": bool(controller.regression_enabled),
        }
        return y_lower, y_upper, progress, extra_meta

    def _resolve_success_adaptive_x_bounds(
        self,
        cfg: Dict[str, Any],
        poses: np.ndarray,
        agent_idx: int,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        controller = self._spawn_x_controller
        if controller is None:
            x_ref = float(poses[agent_idx, 0])
            x_upper_cfg = float(cfg.get("x_upper", cfg.get("x_start", x_ref)))
            x_lower_start_cfg = float(cfg.get("x_lower_start", cfg.get("x_start", x_ref)))
            x_lower_bound_cfg = float(cfg.get("x_lower_bound", cfg.get("x_final", -1.0)))

            x_upper = max(x_upper_cfg, x_lower_start_cfg)
            x_lower_start = min(x_upper_cfg, x_lower_start_cfg)
            x_lower_bound = min(x_lower_bound_cfg, x_upper)

            window = max(1, int(cfg.get("window_episodes", 100)))
            update_every = max(1, int(cfg.get("update_every_episodes", window)))
            sr_low = float(cfg.get("sr_lower_bound", 0.85))
            sr_high = float(cfg.get("sr_upper_bound", 0.95))

            regress_enabled = bool(cfg.get("regression_enabled", False))
            regress_trigger = float(cfg.get("regression_sr_trigger", max(0.0, sr_low - 0.20)))
            regress_floor = float(cfg.get("regression_sr_floor", max(0.0, regress_trigger - 0.20)))
            regress_patience = max(1, int(cfg.get("regression_patience_updates", 1)))
            regress_cooldown = max(0, int(cfg.get("regression_cooldown_updates", 0)))

            inc_min_raw = float(cfg.get("increment_min", -0.001))
            inc_max_raw = float(cfg.get("increment_max", -0.01))
            inc_small = -abs(inc_min_raw)
            inc_large = -abs(inc_max_raw)
            if abs(inc_large) < abs(inc_small):
                inc_small, inc_large = inc_large, inc_small

            regress_inc_min_raw = float(cfg.get("regression_increment_min", abs(inc_min_raw)))
            regress_inc_max_raw = float(cfg.get("regression_increment_max", abs(inc_max_raw)))
            regress_inc_small = abs(regress_inc_min_raw)
            regress_inc_large = abs(regress_inc_max_raw)
            if regress_inc_large < regress_inc_small:
                regress_inc_small, regress_inc_large = regress_inc_large, regress_inc_small

            controller = AdaptiveScalarController(
                initial_value=x_lower_start,
                value_min=x_lower_bound,
                value_max=x_lower_start,
                window_size=window,
                update_every=update_every,
                advance_sr_low=sr_low,
                advance_sr_high=sr_high,
                advance_step_min=inc_small,
                advance_step_max=inc_large,
                regression_enabled=regress_enabled,
                regression_sr_trigger=regress_trigger,
                regression_sr_floor=regress_floor,
                regression_step_min=regress_inc_small,
                regression_step_max=regress_inc_large,
                regression_patience_updates=regress_patience,
                regression_cooldown_updates=regress_cooldown,
                progress_start=x_lower_start,
                progress_end=x_lower_bound,
            )
            self._spawn_x_controller = controller
            self._spawn_x_context = {
                "x_upper": float(x_upper),
                "x_lower_start": float(x_lower_start),
                "x_lower_bound": float(x_lower_bound),
            }

        x_upper = float(self._spawn_x_context.get("x_upper", poses[agent_idx, 0]))
        x_lower = float(controller.value)
        progress = controller.progress()
        if progress is None:
            progress = 0.0

        extra_meta = {
            "spawn_x_window_sr": controller.last_window_sr,
            "spawn_x_last_increment": float(controller.last_delta),
            "spawn_x_completed_episodes": int(controller.completed_events),
            "spawn_x_last_adjustment": str(controller.last_adjustment),
            "spawn_x_regress_low_streak": int(controller.regression_low_streak),
            "spawn_x_regression_enabled": bool(controller.regression_enabled),
        }
        return x_lower, x_upper, progress, extra_meta

    def _resolve_success_adaptive_target_y_bounds(
        self,
        cfg: Dict[str, Any],
        poses: np.ndarray,
        target_idx: int,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        controller = self._target_spawn_y_controller
        if controller is None:
            y_ref = float(poses[target_idx, 1])
            y_upper_cfg = float(cfg.get("y_upper", cfg.get("y_start", y_ref)))
            y_lower_start_cfg = float(cfg.get("y_lower_start", cfg.get("y_start", y_ref)))
            y_lower_bound_cfg = float(cfg.get("y_lower_bound", cfg.get("y_final", y_ref)))

            y_upper = max(y_upper_cfg, y_lower_start_cfg)
            y_lower_start = min(y_upper_cfg, y_lower_start_cfg)
            y_lower_bound = min(y_lower_bound_cfg, y_upper)

            window = max(1, int(cfg.get("window_episodes", 100)))
            update_every = max(1, int(cfg.get("update_every_episodes", window)))
            sr_low = float(cfg.get("sr_lower_bound", 0.85))
            sr_high = float(cfg.get("sr_upper_bound", 0.95))

            regress_enabled = bool(cfg.get("regression_enabled", False))
            regress_trigger = float(cfg.get("regression_sr_trigger", max(0.0, sr_low - 0.20)))
            regress_floor = float(cfg.get("regression_sr_floor", max(0.0, regress_trigger - 0.20)))
            regress_patience = max(1, int(cfg.get("regression_patience_updates", 1)))
            regress_cooldown = max(0, int(cfg.get("regression_cooldown_updates", 0)))

            inc_min_raw = float(cfg.get("increment_min", -0.001))
            inc_max_raw = float(cfg.get("increment_max", -0.01))
            inc_small = -abs(inc_min_raw)
            inc_large = -abs(inc_max_raw)
            if abs(inc_large) < abs(inc_small):
                inc_small, inc_large = inc_large, inc_small

            regress_inc_min_raw = float(cfg.get("regression_increment_min", abs(inc_min_raw)))
            regress_inc_max_raw = float(cfg.get("regression_increment_max", abs(inc_max_raw)))
            regress_inc_small = abs(regress_inc_min_raw)
            regress_inc_large = abs(regress_inc_max_raw)
            if regress_inc_large < regress_inc_small:
                regress_inc_small, regress_inc_large = regress_inc_large, regress_inc_small

            controller = AdaptiveScalarController(
                initial_value=y_lower_start,
                value_min=y_lower_bound,
                value_max=y_lower_start,
                window_size=window,
                update_every=update_every,
                advance_sr_low=sr_low,
                advance_sr_high=sr_high,
                advance_step_min=inc_small,
                advance_step_max=inc_large,
                regression_enabled=regress_enabled,
                regression_sr_trigger=regress_trigger,
                regression_sr_floor=regress_floor,
                regression_step_min=regress_inc_small,
                regression_step_max=regress_inc_large,
                regression_patience_updates=regress_patience,
                regression_cooldown_updates=regress_cooldown,
                progress_start=y_lower_start,
                progress_end=y_lower_bound,
            )
            self._target_spawn_y_controller = controller
            self._target_spawn_y_context = {
                "y_upper": float(y_upper),
                "y_lower_start": float(y_lower_start),
                "y_lower_bound": float(y_lower_bound),
            }

        y_upper = float(self._target_spawn_y_context.get("y_upper", poses[target_idx, 1]))
        y_lower = float(controller.value)
        progress = controller.progress()
        if progress is None:
            progress = 0.0

        extra_meta = {
            "target_spawn_y_window_sr": controller.last_window_sr,
            "target_spawn_y_last_increment": float(controller.last_delta),
            "target_spawn_y_completed_episodes": int(controller.completed_events),
            "target_spawn_y_last_adjustment": str(controller.last_adjustment),
            "target_spawn_y_regress_low_streak": int(controller.regression_low_streak),
            "target_spawn_y_regression_enabled": bool(controller.regression_enabled),
        }
        return y_lower, y_upper, progress, extra_meta

    def _resolve_success_adaptive_target_x_bounds(
        self,
        cfg: Dict[str, Any],
        poses: np.ndarray,
        target_idx: int,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        controller = self._target_spawn_x_controller
        if controller is None:
            x_ref = float(poses[target_idx, 0])
            x_upper_cfg = float(cfg.get("x_upper", cfg.get("x_start", x_ref)))
            x_lower_start_cfg = float(cfg.get("x_lower_start", cfg.get("x_start", x_ref)))
            x_lower_bound_cfg = float(cfg.get("x_lower_bound", cfg.get("x_final", x_ref)))

            x_upper = max(x_upper_cfg, x_lower_start_cfg)
            x_lower_start = min(x_upper_cfg, x_lower_start_cfg)
            x_lower_bound = min(x_lower_bound_cfg, x_upper)

            window = max(1, int(cfg.get("window_episodes", 100)))
            update_every = max(1, int(cfg.get("update_every_episodes", window)))
            sr_low = float(cfg.get("sr_lower_bound", 0.85))
            sr_high = float(cfg.get("sr_upper_bound", 0.95))

            regress_enabled = bool(cfg.get("regression_enabled", False))
            regress_trigger = float(cfg.get("regression_sr_trigger", max(0.0, sr_low - 0.20)))
            regress_floor = float(cfg.get("regression_sr_floor", max(0.0, regress_trigger - 0.20)))
            regress_patience = max(1, int(cfg.get("regression_patience_updates", 1)))
            regress_cooldown = max(0, int(cfg.get("regression_cooldown_updates", 0)))

            inc_min_raw = float(cfg.get("increment_min", -0.001))
            inc_max_raw = float(cfg.get("increment_max", -0.01))
            inc_small = -abs(inc_min_raw)
            inc_large = -abs(inc_max_raw)
            if abs(inc_large) < abs(inc_small):
                inc_small, inc_large = inc_large, inc_small

            regress_inc_min_raw = float(cfg.get("regression_increment_min", abs(inc_min_raw)))
            regress_inc_max_raw = float(cfg.get("regression_increment_max", abs(inc_max_raw)))
            regress_inc_small = abs(regress_inc_min_raw)
            regress_inc_large = abs(regress_inc_max_raw)
            if regress_inc_large < regress_inc_small:
                regress_inc_small, regress_inc_large = regress_inc_large, regress_inc_small

            controller = AdaptiveScalarController(
                initial_value=x_lower_start,
                value_min=x_lower_bound,
                value_max=x_lower_start,
                window_size=window,
                update_every=update_every,
                advance_sr_low=sr_low,
                advance_sr_high=sr_high,
                advance_step_min=inc_small,
                advance_step_max=inc_large,
                regression_enabled=regress_enabled,
                regression_sr_trigger=regress_trigger,
                regression_sr_floor=regress_floor,
                regression_step_min=regress_inc_small,
                regression_step_max=regress_inc_large,
                regression_patience_updates=regress_patience,
                regression_cooldown_updates=regress_cooldown,
                progress_start=x_lower_start,
                progress_end=x_lower_bound,
            )
            self._target_spawn_x_controller = controller
            self._target_spawn_x_context = {
                "x_upper": float(x_upper),
                "x_lower_start": float(x_lower_start),
                "x_lower_bound": float(x_lower_bound),
            }

        x_upper = float(self._target_spawn_x_context.get("x_upper", poses[target_idx, 0]))
        x_lower = float(controller.value)
        progress = controller.progress()
        if progress is None:
            progress = 0.0

        extra_meta = {
            "target_spawn_x_window_sr": controller.last_window_sr,
            "target_spawn_x_last_increment": float(controller.last_delta),
            "target_spawn_x_completed_episodes": int(controller.completed_events),
            "target_spawn_x_last_adjustment": str(controller.last_adjustment),
            "target_spawn_x_regress_low_streak": int(controller.regression_low_streak),
            "target_spawn_x_regression_enabled": bool(controller.regression_enabled),
        }
        return x_lower, x_upper, progress, extra_meta

    def _update_success_adaptive_spawn_y(self, success: bool) -> None:
        cfg = self.spawn_y_curriculum
        controller = self._spawn_y_controller
        if self._spawn_y_stage_mode != "active":
            return
        if not cfg or controller is None:
            return
        controller.observe(bool(success))

    def _update_success_adaptive_spawn_x(self, success: bool) -> None:
        cfg = self.spawn_x_curriculum
        controller = self._spawn_x_controller
        if self._spawn_x_stage_mode != "active":
            return
        if not cfg or controller is None:
            return
        controller.observe(bool(success))

    def _update_success_adaptive_target_spawn_y(self, success: bool) -> None:
        cfg = self.target_spawn_y_curriculum
        controller = self._target_spawn_y_controller
        if self._target_spawn_y_stage_mode != "active":
            return
        if not cfg or controller is None:
            return
        controller.observe(bool(success))

    def _update_success_adaptive_target_spawn_x(self, success: bool) -> None:
        cfg = self.target_spawn_x_curriculum
        controller = self._target_spawn_x_controller
        if self._target_spawn_x_stage_mode != "active":
            return
        if not cfg or controller is None:
            return
        controller.observe(bool(success))

    def _make_mirrored_y_controller(self, cfg: dict) -> AdaptiveScalarController:
        """Build an AdaptiveScalarController with dummy scalar 0→1 for mirrored-y stage."""
        window = max(1, int(cfg.get("window_episodes", 200)))
        update_every = max(1, int(cfg.get("update_every_episodes", window)))
        sr_low = float(cfg.get("sr_lower_bound", 0.85))
        sr_high = float(cfg.get("sr_upper_bound", 0.95))
        inc_min = abs(float(cfg.get("increment_min", 0.10)))
        inc_max = abs(float(cfg.get("increment_max", 0.20)))
        if inc_max < inc_min:
            inc_min, inc_max = inc_max, inc_min
        regress_enabled = bool(cfg.get("regression_enabled", False))
        regress_trigger = float(cfg.get("regression_sr_trigger", max(0.0, sr_low - 0.20)))
        regress_floor = float(cfg.get("regression_sr_floor", max(0.0, regress_trigger - 0.20)))
        regress_patience = max(1, int(cfg.get("regression_patience_updates", 1)))
        regress_cooldown = max(0, int(cfg.get("regression_cooldown_updates", 0)))
        ri_min = abs(float(cfg.get("regression_increment_min", inc_min)))
        ri_max = abs(float(cfg.get("regression_increment_max", inc_max)))
        if ri_max < ri_min:
            ri_min, ri_max = ri_max, ri_min
        return AdaptiveScalarController(
            initial_value=0.0,
            value_min=0.0,
            value_max=1.0,
            window_size=window,
            update_every=update_every,
            advance_sr_low=sr_low,
            advance_sr_high=sr_high,
            advance_step_min=inc_min,
            advance_step_max=inc_max,
            regression_enabled=regress_enabled,
            regression_sr_trigger=regress_trigger,
            regression_sr_floor=regress_floor,
            regression_step_min=ri_min,
            regression_step_max=ri_max,
            regression_patience_updates=regress_patience,
            regression_cooldown_updates=regress_cooldown,
            progress_start=0.0,
            progress_end=1.0,
        )

    def _refresh_spawn_y_runtime_metadata(self) -> None:
        if not self._last_spawn_info:
            return
        y_curriculum = self._last_spawn_info.get("spawn_y_curriculum")
        if not isinstance(y_curriculum, dict):
            return

        controller = self._spawn_y_controller
        if controller is None:
            return
        y_upper = float(self._spawn_y_context.get("y_upper", y_curriculum.get("spawn_y_max", 0.0)))
        y_lower = float(controller.value)
        progress = controller.progress()
        if progress is None:
            progress = 0.0

        y_curriculum.update(
            {
                "spawn_y_min": y_lower,
                "spawn_y_max": y_upper,
                "spawn_y_progress": progress,
                "spawn_y_window_sr": controller.last_window_sr,
                "spawn_y_last_increment": float(controller.last_delta),
                "spawn_y_completed_episodes": int(controller.completed_events),
                "spawn_y_last_adjustment": str(controller.last_adjustment),
                "spawn_y_regress_low_streak": int(controller.regression_low_streak),
                "spawn_y_regression_enabled": bool(controller.regression_enabled),
            }
        )

    def _refresh_spawn_x_runtime_metadata(self) -> None:
        if not self._last_spawn_info:
            return
        x_curriculum = self._last_spawn_info.get("spawn_x_curriculum")
        if not isinstance(x_curriculum, dict):
            return

        controller = self._spawn_x_controller
        if controller is None:
            return
        x_upper = float(self._spawn_x_context.get("x_upper", x_curriculum.get("spawn_x_max", 0.0)))
        x_lower = float(controller.value)
        progress = controller.progress()
        if progress is None:
            progress = 0.0

        x_curriculum.update(
            {
                "spawn_x_min": x_lower,
                "spawn_x_max": x_upper,
                "spawn_x_progress": progress,
                "spawn_x_window_sr": controller.last_window_sr,
                "spawn_x_last_increment": float(controller.last_delta),
                "spawn_x_completed_episodes": int(controller.completed_events),
                "spawn_x_last_adjustment": str(controller.last_adjustment),
                "spawn_x_regress_low_streak": int(controller.regression_low_streak),
                "spawn_x_regression_enabled": bool(controller.regression_enabled),
            }
        )

    def _refresh_target_spawn_y_runtime_metadata(self) -> None:
        if not self._last_spawn_info:
            return
        y_curriculum = self._last_spawn_info.get("target_spawn_y_curriculum")
        if not isinstance(y_curriculum, dict):
            return

        controller = self._target_spawn_y_controller
        if controller is None:
            return
        y_upper = float(
            self._target_spawn_y_context.get("y_upper", y_curriculum.get("target_spawn_y_max", 0.0))
        )
        y_lower = float(controller.value)
        progress = controller.progress()
        if progress is None:
            progress = 0.0

        y_curriculum.update(
            {
                "target_spawn_y_min": y_lower,
                "target_spawn_y_max": y_upper,
                "target_spawn_y_progress": progress,
                "target_spawn_y_window_sr": controller.last_window_sr,
                "target_spawn_y_last_increment": float(controller.last_delta),
                "target_spawn_y_completed_episodes": int(controller.completed_events),
                "target_spawn_y_last_adjustment": str(controller.last_adjustment),
                "target_spawn_y_regress_low_streak": int(controller.regression_low_streak),
                "target_spawn_y_regression_enabled": bool(controller.regression_enabled),
            }
        )

    def _refresh_target_spawn_x_runtime_metadata(self) -> None:
        if not self._last_spawn_info:
            return
        x_curriculum = self._last_spawn_info.get("target_spawn_x_curriculum")
        if not isinstance(x_curriculum, dict):
            return

        controller = self._target_spawn_x_controller
        if controller is None:
            return
        x_upper = float(
            self._target_spawn_x_context.get("x_upper", x_curriculum.get("target_spawn_x_max", 0.0))
        )
        x_lower = float(controller.value)
        progress = controller.progress()
        if progress is None:
            progress = 0.0

        x_curriculum.update(
            {
                "target_spawn_x_min": x_lower,
                "target_spawn_x_max": x_upper,
                "target_spawn_x_progress": progress,
                "target_spawn_x_window_sr": controller.last_window_sr,
                "target_spawn_x_last_increment": float(controller.last_delta),
                "target_spawn_x_completed_episodes": int(controller.completed_events),
                "target_spawn_x_last_adjustment": str(controller.last_adjustment),
                "target_spawn_x_regress_low_streak": int(controller.regression_low_streak),
                "target_spawn_x_regression_enabled": bool(controller.regression_enabled),
            }
        )

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action for controlled agent (discrete index or continuous action)

        Returns:
            obs: Next observation
            reward: Reward
            terminated: Whether episode ended (goal reached or failed)
            truncated: Whether episode was truncated (time limit)
            info: Info dict
        """
        # Convert discrete action to continuous if using action_set
        if self.action_set is not None:
            action_idx = int(action)
            continuous_action = self.action_set[action_idx]
            self._prev_action_norm = np.clip(
                np.asarray(continuous_action, dtype=np.float32),
                -1.0,
                1.0,
            )
        else:
            action_norm = np.asarray(action, dtype=np.float32)
            self._prev_action_norm = np.clip(action_norm, -1.0, 1.0)
            if self._prevent_reverse and 0 <= self._speed_index < self._prev_action_norm.shape[0]:
                if self._prev_action_norm[self._speed_index] < 0.0:
                    self._prev_action_norm = self._prev_action_norm.copy()
                    self._prev_action_norm[self._speed_index] = 0.0
            continuous_action = self._action_low + (self._prev_action_norm + 1.0) * 0.5 * (
                self._action_high - self._action_low
            )
            if self._max_v is not None and 0 <= self._speed_index < continuous_action.shape[0]:
                if continuous_action[self._speed_index] > self._max_v:
                    continuous_action = continuous_action.copy()
                    continuous_action[self._speed_index] = self._max_v

        if self._action_history is not None:
            self._action_history.append(self._prev_action_norm.copy())

        # Build action dict for all agents
        actions = {self.agent_id: continuous_action}

        # Get actions from other agents (if they exist)
        if self.other_agents:
            # Get current observations for other agents from stored obs dict
            for agent_id, agent in self.other_agents.items():
                # Other agents select their own actions
                # (This assumes they have an act() method)
                obs = self.current_obs_dict.get(agent_id) if self.current_obs_dict else None
                if obs is not None and hasattr(agent, 'act'):
                    actions[agent_id] = agent.act(obs)

        prev_obs_dict = self.current_obs_dict
        total_reward = 0.0
        reward_components: Dict[str, float] = {}
        terminated = False
        truncated = False
        obs_dict = None
        info_dict: Dict[str, Any] = {}

        for _ in range(self.action_repeat):
            # Step environment with all actions
            obs_dict, reward_dict, done_dict, truncated_dict, info_dict = self.env.step(actions)

            step_terminated = bool(done_dict[self.agent_id])
            step_truncated = bool(truncated_dict[self.agent_id])

            # Compute reward using custom strategy if provided
            if self.reward_strategy and prev_obs_dict:
                reward_info = self._build_reward_info(
                    prev_obs=prev_obs_dict,
                    next_obs=obs_dict,
                    info=info_dict,
                    terminated=step_terminated,
                    truncated=step_truncated,
                )
                step_reward, components = self.reward_strategy.compute(reward_info)
                if components:
                    for name, value in components.items():
                        reward_components[name] = reward_components.get(name, 0.0) + float(value)
                step_reward_value = float(step_reward)
            else:
                # Use environment's default reward
                step_reward_value = float(reward_dict[self.agent_id])

            total_reward += step_reward_value
            self.current_obs_dict = obs_dict
            self.episode_steps += 1
            prev_obs_dict = obs_dict
            terminated = step_terminated
            truncated = step_truncated

            if terminated or truncated:
                break

        if obs_dict is None:
            raise RuntimeError("Environment returned no observations during step.")

        # Extract results for controlled agent
        obs_with_prev = dict(obs_dict[self.agent_id])
        obs_with_prev["prev_action"] = self._prev_action_norm
        obs = self._flatten_obs(obs_with_prev, all_obs=obs_dict)
        obs = self._append_action_stack(obs)
        obs = self._stack_obs(obs, update=True)
        info = info_dict.get(self.agent_id, {})
        info = dict(info) if isinstance(info, dict) else {}
        target_finished = False
        if self.target_id and isinstance(info_dict, dict):
            target_info = info_dict.get(self.target_id, {})
            if isinstance(target_info, dict):
                target_finished = bool(
                    target_info.get("finish_line", False)
                    or target_info.get("target_finished", False)
                )
        info["target_finished"] = target_finished

        # Determine episode outcome for curriculum tracking
        if terminated or truncated:
            if self.target_id:
                outcome = determine_outcome(info, truncated=truncated)
                info["outcome"] = outcome.value
                info["is_success"] = outcome.is_success()
            elif self.observation_preset == "centerline":
                success = bool(info.get("finish_line", False))
                if success:
                    outcome_value = "finish"
                elif info.get("collision", False):
                    outcome_value = "collision"
                elif truncated:
                    outcome_value = "timeout"
                else:
                    outcome_value = "terminated"
                info["outcome"] = outcome_value
                info["is_success"] = success
            else:
                outcome = determine_outcome(info, truncated=truncated)
                info["outcome"] = outcome.value
                info['is_success'] = outcome.is_success()
            _ep_success = bool(info.get("is_success", False))
            self._update_success_adaptive_spawn_x(_ep_success)
            self._update_success_adaptive_spawn_y(_ep_success)
            self._update_success_adaptive_target_spawn_x(_ep_success)
            self._update_success_adaptive_target_spawn_y(_ep_success)
            self._update_success_adaptive_ftg_curriculum(_ep_success)
            if self._mirrored_y_stage_mode == "active" and self._mirrored_y_controller is not None:
                self._mirrored_y_controller.observe(_ep_success)
            self._refresh_spawn_x_runtime_metadata()
            self._refresh_spawn_y_runtime_metadata()
            self._refresh_target_spawn_x_runtime_metadata()
            self._refresh_target_spawn_y_runtime_metadata()

        if reward_components:
            info['reward_components'] = reward_components

        info = self._attach_spawn_metadata_to_info(info)
        return obs, total_reward, terminated, truncated, info

    def _build_reward_info(
        self,
        prev_obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        """Build reward info dict for custom reward computation.

        Args:
            prev_obs: Previous observations for all agents
            next_obs: Next observations for all agents
            info: Step info from environment
            terminated: Whether episode terminated
            truncated: Whether episode was truncated

        Returns:
            Reward info dict for RewardStrategy.compute()
        """
        # Get timestep from environment if available
        timestep = getattr(self.env, 'timestep', 0.01)

        # Extract info for this agent
        info_for_agent = info.get(self.agent_id, {}) if isinstance(info, dict) else {}

        reward_info = {
            'obs': prev_obs.get(self.agent_id, {}),
            'next_obs': next_obs.get(self.agent_id, {}),
            'info': info_for_agent,
            'step': self.episode_steps,
            'done': terminated or truncated,
            'truncated': truncated,
            'timestep': timestep,
            'action': self._prev_action_norm,
            'centerline': getattr(self.env, 'centerline_points', None),
            'walls': getattr(self.env, 'walls', None),
        }

        # Add target obs if target_id is specified (for adversarial tasks)
        if self.target_id and self.target_id in next_obs:
            reward_info['target_obs'] = next_obs[self.target_id]

        return reward_info

    def render(self):
        """Render environment (delegates to underlying env)."""
        if hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


__all__ = ['SB3SingleAgentWrapper']
