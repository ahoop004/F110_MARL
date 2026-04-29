"""SB3 role wrapper for multi-agent F110 training.

Wraps N same-role agents (e.g. all defenders, or all attackers) as a single
Gymnasium environment for one SB3 model.  Training cycles the "focal" agent
each episode; non-focal same-role agents run the shared model in inference
mode so the environment stays realistic.  Fixed-policy agents from other
roles (e.g. FTG opponents) step via their .act() interface.

Usage
-----
    role_wrapper = SB3RoleWrapper(
        env,
        role_agent_ids=["car_0", "car_1", "car_2"],
        obs_dim=57,
        action_low=np.array([-0.46, -1.0]),
        action_high=np.array([0.46, 4.0]),
        observation_preset="centerline",
        reward_strategy=reward_strategy,
        action_repeat=2,
    )
    role_wrapper.set_other_agents({"car_3": ftg3, "car_4": ftg4, "car_5": ftg5})

    model = PPO("MlpPolicy", role_wrapper, ...)
    role_wrapper.set_shared_model(model)   # must be called before model.learn()
    model.learn(...)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.core.obs_flatten import flatten_observation
from src.rewards.base import RewardStrategy
from src.metrics.outcomes import determine_outcome


class SB3RoleWrapper(gym.Env):
    """Single-policy wrapper for N same-role agents in a PettingZoo ParallelEnv.

    Each episode one agent is "focal" (generates the SB3 training transition).
    The other same-role agents run the shared model in inference mode so the
    environment dynamics remain multi-agent realistic.  Fixed-policy agents
    (other roles) use their own .act() method.

    The focal agent cycles deterministically: episode 0 → role_agents[0],
    episode 1 → role_agents[1], ..., wrapping around.

    Args:
        env: PettingZoo ParallelEnv (F110ParallelEnv).
        role_agent_ids: Ordered list of agent IDs that share this policy.
        obs_dim: Flat observation dimension (must match flatten output).
        action_low: Physical action lower bounds.
        action_high: Physical action upper bounds.
        observation_preset: 'centerline' or 'gaplock'; used for flattening.
        reward_strategy: RewardStrategy applied to the focal agent each step.
        action_repeat: Number of env steps per decision step.
        action_constraints: Optional dict with keys:
            - prevent_reverse (bool): clip negative speed to 0
            - speed_index (int): which action dim is speed (default 1)
            - max_v (float): hard cap on physical speed
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        role_agent_ids: List[str],
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        observation_preset: Optional[str] = None,
        reward_strategy: Optional[RewardStrategy] = None,
        action_repeat: int = 1,
        action_constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if not role_agent_ids:
            raise ValueError("role_agent_ids must contain at least one agent ID")

        self.env = env
        self.render_mode = getattr(env, "render_mode", None)
        self.metadata = getattr(env, "metadata", {"render_modes": []})

        self._role_ids: List[str] = list(role_agent_ids)
        self.observation_preset = observation_preset
        self.reward_strategy = reward_strategy

        try:
            self.action_repeat = max(1, int(action_repeat))
        except (TypeError, ValueError):
            self.action_repeat = 1

        constraints = action_constraints or {}
        self._prevent_reverse = bool(constraints.get("prevent_reverse", False))
        try:
            self._speed_index = int(constraints.get("speed_index", 1))
        except (TypeError, ValueError):
            self._speed_index = 1
        try:
            self._max_v = float(constraints["max_v"]) if "max_v" in constraints else None
        except (TypeError, ValueError):
            self._max_v = None

        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._action_low.shape, dtype=np.float32
        )

        # Shared SB3 model — set by runner after instantiation
        self._shared_model: Optional[Any] = None
        # Fixed-policy agents from other roles
        self._other_agents: Dict[str, Any] = {}

        # Per-episode state
        self._focal_episode: int = 0          # increments each reset()
        self._current_obs_dict: Optional[Dict[str, Any]] = None
        self._episode_steps: int = 0
        self._prev_actions: Dict[str, np.ndarray] = {
            aid: np.zeros(self._action_low.shape, dtype=np.float32)
            for aid in self._role_ids
        }

        # Observation scales resolved from underlying env
        self._obs_scales = self._resolve_obs_scales()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_shared_model(self, model: Any) -> None:
        """Register the SB3 model so non-focal role agents can run inference."""
        self._shared_model = model

    def set_other_agents(self, agents: Dict[str, Any]) -> None:
        """Register fixed-policy agents for other roles."""
        self._other_agents = {
            aid: agent for aid, agent in agents.items()
            if aid not in self._role_ids
        }

    @property
    def focal_id(self) -> str:
        return self._role_ids[self._focal_episode % len(self._role_ids)]

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Advance focal agent for this episode
        self._focal_episode += 1
        focal = self.focal_id

        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self._current_obs_dict = obs_dict
        self._episode_steps = 0

        # Reset prev_actions for all role agents
        for aid in self._role_ids:
            self._prev_actions[aid] = np.zeros(self._action_low.shape, dtype=np.float32)

        obs = self._get_obs(focal)
        return obs, info_dict.get(focal, {})

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        focal = self.focal_id
        phys_focal = self._denormalize(action)
        self._prev_actions[focal] = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )

        # Build actions for all agents once (sticky across action_repeat)
        actions = self._build_all_actions(focal, phys_focal)

        prev_obs_dict = self._current_obs_dict
        total_reward = 0.0
        reward_components: Dict[str, float] = {}
        terminated = False
        truncated = False
        obs_dict: Optional[Dict] = None
        info_dict: Dict[str, Any] = {}

        for _ in range(self.action_repeat):
            obs_dict, reward_dict, done_dict, truncated_dict, info_dict = self.env.step(actions)
            step_terminated = bool(done_dict.get(focal, False))
            step_truncated = bool(truncated_dict.get(focal, False))

            if self.reward_strategy and prev_obs_dict is not None:
                reward_info = self._build_reward_info(
                    prev_obs=prev_obs_dict,
                    next_obs=obs_dict,
                    info=info_dict,
                    terminated=step_terminated,
                    truncated=step_truncated,
                )
                step_reward, components = self.reward_strategy.compute(reward_info)
                for name, val in (components or {}).items():
                    reward_components[name] = reward_components.get(name, 0.0) + float(val)
            else:
                step_reward = float(reward_dict.get(focal, 0.0))

            total_reward += float(step_reward)
            self._current_obs_dict = obs_dict
            prev_obs_dict = obs_dict
            self._episode_steps += 1
            terminated = step_terminated
            truncated = step_truncated

            if terminated or truncated:
                break

        if obs_dict is None:
            raise RuntimeError("Environment returned no observations during step.")

        obs = self._get_obs(focal)
        info = dict(info_dict.get(focal, {}))

        if terminated or truncated:
            outcome = determine_outcome(info, truncated=truncated)
            info["outcome"] = outcome.value
            info["is_success"] = outcome.is_success()

        if reward_components:
            info["reward_components"] = reward_components

        info["focal_agent"] = focal
        info["role_episode"] = self._focal_episode

        return obs, total_reward, terminated, truncated, info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_all_actions(
        self, focal: str, phys_focal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute actions for every agent in the env for one repeat block."""
        actions: Dict[str, np.ndarray] = {focal: phys_focal}

        # Non-focal same-role agents: run shared model in inference mode
        for aid in self._role_ids:
            if aid == focal:
                continue
            obs = (self._current_obs_dict or {}).get(aid)
            if obs is not None and self._shared_model is not None:
                flat_obs = self._flatten_role_obs(aid, obs)
                norm_act, _ = self._shared_model.predict(flat_obs, deterministic=False)
                actions[aid] = self._denormalize(norm_act)
                self._prev_actions[aid] = np.clip(
                    np.asarray(norm_act, dtype=np.float32), -1.0, 1.0
                )
            else:
                # Model not yet set (first episode): drive straight at half speed
                safe = np.zeros(self._action_low.shape, dtype=np.float32)
                actions[aid] = self._denormalize(safe)

        # Fixed-policy agents (other roles)
        for aid, agent in self._other_agents.items():
            obs = (self._current_obs_dict or {}).get(aid)
            if obs is not None and hasattr(agent, "act"):
                actions[aid] = agent.act(obs)

        return actions

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Extract and flatten the focal agent's current observation."""
        raw = (self._current_obs_dict or {}).get(agent_id, {})
        return self._flatten_role_obs(agent_id, raw)

    def _flatten_role_obs(self, agent_id: str, raw_obs: Any) -> np.ndarray:
        """Flatten one role agent's raw observation dict."""
        if isinstance(raw_obs, np.ndarray):
            return raw_obs.astype(np.float32)

        if isinstance(raw_obs, dict) and self.observation_preset:
            obs_with_prev = dict(raw_obs)
            obs_with_prev["prev_action"] = self._prev_actions.get(
                agent_id, np.zeros(self._action_low.shape, dtype=np.float32)
            )
            return flatten_observation(
                obs_with_prev,
                preset=self.observation_preset,
                scales=self._obs_scales,
            )

        try:
            return np.asarray(raw_obs, dtype=np.float32)
        except Exception:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _denormalize(self, norm_action: np.ndarray) -> np.ndarray:
        """Map normalized action [-1,1] → physical action space."""
        norm = np.clip(np.asarray(norm_action, dtype=np.float32), -1.0, 1.0)
        if self._prevent_reverse and 0 <= self._speed_index < norm.shape[0]:
            if norm[self._speed_index] < 0.0:
                norm = norm.copy()
                norm[self._speed_index] = 0.0
        phys = self._action_low + (norm + 1.0) * 0.5 * (self._action_high - self._action_low)
        if self._max_v is not None and 0 <= self._speed_index < phys.shape[0]:
            if phys[self._speed_index] > self._max_v:
                phys = phys.copy()
                phys[self._speed_index] = self._max_v
        return phys

    def _build_reward_info(
        self,
        prev_obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        focal = self.focal_id
        return {
            "obs": prev_obs.get(focal, {}),
            "next_obs": next_obs.get(focal, {}),
            "info": info.get(focal, {}) if isinstance(info, dict) else {},
            "step": self._episode_steps,
            "done": terminated or truncated,
            "truncated": truncated,
            "timestep": getattr(self.env, "timestep", 0.01),
            "action": self._prev_actions.get(focal, np.zeros(2, dtype=np.float32)),
            "centerline": getattr(self.env, "centerline_points", None),
            "walls": getattr(self.env, "walls", None),
            # Expose all agents' info so reward can read cross-agent state
            "all_info": info,
        }

    def _resolve_obs_scales(self) -> Dict[str, float]:
        scales: Dict[str, float] = {}
        lidar_range = getattr(self.env, "lidar_range", None)
        if lidar_range is not None:
            try:
                scales["lidar_range"] = float(lidar_range)
            except (TypeError, ValueError):
                pass
        params = getattr(self.env, "params", None)
        if isinstance(params, dict):
            candidates = [
                abs(float(v))
                for v in (params.get("v_max"), params.get("v_min"))
                if v is not None
            ]
            if candidates:
                speed_scale = max(candidates)
                if speed_scale > 0.0:
                    scales["speed"] = speed_scale
        return scales


__all__ = ["SB3RoleWrapper"]
