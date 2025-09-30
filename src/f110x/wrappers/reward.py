from typing import Dict, Optional, Set, Tuple

from f110x.utils.reward_utils import ScalingParams, apply_reward_scaling


class RewardWrapper:
    def __init__(
        self,
        *,
        target_crash_reward: float = 1.0,
        ego_collision_penalty: float = 0.0,
        truncation_penalty: float = 0.0,
        success_once: bool = True,
        reward_horizon=None,
        reward_clip=None,
        **_ignored_kwargs,
    ) -> None:
        """Sparse reward wrapper that only pays out on successful pursuits."""

        self.target_crash_reward = float(target_crash_reward)
        self.ego_collision_penalty = float(ego_collision_penalty)
        self.truncation_penalty = float(truncation_penalty)
        self.success_once = bool(success_once)

        horizon = self._coerce_positive_float(reward_horizon)
        clip = self._coerce_positive_float(reward_clip)

        self.scaling_params = ScalingParams(horizon=horizon, clip=clip)
        self.reward_horizon = self.scaling_params.horizon
        self.reward_clip = self.scaling_params.clip

        mode_value = _ignored_kwargs.pop('mode', None) if isinstance(_ignored_kwargs, dict) else None
        self.mode = str(mode_value) if mode_value not in (None, '') else 'sparse'
        self._unused_keys = dict(_ignored_kwargs) if _ignored_kwargs else {}

        self._success_awarded: Set[Tuple[str, str]] = set()
        self._returns: Dict[str, float] = {}
        self._last_components: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _coerce_positive_float(value) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            val = float(value)
        except (TypeError, ValueError):
            return None
        return val if val > 0.0 else None

    def reset(self) -> None:
        self._success_awarded.clear()
        self._returns.clear()
        self._last_components.clear()

    def _select_target_obs(self, agent_id: str, all_obs: Optional[Dict[str, Dict]]) -> Optional[Dict]:
        if not all_obs:
            return None
        for other_id, other_obs in all_obs.items():
            if other_id != agent_id:
                return other_obs
        return None

    def __call__(self, obs, agent_id: str, reward: float, done: bool, info, *, all_obs=None) -> float:
        ego_obs = obs[agent_id]
        accum_return = self._returns.get(agent_id, 0.0)

        shaped = 0.0
        components: Dict[str, float] = {}

        env_reward = float(reward)
        if env_reward:
            components["env_reward"] = env_reward

        target_obs = self._select_target_obs(agent_id, all_obs) if all_obs else None
        ego_crashed = bool(ego_obs.get("collision", False))

        if target_obs is not None:
            target_crashed = bool(target_obs.get("collision", False))
            if target_crashed and not ego_crashed:
                key = (agent_id, str(target_obs.get("agent_id", "target")))
                if not self.success_once or key not in self._success_awarded:
                    shaped += self.target_crash_reward
                    components["success_reward"] = (
                        components.get("success_reward", 0.0) + self.target_crash_reward
                    )
                    if self.success_once:
                        self._success_awarded.add(key)

        if ego_crashed and self.ego_collision_penalty:
            shaped += self.ego_collision_penalty
            components["ego_collision_penalty"] = (
                components.get("ego_collision_penalty", 0.0) + self.ego_collision_penalty
            )

        truncated = False
        if done and isinstance(info, dict):
            truncated = bool(info.get("truncated", False))

        if truncated and self.truncation_penalty:
            shaped += self.truncation_penalty
            components["truncation_penalty"] = (
                components.get("truncation_penalty", 0.0) + self.truncation_penalty
            )

        shaped, components = apply_reward_scaling(shaped, components, self.scaling_params)

        self._returns[agent_id] = accum_return + shaped
        self._last_components[agent_id] = components
        return shaped

    def get_last_components(self, agent_id: str) -> Dict[str, float]:
        return dict(self._last_components.get(agent_id, {}))
