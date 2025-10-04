"""Reward strategy orchestration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from f110x.utils.centerline import project_to_centerline
from f110x.utils.reward_utils import ScalingParams, apply_reward_scaling

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from f110x.envs import F110ParallelEnv
    from f110x.utils.map_loader import MapData


# ---------------------------------------------------------------------------
# Dataclasses and step representation
# ---------------------------------------------------------------------------


@dataclass
class RewardRuntimeContext:
    env: "F110ParallelEnv"
    map_data: "MapData"
    roster: Optional[Any] = None


@dataclass
class RewardStep:
    agent_id: str
    obs: Dict[str, Any]
    env_reward: float
    done: bool
    info: Optional[Dict[str, Any]]
    all_obs: Optional[Dict[str, Dict[str, Any]]]
    episode_index: int
    step_index: int
    current_time: float
    timestep: float


# ---------------------------------------------------------------------------
# Strategy primitives
# ---------------------------------------------------------------------------


class RewardStrategy:
    """Base interface for reward strategies."""

    name: str = "base"

    def reset(self, episode_index: int) -> None:  # pragma: no cover - default noop
        return None

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        raise NotImplementedError


class GaplockRewardStrategy(RewardStrategy):
    name = "gaplock"

    def __init__(
        self,
        *,
        target_crash_reward: float = 1.0,
        truncation_penalty: float = 0.0,
        success_once: bool = True,
        reward_horizon: Optional[float] = None,
        reward_clip: Optional[float] = None,
        target_resolver: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.target_crash_reward = float(target_crash_reward)
        self.truncation_penalty = float(truncation_penalty)
        self.success_once = bool(success_once)
        self.scaling_params = ScalingParams(
            horizon=self._coerce_positive_float(reward_horizon),
            clip=self._coerce_positive_float(reward_clip),
        )
        self._success_awarded: Dict[str, set[Tuple[str, str]]] = {}
        self._target_resolver = target_resolver

    @staticmethod
    def _coerce_positive_float(value: Optional[Any]) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        return val if val > 0.0 else None

    def reset(self, episode_index: int) -> None:
        self._success_awarded.clear()

    def _select_target_obs(
        self,
        step: RewardStep,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not step.all_obs:
            return None, None

        if callable(self._target_resolver):
            candidate_id = self._target_resolver(step.agent_id)
            if candidate_id and candidate_id in step.all_obs:
                candidate_obs = step.all_obs.get(candidate_id)
                if candidate_obs is not None:
                    return candidate_obs, candidate_id

        for other_id, other_obs in step.all_obs.items():
            if other_id != step.agent_id:
                return other_obs, other_id
        return None, None

    def _has_awarded(self, agent_id: str, target_id: str) -> bool:
        awarded = self._success_awarded.setdefault(agent_id, set())
        key = (agent_id, target_id)
        if key in awarded:
            return True
        if self.success_once:
            awarded.add(key)
        return False

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        components: Dict[str, float] = {}
        shaped = 0.0

        env_reward = float(step.env_reward)
        if env_reward:
            components["env_reward"] = env_reward

        ego_obs = step.obs
        target_obs, explicit_target_id = self._select_target_obs(step)
        ego_crashed = bool(ego_obs.get("collision", False))

        if target_obs is not None:
            target_crashed = bool(target_obs.get("collision", False))
            if target_crashed and not ego_crashed:
                target_id = explicit_target_id or str(target_obs.get("agent_id", "target"))
                if not self.success_once or not self._has_awarded(step.agent_id, target_id):
                    shaped += self.target_crash_reward
                    components["success_reward"] = (
                        components.get("success_reward", 0.0) + self.target_crash_reward
                    )

        truncated = bool(step.info.get("truncated", False)) if isinstance(step.info, dict) else False
        if truncated and self.truncation_penalty:
            shaped += self.truncation_penalty
            components["truncation_penalty"] = (
                components.get("truncation_penalty", 0.0) + self.truncation_penalty
            )

        shaped, components = apply_reward_scaling(shaped, components, self.scaling_params)
        return shaped, components


class ProgressRewardStrategy(RewardStrategy):
    name = "progress"

    def __init__(
        self,
        *,
        centerline: Optional[np.ndarray],
        progress_weight: float = 1.0,
        speed_weight: float = 0.0,
        lateral_penalty: float = 0.0,
        heading_penalty: float = 0.0,
    ) -> None:
        self.centerline = None if centerline is None else np.asarray(centerline, dtype=np.float32)
        self.progress_weight = float(progress_weight)
        self.speed_weight = float(speed_weight)
        self.lateral_penalty = float(lateral_penalty)
        self.heading_penalty = float(heading_penalty)
        self._last_index: Dict[str, Optional[int]] = {}
        self._last_progress: Dict[str, float] = {}

    def reset(self, episode_index: int) -> None:
        self._last_index.clear()
        self._last_progress.clear()

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        if self.centerline is None or self.centerline.size == 0:
            return 0.0, {}

        pose = step.obs.get("pose")
        if pose is None or len(pose) < 3:
            return 0.0, {}

        position = np.asarray(pose[:2], dtype=np.float32)
        heading = float(pose[2])

        last_idx = self._last_index.get(step.agent_id)
        try:
            projection = project_to_centerline(
                self.centerline,
                position,
                heading,
                last_index=last_idx,
            )
        except ValueError:
            return 0.0, {}

        self._last_index[step.agent_id] = projection.index

        prev_progress = self._last_progress.get(step.agent_id)
        progress = projection.progress
        if prev_progress is None:
            delta = 0.0
        else:
            delta = progress - prev_progress
            if delta < -0.5:
                delta += 1.0
        self._last_progress[step.agent_id] = progress

        reward = 0.0
        components: Dict[str, float] = {}

        if delta:
            progress_term = self.progress_weight * delta
            reward += progress_term
            components["progress"] = progress_term

        if self.speed_weight:
            velocity = np.asarray(step.obs.get("velocity", (0.0, 0.0)), dtype=np.float32)
            speed = float(np.linalg.norm(velocity))
            speed_term = self.speed_weight * speed * step.timestep
            if speed_term:
                reward += speed_term
                components["speed"] = speed_term

        if self.lateral_penalty:
            penalty = -abs(projection.lateral_error) * self.lateral_penalty
            if penalty:
                reward += penalty
                components["lateral_penalty"] = penalty

        if self.heading_penalty:
            penalty = -abs(projection.heading_error) * self.heading_penalty
            if penalty:
                reward += penalty
                components["heading_penalty"] = penalty

        return reward, components


class FastestLapRewardStrategy(RewardStrategy):
    name = "fastest_lap"

    def __init__(
        self,
        *,
        step_penalty: float = 0.0,
        lap_bonus: float = 1.0,
        best_bonus: float = 0.5,
    ) -> None:
        self.step_penalty = float(step_penalty)
        self.lap_bonus = float(lap_bonus)
        self.best_bonus = float(best_bonus)
        self._lap_count: Dict[str, int] = {}
        self._lap_start_time: Dict[str, float] = {}
        self._best_time: Dict[str, Optional[float]] = {}

    def reset(self, episode_index: int) -> None:
        self._lap_count.clear()
        self._lap_start_time.clear()
        self._best_time.clear()

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        components: Dict[str, float] = {}
        reward = 0.0

        if self.step_penalty:
            penalty = -self.step_penalty * step.timestep
            reward += penalty
            components["time_penalty"] = penalty

        lap_info = step.obs.get("lap")
        if lap_info is None or len(lap_info) == 0:
            return reward, components

        current_count = int(lap_info[0])
        last_count = self._lap_count.get(step.agent_id, current_count)

        if step.agent_id not in self._lap_start_time:
            self._lap_start_time[step.agent_id] = float(step.current_time)

        if current_count > last_count:
            lap_finish_time = float(step.current_time)
            lap_start = self._lap_start_time.get(step.agent_id, lap_finish_time - step.timestep)
            lap_duration = max(lap_finish_time - lap_start, step.timestep)
            self._lap_start_time[step.agent_id] = lap_finish_time
            self._lap_count[step.agent_id] = current_count

            lap_reward = self.lap_bonus
            reward += lap_reward
            components["lap_bonus"] = lap_reward

            best_time = self._best_time.get(step.agent_id)
            if best_time is None or lap_duration < best_time:
                improvement = 1.0
                if best_time is not None and best_time > 0.0:
                    improvement = max((best_time - lap_duration) / best_time, 0.0)
                best_reward = self.best_bonus * improvement
                reward += best_reward
                components["best_lap_bonus"] = best_reward
                self._best_time[step.agent_id] = lap_duration
            else:
                self._best_time[step.agent_id] = best_time
        else:
            self._lap_count[step.agent_id] = current_count

        return reward, components


# ---------------------------------------------------------------------------
# Reward wrapper orchestrator
# ---------------------------------------------------------------------------


GAPLOCK_PARAM_KEYS = {
    "target_crash_reward",
    "truncation_penalty",
    "success_once",
    "reward_horizon",
    "reward_clip",
}

PROGRESS_PARAM_KEYS = {
    "progress_weight",
    "speed_weight",
    "lateral_penalty",
    "heading_penalty",
}

FASTEST_LAP_PARAM_KEYS = {
    "step_penalty",
    "lap_bonus",
    "best_bonus",
}


class RewardWrapper:
    """Orchestrates one or more reward strategies with optional weighting."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        context: RewardRuntimeContext,
    ) -> None:
        self.config = dict(config)
        self.context = context
        self.mode = self._normalize_mode(str(self.config.get("mode", "gaplock")))
        self.ego_collision_penalty = float(self.config.get("ego_collision_penalty", 0.0))

        roster = getattr(self.context, "roster", None)
        self._role_members = self._normalise_role_members(roster)
        self._agent_roles = self._extract_agent_roles(roster)
        self._gaplock_target_resolver = self._build_gaplock_target_resolver()

        self._strategies: List[Tuple[RewardStrategy, float]] = self._build_strategies()
        self.modes: Tuple[str, ...] = tuple(strategy.name for strategy, _ in self._strategies)
        self._episode_index = 0
        self._last_components: Dict[str, Dict[str, float]] = {}
        self._step_counter = 0

    # Strategy construction -------------------------------------------------
    def _normalize_mode(self, mode: str) -> str:
        normalized = mode.strip().lower()
        # historical curriculum modes map onto gaplock logic
        if normalized in {"", "sparse", "basic", "pursuit", "adversarial"}:
            return "gaplock"
        return normalized

    @staticmethod
    def _normalise_role_members(roster: Any) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        if roster is None:
            return mapping

        raw_roles = getattr(roster, "roles", None)
        if isinstance(raw_roles, dict):
            for role, members in raw_roles.items():
                if isinstance(members, (list, tuple, set)):
                    normalised = [str(member) for member in members]
                elif members is None:
                    normalised = []
                else:
                    normalised = [str(members)]
                if normalised:
                    mapping[str(role)] = normalised
        return mapping

    @staticmethod
    def _extract_agent_roles(roster: Any) -> Dict[str, str]:
        roles: Dict[str, str] = {}
        if roster is None:
            return roles

        assignments = getattr(roster, "assignments", None)
        if isinstance(assignments, Iterable):
            for assignment in assignments:
                agent_id = getattr(assignment, "agent_id", None)
                spec = getattr(assignment, "spec", None)
                role = getattr(spec, "role", None)
                if agent_id and role:
                    roles[str(agent_id)] = str(role)
        return roles

    def _build_gaplock_target_resolver(self) -> Optional[Callable[[str], Optional[str]]]:
        if not self._agent_roles and not self._role_members:
            return None

        role_members = {role: list(members) for role, members in self._role_members.items()}
        agent_roles = dict(self._agent_roles)

        attackers = role_members.get("attacker", [])
        defenders = role_members.get("defender", [])

        def resolver(agent_id: str) -> Optional[str]:
            role = agent_roles.get(agent_id)
            if role == "defender":
                for candidate in attackers:
                    if candidate != agent_id:
                        return candidate
            else:
                for candidate in defenders:
                    if candidate != agent_id:
                        return candidate

            for candidates in role_members.values():
                for candidate in candidates:
                    if candidate != agent_id:
                        return candidate

            for candidate in agent_roles.keys():
                if candidate != agent_id:
                    return candidate
            return None

        return resolver

    def _merge_mode_params(self, mode: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        params = dict(params or {})

        if mode == "gaplock":
            merged: Dict[str, Any] = {
                key: self.config[key]
                for key in GAPLOCK_PARAM_KEYS
                if key in self.config
            }
            for key, value in params.items():
                if key in GAPLOCK_PARAM_KEYS:
                    merged[key] = value
            return merged

        if mode == "progress":
            merged = {}
            default = self.config.get("progress", {})
            if isinstance(default, dict):
                for key, value in default.items():
                    if key in PROGRESS_PARAM_KEYS:
                        merged[key] = value
            for key, value in self.config.items():
                if key in PROGRESS_PARAM_KEYS and key not in merged:
                    merged[key] = value
            for key, value in params.items():
                if key in PROGRESS_PARAM_KEYS:
                    merged[key] = value
            return merged

        if mode == "fastest_lap":
            merged = {}
            default = self.config.get("fastest_lap", {})
            if isinstance(default, dict):
                for key, value in default.items():
                    if key in FASTEST_LAP_PARAM_KEYS:
                        merged[key] = value
            for key, value in self.config.items():
                if key in FASTEST_LAP_PARAM_KEYS and key not in merged:
                    merged[key] = value
            for key, value in params.items():
                if key in FASTEST_LAP_PARAM_KEYS:
                    merged[key] = value
            return merged

        # Fallback: return params filtered to recognised keys (no defaults known)
        return params

    def _instantiate_strategy(self, mode: str, params: Dict[str, Any]) -> RewardStrategy:
        if mode == "gaplock":
            effective = dict(params)
            if "target_resolver" not in effective and self._gaplock_target_resolver is not None:
                effective["target_resolver"] = self._gaplock_target_resolver
            return GaplockRewardStrategy(**effective)
        if mode == "progress":
            centerline = getattr(self.context.map_data, "centerline", None)
            return ProgressRewardStrategy(centerline=centerline, **params)
        if mode == "fastest_lap":
            return FastestLapRewardStrategy(**params)
        raise ValueError(f"Unknown reward mode '{mode}'")

    def _build_strategies(self) -> List[Tuple[RewardStrategy, float]]:
        strategies: List[Tuple[RewardStrategy, float]] = []

        if self.mode == "composite":
            components_cfg = self.config.get("components", {})
            if not isinstance(components_cfg, dict) or not components_cfg:
                raise ValueError("Composite reward mode requires a non-empty 'components' mapping")
            for name, definition in components_cfg.items():
                if isinstance(definition, dict):
                    sub_mode = self._normalize_mode(str(definition.get("mode", name)))
                    weight = float(definition.get("weight", 1.0))
                    params = definition.get("params", {})
                else:
                    sub_mode = self._normalize_mode(str(name))
                    weight = 1.0
                    params = {}
                merged_params = self._merge_mode_params(sub_mode, params)
                strategy = self._instantiate_strategy(sub_mode, merged_params)
                if weight != 0.0:
                    strategies.append((strategy, weight))
        else:
            merged_params = self._merge_mode_params(self.mode, self.config)
            strategy = self._instantiate_strategy(self.mode, merged_params)
            strategies.append((strategy, 1.0))

        if not strategies:
            raise ValueError("At least one reward strategy must be configured")
        return strategies

    # Public API ------------------------------------------------------------
    def reset(self, episode_index: int = 0) -> None:
        self._episode_index = episode_index
        self._step_counter = 0
        for strategy, _ in self._strategies:
            strategy.reset(episode_index)
        self._last_components.clear()

    def __call__(
        self,
        obs: Dict[str, Dict[str, Any]],
        agent_id: str,
        reward: float,
        *,
        done: bool,
        info: Optional[Dict[str, Any]],
        all_obs: Optional[Dict[str, Dict[str, Any]]] = None,
        step_index: Optional[int] = None,
    ) -> float:
        agent_obs = obs.get(agent_id)
        if agent_obs is None:
            return float(reward)

        timestep = float(getattr(self.context.env, "timestep", 0.01))
        current_time = float(getattr(self.context.env, "current_time", 0.0))

        effective_step = step_index if step_index is not None else self._step_counter
        self._step_counter = effective_step + 1

        step = RewardStep(
            agent_id=agent_id,
            obs=agent_obs,
            env_reward=float(reward),
            done=bool(done),
            info=info,
            all_obs=all_obs if isinstance(all_obs, dict) else None,
            episode_index=self._episode_index,
            step_index=effective_step,
            current_time=current_time,
            timestep=timestep,
        )

        total_reward = 0.0
        components: Dict[str, float] = {}
        multi_strategy = len(self._strategies) > 1

        for strategy, weight in self._strategies:
            base_reward, strat_components = strategy.compute(step)
            weighted_reward = base_reward * weight
            total_reward += weighted_reward

            if strat_components:
                for name, value in strat_components.items():
                    effective_value = value * weight
                    key = f"{strategy.name}/{name}" if multi_strategy else name
                    components[key] = components.get(key, 0.0) + effective_value

        if self.ego_collision_penalty and bool(agent_obs.get("collision", False)):
            total_reward += self.ego_collision_penalty
            components["ego_collision_penalty"] = (
                components.get("ego_collision_penalty", 0.0) + self.ego_collision_penalty
            )

        components["total"] = total_reward
        self._last_components[agent_id] = components
        return total_reward

    def get_last_components(self, agent_id: str) -> Dict[str, float]:
        return dict(self._last_components.get(agent_id, {}))


__all__ = ["RewardWrapper", "RewardRuntimeContext"]
