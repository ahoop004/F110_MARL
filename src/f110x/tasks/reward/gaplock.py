"""Gaplock task reward strategy."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from f110x.utils.reward_utils import ScalingParams, apply_reward_scaling

from .base import RewardRuntimeContext, RewardStep, RewardStrategy
from .components import RewardAccumulator, apply_relative_sector_reward
from .registry import RewardTaskConfig, RewardTaskRegistry, RewardTaskSpec, register_reward_task


GAPLOCK_PARAM_KEYS = (
    "target_crash_reward",
    "self_collision_penalty",
    "truncation_penalty",
    "success_once",
    "reward_horizon",
    "reward_clip",
    "reward_weight",
    "reward_decay",
    "reward_smoothing",
)


class GaplockRewardStrategy(RewardStrategy):
    name = "gaplock"

    def __init__(
        self,
        *,
        target_crash_reward: float = 10.0,
        self_collision_penalty: float = -10.0,
        truncation_penalty: float = 0.0,
        success_once: bool = True,
        reward_horizon: Optional[float] = None,
        reward_clip: Optional[float] = None,
        reward_weight: Optional[float] = None,
        reward_decay: Optional[float] = None,
        reward_smoothing: Optional[float] = None,
        relative_reward: Optional[Dict[str, Any]] = None,
        target_resolver: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.target_crash_reward = float(target_crash_reward)
        self.self_collision_penalty = float(self_collision_penalty)
        self.truncation_penalty = float(truncation_penalty)
        self.success_once = bool(success_once)
        self.scaling_params = ScalingParams(
            horizon=self._coerce_positive_float(reward_horizon),
            clip=self._coerce_positive_float(reward_clip),
            weight=self._coerce_positive_float(reward_weight),
            decay=self._coerce_positive_float(reward_decay),
            smoothing=self._coerce_positive_float(reward_smoothing),
        )
        self._success_awarded: Dict[str, set[Tuple[str, str]]] = {}
        self._target_resolver = target_resolver
        self.relative_reward_cfg = self._prepare_relative_reward(relative_reward)

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
        acc = RewardAccumulator()

        env_reward = float(step.env_reward)
        if env_reward:
            acc.add("env_reward", env_reward)

        ego_obs = step.obs
        target_obs, explicit_target_id = self._select_target_obs(step)
        ego_crashed = bool(ego_obs.get("collision", False))

        if ego_crashed and self.self_collision_penalty:
            acc.add("self_collision_penalty", self.self_collision_penalty)

        if target_obs is not None:
            target_crashed = bool(target_obs.get("collision", False))
            if target_crashed and not ego_crashed:
                target_id = explicit_target_id or str(target_obs.get("agent_id", "target"))
                if not self.success_once or not self._has_awarded(step.agent_id, target_id):
                    acc.add("success_reward", self.target_crash_reward)

        truncated = bool(step.events.get("truncated")) if step.events else False
        if not truncated and isinstance(step.info, dict):
            truncated = bool(step.info.get("truncated", False))
        if truncated and self.truncation_penalty:
            acc.add("truncation_penalty", self.truncation_penalty)

        if self.relative_reward_cfg and target_obs is not None:
            self._apply_relative_reward(acc, ego_obs, target_obs)

        total, components = apply_reward_scaling(acc.total, acc.components, self.scaling_params)
        return total, components

    def _prepare_relative_reward(self, cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cfg:
            return None
        weights_raw = cfg.get("weights", {})
        if not isinstance(weights_raw, Dict):
            weights_raw = {}
        weights = {str(key).lower(): float(value) for key, value in weights_raw.items()}
        if not any(abs(v) > 0.0 for v in weights.values()):
            return None
        return {
            "weights": weights,
            "preferred_radius": float(cfg.get("preferred_radius", 0.0)),
            "inner_tolerance": float(cfg.get("inner_tolerance", 0.0)),
            "outer_tolerance": float(cfg.get("outer_tolerance", 0.0)),
            "falloff": str(cfg.get("falloff", "linear")),
            "scale": float(cfg.get("scale", 1.0)),
        }

    def _apply_relative_reward(
        self,
        acc: RewardAccumulator,
        ego_obs: Dict[str, Any],
        target_obs: Dict[str, Any],
    ) -> None:
        cfg = self.relative_reward_cfg
        if not cfg:
            return
        ego_pose = ego_obs.get("pose")
        target_pose = target_obs.get("pose")
        if ego_pose is None or target_pose is None:
            return
        ego_pose = np.asarray(ego_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)
        if ego_pose.size < 3 or target_pose.size < 2:
            return
        relative_vector = target_pose[:2] - ego_pose[:2]
        ego_heading = float(ego_pose[2])
        apply_relative_sector_reward(
            acc,
            relative_vector=relative_vector,
            ego_heading=ego_heading,
            weights=cfg["weights"],
            preferred_radius=cfg["preferred_radius"],
            inner_tolerance=cfg["inner_tolerance"],
            outer_tolerance=cfg["outer_tolerance"],
            falloff=cfg["falloff"],
            scale=cfg["scale"],
        )


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


def _build_gaplock_target_resolver(context: RewardRuntimeContext) -> Optional[Callable[[str], Optional[str]]]:
    roster = getattr(context, "roster", None)
    role_members = _normalise_role_members(roster)
    agent_roles = _extract_agent_roles(roster)

    if not role_members and not agent_roles:
        return None

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


def _build_gaplock_strategy(
    context: RewardRuntimeContext,
    config: RewardTaskConfig,
    registry: RewardTaskRegistry,
) -> RewardStrategy:
    params = dict(config.get("params", {}))
    if "target_resolver" not in params:
        resolver = _build_gaplock_target_resolver(context)
        if resolver is not None:
            params["target_resolver"] = resolver
    return GaplockRewardStrategy(**params)


register_reward_task(
    RewardTaskSpec(
        name="gaplock",
        factory=_build_gaplock_strategy,
        aliases=("basic", "sparse", "pursuit", "adversarial"),
        param_keys=GAPLOCK_PARAM_KEYS,
    )
)


__all__ = ["GaplockRewardStrategy", "GAPLOCK_PARAM_KEYS"]
