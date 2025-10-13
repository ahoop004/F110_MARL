"""Shared rollout helper functions for training and evaluation runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from f110x.engine.reward import build_reward_wrapper
from f110x.runner.context import RunnerContext
from f110x.utils.builders import AgentTeam
from f110x.utils.start_pose import reset_with_start_poses


Actions = Dict[str, Any]
ProcessedObservations = Dict[str, np.ndarray]
ControllerInfos = Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class RolloutHooks:
    """Container bundling closures used during rollout execution."""

    reward_factory: Callable[[int], Any]
    compute_actions: Callable[[Dict[str, Any], Dict[str, bool]], Tuple[Actions, ProcessedObservations, ControllerInfos]]
    prepare_next_observation: Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    reset_fn: Callable[[], Any]


def build_rollout_hooks(
    context: RunnerContext,
    team: AgentTeam,
    env: Any,
    *,
    deterministic: bool,
) -> RolloutHooks:
    """Create reusable rollout closures shared by train/eval runners."""

    reward_cfg = context.reward_cfg
    map_data = context.map_data
    curriculum = context.curriculum_schedule
    roster = team.roster

    def reward_factory(ep_index: int):
        return build_reward_wrapper(
            reward_cfg,
            env=env,
            map_data=map_data,
            episode_idx=ep_index,
            curriculum=curriculum,
            roster=roster,
        )

    def compute_actions(obs: Dict[str, Any], done: Dict[str, bool]):
        return select_actions(team, obs, done, deterministic=deterministic)

    def prepare_next(agent_id: str, next_obs: Dict[str, Any], infos: Dict[str, Any]):
        return ensure_agent_observation(agent_id, next_obs, infos)

    def reset_env():
        return reset_with_start_poses(
            env,
            context.start_pose_options,
            back_gap=context.start_pose_back_gap,
            min_spacing=context.start_pose_min_spacing,
            map_data=context.map_data,
        )

    return RolloutHooks(
        reward_factory=reward_factory,
        compute_actions=compute_actions,
        prepare_next_observation=prepare_next,
        reset_fn=reset_env,
    )


def select_actions(
    team: AgentTeam,
    obs: Dict[str, Any],
    done: Dict[str, bool],
    *,
    deterministic: bool,
) -> Tuple[Actions, ProcessedObservations, ControllerInfos]:
    """Map observations to controller actions, mirroring legacy runner logic."""

    actions: Actions = {}
    processed_obs: ProcessedObservations = {}
    controller_infos: ControllerInfos = {}

    for bundle in team.agents:
        agent_id = bundle.agent_id
        if done.get(agent_id, False):
            continue
        if agent_id not in obs:
            continue

        controller = bundle.controller
        if bundle.trainer is not None:
            observation = team.observation(agent_id, obs)
            processed_obs[agent_id] = np.asarray(observation, dtype=np.float32)
            action_raw = bundle.trainer.select_action(observation, deterministic=deterministic)
            action_value, wrapper_meta = team.action(agent_id, action_raw, return_info=True)
            actions[agent_id] = action_value

            meta_index: Optional[int] = None
            if wrapper_meta and "action_index" in wrapper_meta:
                try:
                    meta_index = int(wrapper_meta["action_index"])
                except (TypeError, ValueError):
                    meta_index = None
            elif np.isscalar(action_raw) or (isinstance(action_raw, np.ndarray) and action_raw.ndim == 0):
                try:
                    meta_index = int(np.asarray(action_raw).item())
                except (TypeError, ValueError):
                    meta_index = None

            if meta_index is not None:
                controller_infos[agent_id] = {"action_index": meta_index}
        elif hasattr(controller, "get_action"):
            action_space = team.env.action_space(agent_id)
            action = controller.get_action(action_space, obs[agent_id])
            actions[agent_id] = team.action(agent_id, action)
        elif hasattr(controller, "act"):
            transformed = team.observation(agent_id, obs)
            action = controller.act(transformed, agent_id)
            actions[agent_id] = team.action(agent_id, action)
        else:
            raise TypeError(
                f"Controller for agent '{agent_id}' does not expose a supported act/get_action interface"
            )

    return actions, processed_obs, controller_infos


def ensure_agent_observation(
    agent_id: str,
    obs: Dict[str, Any],
    infos: Dict[str, Any],
) -> Dict[str, Any]:
    """Ensure terminal observations propagate when the environment omits them."""

    if agent_id in obs:
        return obs
    terminal = infos.get(agent_id, {}).get("terminal_observation")
    if terminal is not None:
        patched = dict(obs)
        patched[agent_id] = terminal
        return patched
    return obs


__all__ = ["RolloutHooks", "build_rollout_hooks", "ensure_agent_observation", "select_actions"]
