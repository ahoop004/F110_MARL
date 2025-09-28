"""Evaluation entrypoint using the shared env/agent builders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from f110x.envs import F110ParallelEnv
from f110x.utils.builders import AgentBundle, AgentTeam, build_agents, build_env
from f110x.utils.config_models import ExperimentConfig
from f110x.utils.map_loader import MapData
from f110x.utils.start_pose import reset_with_start_poses
from f110x.wrappers.reward import RewardWrapper


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@dataclass
class EvaluationContext:
    cfg: ExperimentConfig
    env: F110ParallelEnv
    map_data: MapData
    start_pose_options: Optional[List[np.ndarray]]
    team: AgentTeam
    ppo_bundle: AgentBundle
    reward_cfg: Dict[str, Any]
    start_pose_back_gap: float
    start_pose_min_spacing: float
    checkpoint_path: Optional[Path]

    @property
    def ppo_agent(self):
        return self.ppo_bundle.controller

    @property
    def ppo_agent_id(self) -> str:
        return self.ppo_bundle.agent_id

    @property
    def opponent_ids(self) -> List[str]:
        return [bundle.agent_id for bundle in self.team.agents if bundle is not self.ppo_bundle]


def _load_checkpoint(bundle: AgentBundle, checkpoint_path: Optional[Path]) -> None:
    if checkpoint_path is None:
        return
    if checkpoint_path.exists():
        bundle.controller.load(str(checkpoint_path))
        print(f"[INFO] Loaded PPO checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No PPO checkpoint found at {checkpoint_path}")


def create_evaluation_context(cfg_path: Path | None = None) -> EvaluationContext:
    cfg_file = cfg_path or DEFAULT_CONFIG_PATH
    cfg = ExperimentConfig.load(cfg_file)

    env, map_data, start_pose_options = build_env(cfg)
    team = build_agents(env, cfg)

    ppo_bundles = [bundle for bundle in team.agents if bundle.algo.lower() == "ppo"]
    if not ppo_bundles:
        raise RuntimeError("Evaluation expects a PPO agent in the roster")
    if len(ppo_bundles) > 1:
        raise RuntimeError("Evaluation currently supports a single PPO learner")
    ppo_bundle = ppo_bundles[0]

    reward_cfg = cfg.reward.to_dict()

    ppo_cfg = ppo_bundle.metadata.get("config", cfg.ppo.to_dict())
    checkpoint_dir = Path(ppo_cfg.get("save_dir", "checkpoints")).expanduser()
    checkpoint_name = ppo_cfg.get("checkpoint_name", "ppo_best.pt")
    default_checkpoint = checkpoint_dir / checkpoint_name
    explicit_checkpoint = cfg.main.checkpoint
    checkpoint_path = None

    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint).expanduser()
    elif default_checkpoint.exists():
        checkpoint_path = default_checkpoint

    _load_checkpoint(ppo_bundle, checkpoint_path)

    start_pose_back_gap = float(cfg.env.get("start_pose_back_gap", 0.0) or 0.0)
    start_pose_min_spacing = float(cfg.env.get("start_pose_min_spacing", 0.0) or 0.0)

    return EvaluationContext(
        cfg=cfg,
        env=env,
        map_data=map_data,
        start_pose_options=start_pose_options,
        team=team,
        ppo_bundle=ppo_bundle,
        reward_cfg=reward_cfg,
        start_pose_back_gap=start_pose_back_gap,
        start_pose_min_spacing=start_pose_min_spacing,
        checkpoint_path=checkpoint_path,
    )


def _collect_actions(
    ctx: EvaluationContext,
    obs: Dict[str, Any],
    done: Dict[str, bool],
) -> Dict[str, Any]:
    actions: Dict[str, Any] = {}
    for bundle in ctx.team.agents:
        aid = bundle.agent_id
        if done.get(aid, False):
            continue
        if aid not in obs:
            continue

        controller = bundle.controller
        if bundle.algo.lower() == "ppo":
            processed = ctx.team.observation(aid, obs)
            if hasattr(controller, "act_deterministic"):
                actions[aid] = controller.act_deterministic(processed, aid)
            else:
                actions[aid] = controller.act(processed, aid)
        elif hasattr(controller, "get_action"):
            actions[aid] = controller.get_action(ctx.env.action_space(aid), obs[aid])
        elif hasattr(controller, "act"):
            processed = ctx.team.observation(aid, obs)
            actions[aid] = controller.act(processed, aid)
        else:
            raise TypeError(f"Controller for agent '{aid}' does not expose an act/get_action method")
    return actions


def evaluate(ctx: EvaluationContext, episodes: int = 20) -> List[Dict[str, Any]]:
    env = ctx.env
    ppo_id = ctx.ppo_agent_id
    render_enabled = ctx.cfg.env.get("render_mode", "") == "human"

    results: List[Dict[str, Any]] = []

    for ep in range(episodes):
        obs, infos = reset_with_start_poses(
            env,
            ctx.start_pose_options,
            back_gap=ctx.start_pose_back_gap,
            min_spacing=ctx.start_pose_min_spacing,
            map_data=ctx.map_data,
        )

        done = {aid: False for aid in env.possible_agents}
        totals = {aid: 0.0 for aid in env.possible_agents}
        reward_wrapper = RewardWrapper(**ctx.reward_cfg)
        reward_wrapper.reset()

        steps = 0
        terms: Dict[str, bool] = {}
        truncs: Dict[str, bool] = {}
        collision_history: List[str] = []
        live_render = render_enabled

        while True:
            actions = _collect_actions(ctx, obs, done)
            if not actions:
                break

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            steps += 1

            for aid in totals:
                reward = rewards.get(aid, 0.0)
                if next_obs.get(aid) is not None:
                    reward = reward_wrapper(
                        next_obs,
                        aid,
                        rewards.get(aid, 0.0),
                        done=terms.get(aid, False) or truncs.get(aid, False),
                        info=infos.get(aid, {}),
                        all_obs=next_obs,
                    )
                totals[aid] += reward

            collision_agents = [
                aid for aid in next_obs.keys() if bool(next_obs.get(aid, {}).get("collision", False))
            ]
            if collision_agents:
                collision_history.extend(collision_agents)

            obs = next_obs
            done = {
                aid: terms.get(aid, False)
                or truncs.get(aid, False)
                or (aid in collision_history)
                for aid in done
            }

            if live_render:
                try:
                    env.render()
                except Exception as exc:
                    print(f"[WARN] Disabling render during eval: {exc}")
                    live_render = False

            if all(done.values()):
                break

        cause_bits: List[str] = []
        if collision_history:
            cause_bits.append("collision")
        for aid in done:
            if terms.get(aid, False):
                cause_bits.append(f"term:{aid}")
            if truncs.get(aid, False):
                cause_bits.append(f"trunc:{aid}")
        if not cause_bits:
            cause_bits.append("unknown")
        cause_str = ",".join(cause_bits)

        print(
            f"[EVAL {ep + 1:03d}/{episodes}] steps={steps} cause={cause_str} "
            f"return_{ppo_id}={totals.get(ppo_id, 0.0):.2f}"
        )

        record = {
            "episode": ep + 1,
            "steps": steps,
            "cause": cause_str,
            "returns": dict(totals),
        }
        for aid, value in totals.items():
            record[f"return_{aid}"] = value
        results.append(record)

    return results


# Instantiate default context for legacy imports
CTX = create_evaluation_context()
cfg = CTX.cfg
env = CTX.env
team = CTX.team
ppo_agent = CTX.ppo_agent
PPO_AGENT = CTX.ppo_agent_id
_opponents = CTX.opponent_ids
GAP_AGENT = _opponents[0] if _opponents else None


if __name__ == "__main__":
    episodes = int(CTX.ppo_bundle.metadata.get("config", {}).get("eval_episodes", 20))
    eval_results = evaluate(CTX, episodes=episodes)
    print("Evaluation finished. Results:")
    for record in eval_results:
        print(record)
