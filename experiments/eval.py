"""Evaluation entrypoint using the shared env/agent builders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pickle

from f110x.envs import F110ParallelEnv
from f110x.utils.builders import AgentBundle, AgentTeam, build_agents, build_env
from f110x.utils.config_models import ExperimentConfig
from f110x.utils.map_loader import MapData
from f110x.utils.start_pose import reset_with_start_poses
from f110x.wrappers.reward import RewardWrapper


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _serialize_obs(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    serial: Dict[str, Any] = {}
    for key, value in obs_dict.items():
        if isinstance(value, dict):
            serial[key] = {sub_k: _to_serializable(sub_v) for sub_k, sub_v in value.items()}
        else:
            serial[key] = _to_serializable(value)
    return serial


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
    save_rollouts: bool = False
    rollout_dir: Optional[Path] = None

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
        print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[WARN] No checkpoint found at {checkpoint_path}; starting from scratch")


def create_evaluation_context(cfg_path: Path | None = None) -> EvaluationContext:
    cfg_file = cfg_path or DEFAULT_CONFIG_PATH
    cfg = ExperimentConfig.load(cfg_file)

    env, map_data, start_pose_options = build_env(cfg)
    team = build_agents(env, cfg)

    ppo_bundles = [bundle for bundle in team.agents if bundle.algo.lower() == "ppo"]
    if ppo_bundles:
        primary_bundle = ppo_bundles[0]
    else:
        trainable = [bundle for bundle in team.agents if bundle.trainer is not None]
        if not trainable:
            raise RuntimeError("Evaluation expects at least one trainer-enabled agent in the roster")
        primary_bundle = trainable[0]

    reward_cfg = cfg.reward.to_dict()

    bundle_cfg = primary_bundle.metadata.get("config", {})
    if (not bundle_cfg) and primary_bundle.algo.lower() == "ppo":
        bundle_cfg = cfg.ppo.to_dict()

    checkpoint_dir = Path(bundle_cfg.get("save_dir", "checkpoints")).expanduser()
    checkpoint_name = bundle_cfg.get("checkpoint_name", f"{primary_bundle.algo.lower()}_best.pt")
    default_checkpoint = checkpoint_dir / checkpoint_name
    explicit_checkpoint = cfg.main.checkpoint
    checkpoint_path = None

    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint).expanduser()
    elif default_checkpoint.exists():
        checkpoint_path = default_checkpoint

    _load_checkpoint(primary_bundle, checkpoint_path)

    start_pose_back_gap = float(cfg.env.get("start_pose_back_gap", 0.0) or 0.0)
    start_pose_min_spacing = float(cfg.env.get("start_pose_min_spacing", 0.0) or 0.0)
    save_rollouts = bool(cfg.main.get("save_eval_rollouts", False))
    rollout_dir: Optional[Path] = None
    if save_rollouts:
        rollout_dir = Path(cfg.main.get("eval_rollout_dir", "eval_rollouts")).expanduser()
        rollout_dir.mkdir(parents=True, exist_ok=True)

    return EvaluationContext(
        cfg=cfg,
        env=env,
        map_data=map_data,
        start_pose_options=start_pose_options,
        team=team,
        ppo_bundle=primary_bundle,
        reward_cfg=reward_cfg,
        start_pose_back_gap=start_pose_back_gap,
        start_pose_min_spacing=start_pose_min_spacing,
        checkpoint_path=checkpoint_path,
        save_rollouts=save_rollouts,
        rollout_dir=rollout_dir,
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
        if bundle.trainer is not None:
            processed = ctx.team.observation(aid, obs)
            action_raw = bundle.trainer.select_action(processed, deterministic=True)
            actions[aid] = ctx.team.action(aid, action_raw)
        elif hasattr(controller, "get_action"):
            action = controller.get_action(ctx.env.action_space(aid), obs[aid])
            actions[aid] = ctx.team.action(aid, action)
        elif hasattr(controller, "act"):
            processed = ctx.team.observation(aid, obs)
            action = controller.act(processed, aid)
            actions[aid] = ctx.team.action(aid, action)
        else:
            raise TypeError(f"Controller for agent '{aid}' does not expose an act/get_action method")
    return actions


def evaluate(ctx: EvaluationContext, episodes: int = 20, force_render: bool = False) -> List[Dict[str, Any]]:
    env = ctx.env
    ppo_id = ctx.ppo_agent_id
    if force_render:
        ctx.env.render_mode = "human"
    render_enabled = force_render or ctx.cfg.env.get("render_mode", "") == "human"

    results: List[Dict[str, Any]] = []
    attacker_id = ctx.team.roles.get("attacker", ctx.ppo_bundle.agent_id)
    defender_id = ctx.team.roles.get("defender")

    for ep in range(episodes):
        obs, infos = reset_with_start_poses(
            env,
            ctx.start_pose_options,
            back_gap=ctx.start_pose_back_gap,
            min_spacing=ctx.start_pose_min_spacing,
            map_data=ctx.map_data,
        )
        ctx.team.reset_actions()

        done = {aid: False for aid in env.possible_agents}
        totals = {aid: 0.0 for aid in env.possible_agents}
        reward_wrapper = RewardWrapper(**ctx.reward_cfg)
        reward_wrapper.reset()

        steps = 0
        terms: Dict[str, bool] = {}
        truncs: Dict[str, bool] = {}
        collision_history: List[str] = []
        live_render = render_enabled
        collision_counts: Dict[str, int] = {aid: 0 for aid in env.possible_agents}
        rollout_trace: Optional[List[Dict[str, Any]]] = [] if ctx.save_rollouts and ctx.rollout_dir else None
        collision_step: Dict[str, Optional[int]] = {aid: None for aid in env.possible_agents}

        while True:
            actions = _collect_actions(ctx, obs, done)
            if not actions:
                break

            step_record: Optional[Dict[str, Any]] = None
            if rollout_trace is not None:
                step_record = {
                    "step": steps,
                    "obs": {aid: _serialize_obs(obs_data) for aid, obs_data in obs.items()},
                    "actions": {aid: _to_serializable(action) for aid, action in actions.items()},
                }

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
                for aid in collision_agents:
                    collision_counts[aid] = collision_counts.get(aid, 0) + 1
                    if collision_step[aid] is None:
                        collision_step[aid] = steps

            obs = next_obs
            done = {
                aid: terms.get(aid, False)
                or truncs.get(aid, False)
                or (aid in collision_history)
                for aid in done
            }

            if step_record is not None:
                step_record["rewards"] = {aid: float(rewards.get(aid, 0.0)) for aid in totals}
                step_record["next_obs"] = {aid: _serialize_obs(next_obs_data) for aid, next_obs_data in next_obs.items()}
                step_record["done"] = {aid: done.get(aid, False) for aid in done}
                step_record["collisions"] = list(collision_agents)
                rollout_trace.append(step_record)

            if live_render:
                try:
                    env.render()
                except Exception as exc:
                    print(f"[WARN] Disabling render during eval: {exc}")
                    live_render = False

            if all(done.values()):
                break

        lap_counts: Dict[str, float] = {}
        if hasattr(env, "lap_counts") and hasattr(env, "_agent_id_to_index"):
            for aid in env.possible_agents:
                idx = getattr(env, "_agent_id_to_index", {}).get(aid)
                if idx is not None and idx < len(getattr(env, "lap_counts", [])):
                    lap_counts[aid] = float(env.lap_counts[idx])

        collision_total = sum(collision_counts.values())

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

        defender_crashed = bool(defender_id and collision_step.get(defender_id) is not None)
        attacker_crashed = bool(collision_step.get(attacker_id))
        defender_crash_step = collision_step.get(defender_id)
        attacker_crash_step = collision_step.get(attacker_id)

        print(
            f"[EVAL {ep + 1:03d}/{episodes}] steps={steps} cause={cause_str} collisions={collision_total} "
            f"defender_crash={defender_crashed} attacker_crash={attacker_crashed} "
            f"return_{ppo_id}={totals.get(ppo_id, 0.0):.2f}"
        )

        record = {
            "episode": ep + 1,
            "steps": steps,
            "cause": cause_str,
            "returns": dict(totals),
            "collision_total": collision_total,
        }
        for aid, value in totals.items():
            record[f"return_{aid}"] = value
        for aid, count in collision_counts.items():
            record[f"collision_count_{aid}"] = count
        for aid, count in lap_counts.items():
            record[f"lap_count_{aid}"] = count
        for aid, step_val in collision_step.items():
            if step_val is not None:
                record[f"collision_step_{aid}"] = step_val

        if defender_id:
            record["defender_crashed"] = defender_crashed
            if defender_crash_step is not None:
                record["defender_crash_step"] = defender_crash_step
                record["defender_survival_steps"] = defender_crash_step
            else:
                record["defender_survival_steps"] = steps
        record["attacker_crashed"] = attacker_crashed
        if attacker_crash_step is not None:
            record["attacker_crash_step"] = attacker_crash_step

        if rollout_trace is not None and ctx.rollout_dir is not None:
            rollout_path = ctx.rollout_dir / f"episode_{ep + 1:03d}.pkl"
            with rollout_path.open("wb") as handle:
                pickle.dump({"trajectory": rollout_trace, "metrics": record}, handle)
            record["rollout_path"] = str(rollout_path)

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
