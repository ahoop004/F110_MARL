"""Training entrypoint using the shared env/agent builders."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from f110x.envs import F110ParallelEnv
from f110x.utils.builders import AgentBundle, AgentTeam, build_agents, build_env
from f110x.utils.config_models import ExperimentConfig
from f110x.utils.map_loader import MapData
from f110x.utils.start_pose import reset_with_start_poses
from f110x.wrappers.reward import RewardWrapper
from f110x.trainers.base import Transition, Trainer


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@dataclass
class TrainingContext:
    cfg: ExperimentConfig
    env: F110ParallelEnv
    map_data: MapData
    start_pose_options: Optional[List[np.ndarray]]
    team: AgentTeam
    ppo_bundle: AgentBundle
    ppo_trainer: Trainer
    trainer_map: Dict[str, Trainer]
    reward_cfg: Dict[str, Any]
    curriculum_schedule: List[Tuple[Optional[int], str]]
    render_interval: int
    update_after: int
    start_pose_back_gap: float
    start_pose_min_spacing: float
    checkpoint_path: Path
    best_path: Path

    @property
    def ppo_agent(self):
        return self.ppo_bundle.controller

    @property
    def ppo_agent_id(self) -> str:
        return self.ppo_bundle.agent_id

    @property
    def opponent_ids(self) -> List[str]:
        return [bundle.agent_id for bundle in self.team.agents if bundle is not self.ppo_bundle]

def _build_curriculum_schedule(raw_curriculum: Iterable[Dict[str, Any]]) -> List[Tuple[Optional[int], str]]:
    schedule: List[Tuple[Optional[int], str]] = []
    for stage in raw_curriculum:
        if not isinstance(stage, dict):
            continue
        mode = stage.get("mode")
        if mode is None:
            continue
        upper = stage.get("until", stage.get("episodes"))
        if upper is not None:
            try:
                upper = int(upper)
            except (TypeError, ValueError):
                upper = None
        schedule.append((upper, str(mode)))
    schedule.sort(key=lambda item: float("inf") if item[0] is None else item[0])
    return schedule


def _resolve_reward_mode(curriculum: List[Tuple[Optional[int], str]], episode_idx: int) -> str:
    if curriculum:
        for threshold, mode in curriculum:
            if threshold is None or episode_idx < threshold:
                return mode
        return curriculum[-1][1]

    if episode_idx < 1000:
        return "basic"
    if episode_idx < 2000:
        return "pursuit"
    return "adversarial"


def create_training_context(cfg_path: Path | None = None) -> TrainingContext:
    cfg_file = cfg_path or DEFAULT_CONFIG_PATH
    cfg = ExperimentConfig.load(cfg_file)

    env, map_data, start_pose_options = build_env(cfg)
    team = build_agents(env, cfg)

    trainer_map: Dict[str, Trainer] = {
        bundle.agent_id: bundle.trainer
        for bundle in team.agents
        if bundle.trainer is not None
    }

    ppo_bundles = [bundle for bundle in team.agents if bundle.algo.lower() == "ppo"]
    if not ppo_bundles:
        raise RuntimeError("Training expects at least one PPO agent in the roster")
    if len(ppo_bundles) > 1:
        raise RuntimeError("Training currently supports a single PPO learner")
    ppo_bundle = ppo_bundles[0]
    ppo_trainer = ppo_bundle.trainer
    if ppo_trainer is None:
        raise RuntimeError("PPO bundle is missing trainer adapter; check builder configuration")

    reward_cfg = cfg.reward.to_dict()
    raw_curriculum = cfg.get("reward_curriculum", [])
    if not isinstance(raw_curriculum, Iterable):
        raw_curriculum = []
    curriculum_schedule = _build_curriculum_schedule(raw_curriculum)

    render_interval = int(cfg.env.get("render_interval", 0) or 0)
    update_after = int(cfg.env.get("update", 1) or 1)

    start_pose_back_gap = float(cfg.env.get("start_pose_back_gap", 0.0) or 0.0)
    start_pose_min_spacing = float(cfg.env.get("start_pose_min_spacing", 0.0) or 0.0)

    ppo_cfg = ppo_bundle.metadata.get("config", cfg.ppo.to_dict())
    checkpoint_dir = Path(ppo_cfg.get("save_dir", "checkpoints")).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = ppo_cfg.get("checkpoint_name", "ppo_best.pt")
    best_path = checkpoint_dir / checkpoint_name
    load_override = ppo_cfg.get("load_path")
    checkpoint_path = Path(load_override).expanduser() if load_override else best_path

    if checkpoint_path.exists():
        try:
            ppo_bundle.controller.load(str(checkpoint_path))
            print(f"[INFO] Loaded PPO checkpoint from {checkpoint_path}")
        except Exception as exc:  # pragma: no cover - defensive load guard
            print(f"[WARN] Failed to load PPO checkpoint at {checkpoint_path}: {exc}")

    return TrainingContext(
        cfg=cfg,
        env=env,
        map_data=map_data,
        start_pose_options=start_pose_options,
        team=team,
        ppo_bundle=ppo_bundle,
        ppo_trainer=ppo_trainer,
        trainer_map=trainer_map,
        reward_cfg=reward_cfg,
        curriculum_schedule=curriculum_schedule,
        render_interval=render_interval,
        update_after=update_after,
        start_pose_back_gap=start_pose_back_gap,
        start_pose_min_spacing=start_pose_min_spacing,
        checkpoint_path=checkpoint_path,
        best_path=best_path,
    )


def _build_reward_wrapper(ctx: TrainingContext, episode_idx: int) -> RewardWrapper:
    mode = _resolve_reward_mode(ctx.curriculum_schedule, episode_idx)
    wrapper_cfg = dict(ctx.reward_cfg)
    wrapper_cfg["mode"] = mode
    wrapper = RewardWrapper(**wrapper_cfg)
    wrapper.reset()
    return wrapper


def _compute_actions(
    ctx: TrainingContext,
    obs: Dict[str, Any],
    done: Dict[str, bool],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    actions: Dict[str, Any] = {}
    processed_obs: Dict[str, np.ndarray] = {}

    for bundle in ctx.team.agents:
        aid = bundle.agent_id
        if done.get(aid, False):
            continue
        if aid not in obs:
            continue

        controller = bundle.controller
        if bundle.trainer is not None:
            obs_vector = ctx.team.observation(aid, obs)
            action = bundle.trainer.select_action(obs_vector, deterministic=False)
            actions[aid] = action
            processed_obs[aid] = np.asarray(obs_vector, dtype=np.float32)
        elif hasattr(controller, "get_action"):
            action_space = ctx.env.action_space(aid)
            actions[aid] = controller.get_action(action_space, obs[aid])
        elif hasattr(controller, "act"):
            processed = ctx.team.observation(aid, obs)
            actions[aid] = controller.act(processed, aid)
        else:
            raise TypeError(f"Controller for agent '{aid}' does not expose an act/get_action method")

    return actions, processed_obs


def _prepare_next_observation(
    ctx: TrainingContext,
    agent_id: str,
    obs: Dict[str, Any],
    infos: Dict[str, Any],
) -> Dict[str, Any]:
    if agent_id in obs:
        return obs

    terminal = infos.get(agent_id, {}).get("terminal_observation")
    if terminal is not None:
        patched = dict(obs)
        patched[agent_id] = terminal
        return patched

    return obs


def run_training(
    ctx: TrainingContext,
    episodes: int,
    *,
    update_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    update_start: int = 0,
) -> List[Dict[str, float]]:
    env = ctx.env
    ppo_agent = ctx.ppo_agent
    ppo_id = ctx.ppo_agent_id

    results: List[Dict[str, float]] = []
    recent_window = max(1, int(ctx.ppo_bundle.metadata.get("config", {}).get("rolling_avg_window", 10)))
    recent_returns: deque[float] = deque(maxlen=recent_window)
    best_return = float("-inf")
    update_count = update_start
    trainers = dict(ctx.trainer_map)

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
        reward_wrapper = _build_reward_wrapper(ctx, ep)

        steps = 0
        collision_history: List[str] = []
        terms: Dict[str, bool] = {}
        truncs: Dict[str, bool] = {}

        while True:
            actions, processed_obs = _compute_actions(ctx, obs, done)
            if not actions:
                break

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            steps += 1

            shaped_rewards: Dict[str, float] = {}
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
                shaped_rewards[aid] = reward

            collision_agents = [
                aid for aid in next_obs.keys()
                if bool(next_obs.get(aid, {}).get("collision", False))
            ]
            if collision_agents:
                collision_history.extend(collision_agents)

            for trainer_id, trainer in trainers.items():
                if trainer_id not in processed_obs:
                    continue

                terminated = terms.get(trainer_id, False) or (trainer_id in collision_agents)
                truncated = truncs.get(trainer_id, False)

                patched_next = _prepare_next_observation(ctx, trainer_id, next_obs, infos)
                try:
                    next_wrapped = ctx.team.observation(trainer_id, patched_next)
                except Exception:
                    next_wrapped = processed_obs[trainer_id]

                shaped = shaped_rewards.get(trainer_id, rewards.get(trainer_id, 0.0))
                transition = Transition(
                    agent_id=trainer_id,
                    obs=processed_obs[trainer_id],
                    action=actions.get(trainer_id),
                    reward=shaped,
                    next_obs=next_wrapped,
                    terminated=terminated,
                    truncated=truncated,
                    info=infos.get(trainer_id),
                    raw_obs=obs,
                    raw_next_obs=next_obs,
                )
                trainer.observe(transition)

            for aid in done:
                done_flag = terms.get(aid, False) or truncs.get(aid, False) or (aid in collision_history)
                done[aid] = done_flag

            obs = next_obs

            if collision_history or all(done.values()):
                break

            if ctx.render_interval and ((ep + 1) % ctx.render_interval == 0):
                try:
                    env.render()
                except Exception as exc:
                    print(f"[WARN] Rendering disabled due to: {exc}")
                    ctx.render_interval = 0

        results.append(totals)

        if (ep + 1) % ctx.update_after == 0:
            for trainer_id, trainer in trainers.items():
                stats = trainer.update()
                if update_callback and stats:
                    update_count += 1
                    payload: Dict[str, float] = {"train/update": update_count}
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            payload[f"train/{key}"] = float(value)
                    update_callback(payload)

        ppo_return = totals.get(ppo_id, 0.0)
        recent_returns.append(ppo_return)
        if len(recent_returns) == recent_returns.maxlen:
            avg_return = float(sum(recent_returns) / len(recent_returns))
            if avg_return > best_return:
                best_return = avg_return
                ppo_agent.save(str(ctx.best_path))
                print(
                    f"[INFO] New best model saved at episode {ep + 1} "
                    f"(avg_return={avg_return:.2f})"
                )

        end_cause = []
        if collision_history:
            end_cause.append("collision")
        for aid in done:
            if terms.get(aid, False):
                end_cause.append(f"term:{aid}")
            if truncs.get(aid, False):
                end_cause.append(f"trunc:{aid}")
        if not end_cause:
            end_cause.append("unknown")
        cause_str = ",".join(end_cause)

        print(
            f"[EP {ep + 1:03d}/{episodes}] mode={reward_wrapper.mode} "
            f"steps={steps} cause={cause_str} "
            f"return_{ppo_id}={totals.get(ppo_id, 0.0):.2f}"
        )

    for trainer_id, trainer in trainers.items():
        stats = trainer.update()
        if update_callback and stats:
            update_count += 1
            payload: Dict[str, float] = {"train/update": update_count}
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    payload[f"train/{key}"] = float(value)
            update_callback(payload)

    return results


# Instantiate a default context for callers expecting module-level objects
CTX = create_training_context()
cfg = CTX.cfg
env = CTX.env
team = CTX.team
ppo_agent = CTX.ppo_agent
ppo_trainer = CTX.ppo_trainer
PPO_AGENT = CTX.ppo_agent_id
_opponents = CTX.opponent_ids
GAP_AGENT = _opponents[0] if _opponents else None


def run_mixed(environment: F110ParallelEnv, episodes: int = 5):
    if environment is not env:
        raise ValueError("run_mixed currently operates on the default env context")
    return run_training(CTX, episodes)


if __name__ == "__main__":
    episodes = int(CTX.ppo_bundle.metadata.get("config", {}).get("train_episodes", 10))
    scores = run_training(CTX, episodes=episodes)
    print("Training completed. Episode returns:", scores)
    CTX.ppo_agent.save(str(CTX.best_path))
