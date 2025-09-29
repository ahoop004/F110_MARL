"""Training entrypoint using the shared env/agent builders."""
from __future__ import annotations

from collections import deque, defaultdict
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
    if ppo_bundles:
        primary_bundle = ppo_bundles[0]
    else:
        trainable_bundles = [bundle for bundle in team.agents if bundle.trainable and bundle.trainer is not None]
        if not trainable_bundles:
            raise RuntimeError("No trainable agent with a trainer adapter found in roster")
        primary_bundle = trainable_bundles[0]

    primary_trainer = primary_bundle.trainer
    if primary_trainer is None:
        raise RuntimeError(
            f"Primary bundle '{primary_bundle.agent_id}' is missing trainer adapter; check configuration"
        )

    reward_cfg = cfg.reward.to_dict()
    raw_curriculum = cfg.get("reward_curriculum", [])
    if not isinstance(raw_curriculum, Iterable):
        raw_curriculum = []
    curriculum_schedule = _build_curriculum_schedule(raw_curriculum)

    render_interval = int(cfg.env.get("render_interval", 0) or 0)
    update_after = int(cfg.env.get("update", 1) or 1)

    start_pose_back_gap = float(cfg.env.get("start_pose_back_gap", 0.0) or 0.0)
    start_pose_min_spacing = float(cfg.env.get("start_pose_min_spacing", 0.0) or 0.0)

    bundle_cfg = primary_bundle.metadata.get("config", {})
    if (not bundle_cfg) and primary_bundle.algo.lower() == "ppo":
        bundle_cfg = cfg.ppo.to_dict()

    checkpoint_dir = Path(bundle_cfg.get("save_dir", "checkpoints")).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"{primary_bundle.algo.lower()}_best.pt"
    checkpoint_name = bundle_cfg.get("checkpoint_name", default_name)
    best_path = checkpoint_dir / checkpoint_name
    load_override = bundle_cfg.get("load_path")
    checkpoint_path = Path(load_override).expanduser() if load_override else best_path

    if checkpoint_path.exists():
        try:
            primary_bundle.controller.load(str(checkpoint_path))
            print(f"[INFO] Loaded {primary_bundle.algo.upper()} checkpoint from {checkpoint_path}")
        except Exception as exc:  # pragma: no cover - defensive load guard
            print(f"[WARN] Failed to load checkpoint at {checkpoint_path}: {exc}")

    return TrainingContext(
        cfg=cfg,
        env=env,
        map_data=map_data,
        start_pose_options=start_pose_options,
        team=team,
        ppo_bundle=primary_bundle,
        ppo_trainer=primary_trainer,
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
            action_raw = bundle.trainer.select_action(obs_vector, deterministic=False)
            action = ctx.team.action(aid, action_raw)
            actions[aid] = action
            processed_obs[aid] = np.asarray(obs_vector, dtype=np.float32)
        elif hasattr(controller, "get_action"):
            action_space = ctx.env.action_space(aid)
            action = controller.get_action(action_space, obs[aid])
            actions[aid] = ctx.team.action(aid, action)
        elif hasattr(controller, "act"):
            processed = ctx.team.observation(aid, obs)
            action = controller.act(processed, aid)
            actions[aid] = ctx.team.action(aid, action)
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
    update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    update_start: int = 0,
) -> List[Dict[str, Any]]:
    env = ctx.env
    ppo_agent = ctx.ppo_agent
    ppo_id = ctx.ppo_agent_id

    results: List[Dict[str, Any]] = []
    recent_window = max(1, int(ctx.ppo_bundle.metadata.get("config", {}).get("rolling_avg_window", 10)))
    recent_returns: deque[float] = deque(maxlen=recent_window)
    best_return = float("-inf")
    update_count = update_start
    trainers = dict(ctx.trainer_map)
    roles = getattr(ctx.team, "roles", {})
    attacker_id = roles.get("attacker", ppo_id)
    defender_id = roles.get("defender")

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
        reward_wrapper = _build_reward_wrapper(ctx, ep)
        reward_breakdown: Dict[str, Dict[str, float]] = {
            aid: defaultdict(float) for aid in env.possible_agents
        }

        steps = 0
        collision_history: List[str] = []
        terms: Dict[str, bool] = {}
        truncs: Dict[str, bool] = {}
        collision_counts: Dict[str, int] = {aid: 0 for aid in env.possible_agents}
        collision_step: Dict[str, Optional[int]] = {aid: None for aid in env.possible_agents}
        speed_sums: Dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
        speed_counts: Dict[str, int] = {aid: 0 for aid in env.possible_agents}

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

                components = reward_wrapper.get_last_components(aid)
                if components:
                    for name, value in components.items():
                        reward_breakdown[aid][name] = reward_breakdown[aid].get(name, 0.0) + float(value)

            collision_agents = [
                aid for aid in next_obs.keys()
                if bool(next_obs.get(aid, {}).get("collision", False))
            ]
            if collision_agents:
                collision_history.extend(collision_agents)
                for aid in collision_agents:
                    collision_counts[aid] = collision_counts.get(aid, 0) + 1
                    if collision_step.get(aid) is None:
                        collision_step[aid] = steps

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

            for aid in env.possible_agents:
                velocity = next_obs.get(aid, {}).get("velocity")
                if velocity is None:
                    continue
                speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float32)))
                if not np.isnan(speed):
                    speed_sums[aid] = speed_sums.get(aid, 0.0) + speed
                    speed_counts[aid] = speed_counts.get(aid, 0) + 1

            if collision_history or all(done.values()):
                break

            if ctx.render_interval and ((ep + 1) % ctx.render_interval == 0):
                try:
                    env.render()
                except Exception as exc:
                    print(f"[WARN] Rendering disabled due to: {exc}")
                    ctx.render_interval = 0

        end_cause: List[str] = []
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

        returns = {aid: float(value) for aid, value in totals.items()}

        defender_crashed = bool(defender_id and collision_step.get(defender_id) is not None)
        attacker_crashed = bool(attacker_id and collision_step.get(attacker_id) is not None)
        success = bool(defender_crashed and not attacker_crashed)
        defender_survival_steps: Optional[int] = None
        if defender_id:
            defender_survival_steps = collision_step.get(defender_id) if defender_crashed else steps

        collisions_total = int(sum(collision_counts.values()))
        episode_record: Dict[str, Any] = {
            "episode": ep + 1,
            "steps": steps,
            "cause": cause_str,
            "reward_mode": reward_wrapper.mode,
            "returns": returns,
            "success": success,
            "defender_crashed": defender_crashed,
            "attacker_crashed": attacker_crashed,
            "defender_survival_steps": defender_survival_steps,
            "collisions_total": collisions_total,
        }

        for aid, count in collision_counts.items():
            episode_record[f"collision_count_{aid}"] = count
        for aid, step_val in collision_step.items():
            if step_val is not None:
                episode_record[f"collision_step_{aid}"] = step_val
        for aid in env.possible_agents:
            count = speed_counts.get(aid, 0)
            if count:
                episode_record[f"avg_speed_{aid}"] = float(speed_sums.get(aid, 0.0) / count)
            else:
                episode_record[f"avg_speed_{aid}"] = 0.0

        if attacker_id in speed_counts:
            count = speed_counts.get(attacker_id, 0)
            episode_record["avg_speed_attacker"] = (
                float(speed_sums.get(attacker_id, 0.0) / count) if count else 0.0
            )
        if defender_id and defender_id in speed_counts:
            count = speed_counts.get(defender_id, 0)
            episode_record["avg_speed_defender"] = (
                float(speed_sums.get(defender_id, 0.0) / count) if count else 0.0
            )

        for aid, components in reward_breakdown.items():
            for name, total in components.items():
                episode_record[f"reward_component_{aid}_{name}"] = float(total)

        if attacker_id in reward_breakdown:
            for name, total in reward_breakdown[attacker_id].items():
                episode_record[f"reward_component_attacker_{name}"] = float(total)
        if defender_id and defender_id in reward_breakdown:
            for name, total in reward_breakdown[defender_id].items():
                episode_record[f"reward_component_defender_{name}"] = float(total)

        results.append(episode_record)

        if update_callback:
            payload: Dict[str, Any] = {"train/episode": float(ep + 1)}
            for aid, value in returns.items():
                payload[f"train/return_{aid}"] = float(value)
            if ppo_id in returns:
                payload["train/return_attacker"] = float(returns[ppo_id])
            if defender_id and defender_id in returns:
                payload["train/return_defender"] = float(returns[defender_id])
            payload["train/steps"] = float(steps)
            payload["train/collisions_total"] = float(collisions_total)
            payload["train/success"] = float(int(success))
            payload["train/defender_crashed"] = float(int(defender_crashed))
            payload["train/attacker_crashed"] = float(int(attacker_crashed))
            if defender_survival_steps is not None:
                payload["train/defender_survival_steps"] = float(defender_survival_steps)

            focus_ids: List[str] = []
            if attacker_id:
                focus_ids.append(attacker_id)
            if defender_id and defender_id not in focus_ids:
                focus_ids.append(defender_id)
            for aid in focus_ids:
                components = reward_breakdown.get(aid)
                if not components:
                    continue
                for name, total in components.items():
                    payload[f"train/reward_component_{aid}_{name}"] = float(total)
            update_callback(payload)

        if (ep + 1) % ctx.update_after == 0:
            for trainer_id, trainer in trainers.items():
                stats = trainer.update()
                if update_callback and stats:
                    update_count += 1
                    payload: Dict[str, Any] = {"train/update": update_count}
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            payload[f"train/{key}"] = float(value)
                        else:
                            payload[f"train/{key}"] = value
                    update_callback(payload)

        ppo_return = returns.get(ppo_id, 0.0)
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

        success_token = "success" if success else "no-success"
        print(
            f"[EP {ep + 1:03d}/{episodes}] mode={reward_wrapper.mode} "
            f"steps={steps} cause={cause_str} {success_token} "
            f"return_{ppo_id}={returns.get(ppo_id, 0.0):.2f}"
        )

    for trainer_id, trainer in trainers.items():
        stats = trainer.update()
        if update_callback and stats:
            update_count += 1
            payload: Dict[str, Any] = {"train/update": update_count}
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    payload[f"train/{key}"] = float(value)
                else:
                    payload[f"train/{key}"] = value
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
