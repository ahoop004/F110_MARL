"""Training entrypoint using the shared env/agent builders."""
from __future__ import annotations

import os
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from f110x.envs import F110ParallelEnv
from f110x.utils.builders import AgentBundle, AgentTeam, build_agents, build_env
from f110x.utils.config_models import ExperimentConfig
from f110x.utils.map_loader import MapData
from f110x.utils.output import resolve_output_dir, resolve_output_file
from f110x.utils.start_pose import reset_with_start_poses
from f110x.wrappers.reward import RewardWrapper
from f110x.trainers.base import Transition, Trainer


DEFAULT_CONFIG_PATH = Path("configs/experiments.yaml")


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
    output_root: Path
    checkpoint_path: Optional[Path]
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


def create_training_context(cfg_path: Path | None = None, *, experiment: str | None = None) -> TrainingContext:
    cfg_file = cfg_path or DEFAULT_CONFIG_PATH
    cfg = ExperimentConfig.load(cfg_file, experiment=experiment)

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

    output_root = Path(cfg.main.get("output_root", "outputs")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    cfg.main.schema.output_root = str(output_root)

    bundle_cfg = primary_bundle.metadata.get("config", {})
    if (not bundle_cfg) and primary_bundle.algo.lower() == "ppo":
        bundle_cfg = cfg.ppo.to_dict()
    bundle_cfg = dict(bundle_cfg)

    save_dir_value = bundle_cfg.get("save_dir", "checkpoints")
    checkpoint_dir = resolve_output_dir(save_dir_value, output_root)
    bundle_cfg["save_dir"] = str(checkpoint_dir)
    default_name = f"{primary_bundle.algo.lower()}_best.pt"
    checkpoint_name = bundle_cfg.get("checkpoint_name", default_name)

    run_suffix = os.environ.get("RUN_ITER") or os.environ.get("RUN_SEED")
    safe_suffix = None
    if run_suffix:
        safe_suffix = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(run_suffix)
        ).strip("-")
    if safe_suffix:
        base_name = Path(checkpoint_name)
        checkpoint_name = f"{base_name.stem}_{safe_suffix}{base_name.suffix}"
    bundle_cfg["checkpoint_name"] = checkpoint_name
    primary_bundle.metadata["config"] = bundle_cfg
    best_path = checkpoint_dir / checkpoint_name

    main_checkpoint = cfg.main.checkpoint
    if main_checkpoint:
        resolved_checkpoint = resolve_output_file(main_checkpoint, output_root)
        cfg.main.schema.checkpoint = str(resolved_checkpoint)

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
        output_root=output_root,
        checkpoint_path=None,
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

    _ = update_start  # retained for compatibility with older callers

    results: List[Dict[str, Any]] = []
    recent_window = max(1, int(ctx.ppo_bundle.metadata.get("config", {}).get("rolling_avg_window", 10)))
    recent_returns: deque[float] = deque(maxlen=recent_window)
    best_return = float("-inf")
    trainers = dict(ctx.trainer_map)
    roles = getattr(ctx.team, "roles", {})
    attacker_id = roles.get("attacker", ppo_id)
    defender_id = roles.get("defender")

    truncation_penalty = float(ctx.reward_cfg.get("truncation_penalty", 0.0))
    idle_speed_threshold = float(ctx.reward_cfg.get("idle_speed_threshold", 0.4))
    idle_patience_steps = int(ctx.reward_cfg.get("idle_patience_steps", 200))

    log_blocklist = {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "action_mean",
        "action_std",
        "action_abs_mean",
        "raw_action_std",
        "value_mean",
        "value_std",
        "adv_mean",
        "adv_std",
        "action_histogram",
        "value_histogram",
    }

    def emit_update_stats(stats: Optional[Dict[str, Any]]) -> None:
        # Caller requested return-only logging; skip optimizer metrics.
        return

    agent_ids = list(env.possible_agents)
    agent_count = len(agent_ids)
    id_to_index = {aid: idx for idx, aid in enumerate(agent_ids)}

    totals_array = np.zeros(agent_count, dtype=np.float32)
    collision_counts_array = np.zeros(agent_count, dtype=np.int32)
    collision_step_array = np.full(agent_count, -1, dtype=np.int32)
    collision_flags = np.zeros(agent_count, dtype=bool)
    speed_sums_array = np.zeros(agent_count, dtype=np.float32)
    speed_counts_array = np.zeros(agent_count, dtype=np.int32)
    done_flags = np.zeros(agent_count, dtype=bool)

    # per-step scratch buffers reused across the episode loop
    step_collision_mask = np.zeros(agent_count, dtype=bool)
    step_speed_values = np.zeros(agent_count, dtype=np.float32)
    step_speed_present = np.zeros(agent_count, dtype=bool)
    shaped_rewards = np.zeros(agent_count, dtype=np.float32)

    class TrajectoryBuffer:
        __slots__ = ("trainer", "_items", "_capacity", "off_policy", "updates_per_step")

        def __init__(
            self,
            trainer: Trainer,
            *,
            capacity: int = 64,
            off_policy: bool = False,
            updates_per_step: int = 1,
        ) -> None:
            self.trainer = trainer
            self._items: List[Transition] = []
            self._capacity = max(int(capacity), 1)
            self.off_policy = bool(off_policy)
            self.updates_per_step = max(int(updates_per_step), 1)

        def append(self, item: Transition) -> bool:
            self._items.append(item)
            if len(self._items) >= self._capacity:
                return self.flush()
            return False

        def flush(self) -> bool:
            if not self._items:
                return False
            observe_batch = getattr(self.trainer, "observe_batch", None)
            if callable(observe_batch):
                observe_batch(self._items)
            else:
                for item in self._items:
                    self.trainer.observe(item)
            self._items.clear()
            return True

    off_policy_algos = {"td3", "sac", "dqn"}
    trajectory_buffers: Dict[str, TrajectoryBuffer] = {}
    off_policy_ids: Set[str] = set()
    for aid, trainer in trainers.items():
        bundle = ctx.team.by_id.get(aid)
        algo_name = bundle.algo.lower() if bundle else ""
        is_off_policy = algo_name in off_policy_algos
        if is_off_policy:
            off_policy_ids.add(aid)
        config_meta = bundle.metadata.get("config", {}) if bundle else {}
        updates_per_step = int(config_meta.get("updates_per_step", config_meta.get("gradient_steps", 1) or 1))
        capacity = 1 if is_off_policy else 64
        trajectory_buffers[aid] = TrajectoryBuffer(
            trainer,
            capacity=capacity,
            off_policy=is_off_policy,
            updates_per_step=updates_per_step,
        )

    for ep in range(episodes):
        obs, infos = reset_with_start_poses(
            env,
            ctx.start_pose_options,
            back_gap=ctx.start_pose_back_gap,
            min_spacing=ctx.start_pose_min_spacing,
            map_data=ctx.map_data,
        )
        ctx.team.reset_actions()

        done = {aid: False for aid in agent_ids}
        done_flags.fill(False)
        totals_array.fill(0.0)
        reward_wrapper = _build_reward_wrapper(ctx, ep)
        reward_breakdown: Dict[str, Dict[str, float]] = {
            aid: defaultdict(float) for aid in agent_ids
        }

        steps = 0
        terms: Dict[str, bool] = {}
        truncs: Dict[str, bool] = {}
        collision_counts_array.fill(0)
        collision_step_array.fill(-1)
        collision_flags.fill(False)
        speed_sums_array.fill(0.0)
        speed_counts_array.fill(0)
        idle_counter = 0
        idle_triggered = False

        while True:
            step_collision_mask.fill(False)
            step_speed_present.fill(False)
            step_speed_values.fill(0.0)
            shaped_rewards.fill(0.0)

            actions, processed_obs = _compute_actions(ctx, obs, done)
            if not actions:
                break

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            steps += 1

            for idx, aid in enumerate(agent_ids):
                base_reward = rewards.get(aid, 0.0)
                if next_obs.get(aid) is not None:
                    base_reward = reward_wrapper(
                        next_obs,
                        aid,
                        rewards.get(aid, 0.0),
                        done=terms.get(aid, False) or truncs.get(aid, False),
                        info=infos.get(aid, {}),
                        all_obs=next_obs,
                    )
                totals_array[idx] += float(base_reward)
                shaped_rewards[idx] = float(base_reward)

                components = reward_wrapper.get_last_components(aid)
                if components:
                    for name, value in components.items():
                        reward_breakdown[aid][name] = reward_breakdown[aid].get(name, 0.0) + float(value)

            for idx, aid in enumerate(agent_ids):
                collided = bool(next_obs.get(aid, {}).get("collision", False))
                if collided:
                    step_collision_mask[idx] = True
                    collision_counts_array[idx] += 1
                    if collision_step_array[idx] < 0:
                        collision_step_array[idx] = steps
                    collision_flags[idx] = True

            for trainer_id, trainer in trainers.items():
                if trainer_id not in processed_obs:
                    continue

                agent_idx = id_to_index.get(trainer_id)
                if agent_idx is None:
                    continue

                terminated = terms.get(trainer_id, False) or bool(step_collision_mask[agent_idx])
                truncated = truncs.get(trainer_id, False)

                patched_next = _prepare_next_observation(ctx, trainer_id, next_obs, infos)
                try:
                    next_wrapped = ctx.team.observation(trainer_id, patched_next)
                except Exception:
                    next_wrapped = processed_obs[trainer_id]

                shaped = float(shaped_rewards[agent_idx]) if agent_idx is not None else rewards.get(trainer_id, 0.0)
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
                buffer = trajectory_buffers[trainer_id]
                flushed = buffer.append(transition)
                if buffer.off_policy and flushed:
                    for _ in range(buffer.updates_per_step):
                        stats = trainer.update()
                        emit_update_stats(stats)

            for idx, aid in enumerate(agent_ids):
                done_flag = terms.get(aid, False) or truncs.get(aid, False) or collision_flags[idx]
                done_flags[idx] = done_flag
                done[aid] = done_flag

            obs = next_obs

            for idx, aid in enumerate(agent_ids):
                velocity = next_obs.get(aid, {}).get("velocity")
                if velocity is None:
                    continue
                speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float32)))
                if np.isnan(speed):
                    speed = 0.0
                step_speed_values[idx] = speed
                step_speed_present[idx] = True
                speed_sums_array[idx] += speed
                speed_counts_array[idx] += 1

            if idle_patience_steps > 0:
                if step_speed_present.any():
                    active = step_speed_values[step_speed_present]
                    if np.all(active < idle_speed_threshold):
                        idle_counter += 1
                    else:
                        idle_counter = 0
                else:
                    idle_counter = 0

                if idle_counter >= idle_patience_steps:
                    idle_triggered = True
                    for aid in env.possible_agents:
                        truncs[aid] = True
                        done[aid] = True
                    break

            if collision_flags.any() or bool(done_flags.all()):
                break

            if ctx.render_interval and ((ep + 1) % ctx.render_interval == 0):
                try:
                    env.render()
                except Exception as exc:
                    print(f"[WARN] Rendering disabled due to: {exc}")
                    ctx.render_interval = 0

        end_cause: List[str] = []
        if collision_flags.any():
            end_cause.append("collision")
        for aid in done:
            if terms.get(aid, False):
                end_cause.append(f"term:{aid}")
            if truncs.get(aid, False):
                end_cause.append(f"trunc:{aid}")
        if idle_triggered:
            end_cause.append("idle")
            print(
                f"[INFO] Idle stop triggered at episode {ep + 1} after {steps} steps "
                f"(threshold={idle_patience_steps}, speed<{idle_speed_threshold})"
            )
        if not end_cause:
            end_cause.append("unknown")
        cause_str = ",".join(end_cause)

        if truncation_penalty:
            for idx, aid in enumerate(agent_ids):
                if truncs.get(aid, False):
                    totals_array[idx] += truncation_penalty
                    reward_breakdown[aid]["truncation_penalty"] = (
                        reward_breakdown[aid].get("truncation_penalty", 0.0) + truncation_penalty
                    )

        returns = {aid: float(totals_array[idx]) for idx, aid in enumerate(agent_ids)}

        epsilon_val: Optional[float] = None
        epsilon_accessor = getattr(ctx.ppo_trainer, "epsilon", None)
        if callable(epsilon_accessor):
            try:
                epsilon_val = float(epsilon_accessor())
            except Exception:  # pragma: no cover - defensive guard
                epsilon_val = None

        defender_idx = id_to_index.get(defender_id) if defender_id else None
        attacker_idx = id_to_index.get(attacker_id) if attacker_id else None

        defender_crashed = bool(
            defender_idx is not None and collision_step_array[defender_idx] >= 0
        )
        attacker_crashed = bool(
            attacker_idx is not None and collision_step_array[attacker_idx] >= 0
        )
        success = bool(defender_crashed and not attacker_crashed)
        defender_survival_steps: Optional[int] = None
        if defender_idx is not None:
            defender_survival_steps = (
                int(collision_step_array[defender_idx]) if defender_crashed else steps
            )

        collisions_total = int(collision_counts_array.sum())
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
            "idle_truncated": idle_triggered,
        }
        if epsilon_val is not None:
            episode_record["epsilon"] = epsilon_val

        for idx, aid in enumerate(agent_ids):
            episode_record[f"collision_count_{aid}"] = int(collision_counts_array[idx])
        for idx, aid in enumerate(agent_ids):
            step_val = collision_step_array[idx]
            if step_val >= 0:
                episode_record[f"collision_step_{aid}"] = int(step_val)
        for idx, aid in enumerate(agent_ids):
            count = speed_counts_array[idx]
            if count > 0:
                episode_record[f"avg_speed_{aid}"] = float(speed_sums_array[idx] / count)
            else:
                episode_record[f"avg_speed_{aid}"] = 0.0

        results.append(episode_record)

        if update_callback:
            payload: Dict[str, Any] = {"train/episode": float(ep + 1)}

            for aid, value in returns.items():
                payload[f"train/return_{aid}"] = float(value)

            update_callback(payload)

        if (ep + 1) % ctx.update_after == 0:
            for trainer_id, trainer in trainers.items():
                if trainer_id in off_policy_ids:
                    continue
                stats = trainer.update()
                emit_update_stats(stats)

        ppo_return = returns.get(ppo_id, 0.0)
        recent_returns.append(ppo_return)
        if len(recent_returns) == recent_returns.maxlen:
            avg_return = float(sum(recent_returns) / len(recent_returns))
            if avg_return > best_return:
                best_return = avg_return
                ppo_agent.save(str(ctx.best_path))
                print(
                    f"[INFO] New best model saved at episode {ep + 1} "
                    f"(avg_return={avg_return:.2f}) → {ctx.best_path}"
                )

        success_token = "success" if success else "no-success"
        eps_fragment = f" epsilon={epsilon_val:.3f}" if epsilon_val is not None else ""
        print(
            f"[EP {ep + 1:03d}/{episodes}] mode={reward_wrapper.mode} "
            f"steps={steps} cause={cause_str} {success_token} "
            f"return_{ppo_id}={returns.get(ppo_id, 0.0):.2f}{eps_fragment}"
        )

    for trainer_id, trainer in trainers.items():
        buffer = trajectory_buffers.get(trainer_id)
        flushed = False
        if buffer is not None:
            flushed = buffer.flush()
        if buffer is not None and buffer.off_policy:
            if flushed:
                for _ in range(buffer.updates_per_step):
                    stats = trainer.update()
                    emit_update_stats(stats)
            continue
        stats = trainer.update()
        emit_update_stats(stats)

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
