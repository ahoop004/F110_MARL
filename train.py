"""Training entrypoint for F110 multi-agent experiments.

Loads config/scenario files, builds the parallel environment, and runs a
lightweight roll-out loop ready to be swapped with an actual learner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import yaml

from f110x.envs import F110ParallelEnv
from f110x.policies import GapFollowPolicy
from f110x.tasks.gap_follow import GapFollowTask
from f110x.tasks.attacker import HerdingAttackTask
from algos.ppo.ppo import PPOConfig, PPOTrainer


@dataclass
class ExperimentSpec:
    config_path: Path
    scenario_path: Path
    episodes: int
    max_steps: int
    algo: Optional[str] = None
    updates: Optional[int] = None
    rollout_steps: Optional[int] = None
    device: Optional[str] = None
    seed: Optional[int] = None


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _prepare_env_config(raw_cfg: dict) -> dict:
    env_cfg = dict(raw_cfg)  # shallow copy to avoid mutating caller state

    map_dir = Path(env_cfg.get("map_dir", ".")).expanduser()
    map_yaml_name = env_cfg.get("map_yaml") or env_cfg.get("map")
    if map_yaml_name is None:
        raise ValueError("env config missing map_yaml/map entry")

    map_yaml_path = (map_dir / map_yaml_name).resolve()
    map_meta = _load_yaml(map_yaml_path)

    image_rel = map_meta.get("image")
    explicit_image = env_cfg.get("map_image")
    if image_rel:
        image_path = (map_yaml_path.parent / image_rel).resolve()
    elif explicit_image:
        image_path = (map_dir / explicit_image).resolve()
    else:
        map_ext = env_cfg.get("map_ext", ".png")
        image_path = map_yaml_path.with_suffix(map_ext)

    env_cfg["map_meta"] = map_meta
    env_cfg["map_image_path"] = str(image_path)

    try:
        from PIL import Image  # local import to keep import cost minimal

        with Image.open(image_path) as img:
            env_cfg["map_image_size"] = img.size
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"map image not found: {image_path}") from exc

    return env_cfg


def _build_env(config_path: Path) -> Tuple[F110ParallelEnv, dict]:
    raw_cfg = _load_yaml(config_path)
    env_cfg = _prepare_env_config(raw_cfg["env"])
    env = F110ParallelEnv(**env_cfg)
    return env, raw_cfg


def _load_scenario(scenario_path: Path) -> dict:
    scenario = _load_yaml(scenario_path)
    if not scenario or "agents" not in scenario:
        raise ValueError(f"scenario file missing 'agents': {scenario_path}")
    return scenario


def _default_policy(action_space, *_):
    return action_space.sample()


_TASK_REGISTRY = {
    "gap_follow": GapFollowTask,
    "attack_target": HerdingAttackTask,
}

_POLICY_REGISTRY = {
    "gap_follow": GapFollowPolicy,
}



def _build_agent_components(
    scenario: dict,
) -> Tuple[Dict[str, GapFollowTask], Dict[str, GapFollowPolicy]]:
    tasks: Dict[str, GapFollowTask] = {}
    policies: Dict[str, GapFollowPolicy] = {}

    for agent in scenario.get("agents", []):
        aid = agent.get("id")
        if not aid:
            continue
        task_spec = agent.get("task", {})
        name = str(task_spec.get("task_name", "")).lower()
        params = task_spec.get("params", {}) or {}

        task_cls = _TASK_REGISTRY.get(name)
        if task_cls is not None:
            task_instance = task_cls(**params)
            tasks[aid] = task_instance
            if isinstance(task_instance, HerdingAttackTask):
                params.setdefault("target_id", task_instance.target_id())
        else:
            task_instance = None

        policy_cls = _POLICY_REGISTRY.get(name)
        if policy_cls is not None:
            policy_kwargs: Dict[str, object] = {}
            if task_instance is not None and hasattr(task_instance, "policy_kwargs"):
                policy_kwargs = task_instance.policy_kwargs()
            elif params:
                policy_kwargs = dict(params)
            policies[aid] = policy_cls(**policy_kwargs)

    return tasks, policies


def _rollout(
    env: F110ParallelEnv,
    policies: Dict[str, GapFollowPolicy],
    episodes: int,
    max_steps: int,
) -> Dict[str, Iterable[float]]:
    episode_rewards: Dict[str, list] = defaultdict(list)
    for episode in range(episodes):
        for policy in policies.values():
            policy.reset()
        obs, _ = env.reset()

        totals = defaultdict(float)
        for step in range(max_steps):
            actions = {}
            for aid in env.agents:
                policy = policies.get(aid)
                if policy is not None:
                    actions[aid] = policy.compute_action(obs[aid], env.action_space(aid))
                else:
                    actions[aid] = _default_policy(env.action_space(aid), obs[aid])
            obs, rewards, terms, truncs, _ = env.step(actions)
            env.render()
            for aid, reward in rewards.items():
                totals[aid] += reward
            if all(terms.get(aid, False) or truncs.get(aid, False) for aid in env.possible_agents):
                break
        for aid in env.possible_agents:
            episode_rewards[aid].append(totals.get(aid, 0.0))
    return episode_rewards


def _summarize(rewards: Dict[str, Iterable[float]]) -> Dict[str, Tuple[float, float]]:
    summary = {}
    for aid, scores in rewards.items():
        if not scores:
            summary[aid] = (0.0, 0.0)
            continue
        values = np.asarray(scores, dtype=np.float32)
        summary[aid] = (float(values.mean()), float(values.std(ddof=0)))
    return summary


def _build_ppo_config(spec: ExperimentSpec, training_cfg: Optional[dict]) -> PPOConfig:
    training_cfg = training_cfg or {}
    ppo_cfg = training_cfg.get("ppo", {}) or {}

    cfg_kwargs = {}
    for name in PPOConfig.__dataclass_fields__:
        value = ppo_cfg.get(name)
        if value is not None:
            cfg_kwargs[name] = value

    if spec.rollout_steps is not None:
        cfg_kwargs["rollout_steps"] = spec.rollout_steps
    elif "rollout_steps" not in cfg_kwargs:
        rollout_steps = training_cfg.get("rollout_steps")
        if rollout_steps is not None:
            cfg_kwargs["rollout_steps"] = rollout_steps

    if spec.device is not None:
        cfg_kwargs["device"] = spec.device
    if spec.seed is not None:
        cfg_kwargs["seed"] = spec.seed

    return PPOConfig(**cfg_kwargs)


def _resolve_update_count(spec: ExperimentSpec, training_cfg: Optional[dict]) -> int:
    training_cfg = training_cfg or {}
    updates = spec.updates
    if updates is None:
        updates = training_cfg.get("updates")
    if updates is None:
        updates = spec.episodes
    updates = int(updates)
    if updates <= 0:
        raise ValueError("Number of PPO updates must be positive")
    return updates


def _print_training_summary(history, start_index: int = 1) -> None:
    for idx, record in enumerate(history, start=start_index):
        metrics = record.get("metrics", {})
        returns = record.get("returns", {})
        ret_summary = ", ".join(
            f"{aid}:{float(val):.2f}" for aid, val in sorted(returns.items())
        ) or "n/a"
        print(
            f"  Update {idx:03d}: policy_loss={metrics.get('policy_loss', 0.0):.4f}, "
            f"value_loss={metrics.get('value_loss', 0.0):.4f}, entropy={metrics.get('entropy', 0.0):.4f}, "
            f"returns={ret_summary}"
        )


# TODO: plug in actual learner & logging backends (e.g., RLlib, MARLlib, WandB) once ready.
def run_training(spec: ExperimentSpec) -> None:
    env, raw_cfg = _build_env(spec.config_path)
    try:
        scenario = _load_scenario(spec.scenario_path)
        tasks, policies = _build_agent_components(scenario)

        print("Loaded config:", spec.config_path)
        print("Loaded scenario agents:", [agent["id"] for agent in scenario.get("agents", [])])
        if tasks:
            print("Tasks:", {aid: task.__class__.__name__ for aid, task in tasks.items()})

        training_cfg = raw_cfg.get("training", {}) or {}
        algo_choice = (spec.algo or training_cfg.get("algo") or "ppo").lower()
        print(f"Selected algorithm: {algo_choice}")

        if algo_choice == "scripted":
            if policies:
                print("Scripted policies:", {aid: policy.__class__.__name__ for aid, policy in policies.items()})
            rewards = _rollout(env, policies, episodes=spec.episodes, max_steps=spec.max_steps)
            stats = _summarize(rewards)

            print("Episode reward stats (mean, std):")
            for aid, (mean_val, std_val) in stats.items():
                print(f"  {aid}: mean={mean_val:.2f}, std={std_val:.2f}")
            return

        if algo_choice != "ppo":
            raise ValueError(f"Unsupported algorithm '{algo_choice}' requested")

        ppo_config = _build_ppo_config(spec, training_cfg)
        updates = _resolve_update_count(spec, training_cfg)
        print(f"PPO config: rollout_steps={ppo_config.rollout_steps}, updates={updates}")

        trainer = PPOTrainer(env, ppo_config)
        history = trainer.train(updates)
        print("PPO training summary:")
        _print_training_summary(history)

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _parse_args() -> ExperimentSpec:
    parser = argparse.ArgumentParser(description="Train agents in F110 parallel environment")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--scenario", type=Path, default=Path("scenarios/single_attacker.yaml"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--algo", type=str, choices=("ppo", "scripted"), default=None)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return ExperimentSpec(
        args.config.expanduser(),
        args.scenario.expanduser(),
        args.episodes,
        args.max_steps,
        algo=args.algo,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    spec = _parse_args()
    run_training(spec)


if __name__ == "__main__":
    main()
