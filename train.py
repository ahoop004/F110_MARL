"""Training entrypoint for F110 multi-agent experiments.

Loads config/scenario files, builds the parallel environment, and runs a
lightweight roll-out loop ready to be swapped with an actual learner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import yaml

from f110x.envs import F110ParallelEnv
from f110x.policies import GapFollowPolicy
from f110x.tasks.gap_follow import GapFollowTask


@dataclass
class ExperimentSpec:
    config_path: Path
    scenario_path: Path
    episodes: int
    max_steps: int


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


# TODO: plug in actual learner & logging backends (e.g., RLlib, MARLlib, WandB) once ready.
def run_training(spec: ExperimentSpec) -> None:
    env, raw_cfg = _build_env(spec.config_path)
    scenario = _load_scenario(spec.scenario_path)
    tasks, policies = _build_agent_components(scenario)

    print("Loaded config:", spec.config_path)
    print("Loaded scenario agents:", [agent["id"] for agent in scenario.get("agents", [])])
    if policies:
        print("Scripted policies:", {aid: policy.__class__.__name__ for aid, policy in policies.items()})
    if tasks:
        print("Tasks:", {aid: task.__class__.__name__ for aid, task in tasks.items()})

    rewards = _rollout(env, policies, episodes=spec.episodes, max_steps=spec.max_steps)
    stats = _summarize(rewards)

    print("Episode reward stats (mean, std):")
    for aid, (mean_val, std_val) in stats.items():
        print(f"  {aid}: mean={mean_val:.2f}, std={std_val:.2f}")


def _parse_args() -> ExperimentSpec:
    parser = argparse.ArgumentParser(description="Train agents in F110 parallel environment")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--scenario", type=Path, default=Path("scenarios/single_attacker.yaml"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()
    return ExperimentSpec(args.config.expanduser(), args.scenario.expanduser(), args.episodes, args.max_steps)


def main() -> None:
    spec = _parse_args()
    run_training(spec)


if __name__ == "__main__":
    main()
