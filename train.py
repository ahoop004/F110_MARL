from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import yaml

from f110x.envs import F110ParallelEnv
from f110x.policies import GapFollowPolicy
from f110x.wrappers import ObservationSanitizerWrapper


def main() -> None:
    with open("configs/config.yaml", "r", encoding="utf-8") as handle:
        full_cfg = yaml.safe_load(handle)

    env_cfg = dict(full_cfg.get("env", {}))
    exec_cfg = full_cfg.get("run", {}) or {}
    agents_cfg = exec_cfg.get("agents") or env_cfg.get("agents") or []
    if "agents" not in env_cfg:
        env_cfg["agents"] = agents_cfg

    episodes = int(exec_cfg.get("episodes", 10))
    max_steps = int(exec_cfg.get("max_steps", 1_000))
    render = bool(exec_cfg.get("render", False))

    env = F110ParallelEnv(**env_cfg)
    env = ObservationSanitizerWrapper(
        env,
        target_beams=env_cfg.get("lidar_beams"),
        max_range=env_cfg.get("lidar_range"),
    )

    controllers: Dict[str, GapFollowPolicy] = {}
    for agent_cfg in agents_cfg:
        agent_id = agent_cfg.get("id")
        task_cfg = agent_cfg.get("task", {})
        task_name = str(task_cfg.get("task_name", "")).lower()
        params = task_cfg.get("params", {}) or {}

        if task_name == "gap_follow" and agent_id:
            controllers[agent_id] = GapFollowPolicy(**params)

    if not controllers:
        raise ValueError("Scenario did not specify any gap_follow agents to control.")

    returns = defaultdict(list)

    try:
        for episode in range(episodes):
            for controller in controllers.values():
                reset_fn = getattr(controller, "reset", None)
                if callable(reset_fn):
                    reset_fn()

            obs, _ = env.reset()
            totals = defaultdict(float)

            for step in range(max_steps):
                actions = {}
                for agent_id in env.agents:
                    controller = controllers.get(agent_id)
                    if controller is not None:
                        actions[agent_id] = controller.compute_action(obs[agent_id], env.action_space(agent_id))
                    else:
                        actions[agent_id] = env.action_space(agent_id).sample()

                obs, rewards, terminations, truncations, _ = env.step(actions)

                for agent_id, reward in rewards.items():
                    totals[agent_id] += reward

                if render:
                    env.render()

                done = all(
                    terminations.get(agent_id, False) or truncations.get(agent_id, False)
                    for agent_id in env.possible_agents
                )
                if done:
                    break

            for agent_id in env.possible_agents:
                returns[agent_id].append(totals.get(agent_id, 0.0))
            print(f"Episode {episode + 1}: " + ", ".join(f"{aid}={totals.get(aid, 0.0):.2f}" for aid in env.possible_agents))

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    print("\nAverage episode returns:")
    for agent_id, scores in returns.items():
        values = np.asarray(scores, dtype=np.float32)
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std(ddof=0)) if values.size else 0.0
        print(f"  {agent_id}: mean={mean:.2f}, std={std:.2f}")


if __name__ == "__main__":
    main()
