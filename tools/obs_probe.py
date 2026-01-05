#!/usr/bin/env python3
"""Quick observation probe for training inputs.

Runs a single reset (and optional steps) to show the raw and flattened
observations that would be passed into the training agents.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import numpy as np

from src.core.enhanced_training import EnhancedTrainingLoop
from src.core.scenario import load_and_expand_scenario, ScenarioError
from src.core.setup import create_training_setup
from src.core.spawn_curriculum import SpawnCurriculumManager


def _extract_observation_presets(scenario: Dict[str, Any]) -> tuple[Dict[str, str], Dict[str, str]]:
    observation_presets: Dict[str, str] = {}
    target_ids: Dict[str, str] = {}

    for agent_id, agent_config in scenario["agents"].items():
        obs_config = agent_config.get("observation")
        if isinstance(obs_config, dict):
            if "preset" in obs_config:
                observation_presets[agent_id] = obs_config["preset"]
            elif len(obs_config) > 0:
                observation_presets[agent_id] = "gaplock"

        if "target_id" in agent_config:
            target_ids[agent_id] = agent_config["target_id"]

    if len(scenario["agents"]) == 2:
        attacker_id = None
        defender_id = None
        for agent_id, agent_config in scenario["agents"].items():
            role = str(agent_config.get("role", "")).lower()
            if role == "attacker":
                attacker_id = agent_id
            elif role == "defender":
                defender_id = agent_id
        if attacker_id and defender_id and attacker_id not in target_ids:
            target_ids[attacker_id] = defender_id

    return observation_presets, target_ids


def _build_spawn_curriculum(
    scenario: Dict[str, Any],
    env,
) -> Optional[SpawnCurriculumManager]:
    env_config = scenario["environment"]
    spawn_configs = env_config.get("spawn_configs", {})
    spawn_config = env_config.get("spawn_curriculum", {})
    if not spawn_configs:
        spawn_configs = spawn_config.get("spawn_configs", {})

    phased_curriculum_enabled = scenario.get("curriculum", {}).get("type") == "phased"

    if spawn_config.get("enabled", False) and spawn_configs:
        env.spawn_configs = spawn_configs
        return SpawnCurriculumManager(config=spawn_config, available_spawn_points=spawn_configs)

    if phased_curriculum_enabled and spawn_configs:
        env.spawn_configs = spawn_configs
        return SpawnCurriculumManager(
            config={
                "window": 1,
                "activation_samples": 1,
                "min_episode": 0,
                "enable_patience": 1,
                "disable_patience": 1,
                "cooldown": 0,
                "lock_speed_steps": 0,
                "stages": [
                    {
                        "name": "phase_sampler",
                        "spawn_points": "all",
                        "speed_range": [0.0, 0.0],
                        "enable_rate": 1.0,
                    }
                ],
            },
            available_spawn_points=spawn_configs,
        )

    return None


def _summarize_tensor(label: str, value: Any) -> None:
    if isinstance(value, np.ndarray):
        finite = np.isfinite(value)
        finite_vals = value[finite]
        min_val = float(np.min(finite_vals)) if finite_vals.size else float("nan")
        max_val = float(np.max(finite_vals)) if finite_vals.size else float("nan")
        mean_val = float(np.mean(finite_vals)) if finite_vals.size else float("nan")
        sample = value.reshape(-1)[:6]
        print(
            f"{label}: shape={value.shape} dtype={value.dtype} "
            f"min={min_val:.4f} max={max_val:.4f} mean={mean_val:.4f} "
            f"sample={sample.tolist()}"
        )
    elif isinstance(value, dict):
        keys = list(value.keys())
        print(f"{label}: dict keys={keys}")
    else:
        print(f"{label}: type={type(value).__name__} value={value}")


def _log_observations(
    loop: EnhancedTrainingLoop,
    obs: Dict[str, Any],
    step_idx: str,
    agent_filter: Optional[str] = None,
) -> None:
    print(f"\n=== Observations at {step_idx} ===")
    for agent_id, raw_obs in obs.items():
        if agent_filter and agent_id != agent_filter:
            continue
        flat_obs = loop._flatten_obs(agent_id, raw_obs, all_obs=obs)
        print(f"\nAgent {agent_id}")
        _summarize_tensor("  raw", raw_obs)
        _summarize_tensor("  flat", flat_obs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe training observations.")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML file")
    parser.add_argument("--steps", type=int, default=1, help="Random steps to take after reset")
    parser.add_argument("--agent", default=None, help="Limit output to a single agent ID")
    args = parser.parse_args()

    try:
        scenario = load_and_expand_scenario(args.scenario)
    except (ScenarioError, FileNotFoundError) as exc:
        raise SystemExit(f"Failed to load scenario: {exc}") from exc

    env, agents, reward_strategies = create_training_setup(scenario)
    observation_presets, target_ids = _extract_observation_presets(scenario)
    spawn_curriculum = _build_spawn_curriculum(scenario, env)

    loop = EnhancedTrainingLoop(
        env=env,
        agents=agents,
        agent_rewards=reward_strategies,
        observation_presets=observation_presets,
        target_ids=target_ids,
        spawn_curriculum=spawn_curriculum,
        max_steps_per_episode=scenario["environment"].get("max_steps", 5000),
    )

    if spawn_curriculum:
        spawn_info = spawn_curriculum.sample_spawn(episode=0)
        obs, _info = env.reset(
            options={
                "poses": spawn_info["poses"],
                "velocities": spawn_info["velocities"],
                "lock_speed_steps": spawn_info["lock_speed_steps"],
            }
        )
    else:
        obs, _info = env.reset()

    _log_observations(loop, obs, "reset", agent_filter=args.agent)

    for step_idx in range(1, max(0, args.steps) + 1):
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in env.possible_agents
        }
        next_obs, _rewards, terminations, truncations, _infos = env.step(actions)
        _log_observations(loop, next_obs, f"step {step_idx}", agent_filter=args.agent)
        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()


if __name__ == "__main__":
    main()
