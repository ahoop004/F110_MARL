"""MARLlib integration scaffolding for single-attacker scenario.

This module prepares environment creators and policy metadata so that a MARLlib
experiment can train the attacker while keeping the gap-follow agent scripted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml

from f110x.envs import F110ParallelEnv
from f110x.policies import GapFollowPolicy
from f110x.wrappers import MarlLibParallelWrapper


@dataclass
class ScenarioSpec:
    attacker_agents: Tuple[str, ...]
    scripted_gap_follow_agents: Tuple[str, ...]
    target_mapping: Dict[str, str]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_scenario(path: Path) -> Tuple[dict, ScenarioSpec]:
    data = _load_yaml(path)
    attackers = []
    gap_followers = []
    target_map: Dict[str, str] = {}

    for entry in data.get("agents", []):
        aid = str(entry.get("id")) if isinstance(entry, dict) and entry.get("id") is not None else None
        task = entry.get("task") if isinstance(entry, dict) else None
        task_name = str(task.get("task_name", "")) if isinstance(task, dict) else ""
        params = task.get("params", {}) if isinstance(task, dict) else {}

        if aid is None:
            continue

        if task_name.lower() == "attack_target":
            attackers.append(aid)
            target_id = params.get("target_id") or entry.get("target_id")
            if target_id:
                target_map[aid] = str(target_id)
        elif task_name.lower() == "gap_follow":
            gap_followers.append(aid)

    spec = ScenarioSpec(
        attacker_agents=tuple(attackers),
        scripted_gap_follow_agents=tuple(gap_followers),
        target_mapping=target_map,
    )
    return data, spec


def marllib_env_creator(env_config: dict) -> MarlLibParallelWrapper:
    env = F110ParallelEnv(**env_config)
    return MarlLibParallelWrapper(env)


def marllib_env_config(
    config_path: Path = Path("configs/config.yaml"),
    scenario_path: Path = Path("scenarios/single_attacker.yaml"),
):
    raw_cfg = _load_yaml(config_path)
    env_cfg = raw_cfg["env"]
    scenario, spec = _load_scenario(scenario_path)

    policy_mapping: Dict[str, str] = {}
    policies: Dict[str, Dict[str, object]] = {}

    attacker_policy_id = "attacker_trainable"
    scripted_policy_id = "gap_follow_scripted"

    for aid in spec.attacker_agents:
        policy_mapping[aid] = attacker_policy_id
    for aid in spec.scripted_gap_follow_agents:
        policy_mapping[aid] = scripted_policy_id

    policies[attacker_policy_id] = {
        "trainable": True,
        "config": {},  # TODO: plug in attacker policy model config for MARLlib.
    }

    scripted = {
        "trainable": False,
        "config": {
            "policy_class": GapFollowPolicy,
        },
    }
    policies[scripted_policy_id] = scripted

    marllib_cfg = {
        "env_creator": marllib_env_creator,
        "env_config": env_cfg,
        "scenario": scenario,
        "policies": policies,
        "policy_mapping": policy_mapping,
        "target_map": spec.target_mapping,
    }

    # TODO: build PPO AlgorithmConfig from marllib_cfg (policy specs, loss settings, evaluation hooks).
    return marllib_cfg


def summarize_marllib_setup(cfg: Dict[str, object]) -> None:
    print("MARLlib environment ready")
    print("  env_config keys:", sorted(cfg["env_config"].keys()))
    print("  policies:")
    for pid, meta in cfg["policies"].items():
        print(f"    {pid}: trainable={meta['trainable']}")
    print("  policy mapping:", cfg["policy_mapping"])


if __name__ == "__main__":
    cfg = marllib_env_config()
    summarize_marllib_setup(cfg)
