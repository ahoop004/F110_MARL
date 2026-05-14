"""Build fixed-policy agents and normalize trainable/fixed roles."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from src.core.config import AgentFactory


PYTORCH_RL_ALGOS = {"ppo", "a2c", "sac", "td3", "ddpg", "dqn", "mappo"}
HEURISTIC_ALGOS = {
    "ftg",
    "follow_gap",
    "gap_follow",
    "followthegap",
    "pure_pursuit",
    "stanley",
    "hybrid_pp_ftg",
}


def build_fixed_policy_agents(agent_configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Instantiate heuristic/fixed-policy agents only.

    Pure PyTorch RL algorithms are instantiated by `run.py`, not setup.
    """

    agents: Dict[str, Any] = {}
    for agent_id, agent_config in agent_configs.items():
        algorithm = str(agent_config["algorithm"]).lower()
        if algorithm in PYTORCH_RL_ALGOS:
            continue
        if algorithm in HEURISTIC_ALGOS:
            heuristic_kwargs = dict(agent_config.get("params", {}))
            agents[agent_id] = AgentFactory.create(algorithm, heuristic_kwargs)
    return agents
