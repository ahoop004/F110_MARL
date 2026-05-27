"""Build fixed-policy agents and normalize trainable/fixed roles.

Agent role resolution
---------------------
Each agent config may declare its training role explicitly::

    agents:
      car_0:
        algorithm: ppo
        trainable: true      # explicit — takes priority

      car_1:
        algorithm: ftg
        trainable: false     # explicit — takes priority

When ``trainable`` is absent the role is inferred from ``algorithm``:

- Algorithms in :data:`PYTORCH_RL_ALGOS` → trainable.
- Algorithms in :data:`HEURISTIC_ALGOS` → fixed.
- Unknown algorithms default to fixed and log a warning.

``fixed_policy_agents`` may also be listed on ``environment:`` in the
scenario; the env uses that list to set up masks.  The agent-config-level
field is the canonical source for training setup.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Tuple

from src.core.config import AgentFactory

logger = logging.getLogger(__name__)


PYTORCH_RL_ALGOS: frozenset[str] = frozenset({
    "ppo", "a2c", "sac", "td3", "ddpg", "dqn", "mappo",
})

HEURISTIC_ALGOS: frozenset[str] = frozenset({
    "ftg",
    "follow_gap",
    "gap_follow",
    "followthegap",
    "pure_pursuit",
    "stanley",
    "hybrid_pp_ftg",
})


# ---------------------------------------------------------------------------
# Single-agent role query
# ---------------------------------------------------------------------------

def is_trainable_agent(agent_cfg: Mapping[str, Any]) -> bool:
    """Return True when *agent_cfg* represents a trainable (RL) agent.

    Checks the explicit ``trainable`` boolean first; falls back to
    algorithm-name inference when the field is absent.

    Parameters
    ----------
    agent_cfg:
        A single agent's config dict (one entry under ``scenario["agents"]``).
    """
    explicit = agent_cfg.get("trainable")
    if explicit is not None:
        return bool(explicit)
    algo = str(agent_cfg.get("algorithm", "")).strip().lower()
    if algo in PYTORCH_RL_ALGOS:
        return True
    if algo in HEURISTIC_ALGOS:
        return False
    logger.warning(
        "Unknown algorithm %r — treating as fixed policy.  "
        "Set 'trainable: true' in the agent config to override.",
        algo,
    )
    return False


# ---------------------------------------------------------------------------
# Multi-agent role split
# ---------------------------------------------------------------------------

def get_trainable_agent_ids(agent_configs: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Return agent IDs whose role resolves to trainable.

    Parameters
    ----------
    agent_configs:
        The full ``scenario["agents"]`` mapping.
    """
    return [aid for aid, cfg in agent_configs.items() if is_trainable_agent(cfg)]


def get_fixed_agent_ids(agent_configs: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Return agent IDs whose role resolves to fixed/heuristic.

    Parameters
    ----------
    agent_configs:
        The full ``scenario["agents"]`` mapping.
    """
    return [aid for aid, cfg in agent_configs.items() if not is_trainable_agent(cfg)]


def split_agent_roles(
    agent_configs: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Split agents into (trainable_ids, fixed_ids).

    Returns
    -------
    trainable_ids, fixed_ids:
        Two lists in ``agent_configs`` iteration order.
    """
    trainable, fixed = [], []
    for aid, cfg in agent_configs.items():
        (trainable if is_trainable_agent(cfg) else fixed).append(aid)
    return trainable, fixed


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def build_fixed_policy_agents(agent_configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Instantiate heuristic/fixed-policy agents only.

    Pure PyTorch RL agents are instantiated by ``run.py``, not here.

    Parameters
    ----------
    agent_configs:
        The full ``scenario["agents"]`` mapping.

    Returns
    -------
    Dict[str, Any]
        ``{agent_id: heuristic_policy_instance}`` for all fixed-role agents.
    """
    agents: Dict[str, Any] = {}
    for agent_id, agent_config in agent_configs.items():
        if is_trainable_agent(agent_config):
            continue
        algorithm = str(agent_config.get("algorithm", "")).strip().lower()
        if algorithm in HEURISTIC_ALGOS:
            heuristic_kwargs = dict(agent_config.get("params", {}))
            agents[agent_id] = AgentFactory.create(algorithm, heuristic_kwargs)
        else:
            logger.warning(
                "Agent '%s' has algorithm '%s' which is neither a known RL algo "
                "nor a known heuristic — skipping instantiation.",
                agent_id,
                algorithm,
            )
    return agents
