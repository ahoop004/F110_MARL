"""Agent protocols — interfaces for RL agents and heuristic/fixed-policy drivers."""
from typing import Any, Dict, Optional, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Agent(Protocol):
    """Protocol defining the interface for all RL agents.

    This protocol eliminates the need for wrapper classes by standardizing
    the interface that all agents must implement.
    """

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given an observation.

        Args:
            obs: Observation from environment
            deterministic: If True, select action deterministically (for eval)

        Returns:
            action: Action to take (continuous array or discrete index)
        """
        ...

    def store(self, *args, **kwargs) -> None:
        """Store a transition in the agent's buffer.

        The signature varies by agent type:
        - On-policy (PPO): store(obs, action, reward, done, terminated)
        - Off-policy (TD3/SAC): store_transition(obs, action, reward, next_obs, done)
        - DQN: store_transition(obs, action, reward, next_obs, done)
        """
        ...

    def update(self) -> Optional[Dict[str, float]]:
        """Perform a learning update.

        Returns:
            stats: Dictionary of training statistics (losses, etc.)
                   Returns None if not ready to update yet.
        """
        ...

    def save(self, path: str) -> None:
        """Save agent checkpoint to disk.

        Args:
            path: Path to save checkpoint
        """
        ...

    def load(self, path: str) -> None:
        """Load agent checkpoint from disk.

        Args:
            path: Path to load checkpoint from
        """
        ...


@runtime_checkable
class OnPolicyAgent(Agent, Protocol):
    """Protocol for on-policy agents (PPO, etc.)."""

    def finish_path(self, **kwargs) -> None:
        """Finish a trajectory and compute advantages/returns.

        Called at the end of an episode or when the buffer is full.
        """
        ...


@runtime_checkable
class OffPolicyAgent(Agent, Protocol):
    """Protocol for off-policy agents (TD3, SAC, DQN, etc.)."""

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        ...


def is_on_policy_agent(agent: Any) -> bool:
    """Check if agent is on-policy."""
    return isinstance(agent, OnPolicyAgent)


def is_off_policy_agent(agent: Any) -> bool:
    """Check if agent is off-policy."""
    return isinstance(agent, OffPolicyAgent)


@runtime_checkable
class HeuristicPolicy(Protocol):
    """Protocol for fixed/heuristic drivers.

    Heuristic policies receive *raw* observation dicts (the same dict the env
    produces) and return a physical action array.  They do **not** depend on
    :class:`ObservationComposer` or any RL-specific normalization.

    Trainers call ``reset`` at the start of each episode and ``act`` at each
    step.  The optional ``info`` argument mirrors the env info dict so policies
    can read spawn metadata, lap counts, etc. without env handles.

    All existing fixed-policy classes (``FTGAgent``, ``PurePursuitPolicy``,
    ``StanleyPolicy``, ``HybridPPFTGPolicy``) satisfy this protocol already;
    the protocol makes the contract explicit and ``isinstance``-checkable.
    """

    def act(
        self,
        obs: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Return a physical action given a raw observation dict.

        Parameters
        ----------
        obs:
            Raw per-agent observation dict from the env.
        info:
            Optional per-agent info dict from the most recent env step/reset.
        deterministic:
            When ``True`` the policy should act greedily (used during eval).

        Returns
        -------
        np.ndarray
            Physical action ``[steer, speed]`` (or equivalent).
        """
        ...

    def reset(
        self,
        agent_id: str,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset episode-scoped state.

        Called at the start of each episode before the first ``act``.

        Parameters
        ----------
        agent_id:
            The env agent ID this policy is driving.
        info:
            Optional reset info dict (spawn metadata, map bundle, etc.).
        """
        ...


def is_heuristic_policy(agent: Any) -> bool:
    """Return True when *agent* satisfies the :class:`HeuristicPolicy` protocol."""
    return isinstance(agent, HeuristicPolicy)
