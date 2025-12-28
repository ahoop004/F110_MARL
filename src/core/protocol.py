"""Agent protocol - defines the interface all RL agents must implement."""
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
