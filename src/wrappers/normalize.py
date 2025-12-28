"""Running normalization wrapper for observations."""

import numpy as np
from typing import Dict, Optional


class RunningMeanStd:
    """Tracks running mean and std of observations for normalization.

    Based on Welford's online algorithm for computing variance.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """Initialize running statistics.

        Args:
            epsilon: Small constant for numerical stability
            shape: Shape of the data to track
        """
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new data.

        Args:
            x: New observation(s), shape (batch, *shape) or (*shape,)
        """
        if x.ndim == 1:
            # Single observation
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        else:
            # Batch of observations
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update from pre-computed batch statistics.

        Args:
            batch_mean: Mean of the batch
            batch_var: Variance of the batch
            batch_count: Number of samples in batch
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize observation using running statistics.

        Args:
            x: Observation to normalize
            clip: Maximum absolute value after normalization

        Returns:
            Normalized observation
        """
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + 1e-8),
            -clip,
            clip
        ).astype(np.float32)


class ObservationNormalizer:
    """Normalizes observations for multiple agents using running statistics."""

    def __init__(self, obs_shape: tuple, clip: float = 10.0):
        """Initialize normalizer.

        Args:
            obs_shape: Shape of observations to normalize
            clip: Maximum absolute value after normalization
        """
        self.obs_shape = obs_shape
        self.clip = clip
        self.running_stats: Dict[str, RunningMeanStd] = {}

    def normalize(
        self,
        obs: np.ndarray,
        agent_id: str,
        update_stats: bool = True
    ) -> np.ndarray:
        """Normalize observation for an agent.

        Args:
            obs: Observation to normalize
            agent_id: Agent identifier
            update_stats: Whether to update running statistics

        Returns:
            Normalized observation
        """
        # Create stats tracker for new agents
        if agent_id not in self.running_stats:
            self.running_stats[agent_id] = RunningMeanStd(shape=self.obs_shape)

        stats = self.running_stats[agent_id]

        # Update statistics if requested
        if update_stats:
            stats.update(obs)

        # Normalize
        return stats.normalize(obs, clip=self.clip)

    def normalize_dict(
        self,
        obs_dict: Dict[str, np.ndarray],
        update_stats: bool = True
    ) -> Dict[str, np.ndarray]:
        """Normalize observations for all agents in a dict.

        Args:
            obs_dict: Dictionary of agent_id -> observation
            update_stats: Whether to update running statistics

        Returns:
            Dictionary of normalized observations
        """
        return {
            agent_id: self.normalize(obs, agent_id, update_stats)
            for agent_id, obs in obs_dict.items()
        }

    def get_stats(self, agent_id: str) -> Optional[RunningMeanStd]:
        """Get running statistics for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Running statistics or None if agent not seen
        """
        return self.running_stats.get(agent_id)

    def state_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get state dict for checkpointing.

        Returns:
            Dictionary of agent stats
        """
        return {
            agent_id: {
                'mean': stats.mean,
                'var': stats.var,
                'count': np.array([stats.count], dtype=np.float32)
            }
            for agent_id, stats in self.running_stats.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Load state dict from checkpoint.

        Args:
            state_dict: Dictionary of agent stats
        """
        for agent_id, stats_dict in state_dict.items():
            if agent_id not in self.running_stats:
                self.running_stats[agent_id] = RunningMeanStd(shape=self.obs_shape)

            stats = self.running_stats[agent_id]
            stats.mean = stats_dict['mean']
            stats.var = stats_dict['var']
            stats.count = float(stats_dict['count'][0])


__all__ = ['RunningMeanStd', 'ObservationNormalizer']
