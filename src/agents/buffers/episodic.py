"""Episodic buffer for storing wavelet-transformed memory chunks."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class EpisodicChunk:
    """Container for a single episodic chunk.

    Attributes:
        chunk_id: Unique identifier for the chunk
        wavelet_coefficients: Wavelet-transformed representation (N, D) or (5, 5, 5, D) for 5x5 grid
        raw_observations: Original observation sequence (N, obs_dim)
        raw_actions: Action sequence (N, act_dim) or (N,) for discrete
        raw_rewards: Reward sequence (N,)
        raw_next_observations: Next observation sequence (N, obs_dim)
        raw_dones: Done flags (N,)
        metadata: Additional information (episode_id, timestamp_start, etc.)
        selection_weight: Priority weight for sampling
    """
    chunk_id: int
    wavelet_coefficients: np.ndarray
    raw_observations: np.ndarray
    raw_actions: np.ndarray
    raw_rewards: np.ndarray
    raw_next_observations: np.ndarray
    raw_dones: np.ndarray
    metadata: Dict[str, Any]
    selection_weight: float = 1.0


class EpisodicBuffer2D:
    """Buffer for storing episodic memory chunks with selection weights.

    Supports both uniform and priority-based sampling strategies.
    Designed to work with 5×5 grid chunks aligned with 5-tuple memory structure.

    Args:
        capacity: Maximum number of chunks to store
        chunk_size: Number of transitions per chunk (should be 25 for 5×5 grid)
        grid_shape: Shape of spatial grid (e.g., (5, 5) for 25 timesteps)
        n_channels: Number of channels (5 for obs/action/reward/next_obs/done)
        selection_mode: 'uniform' for random sampling, 'priority' for weighted sampling
        alpha: Priority exponent (for priority mode)
        beta: Importance sampling weight exponent (for priority mode)
        beta_increment: Per-sample increment for beta annealing
    """

    def __init__(
        self,
        capacity: int,
        chunk_size: int = 25,
        grid_shape: tuple = (5, 5),
        n_channels: int = 5,
        selection_mode: str = "uniform",
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.grid_shape = grid_shape
        self.n_channels = n_channels
        self.selection_mode = selection_mode
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # Validate grid structure
        assert np.prod(grid_shape) == chunk_size, \
            f"Grid shape {grid_shape} must have product equal to chunk_size {chunk_size}"

        # Storage
        self.chunks: List[Optional[EpisodicChunk]] = [None] * capacity
        self.selection_weights: np.ndarray = np.ones(capacity, dtype=np.float32)

        # Circular buffer pointers
        self._idx = 0
        self._size = 0

        # Priority sampling stats
        self._max_weight = 1.0
        self._min_weight = 0.01

    def add_chunk(self, chunk: EpisodicChunk) -> None:
        """Add an episodic chunk to the buffer.

        Args:
            chunk: EpisodicChunk to store
        """
        # Store chunk at current index
        self.chunks[self._idx] = chunk

        # Initialize weight (new chunks get max priority)
        if self.selection_mode == "priority":
            self.selection_weights[self._idx] = self._max_weight
        else:
            self.selection_weights[self._idx] = 1.0

        # Update circular buffer pointers
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of chunks from the buffer.

        Args:
            batch_size: Number of chunks to sample

        Returns:
            Dictionary containing:
                - 'wavelet_chunks': (B, 5, 5, 5, D) wavelet coefficients
                - 'raw_obs': (B, 25, obs_dim) raw observations
                - 'raw_actions': (B, 25, act_dim) raw actions
                - 'raw_rewards': (B, 25) raw rewards
                - 'raw_next_obs': (B, 25, obs_dim) raw next observations
                - 'raw_dones': (B, 25) raw done flags
                - 'indices': (B,) indices of sampled chunks
                - 'weights': (B,) importance sampling weights (for priority mode)
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, self._size)

        # Sample indices
        if self.selection_mode == "uniform":
            indices = np.random.choice(self._size, size=batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)
        else:  # priority mode
            # Compute sampling probabilities
            priorities = self.selection_weights[:self._size] ** self.alpha
            probs = priorities / priorities.sum()

            # Sample with replacement based on priorities
            indices = np.random.choice(self._size, size=batch_size, replace=True, p=probs)

            # Compute importance sampling weights
            weights = (self._size * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize by max weight

            # Anneal beta
            self.beta = min(1.0, self.beta + batch_size * self.beta_increment)

        # Gather chunks
        sampled_chunks = [self.chunks[idx] for idx in indices]

        # Stack into batch tensors
        batch = {
            'wavelet_chunks': np.stack([c.wavelet_coefficients for c in sampled_chunks]),
            'raw_obs': np.stack([c.raw_observations for c in sampled_chunks]),
            'raw_actions': np.stack([c.raw_actions for c in sampled_chunks]),
            'raw_rewards': np.stack([c.raw_rewards for c in sampled_chunks]),
            'raw_next_obs': np.stack([c.raw_next_observations for c in sampled_chunks]),
            'raw_dones': np.stack([c.raw_dones for c in sampled_chunks]),
            'indices': indices,
            'weights': weights,
        }

        return batch

    def update_weights(self, indices: np.ndarray, new_weights: np.ndarray) -> None:
        """Update selection weights for sampled chunks.

        Args:
            indices: Indices of chunks to update
            new_weights: New weight values (e.g., TD-errors, reconstruction errors)
        """
        if self.selection_mode != "priority":
            return  # No-op for uniform sampling

        # Clip weights to reasonable range
        new_weights = np.clip(new_weights, self._min_weight, None)

        # Update weights
        for idx, weight in zip(indices, new_weights):
            self.selection_weights[idx] = float(weight)
            self._max_weight = max(self._max_weight, float(weight))

    def get_all_weights(self) -> np.ndarray:
        """Get all current selection weights.

        Returns:
            weights: Array of selection weights for all stored chunks
        """
        if self._size == 0:
            return np.array([])

        # Return weights for all stored chunks
        return np.array([self.selection_weights[i] for i in range(self._size)])

    def __len__(self) -> int:
        """Return number of chunks currently stored."""
        return self._size

    def __repr__(self) -> str:
        return f"EpisodicBuffer2D(size={self._size}/{self.capacity}, mode={self.selection_mode}, grid={self.grid_shape})"
