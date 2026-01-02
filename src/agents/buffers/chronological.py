"""Chronological buffer for storing experiences in temporal order."""

from typing import Dict, List, Optional
import numpy as np


class ChronologicalBuffer:
    """Buffer that stores transitions in strict chronological order.

    Unlike circular replay buffers, this maintains temporal order and supports
    sliding window extraction for creating episodic chunks.

    Args:
        max_capacity: Maximum number of transitions to store (FIFO eviction if exceeded)
        obs_shape: Shape of observations
        act_shape: Shape of actions
        store_actions: Whether to store continuous actions
        store_action_indices: Whether to store discrete action indices
    """

    def __init__(
        self,
        max_capacity: int,
        obs_shape: tuple,
        act_shape: tuple,
        store_actions: bool = True,
        store_action_indices: bool = False,
    ):
        self.max_capacity = max_capacity
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.store_actions = store_actions
        self.store_action_indices = store_action_indices

        # Storage lists (maintain chronological order)
        self._observations: List[np.ndarray] = []
        self._next_observations: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._timestamps: List[int] = []

        if store_actions:
            self._actions: List[np.ndarray] = []
        if store_action_indices:
            self._action_indices: List[int] = []

        self._infos: List[Optional[Dict]] = []

        # Counters
        self._step_count = 0

    def add(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray] = None,
        reward: float = 0.0,
        next_obs: Optional[np.ndarray] = None,
        done: bool = False,
        action_index: Optional[int] = None,
        info: Optional[Dict] = None,
    ) -> None:
        """Add a transition to the chronological buffer.

        Args:
            obs: Observation
            action: Continuous action (if store_actions=True)
            reward: Reward received
            next_obs: Next observation
            done: Whether episode terminated
            action_index: Discrete action index (if store_action_indices=True)
            info: Optional metadata
        """
        # FIFO eviction if at capacity
        if len(self._observations) >= self.max_capacity:
            self._evict_oldest()

        # Store transition
        self._observations.append(np.array(obs, dtype=np.float32))
        self._next_observations.append(np.array(next_obs, dtype=np.float32) if next_obs is not None else np.zeros_like(obs, dtype=np.float32))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._timestamps.append(self._step_count)

        if self.store_actions and action is not None:
            self._actions.append(np.array(action, dtype=np.float32))
        if self.store_action_indices and action_index is not None:
            self._action_indices.append(int(action_index))

        self._infos.append(info)

        self._step_count += 1

    def _evict_oldest(self) -> None:
        """Remove the oldest transition (FIFO eviction)."""
        self._observations.pop(0)
        self._next_observations.pop(0)
        self._rewards.pop(0)
        self._dones.pop(0)
        self._timestamps.pop(0)

        if self.store_actions:
            self._actions.pop(0)
        if self.store_action_indices:
            self._action_indices.pop(0)

        self._infos.pop(0)

    def get_recent_window(self, window_size: int) -> Dict[str, np.ndarray]:
        """Extract the most recent window of transitions.

        Args:
            window_size: Number of recent transitions to extract

        Returns:
            Dictionary containing:
                - 'observations': (window_size, *obs_shape)
                - 'next_observations': (window_size, *obs_shape)
                - 'rewards': (window_size,)
                - 'dones': (window_size,)
                - 'actions': (window_size, *act_shape) if stored
                - 'action_indices': (window_size,) if stored
                - 'timestamps': (window_size,)
        """
        if len(self) < window_size:
            raise ValueError(f"Buffer has only {len(self)} transitions, cannot extract window of size {window_size}")

        # Get last window_size transitions
        start_idx = len(self) - window_size
        end_idx = len(self)

        window = {
            'observations': np.array(self._observations[start_idx:end_idx], dtype=np.float32),
            'next_observations': np.array(self._next_observations[start_idx:end_idx], dtype=np.float32),
            'rewards': np.array(self._rewards[start_idx:end_idx], dtype=np.float32),
            'dones': np.array(self._dones[start_idx:end_idx], dtype=np.float32),
            'timestamps': np.array(self._timestamps[start_idx:end_idx], dtype=np.int64),
        }

        if self.store_actions:
            window['actions'] = np.array(self._actions[start_idx:end_idx], dtype=np.float32)
        if self.store_action_indices:
            window['action_indices'] = np.array(self._action_indices[start_idx:end_idx], dtype=np.int64)

        return window

    def get_window(self, start_idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """Extract a window of transitions starting at start_idx.

        Args:
            start_idx: Starting index in buffer
            window_size: Number of transitions to extract

        Returns:
            Dictionary of stacked transitions (same format as get_recent_window)
        """
        if start_idx < 0 or start_idx + window_size > len(self):
            raise ValueError(f"Invalid window: start_idx={start_idx}, window_size={window_size}, buffer_size={len(self)}")

        end_idx = start_idx + window_size

        window = {
            'observations': np.array(self._observations[start_idx:end_idx], dtype=np.float32),
            'next_observations': np.array(self._next_observations[start_idx:end_idx], dtype=np.float32),
            'rewards': np.array(self._rewards[start_idx:end_idx], dtype=np.float32),
            'dones': np.array(self._dones[start_idx:end_idx], dtype=np.float32),
            'timestamps': np.array(self._timestamps[start_idx:end_idx], dtype=np.int64),
        }

        if self.store_actions:
            window['actions'] = np.array(self._actions[start_idx:end_idx], dtype=np.float32)
        if self.store_action_indices:
            window['action_indices'] = np.array(self._action_indices[start_idx:end_idx], dtype=np.int64)

        return window

    def has_enough_for_chunk(self, chunk_size: int) -> bool:
        """Check if buffer has enough transitions to create a chunk.

        Args:
            chunk_size: Required number of transitions

        Returns:
            True if buffer size >= chunk_size
        """
        return len(self) >= chunk_size

    def clear(self) -> None:
        """Clear all stored transitions."""
        self._observations.clear()
        self._next_observations.clear()
        self._rewards.clear()
        self._dones.clear()
        self._timestamps.clear()

        if self.store_actions:
            self._actions.clear()
        if self.store_action_indices:
            self._action_indices.clear()

        self._infos.clear()

    def __len__(self) -> int:
        """Return number of transitions currently stored."""
        return len(self._observations)

    def __repr__(self) -> str:
        return f"ChronologicalBuffer(size={len(self)}/{self.max_capacity}, step={self._step_count})"
