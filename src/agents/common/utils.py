"""Common utility functions for RL agents."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class ActionScaler:
    """Helper for scaling actions from [-1, 1] to [action_low, action_high]."""

    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: Optional[torch.device] = None,
    ):
        """Initialize action scaler.

        Args:
            action_low: Minimum action values (shape: [act_dim])
            action_high: Maximum action values (shape: [act_dim])
            device: Torch device for tensor operations (default: CPU)
        """
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_range = self.action_high - self.action_low
        self.action_mid = (self.action_low + self.action_high) * 0.5
        self.action_half_range = self.action_range * 0.5

        self.device = device or torch.device("cpu")
        self.action_low_t = torch.as_tensor(self.action_low, device=self.device)
        self.action_high_t = torch.as_tensor(self.action_high, device=self.device)
        self.action_mid_t = torch.as_tensor(self.action_mid, device=self.device)
        self.action_half_range_t = torch.as_tensor(self.action_half_range, device=self.device)

    def scale_numpy(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to [action_low, action_high] (NumPy)."""
        return np.clip(action, -1.0, 1.0) * self.action_half_range + self.action_mid

    def scale_torch(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to [action_low, action_high] (PyTorch)."""
        return torch.clamp(action, -1.0, 1.0) * self.action_half_range_t + self.action_mid_t


class ExplorationNoiseSchedule:
    """Helper for scheduling exploration noise decay over episodes or steps."""

    def __init__(
        self,
        initial: float,
        final: float,
        decay_steps: int = 50_000,
        decay_episodes: int = 0,
    ):
        """Initialize exploration noise schedule.

        Args:
            initial: Initial noise scale
            final: Final noise scale
            decay_steps: Number of steps over which to decay noise (if decay_episodes=0)
            decay_episodes: Number of episodes over which to decay noise (overrides decay_steps if > 0)
        """
        self.initial = float(initial)
        self.final = float(final)
        self.decay_steps = max(1, int(decay_steps))
        self.decay_episodes = max(0, int(decay_episodes))
        self._current_step = 0
        self._current_episode = 0

    def get_noise(self) -> float:
        """Get current noise scale."""
        if self.decay_episodes > 0:
            frac = min(1.0, self._current_episode / float(self.decay_episodes))
        else:
            if self.decay_steps <= 0:
                return self.final
            frac = min(1.0, self._current_step / float(self.decay_steps))
        return (1.0 - frac) * self.initial + frac * self.final

    def step(self) -> None:
        """Increment step counter."""
        self._current_step += 1

    def step_episode(self) -> None:
        """Increment episode counter and reset step counter (if using episode-based decay)."""
        if self.decay_episodes > 0:
            self._current_step = 0
            self._current_episode = min(self._current_episode + 1, self.decay_episodes)

    def reset(self) -> None:
        """Reset both step and episode counters."""
        self._current_step = 0
        self._current_episode = 0


__all__ = [
    "ActionScaler",
    "ExplorationNoiseSchedule",
]
