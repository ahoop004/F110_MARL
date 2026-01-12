"""Prioritized replay buffer for SB3 off-policy algorithms."""

from typing import Any, Optional, NamedTuple

import numpy as np


try:
    import torch as th
    from stable_baselines3.common.buffers import ReplayBuffer
except ImportError as exc:  # pragma: no cover - optional dependency
    class PrioritizedReplayBuffer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PrioritizedReplayBuffer requires stable-baselines3. "
                "Install with: pip install stable-baselines3"
            ) from exc

    __all__ = ["PrioritizedReplayBuffer"]
else:
    class PrioritizedReplayBufferSamples(NamedTuple):
        observations: th.Tensor
        actions: th.Tensor
        next_observations: th.Tensor
        dones: th.Tensor
        rewards: th.Tensor
        discounts: Optional[th.Tensor]
        weights: th.Tensor
        indices: np.ndarray


    class PrioritizedReplayBuffer(ReplayBuffer):
        """Replay buffer with prioritized experience replay (PER).

        Sampling probability is proportional to priority**alpha, with optional
        importance-sampling weights controlled by beta.
        """

        def __init__(
            self,
            *args,
            alpha: float = 0.6,
            beta: float = 0.4,
            beta_final: float = 1.0,
            beta_anneal_steps: int = 100_000,
            eps: float = 1e-6,
            max_priority: float = 1.0,
            normalize_weights: bool = True,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.alpha = float(alpha)
            self.beta_start = float(beta)
            self.beta_final = float(beta_final)
            self.beta_anneal_steps = max(1, int(beta_anneal_steps))
            self.eps = float(eps)
            self.normalize_weights = bool(normalize_weights)
            self.max_priority = float(max_priority)

            self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
            self._sample_count = 0

        def add(self, *args, **kwargs) -> None:
            super().add(*args, **kwargs)
            insert_idx = (self.pos - 1) % self.buffer_size
            self.priorities[insert_idx] = self.max_priority

        def _current_beta(self) -> float:
            self._sample_count += 1
            fraction = min(1.0, self._sample_count / float(self.beta_anneal_steps))
            return self.beta_start + fraction * (self.beta_final - self.beta_start)

        def _get_probabilities(self, priorities: np.ndarray) -> np.ndarray:
            scaled = np.power(priorities, self.alpha)
            total = np.sum(scaled)
            if not np.isfinite(total) or total <= 0.0:
                scaled = np.ones_like(priorities, dtype=np.float32)
                total = float(np.sum(scaled))
            return scaled / total

        def sample(self, batch_size: int, env: Optional[Any] = None) -> PrioritizedReplayBufferSamples:
            if self.full:
                upper_bound = self.buffer_size
            else:
                upper_bound = self.pos

            if upper_bound == 0:
                raise ValueError("Cannot sample from an empty replay buffer.")

            if self.optimize_memory_usage and self.full:
                valid_indices = np.arange(self.buffer_size, dtype=np.int64)
                valid_indices = valid_indices[valid_indices != self.pos]
                priorities = self.priorities[valid_indices]
                probs = self._get_probabilities(priorities)
                replace = batch_size > valid_indices.shape[0]
                batch_inds = np.random.choice(valid_indices, size=batch_size, replace=replace, p=probs)
                prob_values = probs[np.searchsorted(valid_indices, batch_inds)]
                sample_size = valid_indices.shape[0]
            else:
                priorities = self.priorities[:upper_bound]
                probs = self._get_probabilities(priorities)
                replace = batch_size > upper_bound
                batch_inds = np.random.choice(upper_bound, size=batch_size, replace=replace, p=probs)
                prob_values = probs[batch_inds]
                sample_size = upper_bound

            beta = self._current_beta()
            weights = np.power(sample_size * prob_values, -beta)
            if self.normalize_weights and weights.size > 0:
                weights = weights / weights.max()
            weights = weights.reshape(-1, 1).astype(np.float32)

            replay_data = self._get_samples(batch_inds, env=env)
            weights_t = self.to_torch(weights)

            return PrioritizedReplayBufferSamples(
                observations=replay_data.observations,
                actions=replay_data.actions,
                next_observations=replay_data.next_observations,
                dones=replay_data.dones,
                rewards=replay_data.rewards,
                discounts=replay_data.discounts,
                weights=weights_t,
                indices=batch_inds,
            )

        def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
            if indices is None:
                return
            priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)
            if priorities.shape[0] != len(indices):
                raise ValueError("Priorities and indices must have the same length.")
            priorities = np.abs(priorities) + self.eps
            self.priorities[indices] = priorities
            current_max = float(np.max(priorities)) if priorities.size > 0 else 0.0
            if current_max > self.max_priority:
                self.max_priority = current_max


    __all__ = ["PrioritizedReplayBuffer", "PrioritizedReplayBufferSamples"]
