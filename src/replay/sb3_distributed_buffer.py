"""SB3-compatible distributed replay buffer wrapper.

Wraps SB3's replay buffer to enable cross-sampling from parallel training runs.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
import torch as th

from src.replay.distributed_buffer import DistributedBufferRegistry


class DistributedReplayBuffer(ReplayBuffer):
    """SB3 ReplayBuffer with distributed experience sharing.

    Extends SB3's ReplayBuffer to:
    1. Add transitions to both local and distributed registry
    2. Sample from pool of distributed buffers
    3. Maintain compatibility with SB3 training loop

    Args:
        buffer_size: Local buffer size
        observation_space: Observation space
        action_space: Action space
        device: PyTorch device
        n_envs: Number of parallel environments
        optimize_memory_usage: SB3 optimization flag
        registry: Distributed buffer registry
        buffer_id: This buffer's ID in the registry
        sample_strategy: How to sample from distributed pool
            - 'uniform': Uniform across all buffers
            - 'weighted': Weighted by buffer size
            - 'self_heavy': 80% self, 20% others
            - 'local_only': Only use local buffer (disables distributed)
        cross_sample_ratio: Fraction of samples from distributed pool (0.0-1.0)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        # Distributed parameters
        registry: Optional[DistributedBufferRegistry] = None,
        buffer_id: Optional[str] = None,
        sample_strategy: str = 'self_heavy',
        cross_sample_ratio: float = 0.2,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        self.registry = registry
        self.buffer_id = buffer_id
        self.sample_strategy = sample_strategy
        self.cross_sample_ratio = cross_sample_ratio

        # Track stats
        self.local_samples = 0
        self.distributed_samples = 0

        # Enable/disable distributed sampling
        self.use_distributed = (
            registry is not None
            and buffer_id is not None
            and sample_strategy != 'local_only'
        )

        if self.use_distributed:
            print(f"✓ Distributed replay buffer enabled (strategy={sample_strategy}, "
                  f"cross_sample_ratio={cross_sample_ratio})")
        else:
            print("⚠ Distributed replay buffer disabled (using local only)")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Add transition to both local and distributed buffers."""
        # Add to local buffer (SB3 default behavior)
        super().add(obs, next_obs, action, reward, done, infos)

        # Also add to distributed registry
        if self.use_distributed and self.registry:
            # Convert to serializable format
            transition = (
                obs.copy(),
                action.copy(),
                float(reward[0]),
                next_obs.copy(),
                bool(done[0]),
            )
            self.registry.add_transition(self.buffer_id, transition)

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        """Sample from local or distributed pool based on cross_sample_ratio.

        Args:
            batch_size: Number of samples
            env: Optional VecNormalize environment

        Returns:
            Batch of samples in SB3 format
        """
        if not self.use_distributed or np.random.random() > self.cross_sample_ratio:
            # Sample locally
            self.local_samples += batch_size
            return super().sample(batch_size, env)

        # Sample from distributed pool
        return self._sample_distributed(batch_size, env)

    def _sample_distributed(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        """Sample from distributed buffer pool.

        Falls back to local sampling if distributed pool unavailable.
        """
        # Try to get samples from distributed pool
        transitions = self.registry.sample_from_pool(
            batch_size=batch_size,
            strategy=self.sample_strategy,
            self_buffer_id=self.buffer_id,
        )

        if transitions is None or len(transitions) == 0:
            # Fall back to local sampling
            self.local_samples += batch_size
            return super().sample(batch_size, env)

        # Convert to SB3 format
        self.distributed_samples += len(transitions)
        return self._transitions_to_replay_buffer_samples(transitions, env)

    def _transitions_to_replay_buffer_samples(
        self,
        transitions: List,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        """Convert distributed transitions to SB3 ReplayBufferSamples format."""
        # Unpack transitions
        observations = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([[t[2]] for t in transitions])
        next_observations = np.array([t[3] for t in transitions])
        dones = np.array([[float(t[4])] for t in transitions])

        # Normalize if needed
        if env is not None:
            observations = env.normalize_obs(observations)
            rewards = env.normalize_reward(rewards)
            next_observations = env.normalize_obs(next_observations)

        # Convert to tensors
        observations = self.to_torch(observations)
        actions = self.to_torch(actions)
        next_observations = self.to_torch(next_observations)
        rewards = self.to_torch(rewards).reshape(-1, 1)
        dones = self.to_torch(dones).reshape(-1, 1)

        return ReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics including distributed sampling."""
        total_samples = self.local_samples + self.distributed_samples
        stats = {
            'local_size': self.pos if not self.full else self.buffer_size,
            'local_samples': self.local_samples,
            'distributed_samples': self.distributed_samples,
            'total_samples': total_samples,
            'distributed_ratio': (
                self.distributed_samples / total_samples if total_samples > 0 else 0.0
            ),
        }

        if self.use_distributed and self.registry:
            buffer_stats = self.registry.get_buffer_stats()
            stats['registry_stats'] = buffer_stats
            stats['num_shared_buffers'] = len(buffer_stats)

        return stats


__all__ = ['DistributedReplayBuffer']
