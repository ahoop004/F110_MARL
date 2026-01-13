"""Distributed replay buffer sharing across parallel training runs.

Allows multiple parallel training processes to share experiences by cross-sampling
from each other's replay buffers. This improves sample efficiency and exploration.
"""

import pickle
import time
from collections import deque
from multiprocessing import Manager, Lock
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import threading


class DistributedBufferRegistry:
    """Registry for shared replay buffers across parallel training runs.

    Manages a pool of replay buffers from different training runs, allowing
    cross-sampling. Automatically removes buffers from terminated runs.

    Features:
    - Thread-safe buffer registration/deregistration
    - Automatic cleanup of dead buffers
    - Sampling from multiple buffers with configurable strategies
    - Buffer health monitoring

    Usage:
        # In main process
        registry = DistributedBufferRegistry()
        registry.start()

        # In each training process
        buffer_id = registry.register_buffer(run_id="run_1")

        # During training
        registry.add_transition(buffer_id, transition)
        transitions = registry.sample_from_pool(batch_size=256, buffer_ids=[buffer_id, ...])

        # When run completes
        registry.deregister_buffer(buffer_id)
    """

    def __init__(
        self,
        max_buffer_size: int = 1000000,
        min_buffer_size: int = 1000,
        cleanup_interval: int = 30,
    ):
        """Initialize distributed buffer registry.

        Args:
            max_buffer_size: Maximum size per buffer
            min_buffer_size: Minimum transitions before sampling
            cleanup_interval: Seconds between cleanup checks
        """
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.cleanup_interval = cleanup_interval

        # Shared state using multiprocessing Manager
        self.manager = Manager()
        self.buffers = self.manager.dict()  # buffer_id -> deque of transitions
        self.metadata = self.manager.dict()  # buffer_id -> {run_id, created_at, last_active, size}
        self.lock = self.manager.Lock()

        # Cleanup thread
        self._cleanup_thread = None
        self._running = False

    def start(self):
        """Start the registry cleanup thread."""
        if self._running:
            return

        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        print("✓ Distributed buffer registry started")

    def stop(self):
        """Stop the registry and cleanup."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        print("✓ Distributed buffer registry stopped")

    def register_buffer(
        self,
        run_id: str,
        algorithm: Optional[str] = None,
    ) -> str:
        """Register a new buffer for a training run.

        Args:
            run_id: Unique identifier for this training run
            algorithm: Algorithm name (for logging)

        Returns:
            buffer_id: Unique buffer identifier
        """
        buffer_id = f"{run_id}_{int(time.time() * 1000)}"

        with self.lock:
            self.buffers[buffer_id] = self.manager.list()
            self.metadata[buffer_id] = {
                'run_id': run_id,
                'algorithm': algorithm,
                'created_at': time.time(),
                'last_active': time.time(),
                'size': 0,
                'total_added': 0,
            }

        print(f"✓ Registered buffer {buffer_id} for run {run_id}")
        return buffer_id

    def deregister_buffer(self, buffer_id: str):
        """Remove a buffer from the registry.

        Args:
            buffer_id: Buffer to remove
        """
        with self.lock:
            if buffer_id in self.buffers:
                del self.buffers[buffer_id]
            if buffer_id in self.metadata:
                del self.metadata[buffer_id]

        print(f"✓ Deregistered buffer {buffer_id}")

    def add_transition(
        self,
        buffer_id: str,
        transition: Tuple[Any, Any, float, Any, bool],
    ):
        """Add a transition to a specific buffer.

        Args:
            buffer_id: Target buffer
            transition: (obs, action, reward, next_obs, done)
        """
        with self.lock:
            if buffer_id not in self.buffers:
                return

            buffer = self.buffers[buffer_id]
            buffer.append(transition)

            # Enforce max size
            if len(buffer) > self.max_buffer_size:
                buffer.pop(0)

            # Update metadata
            meta = dict(self.metadata[buffer_id])
            meta['last_active'] = time.time()
            meta['size'] = len(buffer)
            meta['total_added'] = meta.get('total_added', 0) + 1
            self.metadata[buffer_id] = meta

    def sample_from_pool(
        self,
        batch_size: int,
        buffer_ids: Optional[List[str]] = None,
        strategy: str = 'uniform',
        exclude_self: bool = False,
        self_buffer_id: Optional[str] = None,
    ) -> Optional[List[Tuple[Any, Any, float, Any, bool]]]:
        """Sample transitions from a pool of buffers.

        Args:
            batch_size: Number of transitions to sample
            buffer_ids: Specific buffers to sample from (None = all active)
            strategy: Sampling strategy:
                - 'uniform': Sample uniformly from all buffers
                - 'weighted': Weight by buffer size
                - 'self_heavy': 80% from own buffer, 20% from others
                - 'newest': Prefer newer transitions
            exclude_self: Don't sample from own buffer
            self_buffer_id: Own buffer ID (required if exclude_self=True)

        Returns:
            List of sampled transitions, or None if not enough data
        """
        with self.lock:
            # Determine which buffers to use
            if buffer_ids is None:
                available_buffers = list(self.buffers.keys())
            else:
                available_buffers = [bid for bid in buffer_ids if bid in self.buffers]

            # Exclude self if requested
            if exclude_self and self_buffer_id:
                available_buffers = [bid for bid in available_buffers if bid != self_buffer_id]

            if not available_buffers:
                return None

            # Check if we have enough transitions
            total_transitions = sum(len(self.buffers[bid]) for bid in available_buffers)
            if total_transitions < max(batch_size, self.min_buffer_size):
                return None

            # Sample based on strategy
            if strategy == 'uniform':
                samples = self._sample_uniform(available_buffers, batch_size)
            elif strategy == 'weighted':
                samples = self._sample_weighted(available_buffers, batch_size)
            elif strategy == 'self_heavy' and self_buffer_id:
                samples = self._sample_self_heavy(available_buffers, batch_size, self_buffer_id)
            elif strategy == 'newest':
                samples = self._sample_newest(available_buffers, batch_size)
            else:
                samples = self._sample_uniform(available_buffers, batch_size)

            return samples

    def _sample_uniform(self, buffer_ids: List[str], batch_size: int) -> List[Any]:
        """Sample uniformly across all buffers."""
        all_transitions = []
        for bid in buffer_ids:
            all_transitions.extend(list(self.buffers[bid]))

        if len(all_transitions) < batch_size:
            return all_transitions

        indices = np.random.choice(len(all_transitions), batch_size, replace=False)
        return [all_transitions[i] for i in indices]

    def _sample_weighted(self, buffer_ids: List[str], batch_size: int) -> List[Any]:
        """Sample weighted by buffer size."""
        buffer_sizes = [len(self.buffers[bid]) for bid in buffer_ids]
        total_size = sum(buffer_sizes)

        if total_size == 0:
            return []

        # Calculate samples per buffer
        samples_per_buffer = [
            int(batch_size * size / total_size) for size in buffer_sizes
        ]

        # Handle rounding
        remaining = batch_size - sum(samples_per_buffer)
        for i in range(remaining):
            samples_per_buffer[i % len(samples_per_buffer)] += 1

        # Sample from each buffer
        all_samples = []
        for bid, n_samples in zip(buffer_ids, samples_per_buffer):
            if n_samples == 0:
                continue
            buffer = list(self.buffers[bid])
            if len(buffer) <= n_samples:
                all_samples.extend(buffer)
            else:
                indices = np.random.choice(len(buffer), n_samples, replace=False)
                all_samples.extend([buffer[i] for i in indices])

        return all_samples

    def _sample_self_heavy(
        self,
        buffer_ids: List[str],
        batch_size: int,
        self_buffer_id: str,
    ) -> List[Any]:
        """Sample 80% from own buffer, 20% from others."""
        if self_buffer_id not in buffer_ids:
            return self._sample_uniform(buffer_ids, batch_size)

        self_samples = int(batch_size * 0.8)
        other_samples = batch_size - self_samples

        samples = []

        # Sample from own buffer
        self_buffer = list(self.buffers[self_buffer_id])
        if len(self_buffer) > 0:
            n = min(self_samples, len(self_buffer))
            indices = np.random.choice(len(self_buffer), n, replace=False)
            samples.extend([self_buffer[i] for i in indices])

        # Sample from others
        other_buffers = [bid for bid in buffer_ids if bid != self_buffer_id]
        if other_buffers and other_samples > 0:
            other_sample = self._sample_uniform(other_buffers, other_samples)
            samples.extend(other_sample)

        return samples

    def _sample_newest(self, buffer_ids: List[str], batch_size: int) -> List[Any]:
        """Sample most recent transitions across buffers."""
        # Collect recent transitions (last 20% of each buffer)
        recent_transitions = []
        for bid in buffer_ids:
            buffer = list(self.buffers[bid])
            n_recent = max(1, len(buffer) // 5)
            recent_transitions.extend(buffer[-n_recent:])

        if len(recent_transitions) <= batch_size:
            return recent_transitions

        indices = np.random.choice(len(recent_transitions), batch_size, replace=False)
        return [recent_transitions[i] for i in indices]

    def get_buffer_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active buffers.

        Returns:
            Dict mapping buffer_id to stats dict
        """
        with self.lock:
            stats = {}
            for buffer_id in self.buffers.keys():
                meta = dict(self.metadata.get(buffer_id, {}))
                meta['current_size'] = len(self.buffers[buffer_id])
                stats[buffer_id] = meta
            return stats

    def _cleanup_loop(self):
        """Background thread to cleanup stale buffers."""
        while self._running:
            time.sleep(self.cleanup_interval)
            self._cleanup_stale_buffers()

    def _cleanup_stale_buffers(self, timeout: float = 300.0):
        """Remove buffers that haven't been active recently.

        Args:
            timeout: Seconds of inactivity before considering buffer stale
        """
        with self.lock:
            current_time = time.time()
            stale_buffers = []

            for buffer_id, meta in self.metadata.items():
                if current_time - meta.get('last_active', 0) > timeout:
                    stale_buffers.append(buffer_id)

            for buffer_id in stale_buffers:
                print(f"⚠ Cleaning up stale buffer {buffer_id}")
                if buffer_id in self.buffers:
                    del self.buffers[buffer_id]
                if buffer_id in self.metadata:
                    del self.metadata[buffer_id]


def create_distributed_registry(
    shared_dir: Optional[Path] = None,
    max_buffer_size: int = 1000000,
) -> DistributedBufferRegistry:
    """Create and start a distributed buffer registry.

    Args:
        shared_dir: Optional directory for persistent state
        max_buffer_size: Maximum transitions per buffer

    Returns:
        Started registry instance
    """
    registry = DistributedBufferRegistry(max_buffer_size=max_buffer_size)
    registry.start()
    return registry


__all__ = [
    'DistributedBufferRegistry',
    'create_distributed_registry',
]
