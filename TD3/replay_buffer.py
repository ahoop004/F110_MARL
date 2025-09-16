# replay_buffer.py
# Prioritized Replay Buffer (ring buffer, array-backed) for TD3.
# Returns batches as dicts matching agents.py expectations.

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

class PrioritizedReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        priority_eps: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(priority_eps)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)  # normalized actions in [-1,1]
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)          # 1.0 if terminal else 0.0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, s: np.ndarray, a_norm: np.ndarray, r: float, s2: np.ndarray, d: bool, priority: Optional[float] = None) -> None:
        idx = self.ptr
        self.obs[idx] = s
        self.actions[idx] = a_norm
        self.rewards[idx, 0] = float(r)
        self.next_obs[idx] = s2
        self.dones[idx, 0] = 1.0 if d else 0.0

        if priority is None or not np.isfinite(priority):
            # default to max priority so new samples are learnable
            p = self.priorities.max() if self.size > 0 else 1.0
        else:
            p = float(priority)
        self.priorities[idx] = max(p, self.eps)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        assert self.size > 0, "Cannot sample from an empty buffer."
        N = self.size
        ps = self.priorities[:N].astype(np.float32)

        # compute sampling probabilities with stability guards
        ps_alpha = np.power(ps + self.eps, self.alpha, dtype=np.float32)
        denom = ps_alpha.sum()
        if not np.isfinite(denom) or denom <= 0.0:
            probs = np.full(N, 1.0 / N, dtype=np.float32)
        else:
            probs = ps_alpha / denom

        idxs = self.rng.choice(N, size=int(batch_size), replace=False, p=probs)

        # importance sampling weights
        beta = float(beta)
        weights = np.power(N * probs[idxs], -beta, dtype=np.float32)
        weights /= weights.max().clip(min=1e-12)

        batch = {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }
        return batch, idxs.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, new_priorities: np.ndarray) -> None:
        idxs = idxs.astype(np.int64, copy=False)
        p = np.asarray(new_priorities, dtype=np.float32)
        p = np.where(np.isfinite(p), p, 0.0)
        self.priorities[idxs] = np.maximum(p, self.eps)
