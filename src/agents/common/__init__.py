"""Shared PPO/MAPPO networks and advantage calculation."""
import numpy as np
import torch

from agents.common.networks import Actor, Critic, make_mlp

__all__ = ["Actor", "Critic", "make_mlp", "compute_gae"]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the Python-double recurrence, but transfer CUDA scalars in bulk.
    rollout = torch.stack((rewards, values, terminated, truncated), dim=1).cpu().numpy()
    advantages_host = np.zeros(len(rollout), dtype=np.float32)
    last_gae = 0.0
    next_val = float(next_value)
    for t in reversed(range(len(rollout))):
        terminal = float(rollout[t, 2])
        truncation = float(rollout[t, 3])
        # Truncations bootstrap, but both boundaries stop cross-episode GAE.
        bootstrap_mask = 1.0 - terminal
        continuation_mask = 1.0 - float(bool(terminal) or bool(truncation))
        nv = next_val if t == len(rollout) - 1 else float(rollout[t + 1, 1])
        delta = float(rollout[t, 0]) + gamma * nv * bootstrap_mask - float(rollout[t, 1])
        last_gae = delta + gamma * gae_lambda * continuation_mask * last_gae
        advantages_host[t] = last_gae
    advantages = torch.as_tensor(advantages_host, dtype=torch.float32, device=values.device)
    return advantages, advantages + values
