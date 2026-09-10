"""Shared PPO/MAPPO networks, advantage calculation, and optimizer steps."""
import numpy as np
import torch
import torch.nn as nn

from agents.common.networks import Actor, Critic, make_mlp

__all__ = ["Actor", "Critic", "make_mlp", "compute_gae", "ppo_minibatch_step", "mean_update_metrics"]


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


def ppo_minibatch_step(agent, observations, critic_inputs, actions,
                       old_log_probs, advantages, returns) -> torch.Tensor:
    """Update the shared PPO objective; only the critic receives critic_inputs.

    Each caller retains its rollout packing, advantage normalization, and
    minibatch ordering. MAPPO supplies centralized, optionally agent-conditioned
    critic inputs while its actor continues to consume local observations.
    """
    # Score the actions actually collected, preserving PPO's importance ratio.
    log_probs, entropies = agent.actor.evaluate_actions(observations, actions)
    ratio = (log_probs - old_log_probs).exp()
    policy_loss = torch.max(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - agent.clip_range, 1 + agent.clip_range),
    ).mean()
    value_loss = nn.functional.mse_loss(agent.critic(critic_inputs), returns)
    entropy = entropies.mean()
    loss = policy_loss + agent.vf_coef * value_loss - agent.ent_coef * entropy
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent._optim_parameters, agent.max_grad_norm)
    agent.optimizer.step()
    with torch.no_grad():
        approx_kl = (old_log_probs - log_probs).mean().abs()
        return torch.stack((policy_loss, value_loss, entropy, approx_kl))


def mean_update_metrics(rows) -> dict[str, float]:
    """Transfer averaged update metrics to the CPU once, including empty updates."""
    names = ("train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl")
    means = torch.stack(rows).mean(dim=0).cpu().tolist() if rows else [0.0] * len(names)
    return dict(zip(names, means))
