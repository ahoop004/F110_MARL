"""Pure PyTorch PPO agent."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.common.networks import Actor, Critic
from utils.torch_io import resolve_device


class RolloutBuffer:
    """Fixed-capacity on-policy rollout buffer with GAE computation."""

    def __init__(self, n_steps: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.n_steps = n_steps
        self.device = device
        self.obs = torch.zeros(n_steps, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, action_dim, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        i = self.ptr % self.n_steps
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[i] = float(reward)
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)
        self.dones[i] = float(done)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def clear(self) -> None:
        self.ptr = 0

    def size(self) -> int:
        return min(self.ptr, self.n_steps)

    def compute_gae(
        self,
        next_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.size()
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0
        next_val = float(next_value)

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - float(self.dones[t])
            nv = next_val if t == n - 1 else float(self.values[t + 1])
            delta = float(self.rewards[t]) + gamma * nv * next_non_terminal - float(self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:n]
        return advantages, returns

    def iterate_batches(
        self, batch_size: int
    ):
        n = self.size()
        indices = torch.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                self.obs[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.values[idx],
            ), idx


class PPOAgent:
    """Proximal Policy Optimization — pure PyTorch, no SB3."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        params: Dict,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = len(self.action_low)

        # Hyperparameters (merged from training_defaults + scenario params)
        self.lr = float(params.get("learning_rate", 3e-4))
        self.gamma = float(params.get("gamma", 0.99))
        self.gae_lambda = float(params.get("gae_lambda", 0.95))
        self.clip_range = float(params.get("clip_range", 0.2))
        self.ent_coef = float(params.get("ent_coef", 0.01))
        self.vf_coef = float(params.get("vf_coef", 0.5))
        self.max_grad_norm = float(params.get("max_grad_norm", 0.5))
        self.n_steps = int(params.get("n_steps", 2048))
        self.n_epochs = int(params.get("n_epochs", 10))
        self.batch_size = int(params.get("batch_size", 64))

        hidden_dims: List[int] = list(
            params.get("pi_hidden_dims", params.get("hidden_dims", [64, 64]))
        )
        vf_dims: List[int] = list(
            params.get("vf_hidden_dims", params.get("hidden_dims", [64, 64]))
        )
        activation: str = str(params.get("activation", "tanh"))
        self.actor_hidden_dims = list(hidden_dims)
        self.activation = activation

        device_str = str(params.get("device", "cpu"))
        self.device = resolve_device([device_str])

        self.actor = Actor(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.critic = Critic(obs_dim, vf_dims, activation).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
        )

        self.buffer = RolloutBuffer(self.n_steps, obs_dim, self.action_dim, self.device)

    # ------------------------------------------------------------------
    # Denormalize / normalize helpers
    # ------------------------------------------------------------------

    def _denormalize(self, action_norm: np.ndarray) -> np.ndarray:
        """Map [-1, 1] → physical action space."""
        mid = (self.action_high + self.action_low) / 2.0
        half = (self.action_high - self.action_low) / 2.0
        return mid + half * np.clip(action_norm, -1.0, 1.0)

    def _normalize(self, action_phys: np.ndarray) -> np.ndarray:
        mid = (self.action_high + self.action_low) / 2.0
        half = (self.action_high - self.action_low) / 2.0
        return np.clip((action_phys - mid) / half, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action from policy.

        Returns:
            (action_normalized, log_prob, value)
            action_normalized is in [-1, 1] — caller denormalizes for env.step()
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, log_prob_t = self.actor.get_action(obs_t, deterministic=deterministic)
        value_t = self.critic(obs_t)
        return (
            action_t.squeeze(0).cpu().numpy(),
            float(log_prob_t.squeeze()),
            float(value_t.squeeze()),
        )

    def update(self, next_value: float, done: bool) -> Dict[str, float]:
        """Compute GAE and run PPO update epochs.

        Returns dict of training metrics (empty dict if buffer too small to update).
        """
        if self.buffer.size() < 2:
            return {}

        advantages, returns = self.buffer.compute_gae(
            next_value if not done else 0.0,
            self.gamma,
            self.gae_lambda,
        )
        # Normalize advantages — use correction=0 so std is always valid for n>=1
        adv_std = advantages.std(correction=0)
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        n = self.buffer.size()
        total_pi_loss = total_vf_loss = total_ent = total_kl = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for (obs_b, act_b, old_lp_b, _), idx in self.buffer.iterate_batches(self.batch_size):
                adv_b = advantages[idx]
                ret_b = returns[idx]

                action_t, new_lp_t = self.actor.get_action(obs_b)
                value_pred = self.critic(obs_b)

                ratio = (new_lp_t - old_lp_b).exp()
                pi_loss_1 = -adv_b * ratio
                pi_loss_2 = -adv_b * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
                pi_loss = torch.max(pi_loss_1, pi_loss_2).mean()

                vf_loss = nn.functional.mse_loss(value_pred, ret_b)

                _, std = self.actor(obs_b)
                dist = torch.distributions.Normal(action_t, std)
                entropy = dist.entropy().sum(-1).mean()

                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((old_lp_b - new_lp_t).mean()).abs().item()

                total_pi_loss += pi_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += entropy.item()
                total_kl += approx_kl
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            "train/policy_loss": total_pi_loss / denom,
            "train/value_loss": total_vf_loss / denom,
            "train/entropy": total_ent / denom,
            "train/approx_kl": total_kl / denom,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "algorithm": "ppo",
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "actor_hidden_dims": self.actor_hidden_dims,
                "activation": self.activation,
            },
            path,
        )

    def load(self, path: str) -> None:
        from utils.torch_io import safe_load
        ckpt = safe_load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
