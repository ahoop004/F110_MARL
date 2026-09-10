"""Pure PyTorch PPO agent."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.common import Actor, Critic, compute_gae
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
        self.terminated = torch.zeros(n_steps, device=device)
        self.truncated = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        i = self.ptr % self.n_steps
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[i] = float(reward)
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)
        self.terminated[i] = float(terminated)
        self.truncated[i] = float(truncated)
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
        return compute_gae(
            self.rewards[:n], self.values[:n], self.terminated[:n], self.truncated[:n],
            next_value, gamma, gae_lambda,
        )


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
        self._optim_parameters = tuple(self.actor.parameters()) + tuple(self.critic.parameters())
        self.optimizer = optim.Adam(self._optim_parameters, lr=self.lr)

        self.buffer = RolloutBuffer(self.n_steps, obs_dim, self.action_dim, self.device)

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic evaluation action without evaluating the critic."""
        obs_t = torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32, device=self.device)
        return torch.tanh(self.actor.net(obs_t))[0].cpu().numpy()

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action from policy.

        Returns:
            (action_normalized, log_prob, value)
            action_normalized is in [-1, 1] — caller denormalizes for env.step()
        """
        actions, log_probs, values = self.act_batch(np.asarray(obs)[None], deterministic)
        return actions[0], float(log_probs[0]), float(values[0])

    @torch.no_grad()
    def act_batch(self, observations: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions, log_probs = self.actor.get_action(obs_t, deterministic=deterministic)
        values = self.critic(obs_t)
        # One device-to-host transfer for all environments and policy outputs.
        outputs = torch.cat((actions, log_probs[:, None], values[:, None]), dim=1).cpu().numpy()
        return outputs[:, :self.action_dim], outputs[:, -2], outputs[:, -1]

    def update(self, next_value: float) -> Dict[str, float]:
        """Compute GAE and run PPO update epochs.

        Returns dict of training metrics (empty dict if buffer too small to update).
        """
        if self.buffer.size() < 1:
            return {}

        advantages, returns = self.buffer.compute_gae(
            next_value,
            self.gamma,
            self.gae_lambda,
        )
        n = self.buffer.size()
        return self._update(
            self.buffer.obs[:n], self.buffer.actions[:n], self.buffer.log_probs[:n],
            advantages, returns,
        )

    def update_rollouts(self, rollouts) -> Dict[str, float]:
        """Pool independently bootstrapped worker rollouts from one frozen policy."""
        tensors = [
            torch.as_tensor(np.concatenate(parts), dtype=torch.float32, device=self.device)
            for parts in zip(*rollouts)
        ]
        return self._update(*tensors)

    def _update(self, observations, actions, log_probs, advantages, returns) -> Dict[str, float]:
        # Normalize advantages — use correction=0 so std is always valid for n>=1
        adv_std = advantages.std(correction=0)
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        n = len(observations)
        metric_rows = []
        for _ in range(self.n_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                obs_b, act_b, old_lp_b = observations[idx], actions[idx], log_probs[idx]
                adv_b = advantages[idx]
                ret_b = returns[idx]

                # PPO's importance ratio must score the exact action collected
                # during rollout; sampling a replacement action makes the ratio
                # unrelated to the stored transition.
                new_lp_t, entropy_t = self.actor.evaluate_actions(obs_b, act_b)
                value_pred = self.critic(obs_b)

                ratio = (new_lp_t - old_lp_b).exp()
                pi_loss_1 = -adv_b * ratio
                pi_loss_2 = -adv_b * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
                pi_loss = torch.max(pi_loss_1, pi_loss_2).mean()

                vf_loss = nn.functional.mse_loss(value_pred, ret_b)

                entropy = entropy_t.mean()

                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._optim_parameters,
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((old_lp_b - new_lp_t).mean()).abs()
                    metric_rows.append(torch.stack((pi_loss, vf_loss, entropy, approx_kl)))

        names = ("train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl")
        means = torch.stack(metric_rows).mean(dim=0).cpu().tolist() if metric_rows else [0.0] * len(names)
        return dict(zip(names, means))

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
