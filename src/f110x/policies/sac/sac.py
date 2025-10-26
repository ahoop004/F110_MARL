"""Soft Actor-Critic agent implementation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from f110x.policies.common import build_replay_buffer, sample_continuous_replay
from f110x.policies.sac.net import GaussianPolicy, QNetwork, hard_update, soft_update
from f110x.utils.torch_io import resolve_device, safe_load


class SACAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.device = resolve_device([cfg.get("device")])

        self.obs_dim = int(cfg["obs_dim"])
        self.act_dim = int(cfg["act_dim"])

        hidden_dims: Iterable[int] = cfg.get("hidden_dims", [256, 256])
        self.actor = GaussianPolicy(self.obs_dim, self.act_dim, hidden_dims).to(self.device)

        self.q1 = QNetwork(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        self.q1_target = QNetwork(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        self.q2_target = QNetwork(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.get("actor_lr", 3e-4)))
        critic_lr = float(cfg.get("critic_lr", 3e-4))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.batch_size = int(cfg.get("batch_size", 256))
        self.buffer_size = int(cfg.get("buffer_size", 1_000_000))
        self.warmup_steps = int(cfg.get("warmup_steps", 10_000))

        self.auto_alpha = bool(cfg.get("auto_alpha", True))
        init_alpha = float(cfg.get("alpha", 0.2))
        self.alpha_lr = float(cfg.get("alpha_lr", cfg.get("actor_lr", 3e-4)))
        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            target_entropy = cfg.get("target_entropy")
            if target_entropy is None:
                target_entropy = -float(self.act_dim)
            self.target_entropy = float(target_entropy)
            self._alpha_constant = None
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.target_entropy = None
            self._alpha_constant = float(init_alpha)

        self.squash_eps = 1e-6

        self.buffer, self.use_per = build_replay_buffer(
            cfg,
            self.obs_dim,
            self.act_dim,
            store_actions=True,
            store_action_indices=False,
            per_flag_key="use_per",
            default_prioritized=False,
        )

        action_low = np.asarray(cfg.get("action_low"), dtype=np.float32)
        action_high = np.asarray(cfg.get("action_high"), dtype=np.float32)
        if action_low.shape != (self.act_dim,) or action_high.shape != (self.act_dim,):
            raise ValueError("SAC requires action_low/action_high vectors matching act_dim")
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        self.total_it = 0

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, log_std = self.actor(obs_t)
            if deterministic:
                raw_action = mu
            else:
                std = log_std.exp()
                dist = Normal(mu, std)
                raw_action = dist.rsample()
            squashed = torch.tanh(raw_action)
            action = self._scale_action_torch(squashed)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, done, info)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None

        sample = sample_continuous_replay(self.buffer, self.batch_size, self.device)
        obs = sample.obs
        actions = sample.actions
        rewards = sample.rewards
        next_obs = sample.next_obs
        dones = sample.dones

        # Critic update -------------------------------------------------
        with torch.no_grad():
            next_mu, next_log_std = self.actor(next_obs)
            next_std = next_log_std.exp()
            next_dist = Normal(next_mu, next_std)
            next_raw = next_dist.rsample()
            next_squashed = torch.tanh(next_raw)
            next_action = self._scale_action_torch(next_squashed)

            log_prob_next = next_dist.log_prob(next_raw)
            log_prob_next -= torch.log(1 - next_squashed.pow(2) + self.squash_eps)
            log_prob_next = log_prob_next.sum(dim=-1, keepdim=True)

            target_q1 = self.q1_target(next_obs, next_action)
            target_q2 = self.q2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)

            alpha = self._current_alpha(detach=True)
            target_v = target_q - alpha * log_prob_next
            target = rewards + (1.0 - dones) * self.gamma * target_v

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.q1_opt.zero_grad(set_to_none=True)
        self.q2_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

        # Actor update --------------------------------------------------
        mu, log_std = self.actor(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)
        action = self._scale_action_torch(squashed_action)

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - squashed_action.pow(2) + self.squash_eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        q1_pi = self.q1(obs, action)
        q2_pi = self.q2(obs, action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        alpha = self._current_alpha(detach=True)
        actor_loss = (alpha * log_prob - min_q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature update -------------------------------------------
        alpha_value = self._current_alpha(detach=False)
        alpha_loss_value = 0.0
        if self.auto_alpha and self.alpha_opt is not None and self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.detach().cpu().item())
            alpha_value = self.log_alpha.exp()

        # Target networks ----------------------------------------------
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

        self.total_it += 1

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": float(alpha_value.detach().cpu().item()),
            "alpha_loss": float(alpha_loss_value),
            "update_it": float(self.total_it),
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "q1_opt": self.q1_opt.state_dict(),
                "q2_opt": self.q2_opt.state_dict(),
                "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu().item(),
                "alpha_opt": None if self.alpha_opt is None else self.alpha_opt.state_dict(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = safe_load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt.get("q1_target", ckpt["q1"]))
        self.q2_target.load_state_dict(ckpt.get("q2_target", ckpt["q2"]))
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q1_opt.load_state_dict(ckpt["q1_opt"])
        self.q2_opt.load_state_dict(ckpt["q2_opt"])
        log_alpha_val = ckpt.get("log_alpha")
        if self.auto_alpha and log_alpha_val is not None and self.log_alpha is not None:
            self.log_alpha.data = torch.tensor(float(log_alpha_val), dtype=torch.float32, device=self.device)
            if ckpt.get("alpha_opt"):
                self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self.total_it = int(ckpt.get("total_it", 0))
        self.actor.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.q1_target.to(self.device)
        self.q2_target.to(self.device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scale_action_torch(self, squashed: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.action_low, device=squashed.device)
        high = torch.as_tensor(self.action_high, device=squashed.device)
        midpoint = 0.5 * (low + high)
        range_half = 0.5 * (high - low)
        return squashed.clamp(-1.0, 1.0) * range_half + midpoint

    def _current_alpha(self, *, detach: bool) -> torch.Tensor:
        if self.auto_alpha and self.log_alpha is not None:
            alpha = self.log_alpha.exp()
        else:
            alpha = torch.tensor(self._alpha_constant, dtype=torch.float32, device=self.device)
        if detach:
            alpha = alpha.detach()
        return alpha
