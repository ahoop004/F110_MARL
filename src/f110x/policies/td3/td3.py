"""TD3 agent built on top of shared replay utilities."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# try:  # optional dependency for rich logging
#     import wandb  # type: ignore
# except ImportError:  # pragma: no cover - wandb optional
#     wandb = None

from f110x.policies.buffers import PrioritizedReplayBuffer, ReplayBuffer
from f110x.policies.td3.net import TD3Actor, TD3Critic, hard_update, soft_update
from f110x.utils.torch_io import resolve_device, safe_load


class TD3Agent:
    def __init__(self, cfg: Dict[str, Any]):
        self.device = resolve_device([cfg.get("device")])

        self.obs_dim = int(cfg["obs_dim"])
        self.act_dim = int(cfg["act_dim"])

        hidden_dims: Iterable[int] = cfg.get("hidden_dims", [256, 256])
        self.actor = TD3Actor(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        self.actor_target = TD3Actor(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        hard_update(self.actor_target, self.actor)

        self.critic1 = TD3Critic(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        self.critic2 = TD3Critic(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        self.critic_target1 = TD3Critic(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        self.critic_target2 = TD3Critic(self.obs_dim, self.act_dim, hidden_dims).to(self.device)
        hard_update(self.critic_target1, self.critic1)
        hard_update(self.critic_target2, self.critic2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.get("actor_lr", 1e-3)))
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(cfg.get("critic_lr", 1e-3)),
        )

        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.policy_noise = float(cfg.get("policy_noise", 0.2))
        self.noise_clip = float(cfg.get("noise_clip", 0.5))
        self.policy_delay = int(cfg.get("policy_delay", 2))
        self.batch_size = int(cfg.get("batch_size", 128))
        self.warmup_steps = int(cfg.get("warmup_steps", 1000))
        self.exploration_noise_initial = float(cfg.get("exploration_noise", 0.1))
        self.exploration_noise_final = float(
            cfg.get("exploration_noise_final", self.exploration_noise_initial)
        )
        self.exploration_noise_decay_steps = max(
            1, int(cfg.get("exploration_noise_decay_steps", 50_000))
        )
        self._exploration_step = 0
        try:
            decay_episodes_value = int(cfg.get("exploration_noise_decay_episodes", 0))
        except (TypeError, ValueError):
            decay_episodes_value = 0
        self.exploration_noise_decay_episodes = max(decay_episodes_value, 0)
        self._exploration_episode = 0

        buffer_size = int(cfg.get("buffer_size", 100_000))
        self.use_per = bool(cfg.get("use_per", False))
        if self.use_per:
            per_alpha = float(cfg.get("per_alpha", 0.6))
            per_beta = float(cfg.get("per_beta", 0.4))
            per_beta_increment = float(cfg.get("per_beta_increment", 1e-4))
            per_min_priority = float(cfg.get("per_min_priority", 1e-3))
            per_epsilon = float(cfg.get("per_epsilon", 1e-6))
            self.buffer = PrioritizedReplayBuffer(
                buffer_size,
                (self.obs_dim,),
                (self.act_dim,),
                alpha=per_alpha,
                beta=per_beta,
                beta_increment_per_sample=per_beta_increment,
                min_priority=per_min_priority,
                epsilon=per_epsilon,
            )
        else:
            self.buffer = ReplayBuffer(buffer_size, (self.obs_dim,), (self.act_dim,))

        action_low = np.asarray(cfg.get("action_low"), dtype=np.float32)
        action_high = np.asarray(cfg.get("action_high"), dtype=np.float32)
        if action_low.shape != (self.act_dim,) or action_high.shape != (self.act_dim,):
            raise ValueError("action_low/action_high must match act_dim")
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        self.total_it = 0

    # -------------------- Interaction API --------------------

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(obs_t).cpu().numpy().squeeze(0)
        self.actor.train()

        action = self._scale_action(action)
        if not deterministic:
            noise_scale = self._current_exploration_noise()
            if noise_scale > 0.0:
                noise = np.random.normal(0.0, noise_scale, size=self.act_dim)
                action = np.clip(action + noise, self.action_low, self.action_high)
            self._exploration_step += 1
        return action.astype(np.float32)

    def reset_noise_schedule(self, *, restart: bool = False) -> None:
        """Reset or advance the exploration-noise schedule."""
        if restart:
            self._exploration_step = 0
            self._exploration_episode = 0
            return

        if self.exploration_noise_decay_episodes > 0:
            self._exploration_step = 0
            self._exploration_episode = min(
                self._exploration_episode + 1,
                self.exploration_noise_decay_episodes,
            )

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, any]] = None,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, done, info)

    # -------------------- Learning --------------------

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None

        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        if self.use_per:
            weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)
            # Ensure weights broadcast across action dimension
            if weights.ndim == 1:
                weights = weights.view(-1, 1)
        else:
            weights = torch.ones_like(rewards, device=self.device)

        with torch.no_grad():
            # Policy smoothing regularization: perturb target action before evaluating target critics.
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_obs)
            next_action = self._scale_action_torch(next_action)
            next_action = (next_action + noise).clamp(
                torch.as_tensor(self.action_low, device=self.device),
                torch.as_tensor(self.action_high, device=self.device),
            )

            target_q1 = self.critic_target1(next_obs, next_action)
            target_q2 = self.critic_target2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        td_error1 = current_q1 - target
        td_error2 = current_q2 - target

        critic_loss = ((td_error1.pow(2) + td_error2.pow(2)) * weights).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_it % self.policy_delay == 0:
            actor_action = self._scale_action_torch(self.actor(obs))
            actor_loss = -self.critic1(obs, actor_action).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target1, self.critic1, self.tau)
            soft_update(self.critic_target2, self.critic2, self.tau)

        self.total_it += 1

        if self.use_per:
            td_errors = (td_error1.abs() + td_error2.abs()) * 0.5
            indices = batch["indices"]
            self.buffer.update_priorities(
                indices,
                td_errors.detach().cpu().squeeze(1).numpy(),
            )

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "update_it": float(self.total_it),
        }

    def _current_exploration_noise(self) -> float:
        if self.exploration_noise_decay_episodes > 0:
            frac = min(
                1.0,
                self._exploration_episode / float(self.exploration_noise_decay_episodes),
            )
        else:
            if self.exploration_noise_decay_steps <= 0:
                return self.exploration_noise_final
            frac = min(
                1.0,
                self._exploration_step / float(self.exploration_noise_decay_steps),
            )
        return (
            (1.0 - frac) * self.exploration_noise_initial
            + frac * self.exploration_noise_final
        )

    # -------------------- Persistence --------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic_target1": self.critic_target1.state_dict(),
                "critic_target2": self.critic_target2.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = safe_load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic_target1.load_state_dict(ckpt.get("critic_target1", ckpt["critic1"]))
        self.critic_target2.load_state_dict(ckpt.get("critic_target2", ckpt["critic2"]))
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.total_it = int(ckpt.get("total_it", 0))
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.critic_target1.to(self.device)
        self.critic_target2.to(self.device)

    # -------------------- Helpers --------------------

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, -1.0, 1.0) * (self.action_range / 2.0) + (self.action_low + self.action_high) / 2.0

    def _scale_action_torch(self, action: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.action_low, device=action.device)
        high = torch.as_tensor(self.action_high, device=action.device)
        range_half = (high - low) / 2.0
        mid = (high + low) / 2.0
        return torch.clamp(action, -1.0, 1.0) * range_half + mid
