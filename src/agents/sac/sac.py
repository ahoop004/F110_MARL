"""Soft Actor-Critic agent implementation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from agents.common import build_replay_buffer, sample_mixed_continuous_replay
from agents.buffers import ReplayBuffer
from agents.sac.net import GaussianPolicy, QNetwork, hard_update, soft_update
from utils.torch_io import resolve_device, safe_load


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

        self.actor_lr = float(cfg.get("actor_lr", cfg.get("lr_actor", 3e-4)))
        self.critic_lr = float(cfg.get("critic_lr", cfg.get("lr_critic", 3e-4)))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.critic_lr)

        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.batch_size = int(cfg.get("batch_size", 256))
        self.buffer_size = int(cfg.get("buffer_size", 1_000_000))
        self.warmup_steps = int(cfg.get("warmup_steps", cfg.get("learning_starts", 10_000)))

        self.auto_alpha = bool(cfg.get("auto_alpha", True))
        init_alpha = float(cfg.get("alpha", 0.2))
        self.alpha_lr = float(cfg.get("alpha_lr", cfg.get("lr_alpha", self.actor_lr)))
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

        # Prioritized Experience Replay (PER) is disabled by default.
        # Enable with use_per: true in config for potentially faster learning.
        # Trade-off: Higher memory usage and slower sampling vs better sample efficiency.
        self.buffer, self.use_per = build_replay_buffer(
            cfg,
            self.obs_dim,
            self.act_dim,
            store_actions=True,
            store_action_indices=False,
            per_flag_key="use_per",
            default_prioritized=False,
        )
        ratio_raw = cfg.get("success_buffer_ratio", 0.0)
        try:
            ratio_value = float(ratio_raw)
        except (TypeError, ValueError):
            ratio_value = 0.0
        self.success_buffer_ratio = min(max(ratio_value, 0.0), 1.0)
        success_capacity = int(cfg.get("success_buffer_size", 0) or 0)
        self.success_buffer: Optional[ReplayBuffer] = None
        if success_capacity > 0 and self.success_buffer_ratio > 0.0:
            self.success_buffer = ReplayBuffer(
                success_capacity,
                (self.obs_dim,),
                (self.act_dim,),
                store_actions=True,
                store_action_indices=False,
            )
        def _coerce_float_list(value: Any, default: Iterable[float]) -> list[float]:
            if value is None:
                return list(default)
            if isinstance(value, (int, float)):
                return [float(value)]
            if isinstance(value, (list, tuple)):
                items: list[float] = []
                for item in value:
                    try:
                        items.append(float(item))
                    except (TypeError, ValueError):
                        continue
                return items if items else list(default)
            return list(default)

        her_thresholds = _coerce_float_list(cfg.get("her_thresholds"), [0.6, 1.0, 1.5])
        her_bonuses = _coerce_float_list(cfg.get("her_bonuses"), [100.0, 50.0, 20.0])
        if not her_thresholds or not her_bonuses:
            self._her_pairs: list[tuple[float, float]] = []
        else:
            if len(her_bonuses) < len(her_thresholds):
                her_bonuses = her_bonuses + [her_bonuses[-1]] * (len(her_thresholds) - len(her_bonuses))
            elif len(her_bonuses) > len(her_thresholds):
                her_bonuses = her_bonuses[:len(her_thresholds)]
            self._her_pairs = sorted(zip(her_thresholds, her_bonuses), key=lambda pair: pair[0])

        action_low = np.asarray(cfg.get("action_low"), dtype=np.float32)
        action_high = np.asarray(cfg.get("action_high"), dtype=np.float32)
        if action_low.shape != (self.act_dim,) or action_high.shape != (self.act_dim,):
            raise ValueError("SAC requires action_low/action_high vectors matching act_dim")
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Prevent reverse configuration
        self.prevent_reverse = bool(cfg.get("prevent_reverse", False))
        self.prevent_reverse_min_speed = float(cfg.get("prevent_reverse_min_speed", 0.01))
        self.prevent_reverse_speed_index = int(cfg.get("prevent_reverse_speed_index", 1))

        self.exploration_noise_initial = float(cfg.get("exploration_noise", 0.0))
        self.exploration_noise_final = float(
            cfg.get("exploration_noise_final", self.exploration_noise_initial)
        )
        self.exploration_noise_decay_steps = max(
            1, int(cfg.get("exploration_noise_decay_steps", 50_000))
        )
        try:
            decay_episodes_value = int(cfg.get("exploration_noise_decay_episodes", 0))
        except (TypeError, ValueError):
            decay_episodes_value = 0
        self.exploration_noise_decay_episodes = max(decay_episodes_value, 0)
        self._exploration_step = 0
        self._exploration_episode = 0
        self.action_noise_scale = self.action_range / 2.0

        # Gradient clipping
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))

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
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if not deterministic:
            noise_scale = self._current_exploration_noise()
            if noise_scale > 0.0:
                noise = np.random.normal(0.0, noise_scale, size=self.act_dim) * self.action_noise_scale
                action_np = np.clip(action_np + noise, self.action_low, self.action_high)
            self._exploration_step += 1

        # Prevent reverse: clamp speed to minimum positive value
        if self.prevent_reverse:
            action_np[self.prevent_reverse_speed_index] = max(
                action_np[self.prevent_reverse_speed_index],
                self.prevent_reverse_min_speed
            )

        return action_np

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

    def store_success_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.success_buffer is None:
            return
        self.success_buffer.add(obs, action, reward, next_obs, done, info)

    def store_hindsight_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        distance_to_target: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store transition with hindsight experience replay (HER).

        For failed episodes, relabel close approaches as "partial successes"
        and store them in the success buffer with augmented rewards.

        Args:
            obs: Current observation
            action: Action taken
            reward: Original reward
            next_obs: Next observation
            done: Episode done flag
            distance_to_target: Current distance to target (meters)
            info: Additional info dict
        """
        if self.success_buffer is None:
            return

        # Only apply HER to non-success transitions
        # (actual successes are handled by store_success_transition)
        if done and info and (
            info.get('success', False)
            or info.get('target_collision', False)
            or info.get('target_crash', False)
        ):
            return  # Already a real success, don't augment

        if not self._her_pairs:
            return

        # Hindsight relabeling based on distance achieved
        for threshold, bonus in self._her_pairs:
            if distance_to_target < threshold:
                augmented_reward = reward + bonus
                self.success_buffer.add(obs, action, augmented_reward, next_obs, done, info)
                break

    def reset_noise_schedule(self, *, restart: bool = False) -> None:
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

    def current_exploration_noise(self) -> float:
        return float(self._current_exploration_noise())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None

        sample = sample_mixed_continuous_replay(
            self.buffer,
            self.success_buffer,
            self.batch_size,
            self.success_buffer_ratio,
            self.device,
        )
        obs = sample.obs
        actions = sample.actions
        rewards = sample.rewards
        next_obs = sample.next_obs
        dones = sample.dones
        weights = sample.weights if self.use_per else torch.ones_like(rewards, device=self.device)

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
        td_error1 = current_q1 - target
        td_error2 = current_q2 - target
        critic_loss = ((td_error1.pow(2) + td_error2.pow(2)) * weights).mean()

        self.q1_opt.zero_grad(set_to_none=True)
        self.q2_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
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
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
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

        if self.use_per and sample.indices is not None:
            td_errors = (td_error1.abs() + td_error2.abs()) * 0.5
            idx = np.asarray(sample.indices, dtype=np.int64).reshape(-1)
            mask = idx >= 0
            if mask.any():
                td_np = td_errors.detach().cpu().squeeze(1).numpy()
                self.buffer.update_priorities(idx[mask], td_np[mask])

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": float(alpha_value.detach().cpu().item()),
            "alpha_loss": float(alpha_loss_value),
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
        return (1.0 - frac) * self.exploration_noise_initial + frac * self.exploration_noise_final

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.state_dict(include_optim=True), path)

    def load(self, path: str) -> None:
        ckpt = safe_load(path, map_location=self.device)
        self.load_state_dict(ckpt, strict=False, include_optim=True)

    def state_dict(self, *, include_optim: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "total_it": int(self.total_it),
        }
        if self.auto_alpha and self.log_alpha is not None:
            payload["log_alpha"] = float(self.log_alpha.detach().cpu().item())
        if include_optim:
            payload["actor_opt"] = self.actor_opt.state_dict()
            payload["q1_opt"] = self.q1_opt.state_dict()
            payload["q2_opt"] = self.q2_opt.state_dict()
            if self.auto_alpha and self.alpha_opt is not None:
                payload["alpha_opt"] = self.alpha_opt.state_dict()
        return payload

    def load_state_dict(
        self,
        snapshot: Mapping[str, Any],
        *,
        strict: bool = False,
        include_optim: bool = True,
    ) -> None:
        self.actor.load_state_dict(snapshot["actor"], strict=strict)
        self.q1.load_state_dict(snapshot["q1"], strict=strict)
        self.q2.load_state_dict(snapshot["q2"], strict=strict)
        self.q1_target.load_state_dict(snapshot.get("q1_target", snapshot["q1"]), strict=strict)
        self.q2_target.load_state_dict(snapshot.get("q2_target", snapshot["q2"]), strict=strict)

        if include_optim:
            actor_opt_state = snapshot.get("actor_opt")
            if actor_opt_state is not None:
                self.actor_opt.load_state_dict(actor_opt_state)
            critic_opt_state = snapshot.get("q1_opt")
            if critic_opt_state is not None:
                self.q1_opt.load_state_dict(critic_opt_state)
            critic2_opt_state = snapshot.get("q2_opt")
            if critic2_opt_state is not None:
                self.q2_opt.load_state_dict(critic2_opt_state)
            if self.auto_alpha and self.alpha_opt is not None:
                alpha_opt_state = snapshot.get("alpha_opt")
                if alpha_opt_state is not None:
                    self.alpha_opt.load_state_dict(alpha_opt_state)

        log_alpha_val = snapshot.get("log_alpha")
        if self.auto_alpha and log_alpha_val is not None and self.log_alpha is not None:
            self.log_alpha.data = torch.tensor(float(log_alpha_val), dtype=torch.float32, device=self.device)

        self.total_it = int(snapshot.get("total_it", self.total_it))

        self.actor.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.q1_target.to(self.device)
        self.q2_target.to(self.device)

    def reset_optimizers(self) -> None:
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.critic_lr)
        if self.auto_alpha and self.log_alpha is not None:
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

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
