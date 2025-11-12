"""TD3 agent built on top of shared replay utilities."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

# try:  # optional dependency for rich logging
#     import wandb  # type: ignore
# except ImportError:  # pragma: no cover - wandb optional
#     wandb = None

from f110x.policies.common import build_replay_buffer, sample_continuous_replay
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

        self.actor_lr = float(cfg.get("actor_lr", 1e-3))
        self.critic_lr = float(cfg.get("critic_lr", 1e-3))

        self.actor_opt = self._init_optimizer(
            self.actor.parameters(),
            cfg,
            prefix="actor",
            default_lr=self.actor_lr,
        )
        self.critic_opt = self._init_optimizer(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            cfg,
            prefix="critic",
            default_lr=self.critic_lr,
        )

        self.actor_scheduler = self._init_scheduler(self.actor_opt, cfg.get("actor_lr_scheduler"))
        self.critic_scheduler = self._init_scheduler(self.critic_opt, cfg.get("critic_lr_scheduler"))

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

    def current_exploration_noise(self) -> float:
        """Return the exploration noise scale currently applied to actions."""
        return float(self._current_exploration_noise())

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

        sample = sample_continuous_replay(self.buffer, self.batch_size, self.device)
        obs = sample.obs
        actions = sample.actions
        rewards = sample.rewards
        next_obs = sample.next_obs
        dones = sample.dones
        weights = sample.weights if self.use_per else torch.ones_like(rewards, device=self.device)

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
        self._step_scheduler(self.critic_scheduler)

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_it % self.policy_delay == 0:
            actor_action = self._scale_action_torch(self.actor(obs))
            actor_loss = -self.critic1(obs, actor_action).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self._step_scheduler(self.actor_scheduler)

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target1, self.critic1, self.tau)
            soft_update(self.critic_target2, self.critic2, self.tau)

        self.total_it += 1

        if self.use_per and sample.indices is not None:
            td_errors = (td_error1.abs() + td_error2.abs()) * 0.5
            self.buffer.update_priorities(
                sample.indices,
                td_errors.detach().cpu().squeeze(1).numpy(),
            )

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "update_it": float(self.total_it),
        }

    def _init_optimizer(
        self,
        params: Iterable[torch.nn.Parameter],
        cfg: Mapping[str, Any],
        *,
        prefix: str,
        default_lr: float,
    ) -> torch.optim.Optimizer:
        opt_name = str(cfg.get(f"{prefix}_optimizer", cfg.get("optimizer", "adam"))).lower()
        lr_value = float(cfg.get(f"{prefix}_lr", default_lr))
        weight_decay = float(cfg.get(f"{prefix}_weight_decay", 0.0))
        betas = cfg.get(f"{prefix}_betas")
        eps = cfg.get(f"{prefix}_eps")

        opt_kwargs: Dict[str, Any] = {"lr": lr_value, "weight_decay": weight_decay}
        if betas is not None:
            if isinstance(betas, (list, tuple)) and len(betas) == 2:
                opt_kwargs["betas"] = (float(betas[0]), float(betas[1]))
        if eps is not None:
            try:
                opt_kwargs["eps"] = float(eps)
            except (TypeError, ValueError):
                pass

        if opt_name == "adamw":
            return torch.optim.AdamW(params, **opt_kwargs)
        if opt_name == "sgd":
            momentum = float(cfg.get(f"{prefix}_momentum", 0.0))
            opt_kwargs.setdefault("momentum", momentum)
            return torch.optim.SGD(params, **opt_kwargs)
        if opt_name == "rmsprop":
            alpha = cfg.get(f"{prefix}_alpha", 0.99)
            opt_kwargs.setdefault("alpha", float(alpha))
            momentum = cfg.get(f"{prefix}_momentum")
            if momentum is not None:
                opt_kwargs["momentum"] = float(momentum)
            return torch.optim.RMSprop(params, **opt_kwargs)
        # default Adam
        return torch.optim.Adam(params, **opt_kwargs)

    def _init_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: Optional[Mapping[str, Any]],
    ) -> Optional[lr_scheduler._LRScheduler]:
        if not config:
            return None
        sched_type = str(config.get("type", "")).lower()
        if not sched_type:
            return None

        if sched_type == "steplr":
            step_size = int(config.get("step_size", 1000))
            gamma = float(config.get("gamma", 0.5))
            return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if sched_type == "multistep":
            milestones = config.get("milestones", [])
            milestones = [int(m) for m in milestones] if isinstance(milestones, Sequence) else []
            gamma = float(config.get("gamma", 0.5))
            if not milestones:
                return None
            return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        if sched_type == "exponential":
            gamma = float(config.get("gamma", 0.99))
            return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        if sched_type == "cosine":
            t_max = int(config.get("t_max", 1000))
            eta_min = float(config.get("eta_min", 0.0))
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        if sched_type == "cosine_restarts":
            t_0 = int(config.get("t_0", 1000))
            t_mult = int(config.get("t_mult", 1))
            eta_min = float(config.get("eta_min", 0.0))
            return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)
        return None

    @staticmethod
    def _step_scheduler(scheduler_obj: Optional[lr_scheduler._LRScheduler]) -> None:
        if scheduler_obj is None:
            return
        scheduler_obj.step()

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

    def state_dict(self, *, include_optim: bool = True) -> Dict[str, Any]:
        """Return a serialisable snapshot of the agent."""

        state: Dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic_target1": self.critic_target1.state_dict(),
            "critic_target2": self.critic_target2.state_dict(),
            "total_it": int(self.total_it),
        }
        if include_optim:
            state["actor_opt"] = self.actor_opt.state_dict()
            state["critic_opt"] = self.critic_opt.state_dict()
        return state

    def load_state_dict(
        self,
        snapshot: Mapping[str, Any],
        *,
        strict: bool = False,
        include_optim: bool = True,
    ) -> None:
        """Restore agent parameters from :meth:`state_dict` output."""

        modules = {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic1": self.critic1,
            "critic2": self.critic2,
            "critic_target1": self.critic_target1,
            "critic_target2": self.critic_target2,
        }
        for key, module in modules.items():
            weights = snapshot.get(key)
            if weights is None:
                if strict:
                    raise KeyError(f"Missing weights for '{key}'")
                if key.endswith("_target"):
                    source_key = key.replace("_target", "")
                    source_weights = snapshot.get(source_key)
                    if source_weights is not None:
                        module.load_state_dict(source_weights, strict=False)
                continue
            module.load_state_dict(weights, strict=strict)

        self.total_it = int(snapshot.get("total_it", self.total_it))

        if include_optim:
            actor_opt_state = snapshot.get("actor_opt")
            critic_opt_state = snapshot.get("critic_opt")
            if actor_opt_state is None or critic_opt_state is None:
                if strict:
                    missing = ["actor_opt", "critic_opt"]
                    raise KeyError(f"Missing optimizer state in snapshot: {missing}")
            else:
                self.actor_opt.load_state_dict(actor_opt_state)
                self.critic_opt.load_state_dict(critic_opt_state)

        # Ensure modules live on the configured device after loading.
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.critic_target1.to(self.device)
        self.critic_target2.to(self.device)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(include_optim=True), path)

    def load(self, path: str) -> None:
        ckpt = safe_load(path, map_location=self.device)
        self.load_state_dict(ckpt, strict=False, include_optim=True)

    def reset_optimizers(self) -> None:
        """Reinitialise optimiser state while preserving learning rates."""

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.critic_lr,
        )

    # -------------------- Helpers --------------------

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, -1.0, 1.0) * (self.action_range / 2.0) + (self.action_low + self.action_high) / 2.0

    def _scale_action_torch(self, action: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.action_low, device=action.device)
        high = torch.as_tensor(self.action_high, device=action.device)
        range_half = (high - low) / 2.0
        mid = (high + low) / 2.0
        return torch.clamp(action, -1.0, 1.0) * range_half + mid
