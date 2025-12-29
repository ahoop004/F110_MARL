import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.distributions import Normal
from typing import Dict, Any, Optional, Mapping

# try:  # optional dependency for richer logging
#     import wandb  # type: ignore
# except ImportError:  # pragma: no cover - wandb optional
#     wandb = None

from agents.ppo.net import Actor, Critic
from agents.ppo.base import BasePPOAgent
from utils.torch_io import resolve_device, safe_load


class PPOAgent(BasePPOAgent):
    def __init__(self, cfg):
        device = resolve_device([cfg.get("device")])
        super().__init__(cfg, device)
        target_kl = cfg.get("target_kl")
        self.target_kl = float(target_kl) if target_kl is not None else None

        action_low = cfg.get("action_low")
        action_high = cfg.get("action_high")
        if action_low is None or action_high is None:
            action_low = [-1.0] * self.act_dim
            action_high = [1.0] * self.act_dim
        self.action_low_np = np.asarray(action_low, dtype=np.float32)
        self.action_high_np = np.asarray(action_high, dtype=np.float32)
        self.action_low_t = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high_t = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.action_scale_t = self.action_high_t - self.action_low_t
        self.action_scale_t = torch.where(
            torch.abs(self.action_scale_t) < 1e-6,
            torch.ones_like(self.action_scale_t),
            self.action_scale_t,
        )
        self.squash_eps = 1e-6

        # Networks
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # Optimizers (cast LR to float in case YAML gave strings)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=float(cfg.get("actor_lr", 3e-4)))
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=float(cfg.get("critic_lr", 1e-3)))

        # Learning rate schedulers (optional)
        self.actor_scheduler = self._init_scheduler(self.actor_opt, cfg.get("actor_lr_scheduler"))
        self.critic_scheduler = self._init_scheduler(self.critic_opt, cfg.get("critic_lr_scheduler"))

    # ------------------- Buffer -------------------

    # ------------------- Acting -------------------

    def _scale_action(self, squashed):
        return self.action_low_t + 0.5 * (squashed + 1.0) * self.action_scale_t

    def act(self, obs, deterministic=False, aid=None):
        """Select action (protocol-compliant interface).

        Args:
            obs: Observation
            deterministic: If True, select mean action (for eval)
            aid: Optional agent ID (legacy parameter)

        Returns:
            action: Selected action
        """
        if deterministic:
            return self.act_deterministic(obs, aid=aid)
        else:
            return self.act_stochastic(obs, aid=aid)

    def act_stochastic(self, obs, aid=None):
        """Select stochastic action for training."""
        obs_np = np.asarray(obs, dtype=np.float32)
        if not np.isfinite(obs_np).all():
            obs_np = np.nan_to_num(obs_np, copy=False)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        mu, std = self.actor(obs_t)
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        squashed = torch.tanh(raw_action)
        scaled = self._scale_action(squashed)

        logp = dist.log_prob(raw_action).sum(dim=-1)
        logp -= torch.log(1 - squashed.pow(2) + self.squash_eps).sum(dim=-1)
        val = self.critic(obs_t).squeeze(-1)

        self.obs_buf.append(obs_np)
        self.act_buf.append(scaled.detach().cpu().numpy())
        self.logp_buf.append(float(logp.item()))
        self.val_buf.append(float(val.item()))
        self.raw_act_buf.append(raw_action.detach().cpu().numpy())

        return scaled.detach().cpu().numpy()

    def act_deterministic(self, obs, aid=None):
        obs_np = np.asarray(obs, dtype=np.float32)
        if not np.isfinite(obs_np).all():
            obs_np = np.nan_to_num(obs_np, copy=False)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, _ = self.actor(obs_t)
            squashed = torch.tanh(mu)
            scaled = self._scale_action(squashed)
        return scaled.detach().cpu().numpy()

    def store(self, obs, act, rew, done, terminated: bool = False):
        self.store_transition(rew, done, terminated)

    def _estimate_value(self, obs):
        obs_np = np.asarray(obs, dtype=np.float32)
        if not np.isfinite(obs_np).all():
            obs_np = np.nan_to_num(obs_np, copy=False)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        val = self.critic(obs_t).squeeze(-1)
        return float(val.item())

    # ------------------- GAE -------------------

    # ------------------- Update -------------------

    def update(self):
        """Clipped PPO update with safe minibatching and strict length checks.

        Returns a dict of training statistics when an update occurs, otherwise ``None``.
        """
        if len(self.rew_buf) == 0:
            return None

        episodes_progress = self._episodes_since_update or 1
        self.apply_entropy_decay(episodes_progress)
        self._episodes_since_update = 0

        self.finish_path(normalize_advantage=self.normalize_advantage)

        # Convert buffers -> tensors
        obs = torch.as_tensor(np.asarray(self.obs_buf), dtype=torch.float32, device=self.device)
        raw_actions = torch.as_tensor(np.asarray(self.raw_act_buf), dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(np.asarray(self.logp_buf), dtype=torch.float32, device=self.device)
        adv  = torch.as_tensor(self.adv_buf, dtype=torch.float32, device=self.device)
        rets = torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device)
        values_old = torch.as_tensor(np.asarray(self.val_buf), dtype=torch.float32, device=self.device)

        # Hard alignment check
        N = len(self.obs_buf)
        assert N == len(self.act_buf) == len(self.raw_act_buf) == len(self.logp_buf) == len(self.adv_buf) == len(self.ret_buf), \
            f"Buffer length mismatch: obs {len(self.obs_buf)}, acts {len(self.act_buf)}, raw {len(self.raw_act_buf)}, logp {len(self.logp_buf)}, adv {len(self.adv_buf)}, ret {len(self.ret_buf)}"

        idx = np.arange(N)
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        stop_early = False
        mb_size = N if self.episode_batch else self.minibatch_size
        mb_size = max(1, int(mb_size))
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, mb_size):
                end = min(start + mb_size, N)  # clamp
                mb_idx = idx[start:end]

                ob_b = obs[mb_idx]
                raw_b = raw_actions[mb_idx]
                adv_b = adv[mb_idx]
                ret_b = rets[mb_idx]
                logp_b = logp_old[mb_idx]
                val_b = values_old[mb_idx]

                mu, std = self.actor(ob_b)
                if not torch.isfinite(mu).all() or not torch.isfinite(std).all():
                    print("[WARN] Non-finite parameters encountered in PPO update; skipping minibatch")
                    continue
                dist = Normal(mu, std)
                values_pred = self.critic(ob_b).squeeze(-1)

                policy_loss, value_loss, entropy_term, approx_kl = self.compute_losses(
                    dist=dist,
                    raw_actions=raw_b,
                    logp_old=logp_b,
                    advantages=adv_b,
                    returns=ret_b,
                    values_pred=values_pred,
                    values_old=val_b,
                    reduction="mean",
                )

                loss = policy_loss + value_loss - self.ent_coef * entropy_term

                approx_kl_value = float(approx_kl.detach().cpu().item())
                with torch.no_grad():
                    policy_losses.append(float(policy_loss.detach().cpu().item()))
                    value_losses.append(float(value_loss.detach().cpu().item()))
                    entropies.append(float(entropy_term.detach().cpu().item()))
                    approx_kls.append(approx_kl_value)

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                # Step LR schedulers after each minibatch
                self._step_scheduler(self.actor_scheduler)
                self._step_scheduler(self.critic_scheduler)

                if self.target_kl is not None and approx_kl_value > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.reset_buffer()

        if not policy_losses:
            return None

        def _mean_safe(values):
            return float(np.mean(values)) if values else 0.0

        return {
            "policy_loss": _mean_safe(policy_losses),
            "value_loss": _mean_safe(value_losses),
            "entropy": _mean_safe(entropies),
            "approx_kl": _mean_safe(approx_kls),
        }

    # ------------------- Scheduler helpers -------------------

    def _init_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: Optional[Mapping[str, Any]],
    ) -> Optional[lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler from config (adapted from TD3)."""
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
            if isinstance(milestones, list):
                milestones = [int(m) for m in milestones]
            else:
                milestones = []
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
        """Step the scheduler if it exists."""
        if scheduler_obj is None:
            return
        scheduler_obj.step()

    # ------------------- I/O -------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        # Save scheduler state if schedulers exist
        if self.actor_scheduler is not None:
            state["actor_scheduler"] = self.actor_scheduler.state_dict()
        if self.critic_scheduler is not None:
            state["critic_scheduler"] = self.critic_scheduler.state_dict()
        torch.save(state, path)

    def load(self, path):
        ckpt = safe_load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        # Load scheduler state if it exists in checkpoint
        if "actor_scheduler" in ckpt and self.actor_scheduler is not None:
            self.actor_scheduler.load_state_dict(ckpt["actor_scheduler"])
        if "critic_scheduler" in ckpt and self.critic_scheduler is not None:
            self.critic_scheduler.load_state_dict(ckpt["critic_scheduler"])
        self.actor.to(self.device)
        self.critic.to(self.device)
