import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# try:  # optional dependency for richer logging
#     import wandb  # type: ignore
# except ImportError:  # pragma: no cover - wandb optional
#     wandb = None

from f110x.policies.ppo.net import Actor, Critic
from f110x.policies.ppo.base import BasePPOAgent
from f110x.utils.torch_io import resolve_device, safe_load


class PPOAgent(BasePPOAgent):
    def __init__(self, cfg):
        device = resolve_device([cfg.get("device")])
        super().__init__(cfg, device)

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

    # ------------------- Buffer -------------------

    # ------------------- Acting -------------------

    def _scale_action(self, squashed):
        return self.action_low_t + 0.5 * (squashed + 1.0) * self.action_scale_t

    def act(self, obs, aid=None):
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

    def store(self, obs, act, rew, done):
        self.store_transition(rew, done)

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

        self.finish_path(normalize_advantage=True)

        # Convert buffers -> tensors
        obs = torch.as_tensor(np.asarray(self.obs_buf), dtype=torch.float32, device=self.device)
        raw_actions = torch.as_tensor(np.asarray(self.raw_act_buf), dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(np.asarray(self.logp_buf), dtype=torch.float32, device=self.device)
        adv  = torch.as_tensor(self.adv_buf, dtype=torch.float32, device=self.device)
        rets = torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device)

        # Hard alignment check
        N = len(self.obs_buf)
        assert N == len(self.act_buf) == len(self.raw_act_buf) == len(self.logp_buf) == len(self.adv_buf) == len(self.ret_buf), \
            f"Buffer length mismatch: obs {len(self.obs_buf)}, acts {len(self.act_buf)}, raw {len(self.raw_act_buf)}, logp {len(self.logp_buf)}, adv {len(self.adv_buf)}, ret {len(self.ret_buf)}"

        idx = np.arange(N)
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch_size):
                end = min(start + self.minibatch_size, N)  # clamp
                mb_idx = idx[start:end]

                ob_b   = obs[mb_idx]
                raw_b  = raw_actions[mb_idx]
                squashed_b = torch.tanh(raw_b)
                adv_b  = adv[mb_idx]
                ret_b  = rets[mb_idx]
                logp_b = logp_old[mb_idx]

                mu, std = self.actor(ob_b)
                if not torch.isfinite(mu).all() or not torch.isfinite(std).all():
                    print("[WARN] Non-finite parameters encountered in PPO update; skipping minibatch")
                    continue
                dist = Normal(mu, std)
                logp = dist.log_prob(raw_b).sum(dim=-1)
                logp -= torch.log(1 - squashed_b.pow(2) + self.squash_eps).sum(dim=-1)
                ratio = torch.exp(logp - logp_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                v_pred = self.critic(ob_b).squeeze(-1)
                value_loss = F.mse_loss(v_pred, ret_b)

                entropy = dist.entropy().sum(dim=-1).mean()
                # Encourage exploration by subtracting the entropy bonus (maximise entropy)
                loss = policy_loss + value_loss - self.ent_coef * entropy

                with torch.no_grad():
                    policy_losses.append(float(policy_loss.detach().cpu().item()))
                    value_losses.append(float(value_loss.detach().cpu().item()))
                    entropies.append(float(entropy.detach().cpu().item()))
                    approx_kl = (logp_b - logp).mean()
                    approx_kls.append(float(approx_kl.detach().cpu().item()))

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

        self.reset_buffer()

        if not policy_losses:
            return None

        # def _mean_safe(values):
        #     return float(np.mean(values)) if values else 0.0

        # metrics = {
        #     "policy_loss": _mean_safe(policy_losses),
        #     "value_loss": _mean_safe(value_losses),
        #     "entropy": _mean_safe(entropies),
        #     "approx_kl": _mean_safe(approx_kls),
        #     "action_mean": float(action_np.mean()) if action_np.size else 0.0,
        #     "action_std": float(action_np.std()) if action_np.size else 0.0,
        #     "action_abs_mean": float(np.abs(action_np).mean()) if action_np.size else 0.0,
        #     "raw_action_std": float(raw_action_np.std()) if raw_action_np.size else 0.0,
        #     "value_mean": float(value_pred_np.mean()) if value_pred_np.size else 0.0,
        #     "value_std": float(value_pred_np.std()) if value_pred_np.size else 0.0,
        #     "adv_mean": float(adv_np.mean()) if adv_np.size else 0.0,
        #     "adv_std": float(adv_np.std()) if adv_np.size else 0.0,
        # }

        # if wandb is not None:
        #     if action_np.size:
        #         metrics["action_histogram"] = wandb.Histogram(action_np.flatten())
        #     if value_pred_np.size:
        #         metrics["value_histogram"] = wandb.Histogram(value_pred_np.flatten())

        # return metrics

    # ------------------- I/O -------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = safe_load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.actor.to(self.device)
        self.critic.to(self.device)
