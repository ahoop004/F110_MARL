import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from f110x.policies.ppo.net import Actor, Critic
from f110x.utils.torch_io import safe_load


class PPOAgent:
    def __init__(self, cfg):
        self.obs_dim = cfg["obs_dim"]
        self.act_dim = cfg["act_dim"]

        # Hyperparameters
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("lam", 0.95))
        self.clip_eps = float(cfg.get("clip_eps", 0.2))
        self.update_epochs = int(cfg.get("update_epochs", 10))
        self.minibatch_size = int(cfg.get("minibatch_size", 64))
        self.ent_coef = float(cfg.get("ent_coef", 0.0))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.reset_buffer()

    # ------------------- Buffer -------------------

    def reset_buffer(self):
        self.obs_buf, self.act_buf = [], []
        self.rew_buf, self.done_buf = [], []
        self.logp_buf, self.val_buf = [], []
        self.raw_act_buf = []

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
        self.rew_buf.append(float(rew))
        self.done_buf.append(bool(done))

    def record_final_value(self, obs):
        obs_np = np.asarray(obs, dtype=np.float32)
        if not np.isfinite(obs_np).all():
            obs_np = np.nan_to_num(obs_np, copy=False)
        self.obs_buf.append(obs_np)
        self.act_buf.append(np.zeros(self.act_dim, dtype=np.float32))
        self.raw_act_buf.append(np.zeros(self.act_dim, dtype=np.float32))
        self.logp_buf.append(0.0)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        val = self.critic(obs_t).squeeze(-1)
        self.val_buf.append(float(val.item()))

    # ------------------- GAE -------------------

    def finish_path(self):
        T = len(self.rew_buf)
        if T == 0:
            self.adv_buf = np.zeros(0, dtype=np.float32)
            self.ret_buf = np.zeros(0, dtype=np.float32)
            return

        if len(self.done_buf) != T:
            raise ValueError(
                f"rollout length mismatch: rewards {T}, dones {len(self.done_buf)}"
            )

        if len(self.val_buf) == T + 1:
            # keep the final critic estimate for bootstrap, but drop the extra
            bootstrap_v = float(self.val_buf.pop())
            self.obs_buf.pop()
            self.act_buf.pop()
            self.logp_buf.pop()
            self.raw_act_buf.pop()
        elif len(self.val_buf) == T:
            bootstrap_v = 0.0
        else:
            raise ValueError(
                f"rollout length mismatch: rewards {T}, values {len(self.val_buf)}"
            )

        rewards = np.asarray(self.rew_buf, dtype=np.float32)
        values  = np.asarray(self.val_buf, dtype=np.float32)
        dones   = np.asarray(self.done_buf, dtype=np.float32)

        values_ext = np.concatenate([values, np.array([bootstrap_v], dtype=np.float32)])

        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values_ext[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.adv_buf = adv
        self.ret_buf = adv + values


    # ------------------- Update -------------------

    def update(self):
        """Clipped PPO update with safe minibatching and strict length checks.

        Returns a dict of training statistics when an update occurs, otherwise ``None``.
        """
        if len(self.rew_buf) == 0:
            return None

        self.finish_path()

        # Convert buffers -> tensors
        obs = torch.as_tensor(np.asarray(self.obs_buf), dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(np.asarray(self.act_buf), dtype=torch.float32, device=self.device)
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
                act_b  = acts[mb_idx]
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
                loss = policy_loss + value_loss + self.ent_coef * entropy

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

        # fresh rollout next time
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
