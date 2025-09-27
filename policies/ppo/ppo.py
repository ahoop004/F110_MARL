import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from collections import deque
import numpy as np
import os

from policies.ppo.net import Actor, Critic

class PPOAgent:
    def __init__(self, cfg):
        self.obs_dim = cfg["obs_dim"]
        self.act_dim = cfg["act_dim"]
        self.gamma = cfg.get("gamma", 0.99)
        self.lam = cfg.get("lam", 0.95)
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.update_epochs = cfg.get("update_epochs", 10)
        self.minibatch_size = cfg.get("minibatch_size", 64)

        # Networks
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.get("actor_lr", 3e-4))
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.get("critic_lr", 1e-3))

        # Storage
        self.reset_buffer()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    def reset_buffer(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.logp_buf = []
        self.val_buf = []

    def act(self, obs, aid=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        mu, std = self.actor(obs_t)
        dist = Normal(mu, std)
        act = dist.sample()
        logp = dist.log_prob(act).sum()
        val = self.critic(obs_t)

        # Save to buffer
        self.obs_buf.append(obs)
        self.act_buf.append(act.cpu().numpy())
        self.logp_buf.append(logp.item())
        self.val_buf.append(val.item())

        return act.cpu().numpy()

    def store(self, obs, act, rew, done):
        self.rew_buf.append(rew)
        self.done_buf.append(done)

    def finish_path(self):
        """
        Compute GAE(λ) advantages and returns for the current rollout in buffers.
        Assumes:
        - self.rew_buf: [r_0, ..., r_{T-1}]
        - self.val_buf: [V(s_0), ..., V(s_{T-1})]
        Produces:
        - self.adv_buf: [A_0, ..., A_{T-1}]
        - self.ret_buf: [R_0, ..., R_{T-1}] where R_t = A_t + V(s_t)
        """
        rewards = np.asarray(self.rew_buf, dtype=np.float32)            # (T,)
        values  = np.asarray(self.val_buf, dtype=np.float32)            # (T,)

        # ensure one-step bootstrap value exists
        # if episode ended, bootstrap = 0; if truncated you can inject critic value here instead
        if len(values) == len(rewards):
            values = np.append(values, 0.0)                              # (T+1,)

        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t+1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae

        # returns
        rets = adv + values[:-1]                                         # (T,)

        # advantage normalization improves stability
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.adv_buf = adv
        self.ret_buf = rets
    # drop bootstrap value

    def update(self):
        """
        PPO clipped surrogate update with minibatching and safe indexing.
        Uses buffers filled by act/store and advantages/returns from finish_path().
        Resets buffers at the end.
        """
        # Guard: nothing collected
        if len(self.obs_buf) == 0:
            return

        # 1) finalize advantages/returns
        self.finish_path()

        # 2) build dataset
        obs = torch.as_tensor(np.asarray(self.obs_buf), dtype=torch.float32, device=self.device)   # (N, obs_dim)
        acts = torch.as_tensor(np.asarray(self.act_buf), dtype=torch.float32, device=self.device)  # (N, act_dim)
        adv  = torch.as_tensor(self.adv_buf, dtype=torch.float32, device=self.device)              # (N,)
        rets = torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device)              # (N,)
        logp_old = torch.as_tensor(np.asarray(self.logp_buf), dtype=torch.float32, device=self.device)  # (N,)

        N = obs.shape[0]
        idx = np.arange(N)

        # 3) update epochs
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch_size):
                end = min(start + self.minibatch_size, N)               # safe end
                mb_idx = idx[start:end]

                ob_b   = obs[mb_idx]
                act_b  = acts[mb_idx]
                adv_b  = adv[mb_idx]
                ret_b  = rets[mb_idx]
                logp_b = logp_old[mb_idx]

                # policy forward
                mu, std = self.actor(ob_b)                               # (B, act_dim) each
                dist = Normal(mu, std)
                logp = dist.log_prob(act_b).sum(dim=-1)                  # (B,)
                ratio = torch.exp(logp - logp_b)                         # (B,)

                # clipped surrogate
                clip_eps = float(self.clip_eps)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                v_pred = self.critic(ob_b).squeeze(-1)                   # (B,)
                value_loss = F.mse_loss(v_pred, ret_b)

                # entropy bonus (optional but helpful)
                entropy = dist.entropy().sum(dim=-1).mean()
                ent_coef = 0.0  # set >0 if desired, e.g. 0.001
                loss = policy_loss + value_loss + ent_coef * entropy

                # optimize
                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_opt.step()
                self.critic_opt.step()

        # 4) reset buffers to start a fresh rollout
        self.reset_buffer()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])
