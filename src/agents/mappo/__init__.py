"""Multi-Agent PPO (MAPPO) agent.

Architecture
------------
Shared actor
    All trainable agents share one :class:`~agents.common.networks.Actor`.
    Each agent uses its own local observation to select actions.

Centralized critic
    A single :class:`~agents.common.networks.Critic` takes the **global state**
    (``env.get_global_state().vector``) and outputs one scalar value per call.
    This is the CTDE (Centralized Training, Decentralized Execution) pattern:
    the critic sees everything during training but the actor only uses local obs.

Per-agent rollout buffers
    One :class:`MAPPORolloutBuffer` per trainable agent.  Each buffer stores
    ``(local_obs, action, reward, log_prob, global_state, done)`` so GAE can
    be re-evaluated against the centralized critic during the update.

Update
    Advantages and returns are computed per-agent using centralized value
    estimates.  All agents' data is then pooled into a single minibatch set
    and the shared actor + centralized critic are updated jointly via PPO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.common.networks import Actor, Critic
from utils.torch_io import resolve_device


# ---------------------------------------------------------------------------
# Rollout buffer with global-state storage
# ---------------------------------------------------------------------------

class MAPPORolloutBuffer:
    """Per-agent rollout buffer for MAPPO.

    Extends the standard PPO buffer by also storing the global state at each
    timestep so the centralized critic can re-evaluate values during the update.

    Parameters
    ----------
    n_steps:
        Maximum steps before a forced update (same as PPO ``n_steps``).
    obs_dim:
        Dimension of the local observation for this agent.
    global_state_dim:
        Dimension of the global state vector from ``env.get_global_state()``.
    action_dim:
        Number of action dimensions.
    device:
        PyTorch device to place tensors on.
    """

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        global_state_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.device = device
        self.obs = torch.zeros(n_steps, obs_dim, device=device)
        self.global_states = torch.zeros(n_steps, global_state_dim, device=device)
        self.actions = torch.zeros(n_steps, action_dim, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        i = self.ptr % self.n_steps
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.global_states[i] = torch.as_tensor(
            global_state, dtype=torch.float32, device=self.device
        )
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
            delta = (
                float(self.rewards[t])
                + gamma * nv * next_non_terminal
                - float(self.values[t])
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:n]
        return advantages, returns

    def iterate_batches(self, batch_size: int):
        n = self.size()
        indices = torch.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                self.obs[idx],
                self.global_states[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.values[idx],
            ), idx


# ---------------------------------------------------------------------------
# MAPPO agent
# ---------------------------------------------------------------------------

class MAPPOAgent:
    """Multi-Agent PPO with shared actor and centralized critic.

    Parameters
    ----------
    obs_dim:
        Local observation dimension (same for all agents when shared policy).
    global_state_dim:
        Dimension of ``env.get_global_state().vector``.
    action_low, action_high:
        Physical action bounds (numpy arrays).
    agent_ids:
        Ordered list of trainable agent IDs.  One rollout buffer is created
        per agent.
    params:
        Hyperparameter dict (merged from ``training_defaults`` + scenario).
    """

    def __init__(
        self,
        obs_dim: int,
        global_state_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        agent_ids: List[str],
        params: Dict,
    ) -> None:
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = len(self.action_low)
        self.agent_ids = list(agent_ids)

        # Hyperparameters
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
            params.get("pi_hidden_dims", params.get("hidden_dims", [256, 256]))
        )
        vf_dims: List[int] = list(
            params.get("vf_hidden_dims", params.get("hidden_dims", [256, 256]))
        )
        activation: str = str(params.get("activation", "tanh"))

        device_str = str(params.get("device", "cpu"))
        self.device = resolve_device([device_str])

        # Shared actor (local obs → action)
        self.actor = Actor(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)

        # Centralized critic (global state → value)
        self.critic = Critic(global_state_dim, vf_dims, activation).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
        )

        # Per-agent rollout buffers
        self.buffers: Dict[str, MAPPORolloutBuffer] = {
            aid: MAPPORolloutBuffer(
                self.n_steps, obs_dim, global_state_dim, self.action_dim, self.device
            )
            for aid in self.agent_ids
        }

    # ------------------------------------------------------------------
    # Action selection (decentralized execution — uses local obs only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """Sample action from the shared actor using a local observation.

        Returns
        -------
        action_normalized : np.ndarray
            Action in ``[-1, 1]`` — caller denormalizes for ``env.step()``.
        log_prob : float
            Log probability of the sampled action.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, log_prob_t = self.actor.get_action(obs_t, deterministic=deterministic)
        return (
            action_t.squeeze(0).cpu().numpy(),
            float(log_prob_t.squeeze()),
        )

    @torch.no_grad()
    def evaluate_state(self, global_state: np.ndarray) -> float:
        """Estimate value of a global state using the centralized critic.

        Parameters
        ----------
        global_state:
            Flat numpy array from ``env.get_global_state().vector``.
        """
        gs_t = torch.as_tensor(
            global_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return float(self.critic(gs_t).squeeze())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        agent_id: str,
        obs: np.ndarray,
        global_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """Store one transition in *agent_id*'s rollout buffer."""
        self.buffers[agent_id].add(obs, global_state, action, reward, log_prob, value, done)

    def any_buffer_full(self) -> bool:
        """True when any agent's buffer has reached ``n_steps``."""
        return any(buf.is_full() for buf in self.buffers.values())

    def clear_buffers(self) -> None:
        for buf in self.buffers.values():
            buf.clear()

    # ------------------------------------------------------------------
    # PPO update — all agents' data pooled into shared actor + critic update
    # ------------------------------------------------------------------

    def update(
        self,
        next_global_state: np.ndarray,
        dones: Dict[str, bool],
    ) -> Dict[str, float]:
        """Compute GAE for each agent and run PPO update.

        Parameters
        ----------
        next_global_state:
            Global state at the end of the rollout (for bootstrapping).
        dones:
            Per-agent done flags at the end of the rollout.

        Returns
        -------
        Dict with average training losses across all minibatch updates.
        """
        next_value = self.evaluate_state(next_global_state)

        # Collect per-agent advantages and returns
        all_obs: List[torch.Tensor] = []
        all_gs: List[torch.Tensor] = []
        all_acts: List[torch.Tensor] = []
        all_old_lp: List[torch.Tensor] = []
        all_adv: List[torch.Tensor] = []
        all_ret: List[torch.Tensor] = []

        for aid in self.agent_ids:
            buf = self.buffers[aid]
            n = buf.size()
            if n < 2:
                continue

            boot_value = 0.0 if dones.get(aid, False) else next_value
            adv, ret = buf.compute_gae(boot_value, self.gamma, self.gae_lambda)

            all_obs.append(buf.obs[:n])
            all_gs.append(buf.global_states[:n])
            all_acts.append(buf.actions[:n])
            all_old_lp.append(buf.log_probs[:n])
            all_adv.append(adv)
            all_ret.append(ret)

        if not all_obs:
            return {}

        # Pool all agents' data
        obs_pool = torch.cat(all_obs, dim=0)
        gs_pool = torch.cat(all_gs, dim=0)
        acts_pool = torch.cat(all_acts, dim=0)
        old_lp_pool = torch.cat(all_old_lp, dim=0)
        adv_pool = torch.cat(all_adv, dim=0)
        ret_pool = torch.cat(all_ret, dim=0)

        # Normalize advantages over the pooled set
        adv_std = adv_pool.std(correction=0)
        adv_pool = (adv_pool - adv_pool.mean()) / (adv_std + 1e-8)

        total_pi_loss = total_vf_loss = total_ent = total_kl = 0.0
        n_updates = 0
        n_pool = obs_pool.shape[0]

        for _ in range(self.n_epochs):
            idx_all = torch.randperm(n_pool, device=self.device)
            for start in range(0, n_pool, self.batch_size):
                idx = idx_all[start:start + self.batch_size]

                obs_b = obs_pool[idx]
                gs_b = gs_pool[idx]
                acts_b = acts_pool[idx]
                old_lp_b = old_lp_pool[idx]
                adv_b = adv_pool[idx]
                ret_b = ret_pool[idx]

                # Actor loss
                _, new_lp_b = self.actor.get_action(obs_b)
                ratio = (new_lp_b - old_lp_b).exp()
                pi_loss = torch.max(
                    -adv_b * ratio,
                    -adv_b * ratio.clamp(1 - self.clip_range, 1 + self.clip_range),
                ).mean()

                # Centralized critic loss
                value_pred = self.critic(gs_b)
                vf_loss = nn.functional.mse_loss(value_pred, ret_b)

                # Entropy bonus
                mean_b, std_b = self.actor(obs_b)
                dist = torch.distributions.Normal(mean_b, std_b)
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
                    approx_kl = ((old_lp_b - new_lp_b).mean()).abs().item()

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

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "agent_ids": self.agent_ids,
                "obs_dim": self.obs_dim,
                "global_state_dim": self.global_state_dim,
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
