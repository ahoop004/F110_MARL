"""Multi-Agent PPO (MAPPO) agent.

Architecture
------------
Shared actor
    All trainable agents share one :class:`~agents.common.networks.Actor`.
    Each agent uses its own local observation to select actions.

Centralized critic
    A single :class:`~agents.common.networks.Critic` takes the **global state**
    (``env.get_global_state().vector``) plus a one-hot trainable-agent identity.
    It therefore estimates a distinct ``V_i(s)`` for each independently
    rewarded agent while sharing critic parameters.  This is the CTDE
    (Centralized Training, Decentralized Execution) pattern: the critic sees
    everything during training but the actor only uses local observations.

Per-agent rollout buffers
    One :class:`MAPPORolloutBuffer` per trainable agent.  Each buffer stores
    ``(local_obs, action, reward, log_prob, global_state, terminated, truncated)``
    so GAE can
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
        self.terminated = torch.zeros(n_steps, device=device)
        self.truncated = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        terminated: bool,
        truncated: bool,
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
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0
        next_val = float(next_value)

        for t in reversed(range(n)):
            bootstrap_mask = 1.0 - float(self.terminated[t])
            continuation_mask = 1.0 - float(
                bool(self.terminated[t]) or bool(self.truncated[t])
            )
            nv = next_val if t == n - 1 else float(self.values[t + 1])
            delta = (
                float(self.rewards[t])
                + gamma * nv * bootstrap_mask
                - float(self.values[t])
            )
            last_gae = delta + gamma * gae_lambda * continuation_mask * last_gae
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

        # Agent-conditioned centralized critic.  Reward composers are
        # independent per agent, so V(s) alone would receive conflicting return
        # targets for the same global state.  A one-hot agent identity preserves
        # centralized state access while allowing V_i(s) estimates.
        self.critic_input_dim = global_state_dim + len(self.agent_ids)
        self.critic = Critic(self.critic_input_dim, vf_dims, activation).to(self.device)

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

    def _critic_inputs(
        self,
        global_states: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Append one-hot agent identities to batched global states."""
        identities = torch.nn.functional.one_hot(
            agent_indices.long(), num_classes=len(self.agent_ids)
        ).to(dtype=global_states.dtype, device=global_states.device)
        return torch.cat((global_states, identities), dim=-1)

    @torch.no_grad()
    def evaluate_state(self, global_state: np.ndarray, agent_id: str) -> float:
        """Estimate one agent's value from the centralized global state.

        Parameters
        ----------
        global_state:
            Flat numpy array from ``env.get_global_state().vector``.
        agent_id:
            Trainable agent whose independently rewarded return is estimated.
        """
        if agent_id not in self.buffers:
            raise KeyError(f"Unknown MAPPO agent_id: {agent_id}")
        agent_index = self.agent_ids.index(agent_id)
        gs_t = torch.as_tensor(
            global_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        index_t = torch.tensor([agent_index], dtype=torch.long, device=self.device)
        critic_input = self._critic_inputs(gs_t, index_t)
        return float(self.critic(critic_input).squeeze())

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
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one transition in *agent_id*'s rollout buffer."""
        self.buffers[agent_id].add(
            obs,
            global_state,
            action,
            reward,
            log_prob,
            value,
            terminated,
            truncated,
        )

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
    ) -> Dict[str, float]:
        """Compute GAE for each agent and run PPO update.

        Parameters
        ----------
        next_global_state:
            Global state at the end of the rollout (for bootstrapping).
        Returns
        -------
        Dict with average training losses across all minibatch updates.
        """
        next_values = {
            aid: self.evaluate_state(next_global_state, aid) for aid in self.agent_ids
        }

        # Collect per-agent advantages and returns
        all_obs: List[torch.Tensor] = []
        all_gs: List[torch.Tensor] = []
        all_acts: List[torch.Tensor] = []
        all_old_lp: List[torch.Tensor] = []
        all_agent_indices: List[torch.Tensor] = []
        all_adv: List[torch.Tensor] = []
        all_ret: List[torch.Tensor] = []

        for agent_index, aid in enumerate(self.agent_ids):
            buf = self.buffers[aid]
            n = buf.size()
            if n == 0:
                continue

            adv, ret = buf.compute_gae(
                next_values[aid], self.gamma, self.gae_lambda
            )

            all_obs.append(buf.obs[:n])
            all_gs.append(buf.global_states[:n])
            all_acts.append(buf.actions[:n])
            all_old_lp.append(buf.log_probs[:n])
            all_agent_indices.append(
                torch.full((n,), agent_index, dtype=torch.long, device=self.device)
            )
            all_adv.append(adv)
            all_ret.append(ret)

        if not all_obs:
            return {}

        # Pool all agents' data
        obs_pool = torch.cat(all_obs, dim=0)
        gs_pool = torch.cat(all_gs, dim=0)
        acts_pool = torch.cat(all_acts, dim=0)
        old_lp_pool = torch.cat(all_old_lp, dim=0)
        agent_index_pool = torch.cat(all_agent_indices, dim=0)
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
                agent_indices_b = agent_index_pool[idx]
                adv_b = adv_pool[idx]
                ret_b = ret_pool[idx]

                # Actor loss
                new_lp_b, entropy_b = self.actor.evaluate_actions(obs_b, acts_b)
                ratio = (new_lp_b - old_lp_b).exp()
                pi_loss = torch.max(
                    -adv_b * ratio,
                    -adv_b * ratio.clamp(1 - self.clip_range, 1 + self.clip_range),
                ).mean()

                # Centralized critic loss
                critic_input_b = self._critic_inputs(gs_b, agent_indices_b)
                value_pred = self.critic(critic_input_b)
                vf_loss = nn.functional.mse_loss(value_pred, ret_b)

                # Entropy bonus
                entropy = entropy_b.mean()

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
                "critic_mode": "agent_conditioned_v1",
            },
            path,
        )

    def load(self, path: str) -> None:
        from utils.torch_io import safe_load
        ckpt = safe_load(path, map_location=self.device)
        checkpoint_obs_dim = int(ckpt.get("obs_dim", self.obs_dim))
        checkpoint_global_dim = int(
            ckpt.get("global_state_dim", self.global_state_dim)
        )
        checkpoint_critic_mode = ckpt.get("critic_mode")
        checkpoint_agent_ids = list(ckpt.get("agent_ids", []))
        if (
            checkpoint_obs_dim != self.obs_dim
            or checkpoint_global_dim != self.global_state_dim
            or checkpoint_critic_mode != "agent_conditioned_v1"
            or checkpoint_agent_ids != self.agent_ids
        ):
            raise ValueError(
                "Incompatible MAPPO checkpoint: "
                f"checkpoint obs/global={checkpoint_obs_dim}/{checkpoint_global_dim}, "
                f"current={self.obs_dim}/{self.global_state_dim}; "
                f"checkpoint agents={checkpoint_agent_ids}, current={self.agent_ids}. "
                "Agent-conditioned critic checkpoints require matching dimensions "
                "and agent order."
            )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
