"""Multi-Agent PPO (MAPPO) agent.

Architecture
------------
Shared actor
    All trainable agents share one :class:`~agents.common.networks.Actor`.
    Each agent uses its own local observation to select actions.

Centralized critic
    A single :class:`~agents.common.networks.Critic` takes the **global state**
    (``env.get_global_state().vector``). It either estimates one team value
    ``V(s)`` or appends focal-agent identity to estimate ``V_i(s)``.
    This is the CTDE (Centralized Training, Decentralized Execution) pattern:
    the critic sees everything during training but the actor only uses local obs.

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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
        packed_storage: Optional[torch.Tensor] = None,
    ) -> None:
        self.n_steps = n_steps
        self.device = device
        row_dim = obs_dim + global_state_dim + action_dim + 5
        storage = packed_storage
        if storage is None:
            storage = torch.zeros(n_steps, row_dim, device=device)
        if storage.shape != (n_steps, row_dim):
            raise ValueError(
                f"Expected packed rollout storage shape {(n_steps, row_dim)}, "
                f"got {tuple(storage.shape)}."
            )
        self._packed = storage
        obs_end = obs_dim
        global_state_end = obs_end + global_state_dim
        action_end = global_state_end + action_dim
        self.obs = storage[:, :obs_end]
        self.global_states = storage[:, obs_end:global_state_end]
        self.actions = storage[:, global_state_end:action_end]
        self.rewards = storage[:, action_end]
        self.log_probs = storage[:, action_end + 1]
        self.values = storage[:, action_end + 2]
        self.terminated = storage[:, action_end + 3]
        self.truncated = storage[:, action_end + 4]
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
        if self.device.type == "cuda":
            # Preserve the existing Python-double recurrence while replacing
            # thousands of individual CUDA scalar reads with one bulk copy.
            rollout = torch.stack(
                (
                    self.rewards[:n],
                    self.values[:n],
                    self.terminated[:n],
                    self.truncated[:n],
                ),
                dim=1,
            ).cpu().numpy()
            advantages_host = np.zeros(n, dtype=np.float32)
            last_gae = 0.0
            next_val = float(next_value)
            for t in reversed(range(n)):
                terminated = float(rollout[t, 2])
                truncated = float(rollout[t, 3])
                bootstrap_mask = 1.0 - terminated
                continuation_mask = 1.0 - float(
                    bool(terminated) or bool(truncated)
                )
                nv = next_val if t == n - 1 else float(rollout[t + 1, 1])
                delta = (
                    float(rollout[t, 0])
                    + gamma * nv * bootstrap_mask
                    - float(rollout[t, 1])
                )
                last_gae = (
                    delta
                    + gamma * gae_lambda * continuation_mask * last_gae
                )
                advantages_host[t] = last_gae
            advantages = torch.as_tensor(
                advantages_host, dtype=torch.float32, device=self.device
            )
            returns = advantages + self.values[:n]
            return advantages, returns

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
        if not agent_ids:
            raise ValueError("MAPPO requires at least one trainable agent ID.")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("MAPPO trainable agent IDs must be unique and ordered.")
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = len(self.action_low)
        self.agent_ids = list(agent_ids)
        self._agent_index = {aid: idx for idx, aid in enumerate(self.agent_ids)}

        self.critic_mode = str(params.get("critic_mode", "agent_conditioned")).strip().lower()
        if self.critic_mode not in {"shared_team", "agent_conditioned"}:
            raise ValueError(
                "MAPPO critic_mode must be 'shared_team' or 'agent_conditioned', "
                f"got {self.critic_mode!r}."
            )
        self.reward_mode = str(params.get("reward_mode", "individual")).strip().lower()
        self.team_reward_reduction = str(
            params.get("team_reward_reduction", "mean")
        ).strip().lower()
        if self.reward_mode not in {"individual", "team_shared"}:
            raise ValueError(
                "MAPPO reward_mode must be 'individual' or 'team_shared', "
                f"got {self.reward_mode!r}."
            )
        if self.team_reward_reduction not in {"mean", "sum"}:
            raise ValueError(
                "MAPPO team_reward_reduction must be 'mean' or 'sum', "
                f"got {self.team_reward_reduction!r}."
            )
        if self.reward_mode == "individual" and self.critic_mode == "shared_team":
            raise ValueError(
                "MAPPO individual rewards require critic_mode='agent_conditioned'."
            )

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
        self.actor_hidden_dims = list(hidden_dims)
        self.critic_hidden_dims = list(vf_dims)
        self.activation = activation

        device_str = str(params.get("device", "cpu"))
        self.device = resolve_device([device_str])

        # Shared actor (local obs → action)
        self.actor = Actor(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)

        # The team critic estimates one shared V(s).  The agent-conditioned
        # critic estimates V_i(s) by appending a focal-agent one-hot vector.
        # In both cases the actor remains decentralized and sees local obs only.
        self.critic_input_dim = global_state_dim + (
            len(self.agent_ids) if self.critic_mode == "agent_conditioned" else 0
        )
        self.critic = Critic(self.critic_input_dim, vf_dims, activation).to(self.device)
        self._agent_identity = torch.eye(
            len(self.agent_ids), dtype=torch.float32, device=self.device
        )

        self._optim_parameters = tuple(self.actor.parameters()) + tuple(
            self.critic.parameters()
        )
        self.optimizer = optim.Adam(self._optim_parameters, lr=self.lr)

        # Per-agent rollout buffers
        rollout_row_dim = obs_dim + global_state_dim + self.action_dim + 5
        self._rollout_storage = torch.zeros(
            len(self.agent_ids), self.n_steps, rollout_row_dim, device=self.device
        )
        self._rollout_agent_indices: Dict[Tuple[str, ...], torch.Tensor] = {
            tuple(self.agent_ids): torch.arange(
                len(self.agent_ids), dtype=torch.long, device=self.device
            )
        }
        self.buffers: Dict[str, MAPPORolloutBuffer] = {
            aid: MAPPORolloutBuffer(
                self.n_steps,
                obs_dim,
                global_state_dim,
                self.action_dim,
                self.device,
                packed_storage=self._rollout_storage[index],
            )
            for index, aid in enumerate(self.agent_ids)
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
    def act_batch(
        self,
        agent_ids: Sequence[str],
        observations: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Select actions for an ordered active-agent batch with one actor call."""
        ordered_ids = self._validate_agent_batch(agent_ids)
        if not ordered_ids:
            return {}, {}
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (len(ordered_ids), self.obs_dim):
            raise ValueError(
                "Expected batched local observations with shape "
                f"({len(ordered_ids)}, {self.obs_dim}), got {obs.shape}."
            )
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_t, log_prob_t = self.actor.get_action(
            obs_t, deterministic=deterministic
        )
        # One device-to-host transfer for the complete joint decision.
        result = torch.cat((action_t, log_prob_t.unsqueeze(-1)), dim=-1).cpu().numpy()
        actions = {
            agent_id: result[index, : self.action_dim].copy()
            for index, agent_id in enumerate(ordered_ids)
        }
        log_probs = {
            agent_id: float(result[index, self.action_dim])
            for index, agent_id in enumerate(ordered_ids)
        }
        return actions, log_probs

    def _validate_agent_batch(self, agent_ids: Sequence[str]) -> List[str]:
        ordered_ids = [str(agent_id) for agent_id in agent_ids]
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("MAPPO inference batches cannot contain duplicate agent IDs.")
        unknown = [agent_id for agent_id in ordered_ids if agent_id not in self._agent_index]
        if unknown:
            raise ValueError(f"MAPPO inference batch contains unknown agent IDs: {unknown}.")
        return ordered_ids

    def _critic_input(
        self,
        global_state: np.ndarray,
        agent_id: Optional[str] = None,
    ) -> np.ndarray:
        state = np.asarray(global_state, dtype=np.float32).reshape(-1)
        if state.size != self.global_state_dim:
            raise ValueError(
                f"Expected global state dimension {self.global_state_dim}, got {state.size}."
            )
        if self.critic_mode == "shared_team":
            return state
        if agent_id not in self._agent_index:
            raise ValueError(
                "agent_conditioned critic requires a known agent_id; "
                f"got {agent_id!r}."
            )
        identity = np.zeros(len(self.agent_ids), dtype=np.float32)
        identity[self._agent_index[agent_id]] = 1.0
        return np.concatenate((state, identity))

    def _critic_batch(self, global_states: torch.Tensor, agent_id: str) -> torch.Tensor:
        if self.critic_mode == "shared_team":
            return global_states
        identity = self._agent_identity[self._agent_index[agent_id]].expand(
            global_states.shape[0], -1
        )
        return torch.cat((global_states, identity), dim=-1)

    @torch.no_grad()
    def evaluate_state(
        self,
        global_state: np.ndarray,
        agent_id: Optional[str] = None,
    ) -> float:
        """Estimate value of a global state using the centralized critic.

        Parameters
        ----------
        global_state:
            Flat numpy array from ``env.get_global_state().vector``.
        """
        critic_input = self._critic_input(global_state, agent_id)
        # GlobalState vectors are intentionally read-only. Copy into owned
        # tensor storage rather than aliasing immutable NumPy memory.
        gs_t = torch.tensor(
            critic_input, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return float(self.critic(gs_t).squeeze())

    @torch.no_grad()
    def evaluate_states(
        self,
        global_state: np.ndarray,
        agent_ids: Sequence[str],
    ) -> Dict[str, float]:
        """Estimate ordered per-agent centralized values with one critic call."""
        ordered_ids = self._validate_agent_batch(agent_ids)
        if not ordered_ids:
            return {}
        state = np.asarray(global_state, dtype=np.float32).reshape(-1)
        if state.size != self.global_state_dim:
            raise ValueError(
                f"Expected global state dimension {self.global_state_dim}, got {state.size}."
            )
        # GlobalState vectors are intentionally read-only. Copy into owned
        # tensor storage rather than aliasing immutable NumPy memory.
        states_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(len(ordered_ids), -1)
        if self.critic_mode == "agent_conditioned":
            indices = torch.as_tensor(
                [self._agent_index[agent_id] for agent_id in ordered_ids],
                dtype=torch.long,
                device=self.device,
            )
            identities = self._agent_identity.index_select(0, indices)
            states_t = torch.cat((states_t, identities), dim=-1)
        values = self.critic(states_t).cpu().numpy()
        return {
            agent_id: float(values[index])
            for index, agent_id in enumerate(ordered_ids)
        }

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

    def store_batch(
        self,
        agent_ids: Sequence[str],
        *,
        observations: Mapping[str, np.ndarray],
        global_state: np.ndarray,
        actions: Mapping[str, np.ndarray],
        rewards: Mapping[str, float],
        log_probs: Mapping[str, float],
        values: Mapping[str, float],
        terminated: Mapping[str, bool],
        truncated: Mapping[str, bool],
    ) -> None:
        """Insert one transition per ordered active agent with one tensor transfer."""
        ordered_ids = self._validate_agent_batch(agent_ids)
        if not ordered_ids:
            return
        state = np.asarray(global_state, dtype=np.float32).reshape(-1)
        if state.size != self.global_state_dim:
            raise ValueError(
                f"Expected global state dimension {self.global_state_dim}, got {state.size}."
            )
        obs_batch = np.stack(
            [np.asarray(observations[aid], dtype=np.float32) for aid in ordered_ids]
        )
        action_batch = np.stack(
            [np.asarray(actions[aid], dtype=np.float32) for aid in ordered_ids]
        )
        if obs_batch.shape != (len(ordered_ids), self.obs_dim):
            raise ValueError(
                f"Expected observation batch shape ({len(ordered_ids)}, {self.obs_dim}), "
                f"got {obs_batch.shape}."
            )
        if action_batch.shape != (len(ordered_ids), self.action_dim):
            raise ValueError(
                f"Expected action batch shape ({len(ordered_ids)}, {self.action_dim}), "
                f"got {action_batch.shape}."
            )
        scalars = np.asarray(
            [
                [
                    rewards[aid],
                    log_probs[aid],
                    values[aid],
                    terminated[aid],
                    truncated[aid],
                ]
                for aid in ordered_ids
            ],
            dtype=np.float32,
        )
        packed = np.concatenate(
            (
                obs_batch,
                np.broadcast_to(state, (len(ordered_ids), state.size)),
                action_batch,
                scalars,
            ),
            axis=1,
        )
        packed_t = torch.as_tensor(
            packed, dtype=torch.float32, device=self.device
        )
        id_key = tuple(ordered_ids)
        agent_indices = self._rollout_agent_indices.get(id_key)
        if agent_indices is None:
            agent_indices = torch.as_tensor(
                [self._agent_index[agent_id] for agent_id in ordered_ids],
                dtype=torch.long,
                device=self.device,
            )
            self._rollout_agent_indices[id_key] = agent_indices
        buffer_indices = [
            self.buffers[agent_id].ptr % self.n_steps for agent_id in ordered_ids
        ]
        if len(set(buffer_indices)) == 1:
            self._rollout_storage[agent_indices, buffer_indices[0]] = packed_t
        else:
            step_indices = torch.as_tensor(
                buffer_indices, dtype=torch.long, device=self.device
            )
            self._rollout_storage[agent_indices, step_indices] = packed_t
        for agent_id in ordered_ids:
            self.buffers[agent_id].ptr += 1

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
        rollout_agent_ids = [
            aid for aid in self.agent_ids if self.buffers[aid].size() > 0
        ]
        if not rollout_agent_ids:
            return {}

        # Assemble one packed update tensor. Every minibatch then needs one
        # advanced-index gather instead of six independent gathers, and the
        # preallocated identity basis is copied directly into critic inputs.
        n_pool = sum(self.buffers[aid].size() for aid in rollout_agent_ids)
        obs_end = self.obs_dim
        gs_end = obs_end + self.critic_input_dim
        acts_end = gs_end + self.action_dim
        old_lp_index = acts_end
        adv_index = acts_end + 1
        ret_index = acts_end + 2
        update_pool = torch.empty(
            (n_pool, ret_index + 1), dtype=torch.float32, device=self.device
        )

        next_values = self.evaluate_states(next_global_state, rollout_agent_ids)
        row_start = 0
        for aid in rollout_agent_ids:
            buf = self.buffers[aid]
            n = buf.size()
            next_value = next_values[aid]
            adv, ret = buf.compute_gae(next_value, self.gamma, self.gae_lambda)
            rows = update_pool[row_start : row_start + n]
            rows[:, :obs_end] = buf.obs[:n]
            rows[:, obs_end : obs_end + self.global_state_dim] = (
                buf.global_states[:n]
            )
            if self.critic_mode == "agent_conditioned":
                identity = self._agent_identity[self._agent_index[aid]]
                rows[:, obs_end + self.global_state_dim : gs_end] = identity
            rows[:, gs_end:acts_end] = buf.actions[:n]
            rows[:, old_lp_index] = buf.log_probs[:n]
            rows[:, adv_index] = adv
            rows[:, ret_index] = ret
            row_start += n

        # Normalize advantages over the pooled set
        adv_pool = update_pool[:, adv_index]
        adv_std = adv_pool.std(correction=0)
        update_pool[:, adv_index] = (
            adv_pool - adv_pool.mean()
        ) / (adv_std + 1e-8)

        metric_rows: List[torch.Tensor] = []

        for _ in range(self.n_epochs):
            idx_all = torch.randperm(n_pool, device=self.device)
            for start in range(0, n_pool, self.batch_size):
                idx = idx_all[start:start + self.batch_size]
                batch = update_pool[idx]
                obs_b = batch[:, :obs_end]
                gs_b = batch[:, obs_end:gs_end]
                acts_b = batch[:, gs_end:acts_end]
                old_lp_b = batch[:, old_lp_index]
                adv_b = batch[:, adv_index]
                ret_b = batch[:, ret_index]

                # PPO compares the current policy probability with the
                # probability recorded for the same rollout action.  Sampling
                # a replacement action here would make the importance ratio
                # unrelated to the collected transition.
                new_lp_b, entropy_b = self.actor.evaluate_actions(obs_b, acts_b)
                ratio = (new_lp_b - old_lp_b).exp()
                pi_loss = torch.max(
                    -adv_b * ratio,
                    -adv_b * ratio.clamp(1 - self.clip_range, 1 + self.clip_range),
                ).mean()

                # Centralized critic loss
                value_pred = self.critic(gs_b)
                vf_loss = nn.functional.mse_loss(value_pred, ret_b)

                # Entropy of the current unsquashed Gaussian policy.
                entropy = entropy_b.mean()

                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._optim_parameters,
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((old_lp_b - new_lp_b).mean()).abs()
                    metric_rows.append(
                        torch.stack(
                            (
                                pi_loss.detach(),
                                vf_loss.detach(),
                                entropy.detach(),
                                approx_kl,
                            )
                        )
                    )

        if not metric_rows:
            return {
                "train/policy_loss": 0.0,
                "train/value_loss": 0.0,
                "train/entropy": 0.0,
                "train/approx_kl": 0.0,
            }
        averages = torch.stack(metric_rows).mean(dim=0).cpu().tolist()
        return {
            "train/policy_loss": averages[0],
            "train/value_loss": averages[1],
            "train/entropy": averages[2],
            "train/approx_kl": averages[3],
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def load_pretrained_actor(self, path: str) -> None:
        """Initialize only the shared actor from a PPO checkpoint.

        MAPPO's centralized critic and fresh optimizer state are intentionally
        retained.  Local observation and physical action contracts must match.
        """
        from utils.torch_io import safe_load

        ckpt = safe_load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt:
            raise ValueError(f"Pretrained PPO checkpoint has no actor state: {path}")
        if "algorithm" in ckpt and str(ckpt["algorithm"]).lower() != "ppo":
            raise ValueError(
                "Pretrained actor checkpoint must come from PPO; "
                f"found algorithm={ckpt['algorithm']!r}."
            )

        checks = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_hidden_dims": self.actor_hidden_dims,
            "activation": self.activation,
        }
        for key, expected in checks.items():
            if key in ckpt and ckpt[key] != expected:
                raise ValueError(
                    f"Incompatible pretrained PPO actor {key}: "
                    f"checkpoint={ckpt[key]!r}, MAPPO={expected!r}."
                )
        for key, expected in (
            ("action_low", self.action_low),
            ("action_high", self.action_high),
        ):
            if key in ckpt and not np.allclose(
                np.asarray(ckpt[key], dtype=np.float32), expected
            ):
                raise ValueError(
                    f"Incompatible pretrained PPO actor {key}: physical action bounds differ."
                )
        try:
            self.actor.load_state_dict(ckpt["actor"], strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Incompatible pretrained PPO actor network architecture: " + str(exc)
            ) from exc

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "algorithm": "mappo",
                "agent_ids": self.agent_ids,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "global_state_dim": self.global_state_dim,
                "critic_input_dim": self.critic_input_dim,
                "critic_mode": self.critic_mode,
                "reward_mode": self.reward_mode,
                "team_reward_reduction": self.team_reward_reduction,
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "activation": self.activation,
            },
            path,
        )

    def load(self, path: str) -> None:
        from utils.torch_io import safe_load
        ckpt = safe_load(path, map_location=self.device)
        if "critic_mode" not in ckpt or "reward_mode" not in ckpt:
            raise ValueError(
                "MAPPO checkpoint predates the explicit reward/critic contract; "
                "start a new experiment with a contract-aware checkpoint."
            )
        checkpoint_obs_dim = int(ckpt.get("obs_dim", self.obs_dim))
        checkpoint_global_dim = int(
            ckpt.get("global_state_dim", self.global_state_dim)
        )
        checkpoint_critic_mode = str(ckpt["critic_mode"])
        checkpoint_reward_mode = str(ckpt["reward_mode"])
        checkpoint_reduction = str(ckpt.get("team_reward_reduction", "mean"))
        checkpoint_agent_ids = list(ckpt.get("agent_ids", self.agent_ids))
        if (
            checkpoint_obs_dim != self.obs_dim
            or checkpoint_global_dim != self.global_state_dim
            or checkpoint_agent_ids != self.agent_ids
            or checkpoint_critic_mode != self.critic_mode
            or checkpoint_reward_mode != self.reward_mode
            or checkpoint_reduction != self.team_reward_reduction
        ):
            raise ValueError(
                "Incompatible MAPPO checkpoint contract: "
                f"checkpoint obs/global={checkpoint_obs_dim}/{checkpoint_global_dim}, "
                f"current={self.obs_dim}/{self.global_state_dim}; "
                f"checkpoint agents={checkpoint_agent_ids!r}, "
                f"current={self.agent_ids!r}; "
                f"checkpoint critic_mode={checkpoint_critic_mode!r}, "
                f"current={self.critic_mode!r}; "
                f"checkpoint reward={checkpoint_reward_mode}/{checkpoint_reduction}, "
                f"current={self.reward_mode}/{self.team_reward_reduction}. "
                "Use a checkpoint created with the same lifecycle-state dimensions "
                "and MAPPO reward/critic contract."
            )
        scalar_checks = {
            "algorithm": "mappo",
            "action_dim": self.action_dim,
            "actor_hidden_dims": self.actor_hidden_dims,
            "critic_hidden_dims": self.critic_hidden_dims,
            "activation": self.activation,
        }
        for key, expected in scalar_checks.items():
            if key in ckpt and ckpt[key] != expected:
                raise ValueError(
                    f"Incompatible MAPPO checkpoint {key}: "
                    f"checkpoint={ckpt[key]!r}, current={expected!r}."
                )
        for key, expected in (
            ("action_low", self.action_low),
            ("action_high", self.action_high),
        ):
            if key in ckpt:
                actual = np.asarray(ckpt[key], dtype=np.float32)
                if actual.shape != expected.shape or not np.allclose(actual, expected):
                    raise ValueError(
                        f"Incompatible MAPPO checkpoint {key}: action bounds differ."
                    )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
