"""Multi-Agent PPO (MAPPO) agent.

Architecture
------------
Actors
    Shared, independent full actors, or a frozen base with routed LoRA adapters.
    Each agent uses its own local observation; routed policies retain learner
    identity through inference and PPO minibatches.

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
    and the selected actors plus the centralized critic are updated via PPO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim

from agents.common import Actor, Critic, compute_gae, ppo_minibatch_step, mean_update_metrics
from agents.common.lora import LoRAActor, resolve_lora_config
from agents.common.independent import IndependentActors
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
        self.raw_actions = torch.full_like(self.actions, float("nan"))
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
        raw_action: Optional[np.ndarray] = None,
    ) -> None:
        i = self.ptr % self.n_steps
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.global_states[i] = torch.as_tensor(
            global_state, dtype=torch.float32, device=self.device
        )
        self.actions[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.raw_actions[i] = (float("nan") if raw_action is None else
                               torch.as_tensor(raw_action, dtype=torch.float32, device=self.device))
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
        self.raw_actions.fill_(float("nan"))

    def size(self) -> int:
        return min(self.ptr, self.n_steps)

    def compute_gae(
        self,
        next_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.size()
        return compute_gae(
            self.rewards[:n], self.values[:n], self.terminated[:n], self.truncated[:n],
            next_value, gamma, gae_lambda,
        )

# ---------------------------------------------------------------------------
# MAPPO agent
# ---------------------------------------------------------------------------

class MAPPOAgent:
    """Multi-Agent PPO with shared or independent actors and one centralized critic.

    Optional LoRA residuals are shared or selected by trainable agent identity.

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
        self.global_state_contract_version = str(
            params.get("_global_state_contract_version", "legacy_unspecified")
        )
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = len(self.action_low)
        self.action_contract = dict(params.get("_action_contract", {"speed_control": "direct"}))
        self.physics_contract = params.get("_physics_contract")
        self.observation_contract = params.get("_observation_contract")
        self.pretrained_actor_observation_extension = params.get("pretrained_actor_observation_extension")
        if self.pretrained_actor_observation_extension not in (None, "frenet_neighbors"):
            raise ValueError("pretrained_actor_observation_extension must be null or frenet_neighbors")
        self.agent_ids = list(agent_ids)
        self._agent_index = {aid: idx for idx, aid in enumerate(self.agent_ids)}
        self.actor_mode = str(params.get("actor_mode", "shared"))
        if self.actor_mode not in {"shared", "independent"}:
            raise ValueError("actor_mode must be shared or independent")
        self.lora_config = resolve_lora_config(params.get("lora"))
        if self.actor_mode == "independent" and self.lora_config is not None:
            raise ValueError("Independent actors cannot also use LoRA; use shared with per_agent adapters")
        self.pretrained_actor_source = None
        self._lora_ready = self.lora_config is None
        self.last_raw_actions: Dict[str, np.ndarray] = {}

        self.critic_mode = str(params.get("critic_mode", "agent_conditioned")).strip().lower()
        if self.critic_mode not in {"shared_team", "agent_conditioned"}:
            raise ValueError(
                "MAPPO critic_mode must be 'shared_team' or 'agent_conditioned', "
                f"got {self.critic_mode!r}."
            )
        self.reward_mode = str(params.get("reward_mode", "individual")).strip().lower()
        self.team_return_mode = str(params.get("team_return_mode", "per_agent"))
        if self.team_return_mode not in {"per_agent", "joint"}:
            raise ValueError("team_return_mode must be per_agent or joint")
        if self.team_return_mode == "joint" and (
            self.reward_mode != "team_shared" or self.critic_mode != "shared_team"
        ):
            raise ValueError("Joint team returns require team_shared rewards and shared_team critic")
        self._team_rollout: List[Tuple[float, float, bool]] = []
        self._team_step_indices: Dict[str, List[int]] = {aid: [] for aid in agent_ids}
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

        # Build adapters after the critic so its initialization matches the
        # full-fine-tuning control under the same seed.
        if self.actor_mode == "independent":
            self.actor = IndependentActors(self.actor, self.agent_ids).to(self.device)
        self.lora_contract = None
        if self.lora_config is not None:
            self.actor = LoRAActor(self.actor, self.lora_config, len(self.agent_ids)).to(self.device)
            self.lora_contract = {
                "version": 1, **self.lora_config,
                "target_layers": self.actor.target_layers,
                "agent_to_adapter": {aid: (i if self.lora_config["mode"] == "per_agent" else 0)
                                     for i, aid in enumerate(self.agent_ids)},
            }
        self._optim_parameters = tuple(p for p in self.actor.parameters() if p.requires_grad) + tuple(
            self.critic.parameters())
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

    @property
    def per_agent_adapters(self) -> bool:
        return self.lora_config is not None and self.lora_config["mode"] == "per_agent"

    @property
    def routed_actor(self) -> bool:
        return self.actor_mode == "independent" or self.per_agent_adapters

    def _require_lora_source(self):
        if not self._lora_ready:
            raise ValueError("LoRA requires a pretrained PPO actor or a matching MAPPO checkpoint before use")

    def actor_actions(self, observations, agent_ids, *, deterministic=False, return_raw=False):
        """Route rows, including repeated IDs from independent environments."""
        self._require_lora_source()
        if len(agent_ids) != len(observations) or any(aid not in self._agent_index for aid in agent_ids):
            raise ValueError("Actor rows require matching, known agent IDs")
        kwargs = {}
        if self.routed_actor:
            kwargs["adapter_indices"] = torch.tensor(
                [self._agent_index[aid] for aid in agent_ids], device=self.device, dtype=torch.long)
        return self.actor.get_action(observations, deterministic=deterministic,
                                     return_raw=return_raw, **kwargs)

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False, *, agent_id: Optional[str] = None
    ) -> Tuple[np.ndarray, float]:
        """Sample a local action; routed policies require the agent ID.

        Returns
        -------
        action_normalized : np.ndarray
            Action in ``[-1, 1]`` — caller denormalizes for ``env.step()``.
        log_prob : float
            Log probability of the sampled action.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.routed_actor and agent_id is None:
            raise ValueError("Routed actor act requires agent_id")
        action_t, log_prob_t = self.actor_actions(
            obs_t, [agent_id if agent_id is not None else self.agent_ids[0]], deterministic=deterministic)
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
        action_t, log_prob_t, raw_t = self.actor_actions(
            obs_t, ordered_ids, deterministic=deterministic, return_raw=True
        )
        # One device-to-host transfer for the complete joint decision.
        result = torch.cat((action_t, log_prob_t.unsqueeze(-1), raw_t), dim=-1).cpu().numpy()
        self.last_raw_actions = {
            aid: result[index, self.action_dim + 1:].copy()
            for index, aid in enumerate(ordered_ids)
        }
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
        raw_actions: Optional[Mapping[str, np.ndarray]] = None,
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
        if raw_actions is not None:
            raw_batch = np.stack([raw_actions[aid] for aid in ordered_ids])
            if raw_batch.shape != action_batch.shape or not np.isfinite(raw_batch).all():
                raise ValueError("Pre-tanh actions must be finite and match the action batch shape")
            packed = np.concatenate((packed, raw_batch), axis=1)
        packed_t = torch.as_tensor(
            packed, dtype=torch.float32, device=self.device
        )
        raw_t = None
        if raw_actions is not None:
            packed_t, raw_t = packed_t[:, :-self.action_dim], packed_t[:, -self.action_dim:]
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
        for index, agent_id in enumerate(ordered_ids):
            buffer = self.buffers[agent_id]
            buffer.raw_actions[buffer.ptr % self.n_steps] = (
                float("nan") if raw_t is None else raw_t[index]
            )
            self.buffers[agent_id].ptr += 1

    def any_buffer_full(self) -> bool:
        """True when any agent's buffer has reached ``n_steps``."""
        return len(self._team_rollout) >= self.n_steps or any(buf.is_full() for buf in self.buffers.values())

    def store_team_step(self, agent_ids: Sequence[str], *, reward: float,
                        value: float, terminal: bool) -> None:
        """Record one joint reward/value, with indices only for actual decisions."""
        if self.team_return_mode != "joint" or len(self._team_rollout) >= self.n_steps:
            raise ValueError("Joint team rollout is disabled or full")
        for aid in agent_ids:
            if self.buffers[aid].size() != len(self._team_step_indices[aid]) + 1:
                raise ValueError("Store each agent decision before its joint team step")
            self._team_step_indices[aid].append(len(self._team_rollout))
        self._team_rollout.append((float(reward), float(value), bool(terminal)))

    def compute_team_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Continue shared credit after an individual car finishes or crashes.

        Rollout cuts bootstrap V(s); the finite race horizon (including timeout)
        is terminal only when no teammate can act. No inactive actor samples
        are manufactured. Earlier rollout fragments receive continuation through
        the shared critic, as in ordinary truncated PPO collection.
        """
        if not self._team_rollout:
            raise ValueError("Missing joint team rollout")
        rows = torch.tensor(self._team_rollout, dtype=torch.float32, device=self.device)
        return compute_gae(rows[:, 0], rows[:, 1], rows[:, 2], torch.zeros_like(rows[:, 2]),
                           next_value, self.gamma, self.gae_lambda)

    def clear_buffers(self) -> None:
        for buf in self.buffers.values():
            buf.clear()
        self._team_rollout.clear()
        for indices in self._team_step_indices.values():
            indices.clear()

    # ------------------------------------------------------------------
    # PPO update — pooled data with explicit actor routing and centralized values
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
        raw_pool = torch.cat([self.buffers[aid].raw_actions[:self.buffers[aid].size()]
                              for aid in rollout_agent_ids])

        next_values = self.evaluate_states(next_global_state, rollout_agent_ids)
        team_gae = (
            self.compute_team_gae(next_values[rollout_agent_ids[0]])
            if self.team_return_mode == "joint" else None
        )
        row_start = 0
        for aid in rollout_agent_ids:
            buf = self.buffers[aid]
            n = buf.size()
            next_value = next_values[aid]
            if team_gae is None:
                adv, ret = buf.compute_gae(next_value, self.gamma, self.gae_lambda)
            else:
                indices = self._team_step_indices[aid]
                if len(indices) != n:
                    raise ValueError("Joint team return indices must match actual agent decisions")
                adv, ret = (values[indices] for values in team_gae)
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

        agent_indices = None
        if self.routed_actor:
            agent_indices = torch.cat([torch.full(
                (self.buffers[aid].size(),), self._agent_index[aid], dtype=torch.long, device=self.device)
                for aid in rollout_agent_ids])
        return self._update_pool(update_pool, raw_pool, agent_indices)

    def update_rollouts(self, rollouts) -> Dict[str, float]:
        """Pool independently bootstrapped collector fragments for one PPO update."""
        rollouts = [item for item in rollouts if len(item[0])]
        if not rollouts:
            return {}
        pool = torch.as_tensor(np.concatenate([r[0] for r in rollouts]),
                               dtype=torch.float32, device=self.device)
        raw = torch.as_tensor(np.concatenate([r[1] for r in rollouts]),
                              dtype=torch.float32, device=self.device)
        agent_indices = None
        if self.routed_actor:
            if any(len(r) != 3 for r in rollouts):
                raise ValueError("Routed actor rollouts require actor identity")
            indices = np.concatenate([r[2] for r in rollouts])
            if (indices.shape != (len(pool),) or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= len(self.agent_ids))):
                raise ValueError("Invalid rollout actor identity")
            agent_indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self._update_pool(pool, raw, agent_indices)

    def _update_pool(self, update_pool, raw_pool, agent_indices=None) -> Dict[str, float]:
        self._require_lora_source()
        if self.routed_actor and (agent_indices is None or agent_indices.shape != (len(update_pool),)):
            raise ValueError("Routed actor update requires actor identity")
        n_pool = len(update_pool)
        obs_end = self.obs_dim
        gs_end = obs_end + self.critic_input_dim
        acts_end = gs_end + self.action_dim
        old_lp_index, adv_index, ret_index = acts_end, acts_end + 1, acts_end + 2
        # Specialists normalize their own advantages; shared/team policies pool them.
        adv_pool = update_pool[:, adv_index]
        if self.routed_actor and self.team_return_mode == "per_agent":
            for index in range(len(self.agent_ids)):
                rows = agent_indices == index
                if rows.any():
                    values = adv_pool[rows]
                    update_pool[rows, adv_index] = (values - values.mean()) / (values.std(correction=0) + 1e-8)
        else:
            adv_std = adv_pool.std(correction=0)
            update_pool[:, adv_index] = (adv_pool - adv_pool.mean()) / (adv_std + 1e-8)

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

                metric_rows.append(ppo_minibatch_step(
                    self, obs_b, gs_b, acts_b, old_lp_b, adv_b, ret_b,
                    raw_actions=raw_pool[idx],
                    adapter_indices=agent_indices[idx] if self.routed_actor else None,
                ))

        return mean_update_metrics(metric_rows)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _pretrained_observation_dim(self, checkpoint: Dict) -> int:
        """Allow only an explicit, appended neighbor block after the driving state."""
        source = checkpoint.get("observation_contract")
        if source == self.observation_contract:
            return self.obs_dim
        if self.pretrained_actor_observation_extension != "frenet_neighbors":
            raise ValueError("Incompatible checkpoint observation_contract; observation semantics differ")
        from copy import deepcopy
        target = deepcopy(self.observation_contract)
        if not isinstance(source, dict) or not isinstance(target, dict):
            raise ValueError("Neighbor extension requires explicit observation contracts")
        obs = target.get("observation", {})
        neighbors = obs.pop("frenet_neighbors", {})
        source_obs = source.get("observation", {})
        # The composer appends neighbors immediately after Frenet state. Restrict
        # this migration to that layout; never pad arbitrary or reordered inputs.
        enabled = {key for key, value in source_obs.items()
                   if isinstance(value, dict) and value.get("enabled", False)}
        if (target != source or enabled not in ({"frenet_vehicle_track"}, {"lidar", "frenet_vehicle_track"})
                or not neighbors.get("enabled", False)):
            raise ValueError("Neighbor extension requires an unchanged LiDAR/Frenet observation prefix")
        points = int(source_obs["frenet_vehicle_track"].get("points", 20))
        source_dim = 10 + 2 * points
        if "lidar" in enabled:
            source_dim += int(source.get("lidar_beams", 108))
        from wrappers.observations.neighbors import FrenetNeighborsComponent
        added_dim = FrenetNeighborsComponent(
            max_neighbors=int(neighbors.get("max_neighbors", 1)),
            include_team=bool(neighbors.get("include_team", False)),
            agent_ids=neighbors.get("agent_ids"),
        ).dim
        if (checkpoint.get("obs_dim") != source_dim or added_dim <= 0
                or self.obs_dim != source_dim + added_dim):
            raise ValueError("Neighbor extension observation dimensions do not match the contracts")
        return source_dim

    def load_pretrained_actor(self, path: str) -> None:
        """Initialize actors from a PPO or plain shared-MAPPO policy.

        MAPPO's centralized critic and fresh optimizer state are intentionally
        retained. Physical contracts must match. An explicitly configured
        neighbor extension preserves the existing actor with zero new weights.
        """
        from utils.torch_io import safe_load

        ckpt = safe_load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt:
            raise ValueError(f"Pretrained checkpoint has no single shared actor state: {path}")
        if ckpt.get("physics_contract") != self.physics_contract:
            raise ValueError("Incompatible checkpoint physics_contract; physics semantics differ")
        source_obs_dim = self._pretrained_observation_dim(ckpt)
        checkpoint_contract = ckpt.get("action_contract", {"speed_control": "direct"})
        if checkpoint_contract != self.action_contract:
            raise ValueError(
                "Incompatible pretrained PPO action contract (speed control semantics differ): "
                f"checkpoint={checkpoint_contract!r}, MAPPO={self.action_contract!r}. "
                "Match the MAPPO action_constraints and decision interval to the PPO "
                "training configuration, or select a compatible checkpoint."
            )
        source_algorithm = str(ckpt.get("algorithm", "ppo")).lower()
        if source_algorithm not in {"ppo", "mappo"}:
            raise ValueError("Pretrained actor checkpoint must come from PPO or shared MAPPO")
        if source_algorithm == "mappo" and (
            ckpt.get("actor_mode", "shared") != "shared" or ckpt.get("lora_contract") is not None
        ):
            raise ValueError("MAPPO initialization requires a plain shared actor, without adapters")

        checks = {
            "obs_dim": source_obs_dim,
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
        recipient = (self.actor.actors[self.agent_ids[0]]
                     if self.actor_mode == "independent" else self.actor)
        actor_state = dict(ckpt["actor"])
        if source_obs_dim != self.obs_dim:
            old_weight = actor_state.get("net.0.weight")
            expected = recipient.state_dict()["net.0.weight"]
            if old_weight is None or old_weight.shape != (expected.shape[0], source_obs_dim):
                raise ValueError("Incompatible pretrained actor first layer for neighbor extension")
            expanded = torch.zeros_like(expected)
            expanded[:, :source_obs_dim] = old_weight
            actor_state["net.0.weight"] = expanded
        # Validate all tensors before modifying the recipient, including failures
        # after the first layer (load_state_dict itself can partially mutate).
        expected_state = (self.actor.base_state_dict() if self.lora_config is not None
                          else recipient.state_dict())
        if (actor_state.keys() != expected_state.keys()
                or any(actor_state[key].shape != value.shape for key, value in expected_state.items())):
            raise ValueError("Incompatible pretrained PPO actor network architecture")
        try:
            if self.lora_config is not None:
                if self.optimizer.state:
                    raise ValueError("Initialize a pretrained LoRA base on a fresh agent")
                self.actor.reset_adapters()
                if self.lora_config.get("per_agent_log_std"):
                    actor_state.update({f"log_stds.{i}": actor_state["log_std"].clone()
                                        for i in range(len(self.agent_ids))})
                actor_state = {**self.actor.state_dict(), **actor_state}
            if self.actor_mode == "independent":
                for actor in self.actor.actors.values():
                    actor.load_state_dict(actor_state, strict=True)
            else:
                self.actor.load_state_dict(actor_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Incompatible pretrained PPO actor network architecture: " + str(exc)
            ) from exc
        import hashlib
        self.pretrained_actor_source = {
            "path": str(Path(path).resolve()),
            "algorithm": source_algorithm,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        self._lora_ready = True

    def save(self, path: str) -> None:
        self._require_lora_source()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **({"actors": {aid: actor.state_dict() for aid, actor in self.actor.actors.items()}}
                   if self.actor_mode == "independent" else {"actor": self.actor.state_dict()}),
                "actor_mode": self.actor_mode,
                "actor_routing": dict(self._agent_index),
                "advantage_normalization": ("per_agent" if self.routed_actor and self.team_return_mode == "per_agent" else "pooled"),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "algorithm": "mappo",
                "agent_ids": self.agent_ids,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "action_contract": self.action_contract,
                "physics_contract": self.physics_contract,
                "observation_contract": self.observation_contract,
                "global_state_dim": self.global_state_dim,
                "global_state_contract_version": self.global_state_contract_version,
                "critic_input_dim": self.critic_input_dim,
                "critic_mode": self.critic_mode,
                "reward_mode": self.reward_mode,
                "team_reward_reduction": self.team_reward_reduction,
                "team_return_mode": self.team_return_mode,
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "activation": self.activation,
                "lora_contract": self.lora_contract,
                "pretrained_actor_source": self.pretrained_actor_source,
            },
            path,
        )

    def load(self, path: str) -> None:
        from utils.torch_io import safe_load
        ckpt = safe_load(path, map_location=self.device)
        if ckpt.get("actor_mode", "shared") != self.actor_mode:
            raise ValueError("Incompatible MAPPO actor_mode; shared and independent checkpoints are distinct")
        if ckpt.get("actor_routing", self._agent_index) != self._agent_index:
            raise ValueError("Incompatible MAPPO actor routing contract")
        if self.actor_mode == "independent":
            actors = ckpt.get("actors", {})
            if set(actors) != set(self.agent_ids):
                raise ValueError("Independent checkpoint must contain every learner actor")
            ckpt["actor"] = {f"actors.{aid}.{key}": value for aid, state in actors.items()
                             for key, value in state.items()}
        if ckpt.get("lora_contract") != self.lora_contract:
            raise ValueError("Incompatible MAPPO LoRA contract; mode, rank, alpha, layers and routing must match")
        if self.lora_config is not None and not isinstance(ckpt.get("pretrained_actor_source"), dict):
            raise ValueError("LoRA checkpoint is missing its pretrained actor source")
        for key in ("physics_contract", "observation_contract"):
            if ckpt.get(key) != getattr(self, key):
                raise ValueError(f"Incompatible checkpoint {key}; physics/observation semantics differ")
        if ckpt.get("team_return_mode", "per_agent") != self.team_return_mode:
            raise ValueError("Incompatible MAPPO checkpoint team return contract")
        if ckpt.get("action_contract", {"speed_control": "direct"}) != self.action_contract:
            raise ValueError("Incompatible MAPPO checkpoint action contract (speed control semantics differ).")
        if "critic_mode" not in ckpt or "reward_mode" not in ckpt:
            raise ValueError(
                "MAPPO checkpoint predates the explicit reward/critic contract; "
                "start a new experiment with a contract-aware checkpoint."
            )
        checkpoint_obs_dim = int(ckpt.get("obs_dim", self.obs_dim))
        checkpoint_global_dim = int(
            ckpt.get("global_state_dim", self.global_state_dim)
        )
        checkpoint_global_contract = str(
            ckpt.get("global_state_contract_version", "legacy_unspecified")
        )
        checkpoint_critic_mode = str(ckpt["critic_mode"])
        checkpoint_reward_mode = str(ckpt["reward_mode"])
        checkpoint_reduction = str(ckpt.get("team_reward_reduction", "mean"))
        checkpoint_agent_ids = list(ckpt.get("agent_ids", self.agent_ids))
        if (
            checkpoint_obs_dim != self.obs_dim
            or checkpoint_global_dim != self.global_state_dim
            or checkpoint_global_contract != self.global_state_contract_version
            or checkpoint_agent_ids != self.agent_ids
            or checkpoint_critic_mode != self.critic_mode
            or checkpoint_reward_mode != self.reward_mode
            or checkpoint_reduction != self.team_reward_reduction
        ):
            raise ValueError(
                "Incompatible MAPPO checkpoint contract: "
                f"checkpoint obs/global={checkpoint_obs_dim}/{checkpoint_global_dim}, "
                f"current={self.obs_dim}/{self.global_state_dim}; "
                f"checkpoint global contract={checkpoint_global_contract!r}, "
                f"current={self.global_state_contract_version!r}; "
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
        for key, module in (("actor", self.actor), ("critic", self.critic)):
            expected = module.state_dict()
            actual = ckpt[key]
            if (actual.keys() != expected.keys()
                    or any(actual[name].shape != value.shape for name, value in expected.items())):
                raise ValueError(f"Incompatible MAPPO {key} network architecture")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.pretrained_actor_source = ckpt.get("pretrained_actor_source")
        self._lora_ready = True
