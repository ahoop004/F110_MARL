"""Shared discrete-action utilities for DQN-style agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import warnings
import torch

from f110x.policies.buffers import PrioritizedReplayBuffer, ReplayBuffer


@dataclass(slots=True)
class ReplaySample:
    """Structured batch sampled from replay for discrete-action agents."""

    obs: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    action_indices: torch.Tensor
    weights: torch.Tensor
    infos: Iterable[Optional[Dict[str, Any]]]
    indices: Optional[np.ndarray]
    actions: Optional[torch.Tensor] = None


@dataclass(slots=True)
class ContinuousReplaySample:
    """Structured batch sampled from replay for continuous-action agents."""

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor
    indices: Optional[np.ndarray]


class DiscreteAgentBase:
    """Common scaffold for discrete-action value-based agents."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        obs_dim: int,
        store_actions: bool,
        store_action_indices: bool,
        per_flag_key: str = "prioritized_replay",
        default_prioritized: bool = True,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_mode = str(cfg.get("action_mode", "absolute")).lower()
        self._action_helper = DiscreteActionAdapter(cfg.get("action_set"), action_mode=self.action_mode)
        self.action_set = self._action_helper.action_set
        self.n_actions = self._action_helper.n_actions
        self.act_dim = self._action_helper.act_dim
        self.buffer, self._use_per = build_replay_buffer(
            cfg,
            self.obs_dim,
            self.act_dim,
            store_actions=store_actions,
            store_action_indices=store_action_indices,
            per_flag_key=per_flag_key,
            default_prioritized=default_prioritized,
        )
        self.step_count = 0
        self._updates = 0
        self.episode_count = 0
        self._episode_done = False
        self.epsilon_unit = "episode"
        self.epsilon_start = 0.0
        self.epsilon_end = 0.0
        self.epsilon_decay = 1
        self.epsilon_decay_rate = 0.0
        self.epsilon_enabled = True
        self._epsilon_value = 0.0

    # ------------------------------------------------------------------ #
    # Epsilon management
    # ------------------------------------------------------------------ #
    def configure_epsilon(
        self,
        *,
        start: float,
        end: float,
        decay: Optional[int],
        decay_rate: float = 0.0,
        unit: str = "episode",
        enabled: bool = True,
    ) -> None:
        decay_value = int(decay) if decay is not None else 1
        if decay_value <= 0:
            decay_value = 1
        self.epsilon_start = float(start)
        self.epsilon_end = float(end)
        self.epsilon_decay = decay_value
        self.epsilon_decay_rate = float(decay_rate)
        self.epsilon_unit = str(unit).lower()
        self.epsilon_enabled = bool(enabled)
        self._epsilon_value = self._initial_epsilon()

    def epsilon(self) -> float:
        return self._epsilon_value if self.epsilon_enabled else 0.0

    def _initial_epsilon(self) -> float:
        if not self.epsilon_enabled:
            return 0.0
        if self.epsilon_decay_rate:
            return self.epsilon_start
        return self._epsilon_from_progress()

    def _epsilon_from_progress(self) -> float:
        progress = self.episode_count if self.epsilon_unit == "episode" else self.step_count
        fraction = min(1.0, progress / float(max(1, self.epsilon_decay)))
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1.0 - fraction)

    def _epsilon_from_counts(self) -> float:
        fraction = min(1.0, self.episode_count / float(max(1, self.epsilon_decay)))
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - fraction)

    def _advance_episode(self) -> None:
        self.episode_count += 1
        if not self.epsilon_enabled:
            self._epsilon_value = 0.0
            return
        if self.epsilon_decay_rate:
            next_eps = self._epsilon_value * self.epsilon_decay_rate
            self._epsilon_value = max(self.epsilon_end, next_eps)
        else:
            self._epsilon_value = self._epsilon_from_counts()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def refresh_action_helper(self, action_set: np.ndarray) -> None:
        """Update action metadata after loading checkpoints."""

        self._action_helper = DiscreteActionAdapter(action_set, action_mode=self.action_mode)
        self.action_set = self._action_helper.action_set
        self.n_actions = self._action_helper.n_actions
        self.act_dim = self._action_helper.act_dim


class ActionValueAgent(DiscreteAgentBase):
    """Augments :class:`DiscreteAgentBase` with Q-learning utilities."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        obs_dim: int,
        store_actions: bool,
        store_action_indices: bool,
        per_flag_key: str = "prioritized_replay",
        default_prioritized: bool = True,
    ) -> None:
        super().__init__(
            cfg,
            obs_dim=obs_dim,
            store_actions=store_actions,
            store_action_indices=store_action_indices,
            per_flag_key=per_flag_key,
            default_prioritized=default_prioritized,
        )
        self.gamma = float(cfg.get("gamma", 0.99))
        self.batch_size = int(cfg.get("batch_size", 64))
        base_learning = int(cfg.get("learning_starts", self.batch_size))
        self.learning_starts = max(self.batch_size, base_learning)
        capacity = getattr(self.buffer, "capacity", None)
        if capacity is not None:
            self.learning_starts = min(self.learning_starts, int(capacity))
        self.target_update_interval = int(cfg.get("target_update_interval", 500))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.0))
        self._target_pair: Optional[Tuple[Any, Any]] = None

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def register_target_networks(self, online: Any, target: Any) -> None:
        self._target_pair = (online, target)

    def ready_to_update(self) -> bool:
        if len(self.buffer) < self.batch_size:
            return False
        if len(self.buffer) < self.learning_starts:
            return False
        if self.step_count < self.learning_starts:
            return False
        return True

    def sample_batch(self) -> ReplaySample:
        return sample_replay_batch(self.buffer, self.batch_size, self.device, self._action_helper)

    def finalize_update(self, indices: Optional[np.ndarray], td_errors: Any) -> None:
        self._updates += 1
        if self._target_pair and self.target_update_interval > 0:
            if self._updates % self.target_update_interval == 0:
                online, target = self._target_pair
                target.load_state_dict(online.state_dict())
        if not self._use_per or indices is None or td_errors is None:
            return
        if hasattr(td_errors, "detach"):
            td_np = td_errors.detach().cpu().numpy()
        else:
            td_np = np.asarray(td_errors, dtype=np.float32)
        self.buffer.update_priorities(indices, np.abs(td_np))


class DiscreteActionAdapter:
    """Normalises action inputs and metadata for discrete-action agents."""

    def __init__(self, action_set: Any, *, action_mode: str = "absolute") -> None:
        action_array = np.asarray(action_set, dtype=np.float32)
        if action_array.ndim != 2:
            raise ValueError("DiscreteActionAdapter requires `action_set` shaped (n_actions, act_dim)")
        self.action_set = action_array
        self.n_actions = action_array.shape[0]
        self.act_dim = action_array.shape[1]
        self.action_mode = str(action_mode).lower()
        self.requires_action_index = self.action_mode in {"delta", "rate"}
        self._warned_fallback = False

    # ------------------------------------------------------------------ #
    # Index resolution helpers
    # ------------------------------------------------------------------ #
    def infer_index(self, action_vec: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        """Determine the discrete action index from vector metadata."""

        if info and "action_index" in info:
            return int(info["action_index"])

        if self.requires_action_index:
            raise RuntimeError(
                "Discrete action mode 'rate' or 'delta' requires 'action_index' metadata with each transition."
            )

        diffs = np.linalg.norm(self.action_set - np.asarray(action_vec, dtype=np.float32), axis=1)
        if not self._warned_fallback:
            warnings.warn(
                "Falling back to nearest-action lookup because no 'action_index' metadata was provided; "
                "ensure discrete action indices are recorded alongside transitions.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned_fallback = True
        return int(np.argmin(diffs))

    # ------------------------------------------------------------------ #
    # Transition preparation
    # ------------------------------------------------------------------ #
    def prepare_action(
        self,
        action: Any,
        info: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any], int]:
        """Normalise an action payload and ensure metadata includes `action_index`."""

        info_dict: Dict[str, Any] = dict(info or {})
        if np.isscalar(action) or (
            isinstance(action, np.ndarray) and action.ndim == 0
        ):
            action_idx_fallback = int(np.asarray(action).item())
            if action_idx_fallback < 0 or action_idx_fallback >= self.n_actions:
                raise IndexError(f"Discrete action index {action_idx_fallback} out of bounds for {self.n_actions} actions")
            action_vec = self.action_set[action_idx_fallback]
        else:
            action_vec = np.asarray(action, dtype=np.float32)
            action_idx_fallback = self.infer_index(action_vec, info_dict)

        action_idx = int(info_dict.get("action_index", action_idx_fallback))
        info_dict["action_index"] = action_idx
        return np.asarray(action_vec, dtype=np.float32), info_dict, action_idx


def build_replay_buffer(
    cfg: Dict[str, Any],
    obs_dim: int,
    act_dim: int,
    *,
    store_actions: bool,
    store_action_indices: bool = True,
    per_flag_key: str = "prioritized_replay",
    default_prioritized: bool = True,
) -> Tuple[ReplayBuffer, bool]:
    """Instantiate a replay buffer (optionally prioritized) from configuration."""

    buffer_size = int(cfg.get("buffer_size", 50_000))
    prioritized = bool(cfg.get(per_flag_key, default_prioritized))
    buffer: ReplayBuffer
    if prioritized:
        beta_value = cfg.get("per_beta_start", cfg.get("per_beta", 0.4))
        beta_increment = cfg.get("per_beta_increment", cfg.get("per_beta_step", 1e-4))
        beta_final = cfg.get("per_beta_final", 1.0)
        per_args = dict(
            alpha=float(cfg.get("per_alpha", 0.6)),
            beta=float(beta_value),
            beta_increment_per_sample=float(beta_increment),
            beta_final=float(beta_final),
            min_priority=float(cfg.get("per_min_priority", 1e-3)),
            epsilon=float(cfg.get("per_epsilon", 1e-6)),
        )
        buffer = PrioritizedReplayBuffer(
            buffer_size,
            (obs_dim,),
            (act_dim,),
            store_actions=store_actions,
            store_action_indices=store_action_indices,
            **per_args,
        )
    else:
        buffer = ReplayBuffer(
            buffer_size,
            (obs_dim,),
            (act_dim,),
            store_actions=store_actions,
            store_action_indices=store_action_indices,
        )
    return buffer, prioritized


def sample_replay_batch(
    buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    action_helper: DiscreteActionAdapter,
) -> ReplaySample:
    """Sample a batch from replay and normalise tensors for update logic."""

    batch = buffer.sample(batch_size)
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)

    infos = batch.get("infos")
    if infos is None:
        infos = [{}] * batch_size

    weights_arr = batch.get("weights")
    if weights_arr is None:
        weights = torch.ones((batch_size,), dtype=torch.float32, device=device)
    else:
        weights = torch.as_tensor(weights_arr, dtype=torch.float32, device=device).view(-1)

    action_indices_arr = batch.get("action_indices")
    if action_indices_arr is not None:
        action_indices_np = np.asarray(action_indices_arr, dtype=np.int64).reshape(-1)
        if (action_indices_np < 0).any():
            raise RuntimeError("Replay buffer returned invalid action indices")
        action_indices = torch.as_tensor(action_indices_np, dtype=torch.long, device=device)
    else:
        actions_arr = batch.get("actions")
        if actions_arr is None:
            raise RuntimeError("Replay batch missing actions and action indices")
        actions_np = np.asarray(actions_arr, dtype=np.float32)
        resolved = [action_helper.infer_index(act, info) for act, info in zip(actions_np, infos)]
        action_indices = torch.as_tensor(resolved, dtype=torch.long, device=device)

    actions_tensor: Optional[torch.Tensor] = None
    if batch.get("actions") is not None:
        actions_tensor = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)

    indices_arr = batch.get("indices")
    indices = None if indices_arr is None else np.asarray(indices_arr, dtype=np.int64)

    return ReplaySample(
        obs=obs,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        action_indices=action_indices,
        weights=weights,
        infos=infos,
        indices=indices,
        actions=actions_tensor,
    )


def sample_continuous_replay(
    buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> ContinuousReplaySample:
    """Sample a batch for continuous-action agents (TD3, SAC, etc.)."""

    batch = buffer.sample(batch_size)
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions_arr = batch.get("actions")
    if actions_arr is None:
        raise RuntimeError("Continuous replay sample requires stored actions.")
    actions = torch.as_tensor(actions_arr, dtype=torch.float32, device=device)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)

    weights_arr = batch.get("weights")
    if weights_arr is None:
        weights = torch.ones_like(rewards, dtype=torch.float32, device=device)
    else:
        weights = torch.as_tensor(weights_arr, dtype=torch.float32, device=device)
        if weights.ndim == 1:
            weights = weights.view(-1, 1)

    indices_arr = batch.get("indices")
    indices = None if indices_arr is None else np.asarray(indices_arr, dtype=np.int64)

    return ContinuousReplaySample(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        weights=weights,
        indices=indices,
    )


__all__ = [
    "ContinuousReplaySample",
    "ActionValueAgent",
    "DiscreteAgentBase",
    "DiscreteActionAdapter",
    "ReplaySample",
    "build_replay_buffer",
    "sample_continuous_replay",
    "sample_replay_batch",
]
