"""Rainbow DQN agent that integrates all Rainbow components."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Tuple
import warnings

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from f110x.policies.buffers.replay import ReplayBuffer
from f110x.policies.rainbow.per import PrioritizedReplayBuffer
from f110x.policies.rainbow.r_dqn_net import RainbowQNetwork
from f110x.utils.torch_io import resolve_device, safe_load


NStepEntry = Tuple[
    np.ndarray,
    Optional[np.ndarray],
    float,
    np.ndarray,
    bool,
    Optional[Dict[str, Any]],
    int,
]


class RainbowDQNAgent:
    """Rainbow-style DQN agent operating on discrete action sets."""

    def __init__(self, cfg: Dict[str, Any]):
        self.device = resolve_device([cfg.get("device")])

        self.obs_dim = int(cfg["obs_dim"])
        self.action_mode = str(cfg.get("action_mode", "absolute")).lower()
        self._requires_action_index = self.action_mode in {"delta", "rate"}
        self._warned_action_index_fallback = False

        action_set = np.asarray(cfg.get("action_set"), dtype=np.float32)
        if action_set.ndim != 2:
            raise ValueError("RainbowDQN requires `action_set` as a list of action vectors")
        self.action_set = action_set
        self.n_actions = action_set.shape[0]
        self.act_dim = action_set.shape[1]

        self.use_noisy = bool(cfg.get("noisy_layers", True))
        self.noisy_sigma0 = float(cfg.get("noisy_sigma0", 0.5))
        self.atoms = int(cfg.get("atoms", 51))
        self.v_min = float(cfg.get("v_min", -10.0))
        self.v_max = float(cfg.get("v_max", 10.0))
        if self.atoms <= 1:
            raise ValueError("RainbowDQN requires at least two atoms for the categorical support")
        if self.v_max <= self.v_min:
            raise ValueError("RainbowDQN requires v_max > v_min for the categorical support")
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)

        hidden_dims: Iterable[int] = cfg.get("hidden_dims", [256, 256])
        self.q_net = RainbowQNetwork(
            self.obs_dim,
            self.n_actions,
            hidden_dims,
            atoms=self.atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            noisy=self.use_noisy,
            sigma0=self.noisy_sigma0,
        ).to(self.device)
        self.target_q_net = RainbowQNetwork(
            self.obs_dim,
            self.n_actions,
            hidden_dims,
            atoms=self.atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            noisy=self.use_noisy,
            sigma0=self.noisy_sigma0,
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net.train()
        self.target_q_net.train()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(cfg.get("lr", 5e-4)))

        self.gamma = float(cfg.get("gamma", 0.99))
        self.n_step = max(1, int(cfg.get("n_step", 3)))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.target_update_interval = int(cfg.get("target_update_interval", 500))
        self.learning_starts = max(
            self.batch_size,
            int(cfg.get("learning_starts", 1000)),
        )
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.0))

        epsilon_flag = cfg.get("epsilon_enabled")
        self.use_epsilon = bool(epsilon_flag if epsilon_flag is not None else not self.use_noisy)
        if self.use_noisy and self.use_epsilon:
            raise ValueError(
                "RainbowDQN cannot enable epsilon-greedy when noisy exploration layers are active. "
                "Set `epsilon_enabled=false` or disable `noisy_layers`."
            )
        self.epsilon_start = float(cfg.get("epsilon_start", 0.9 if self.use_epsilon else 0.0))
        self.epsilon_end = float(cfg.get("epsilon_end", 0.05 if self.use_epsilon else 0.0))
        self.epsilon_decay = max(1, int(cfg.get("epsilon_decay", 20000))) if self.use_epsilon else 1
        self.epsilon_decay_rate = float(cfg.get("epsilon_decay_rate", 0.0))
        if self.use_epsilon and self.epsilon_decay_rate:
            if not 0.0 < self.epsilon_decay_rate < 1.0:
                raise ValueError("epsilon_decay_rate must be in (0, 1) for multiplicative decay")
        self.episode_count = 0
        self._epsilon_value = self._initial_epsilon()
        self._episode_done = False

        buffer_size = int(cfg.get("buffer_size", 50000))
        self.learning_starts = min(self.learning_starts, buffer_size)
        prioritized = bool(cfg.get("prioritized_replay", True))
        store_actions = not bool(cfg.get("store_action_indices_only", True))
        if prioritized:
            per_args = dict(
                alpha=float(cfg.get("per_alpha", 0.6)),
                beta=float(cfg.get("per_beta_start", 0.4)),
                beta_increment_per_sample=float(cfg.get("per_beta_increment", 1e-4)),
                min_priority=float(cfg.get("per_min_priority", 1e-3)),
                epsilon=float(cfg.get("per_epsilon", 1e-6)),
            )
            self.buffer: ReplayBuffer = PrioritizedReplayBuffer(
                buffer_size,
                (self.obs_dim,),
                (self.act_dim,),
                store_actions=store_actions,
                store_action_indices=True,
                **per_args,
            )
        else:
            self.buffer = ReplayBuffer(
                buffer_size,
                (self.obs_dim,),
                (self.act_dim,),
                store_actions=store_actions,
                store_action_indices=True,
            )
        self._use_per = prioritized
        self._store_actions = getattr(self.buffer, "store_actions", True)
        self._n_step_buffer: Deque[NStepEntry] = deque()

        self.step_count = 0
        self._updates = 0

    # -------------------- Interaction --------------------

    def epsilon(self) -> float:
        return self._epsilon_value if self.use_epsilon else 0.0

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if self.use_noisy:
            if deterministic:
                previous_mode = self.q_net.training
                self.q_net.eval()
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_values = self.q_net.q_values(obs_t)
                    idx = int(torch.argmax(q_values, dim=-1).item())
                if previous_mode:
                    self.q_net.train()
                return idx
            self.q_net.train()
            self.q_net.reset_noise()

        eps = 0.0
        if not deterministic and self.use_epsilon:
            eps = self.epsilon()

        if not deterministic and self.use_epsilon and np.random.rand() < eps:
            idx = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_net.q_values(obs_t)
                idx = int(torch.argmax(q_values, dim=-1).item())
        return idx

    def store_transition(
        self,
        obs: np.ndarray,
        action: Iterable[float],
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if info is None:
            info = {}
        obs_arr = np.asarray(obs, dtype=np.float32)
        next_obs_arr = np.asarray(next_obs, dtype=np.float32)

        action_arr: Optional[np.ndarray]
        if np.isscalar(action) or (
            isinstance(action, np.ndarray) and action.ndim == 0
        ):
            action_idx_fallback = int(np.asarray(action).item())
            action_arr = self.action_set[action_idx_fallback]
        else:
            action_arr = np.asarray(action, dtype=np.float32)
            action_idx_fallback = self._action_to_index(action_arr, info)

        info = dict(info)
        action_idx: int
        if "action_index" in info:
            action_idx = int(info["action_index"])
        else:
            action_idx = action_idx_fallback
        info["action_index"] = action_idx

        action_vec = action_arr if self._store_actions else None
        transition: NStepEntry = (
            obs_arr,
            None if action_vec is None else np.asarray(action_vec, dtype=np.float32),
            float(reward),
            next_obs_arr,
            bool(done),
            info,
            action_idx,
        )

        self._n_step_buffer.append(transition)
        self._maybe_append_transition(done)

        if done and not self._episode_done:
            self._advance_episode()
            self._episode_done = True
        elif not done and self._episode_done:
            self._episode_done = False

    # -------------------- Learning --------------------

    def update(self) -> Optional[Dict[str, Any]]:
        if len(self.buffer) < self.batch_size:
            return None
        if len(self.buffer) < self.learning_starts:
            return None
        if self.step_count < self.learning_starts:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        infos = batch.get("infos")
        if infos is None:
            infos = [{}] * self.batch_size
        weights = batch.get("weights")
        if weights is None:
            weights_t = torch.ones((self.batch_size,), dtype=torch.float32, device=self.device)
        else:
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).view(-1)
        indices = batch.get("indices")

        action_indices_data = batch.get("action_indices")
        if action_indices_data is not None:
            action_indices_np = np.asarray(action_indices_data, dtype=np.int64).reshape(-1)
            if (action_indices_np < 0).any():
                raise RuntimeError("ReplayBuffer returned invalid action indices for Rainbow DQN update")
            action_indices = torch.as_tensor(action_indices_np, dtype=torch.long, device=self.device)
        else:
            actions_data = batch.get("actions")
            if actions_data is None:
                raise RuntimeError("ReplayBatch missing actions and action indices")
            actions_np = np.asarray(actions_data, dtype=np.float32)
            inferred: list[int] = []
            for act, info in zip(actions_np, infos):
                if info and "action_index" in info:
                    inferred.append(int(info["action_index"]))
                else:
                    inferred.append(self._action_to_index(act, info))
            action_indices = torch.as_tensor(inferred, dtype=torch.long, device=self.device)

        discounts = self._gather_discounts(infos, obs.size(0)).unsqueeze(-1)

        if self.use_noisy:
            self.q_net.reset_noise()
        logits = self.q_net(obs)

        gather_index = action_indices.view(-1, 1, 1).expand(-1, 1, self.atoms)
        chosen_logits = logits.gather(1, gather_index).squeeze(1)
        chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
        chosen_probs = torch.softmax(chosen_logits, dim=-1)
        chosen_q = torch.sum(chosen_probs * self.q_net.support, dim=-1)

        with torch.no_grad():
            next_logits_online = self.q_net(next_obs)
            next_probs = torch.softmax(next_logits_online, dim=-1)
            next_q = torch.sum(next_probs * self.q_net.support, dim=-1)
            next_actions = torch.argmax(next_q, dim=-1)

            if self.use_noisy:
                self.target_q_net.reset_noise()
            target_logits = self.target_q_net(next_obs)
            target_probs = torch.softmax(target_logits, dim=-1)
            next_dist = target_probs.gather(
                1, next_actions.view(-1, 1, 1).expand(-1, 1, self.atoms)
            ).squeeze(1)
            projected_dist = self._project_distribution(
                next_dist,
                rewards.squeeze(-1),
                dones.squeeze(-1),
                discounts.squeeze(-1),
            )
            target_q = torch.sum(projected_dist * self.q_net.support, dim=-1)

        loss_per_sample = -torch.sum(projected_dist * chosen_log_probs, dim=-1)
        loss = (weights_t * loss_per_sample).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if self.max_grad_norm > 0.0:
            grad_norm = clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self._updates += 1
        if self.target_update_interval > 0 and self._updates % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        if self._use_per and indices is not None:
            td_errors = (chosen_q - target_q).detach().cpu().numpy()
            self.buffer.update_priorities(np.asarray(indices), np.abs(td_errors))

        buffer_capacity = float(getattr(self.buffer, "capacity", max(len(self.buffer), 1)))
        buffer_fraction = float(len(self.buffer) / buffer_capacity)
        stats: Dict[str, Any] = {
            "loss": float(loss.item()),
            "kl": float(loss_per_sample.detach().mean().cpu().item()),
            "q_mean": float(chosen_q.detach().mean().cpu().item()),
            "q_max": float(chosen_q.detach().max().cpu().item()),
            "target_q_mean": float(target_q.detach().mean().cpu().item()),
            "buffer_fraction": buffer_fraction,
            "epsilon": float(self.epsilon()),
            "updates": float(self._updates),
            "n_step": float(self.n_step),
        }
        if grad_norm is not None:
            try:
                stats["grad_norm"] = float(grad_norm.detach().cpu().item())
            except AttributeError:
                stats["grad_norm"] = float(grad_norm)
        if self._use_per and weights is not None:
            stats["is_weight_mean"] = float(weights_t.detach().mean().cpu().item())
        return stats

    # -------------------- Persistence --------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "updates": self._updates,
                "episode_count": self.episode_count,
                "epsilon_value": self._epsilon_value,
                "action_set": self.action_set,
                "obs_dim": self.obs_dim,
                "atoms": self.atoms,
                "v_min": self.v_min,
                "v_max": self.v_max,
                "use_noisy": self.use_noisy,
                "noisy_sigma0": self.noisy_sigma0,
                "n_step": self.n_step,
                "use_epsilon": self.use_epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = safe_load(path, map_location=self.device)
        stored_obs_dim = int(ckpt.get("obs_dim", self.obs_dim))
        if stored_obs_dim != self.obs_dim:
            raise RuntimeError(
                "Checkpoint observation size mismatch for "
                f"'{path}': checkpoint obs_dim={stored_obs_dim}, "
                f"expected {self.obs_dim}. "
                "Ensure the observation wrapper configuration matches the saved model."
            )

        stored_atoms = int(ckpt.get("atoms", self.atoms))
        stored_v_min = float(ckpt.get("v_min", self.v_min))
        stored_v_max = float(ckpt.get("v_max", self.v_max))
        if stored_atoms != self.atoms or stored_v_min != self.v_min or stored_v_max != self.v_max:
            raise RuntimeError(
                "Checkpoint categorical support does not match current configuration "
                f"(atoms={stored_atoms}, v_min={stored_v_min}, v_max={stored_v_max})."
            )

        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_q_net.load_state_dict(ckpt.get("target_q_net", ckpt["q_net"]))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = int(ckpt.get("step_count", 0))
        self._updates = int(ckpt.get("updates", 0))
        self.episode_count = int(ckpt.get("episode_count", 0))
        self._epsilon_value = float(ckpt.get("epsilon_value", self._initial_epsilon()))
        if "action_set" in ckpt:
            self.action_set = np.asarray(ckpt["action_set"], dtype=np.float32)
            self.n_actions = self.action_set.shape[0]
            self.act_dim = self.action_set.shape[1]
        self.use_noisy = bool(ckpt.get("use_noisy", self.use_noisy))
        self.noisy_sigma0 = float(ckpt.get("noisy_sigma0", self.noisy_sigma0))
        self.n_step = max(1, int(ckpt.get("n_step", self.n_step)))
        self.use_epsilon = bool(ckpt.get("use_epsilon", self.use_epsilon))
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self._n_step_buffer.clear()
        self._episode_done = False
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)

    # -------------------- Helpers --------------------

    def _advance_episode(self) -> None:
        self.episode_count += 1
        if not self.use_epsilon:
            self._epsilon_value = 0.0
            return
        if self.epsilon_decay_rate:
            next_eps = self._epsilon_value * self.epsilon_decay_rate
            self._epsilon_value = max(self.epsilon_end, next_eps)
        else:
            self._epsilon_value = self._epsilon_from_counts()

    def _initial_epsilon(self) -> float:
        if not self.use_epsilon:
            return 0.0
        if self.epsilon_decay_rate:
            return self.epsilon_start
        return self._epsilon_from_counts()

    def _epsilon_from_counts(self) -> float:
        fraction = min(1.0, self.episode_count / self.epsilon_decay)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - fraction)

    def _maybe_append_transition(self, terminal: bool) -> None:
        while len(self._n_step_buffer) >= self.n_step or (terminal and self._n_step_buffer):
            reward, discount, next_obs, done_flag = self._compute_n_step_target()
            obs0, action0, _r0, _next0, _done0, info0, action_idx0 = self._n_step_buffer.popleft()
            info_store = dict(info0 or {})
            info_store["action_index"] = action_idx0
            info_store["n_step_gamma"] = discount
            action_for_buffer = action0 if self._store_actions else None
            self.buffer.add(
                obs0,
                action_for_buffer,
                reward,
                next_obs,
                done_flag,
                info_store,
                action_index=action_idx0,
            )
            self.step_count += 1

    def _compute_n_step_target(self) -> Tuple[float, float, np.ndarray, bool]:
        reward = 0.0
        next_obs = self._n_step_buffer[0][3]
        done_flag = False
        steps = 0
        for idx, (_obs, _action, r, next_o, done, _info, _aidx) in enumerate(self._n_step_buffer):
            reward += (self.gamma ** idx) * float(r)
            next_obs = next_o
            steps = idx + 1
            done_flag = done
            if done or steps >= self.n_step:
                break
        discount = self.gamma ** steps
        return float(reward), float(discount), next_obs, bool(done_flag)

    def _gather_discounts(self, infos: Iterable[Optional[Dict[str, Any]]], batch_size: int) -> torch.Tensor:
        base = self.gamma ** self.n_step
        values = np.full((batch_size,), base, dtype=np.float32)
        for idx, info in enumerate(infos):
            if info and "n_step_gamma" in info:
                values[idx] = float(info["n_step_gamma"])
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _project_distribution(
        self,
        next_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rewards.size(0)
        support = self.q_net.support.unsqueeze(0).expand(batch_size, -1)
        tz = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * discounts.unsqueeze(-1) * support
        tz = tz.clamp(self.v_min, self.v_max)
        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        l = l.clamp(0, self.atoms - 1)
        u = u.clamp(0, self.atoms - 1)

        offset = (torch.arange(batch_size, device=self.device).unsqueeze(1) * self.atoms)
        projected = torch.zeros_like(next_dist)
        projected.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (next_dist * (u.float() - b)).view(-1),
        )
        projected.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (next_dist * (b - l.float())).view(-1),
        )
        projected = projected.clamp_min_(1e-8)
        projected = projected / projected.sum(dim=-1, keepdim=True)
        return projected

    def _action_to_index(self, action: Iterable[float], info: Optional[Dict[str, Any]] = None) -> int:
        if info and "action_index" in info:
            return int(info["action_index"])
        if self._requires_action_index:
            raise RuntimeError(
                "RainbowDQNAgent requires 'action_index' metadata when using a ``rate`` or ``delta`` action mode"
            )
        action_arr = np.asarray(action, dtype=np.float32)
        diffs = np.linalg.norm(self.action_set - action_arr, axis=1)
        if not self._warned_action_index_fallback:
            warnings.warn(
                "Falling back to nearest-action lookup because no 'action_index' metadata was provided; "
                "ensure discrete action indices are recorded alongside transitions.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_action_index_fallback = True
        return int(np.argmin(diffs))


# Backwards compatibility alias for code that still expects `DQNAgent`.
DQNAgent = RainbowDQNAgent

__all__ = ["RainbowDQNAgent", "DQNAgent"]
