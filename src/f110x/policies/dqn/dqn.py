"""Vanilla DQN agent operating on discrete action sets."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# try:  # optional dependency for richer logging
#     import wandb  # type: ignore
# except ImportError:  # pragma: no cover - wandb optional
#     wandb = None

from f110x.policies.buffers import PrioritizedReplayBuffer, ReplayBuffer
from f110x.utils.torch_io import resolve_device, safe_load
from f110x.policies.dqn.net import QNetwork


class DQNAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.device = resolve_device([cfg.get("device")])

        self.obs_dim = int(cfg["obs_dim"])
        action_set = np.asarray(cfg.get("action_set"), dtype=np.float32)
        if action_set.ndim != 2:
            raise ValueError("DQN requires `action_set` as a list of action vectors")
        self.action_set = action_set
        self.n_actions = action_set.shape[0]
        self.act_dim = action_set.shape[1]

        hidden_dims: Iterable[int] = cfg.get("hidden_dims", [256, 256])
        self.q_net = QNetwork(self.obs_dim, self.n_actions, hidden_dims).to(self.device)
        self.target_q_net = QNetwork(self.obs_dim, self.n_actions, hidden_dims).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(cfg.get("lr", 5e-4)))

        self.gamma = float(cfg.get("gamma", 0.99))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.target_update_interval = int(cfg.get("target_update_interval", 500))

        self.epsilon_start = float(cfg.get("epsilon_start", 0.9))
        self.epsilon_end = float(cfg.get("epsilon_end", 0.05))
        self.epsilon_decay = max(1, int(cfg.get("epsilon_decay", 20000)))
        self.epsilon_decay_rate = float(cfg.get("epsilon_decay_rate", 0.0))
        if self.epsilon_decay_rate:
            if not 0.0 < self.epsilon_decay_rate < 1.0:
                raise ValueError("epsilon_decay_rate must be in (0, 1) for multiplicative decay")
        self.episode_count = 0
        self._epsilon_value = self._initial_epsilon()

        buffer_size = int(cfg.get("buffer_size", 50000))
        prioritized = bool(cfg.get("prioritized_replay", True))
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
                **per_args,
            )
        else:
            self.buffer = ReplayBuffer(buffer_size, (self.obs_dim,), (self.act_dim,))
        self._use_per = prioritized

        self.step_count = 0

    # -------------------- Interaction --------------------

    def epsilon(self) -> float:
        return self._epsilon_value

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        eps = 0.0 if deterministic else self.epsilon()
        if not deterministic and np.random.rand() < eps:
            idx = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_net(obs_t)
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
        # allow action passed as index or vector
        if np.isscalar(action):
            action_idx = int(action)
            action_vec = self.action_set[action_idx]
        else:
            action_vec = np.asarray(action, dtype=np.float32)
            action_idx = self._action_to_index(action_vec)
        info = dict(info)
        info["action_index"] = action_idx
        self.buffer.add(obs, action_vec, reward, next_obs, done, info)
        self.step_count += 1

        if done:
            self._advance_episode()

    # -------------------- Learning --------------------

    def update(self) -> Optional[Dict[str, Any]]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        infos = batch.get("infos", [{}] * self.batch_size)
        weights = batch.get("weights")
        indices = batch.get("indices")
        if weights is None:
            weights_t = torch.ones((self.batch_size,), dtype=torch.float32, device=self.device)
        else:
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).squeeze(-1)

        action_indices = [self._action_to_index(act, info) for act, info in zip(actions.cpu().numpy(), infos)]
        action_indices = torch.as_tensor(action_indices, dtype=torch.long, device=self.device)

        q_values = self.q_net(obs)
        chosen_q = q_values.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_online_q = self.q_net(next_obs)
            next_actions = torch.argmax(next_online_q, dim=-1)
            next_target_q = self.target_q_net(next_obs)
            next_q_values = next_target_q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = rewards.squeeze(-1) + (1 - dones.squeeze(-1)) * self.gamma * next_q_values

        td_errors = chosen_q - target
        loss = (weights_t * td_errors.pow(2)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        if self._use_per and indices is not None:
            td_error_np = td_errors.detach().cpu().numpy()
            self.buffer.update_priorities(np.asarray(indices), td_error_np)

        # metrics: Dict[str, Any] = {
        #     "loss": float(loss.detach().cpu().item()),
        #     "epsilon": float(self.epsilon()),
        #     "q_mean": float(q_values_np.mean()),
        #     "q_std": float(q_values_np.std()),
        #     "chosen_q_mean": float(chosen_q_np.mean()),
        #     "chosen_q_std": float(chosen_q_np.std()),
        #     "target_q_mean": float(target_np.mean()),
        #     "target_q_std": float(target_np.std()),
        #     "td_error_mean": float(td_error_np.mean()),
        #     "td_error_std": float(td_error_np.std()),
        #     "action_index_mean": float(action_indices_np.mean()),
        #     "action_index_std": float(action_indices_np.std()),
        # }
        # if weights is not None:
        #     weights_np = np.asarray(weights, dtype=np.float32)
        #     metrics["per_beta"] = float(self.buffer.beta)
        #     metrics["per_weight_mean"] = float(weights_np.mean())
        #     metrics["per_weight_max"] = float(weights_np.max())

        # if wandb is not None:
        #     metrics["action_index_histogram"] = wandb.Histogram(action_indices_np.astype(np.int64))
        #     metrics["q_histogram"] = wandb.Histogram(q_values_np.flatten())

        # return metrics

    def _advance_episode(self) -> None:
        self.episode_count += 1
        if self.epsilon_decay_rate:
            next_eps = self._epsilon_value * self.epsilon_decay_rate
            self._epsilon_value = max(self.epsilon_end, next_eps)
        else:
            self._epsilon_value = self._epsilon_from_counts()

    # -------------------- Persistence --------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "episode_count": self.episode_count,
                "epsilon_value": self._epsilon_value,
                "action_set": self.action_set,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = safe_load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_q_net.load_state_dict(ckpt.get("target_q_net", ckpt["q_net"]))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = int(ckpt.get("step_count", 0))
        self.episode_count = int(ckpt.get("episode_count", 0))
        self._epsilon_value = float(ckpt.get("epsilon_value", self._initial_epsilon()))
        if "action_set" in ckpt:
            self.action_set = np.asarray(ckpt["action_set"], dtype=np.float32)
            self.n_actions = self.action_set.shape[0]
            self.act_dim = self.action_set.shape[1]
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)

    def _initial_epsilon(self) -> float:
        if self.epsilon_decay_rate:
            if self.episode_count:
                decayed = self.epsilon_start * (self.epsilon_decay_rate ** self.episode_count)
                return max(self.epsilon_end, decayed)
            return self.epsilon_start
        return self._epsilon_from_counts()

    def _epsilon_from_counts(self) -> float:
        fraction = min(1.0, self.episode_count / self.epsilon_decay)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - fraction)

    # -------------------- Helpers --------------------

    def _action_to_index(self, action: Iterable[float], info: Optional[Dict[str, Any]] = None) -> int:
        if info and "action_index" in info:
            return int(info["action_index"])
        action_arr = np.asarray(action, dtype=np.float32)
        diffs = np.linalg.norm(self.action_set - action_arr, axis=1)
        return int(np.argmin(diffs))
