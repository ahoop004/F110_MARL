"""Vanilla DQN agent operating on discrete action sets."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

# try:  # optional dependency for richer logging
#     import wandb  # type: ignore
# except ImportError:  # pragma: no cover - wandb optional
#     wandb = None

from f110x.policies.common import ActionValueAgent
from f110x.utils.torch_io import resolve_device, safe_load
from f110x.policies.dqn.net import QNetwork


class DQNAgent(ActionValueAgent):
    def __init__(self, cfg: Dict[str, Any]):
        self.device = resolve_device([cfg.get("device")])
        super().__init__(
            cfg,
            obs_dim=int(cfg["obs_dim"]),
            store_actions=False,
            store_action_indices=True,
            default_prioritized=True,
        )

        hidden_dims: Iterable[int] = cfg.get("hidden_dims", [256, 256])
        self.q_net = QNetwork(self.obs_dim, self.n_actions, hidden_dims).to(self.device)
        self.target_q_net = QNetwork(self.obs_dim, self.n_actions, hidden_dims).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.register_target_networks(self.q_net, self.target_q_net)

        self.lr = float(cfg.get("lr", 5e-4))
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        epsilon_decay_rate = float(cfg.get("epsilon_decay_rate", 0.0))
        if epsilon_decay_rate and not 0.0 < epsilon_decay_rate < 1.0:
            raise ValueError("epsilon_decay_rate must be in (0, 1) for multiplicative decay")
        self.configure_epsilon(
            start=float(cfg.get("epsilon_start", 0.9)),
            end=float(cfg.get("epsilon_end", 0.05)),
            decay=cfg.get("epsilon_decay", 20000),
            decay_rate=epsilon_decay_rate,
            unit=cfg.get("epsilon_unit", "episode"),
            enabled=True,
        )
        self.use_huber = bool(cfg.get("use_huber", True))

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
        action_vec, info_dict, action_idx = self._action_helper.prepare_action(action, info)
        action_payload = action_vec if getattr(self.buffer, "store_actions", True) else None
        self.buffer.add(
            obs,
            action_payload,
            reward,
            next_obs,
            done,
            info_dict,
            action_index=action_idx,
        )
        self.step_count += 1
        if done and not self._episode_done:
            self._advance_episode()
            self._episode_done = True
        elif not done and self._episode_done:
            # First transition of a new episode; allow the next terminal to trigger decay.
            self._episode_done = False

    # -------------------- Learning --------------------

    def update(self) -> Optional[Dict[str, Any]]:
        if not self.ready_to_update():
            return None

        sample = self.sample_batch()
        obs = sample.obs
        rewards = sample.rewards
        next_obs = sample.next_obs
        dones = sample.dones
        action_indices = sample.action_indices
        weights_t = sample.weights
        indices = sample.indices

        q_values = self.q_net(obs)
        chosen_q = q_values.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_online_q = self.q_net(next_obs)
            next_actions = torch.argmax(next_online_q, dim=-1)
            next_target_q = self.target_q_net(next_obs)
            next_q_values = next_target_q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = rewards.squeeze(-1) + (1 - dones.squeeze(-1)) * self.gamma * next_q_values

        td_errors = chosen_q - target
        per_sample = (
            F.smooth_l1_loss(chosen_q, target, reduction="none") if getattr(self, "use_huber", True)
            else td_errors.pow(2)
        )
        loss = (weights_t * per_sample).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if self.max_grad_norm > 0.0:
            grad_norm = clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.finalize_update(indices, td_errors)

        td_mean = float(td_errors.detach().mean().cpu().item())
        td_abs_mean = float(td_errors.detach().abs().mean().cpu().item())
        q_mean = float(chosen_q.detach().mean().cpu().item())
        q_max = float(chosen_q.detach().max().cpu().item())
        target_mean = float(target.detach().mean().cpu().item())
        buffer_fill = float(len(self.buffer) / float(getattr(self.buffer, "capacity", max(len(self.buffer), 1))))
        lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        stats: Dict[str, Any] = {
            "loss": float(loss.item()),
            "td_error_mean": td_mean,
            "td_error_abs": td_abs_mean,
            "q_mean": q_mean,
            "q_max": q_max,
            "target_mean": target_mean,
            "buffer_fraction": buffer_fill,
            "epsilon": float(self.epsilon()),
            "lr": lr,
            "updates": float(self._updates),
        }
        if grad_norm is not None:
            try:
                stats["grad_norm"] = float(grad_norm.detach().cpu().item())
            except AttributeError:
                stats["grad_norm"] = float(grad_norm)
        if self._use_per and weights_t is not None:
            stats["is_weight_mean"] = float(weights_t.detach().mean().cpu().item())
        return stats

    # -------------------- Persistence --------------------

    def save(self, path: str) -> None:
        torch.save(self.state_dict(include_optim=True), path)

    def load(self, path: str) -> None:
        ckpt = safe_load(path, map_location=self.device)
        self.load_state_dict(ckpt, strict=True, include_optim=True)

    def state_dict(self, *, include_optim: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "step_count": self.step_count,
            "updates": self._updates,
            "episode_count": self.episode_count,
            "epsilon_value": self._epsilon_value,
            "action_set": self.action_set,
            "obs_dim": self.obs_dim,
        }
        if include_optim:
            payload["optimizer"] = self.optimizer.state_dict()
        return payload

    def load_state_dict(
        self,
        snapshot: Mapping[str, Any],
        *,
        strict: bool = True,
        include_optim: bool = True,
    ) -> None:
        stored_obs_dim = int(snapshot.get("obs_dim", self.obs_dim))
        if stored_obs_dim != self.obs_dim:
            raise RuntimeError(
                "Checkpoint observation size mismatch: "
                f"checkpoint obs_dim={stored_obs_dim}, expected {self.obs_dim}."
            )
        self.q_net.load_state_dict(snapshot["q_net"], strict=strict)
        self.target_q_net.load_state_dict(snapshot.get("target_q_net", snapshot["q_net"]), strict=strict)
        if include_optim:
            opt_state = snapshot.get("optimizer")
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
        self.step_count = int(snapshot.get("step_count", self.step_count))
        self._updates = float(snapshot.get("updates", self._updates))
        self.episode_count = int(snapshot.get("episode_count", self.episode_count))
        self._epsilon_value = float(snapshot.get("epsilon_value", self._epsilon_value))

    def reset_optimizers(self) -> None:
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.target_q_net.load_state_dict(ckpt.get("target_q_net", ckpt["q_net"]))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = int(ckpt.get("step_count", 0))
        self._updates = int(ckpt.get("updates", 0))
        self.episode_count = int(ckpt.get("episode_count", 0))
        self._epsilon_value = float(ckpt.get("epsilon_value", self._initial_epsilon()))
        if "action_set" in ckpt:
            self.refresh_action_helper(np.asarray(ckpt["action_set"], dtype=np.float32))
        self._episode_done = False
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)
