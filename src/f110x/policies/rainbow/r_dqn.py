"""Rainbow DQN agent that integrates all Rainbow components."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from f110x.policies.common import ActionValueAgent
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


class RainbowDQNAgent(ActionValueAgent):
    """Rainbow-style DQN agent operating on discrete action sets."""

    def __init__(self, cfg: Dict[str, Any]):
        self.device = resolve_device([cfg.get("device")])
        store_actions_flag = not bool(cfg.get("store_action_indices_only", True))
        super().__init__(
            cfg,
            obs_dim=int(cfg["obs_dim"]),
            store_actions=store_actions_flag,
            store_action_indices=True,
            default_prioritized=True,
        )

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
        self.register_target_networks(self.q_net, self.target_q_net)

        self.n_step = max(1, int(cfg.get("n_step", 3)))

        epsilon_flag = cfg.get("epsilon_enabled")
        self.use_epsilon = bool(epsilon_flag if epsilon_flag is not None else not self.use_noisy)
        if self.use_noisy and self.use_epsilon:
            raise ValueError(
                "RainbowDQN cannot enable epsilon-greedy when noisy exploration layers are active. "
                "Set `epsilon_enabled=false` or disable `noisy_layers`."
            )
        epsilon_decay_rate = float(cfg.get("epsilon_decay_rate", 0.0))
        if self.use_epsilon and epsilon_decay_rate and not 0.0 < epsilon_decay_rate < 1.0:
            raise ValueError("epsilon_decay_rate must be in (0, 1) for multiplicative decay")
        self.configure_epsilon(
            start=float(cfg.get("epsilon_start", 0.9 if self.use_epsilon else 0.0)),
            end=float(cfg.get("epsilon_end", 0.05 if self.use_epsilon else 0.0)),
            decay=cfg.get("epsilon_decay", 20000 if self.use_epsilon else 1),
            decay_rate=epsilon_decay_rate if self.use_epsilon else 0.0,
            unit=cfg.get("epsilon_unit", "episode"),
            enabled=self.use_epsilon,
        )

        self._store_actions = getattr(self.buffer, "store_actions", store_actions_flag)
        self._n_step_buffer: Deque[NStepEntry] = deque()

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
        obs_arr = np.asarray(obs, dtype=np.float32)
        next_obs_arr = np.asarray(next_obs, dtype=np.float32)

        action_vec, info_dict, action_idx = self._action_helper.prepare_action(action, info)
        action_vec = action_vec if self._store_actions else None
        transition: NStepEntry = (
            obs_arr,
            None if action_vec is None else np.asarray(action_vec, dtype=np.float32),
            float(reward),
            next_obs_arr,
            bool(done),
            info_dict,
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
        if not self.ready_to_update():
            return None

        sample = self.sample_batch()
        obs = sample.obs
        rewards = sample.rewards
        next_obs = sample.next_obs
        dones = sample.dones
        action_indices = sample.action_indices
        weights_t = sample.weights
        infos = sample.infos
        indices = sample.indices

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

        td_delta = chosen_q - target_q
        self.finalize_update(indices, td_delta)

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
        if self._use_per:
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
            self.refresh_action_helper(np.asarray(ckpt["action_set"], dtype=np.float32))
        self.use_noisy = bool(ckpt.get("use_noisy", self.use_noisy))
        self.noisy_sigma0 = float(ckpt.get("noisy_sigma0", self.noisy_sigma0))
        self.n_step = max(1, int(ckpt.get("n_step", self.n_step)))
        self.use_epsilon = bool(ckpt.get("use_epsilon", self.use_epsilon))
        self.epsilon_enabled = self.use_epsilon
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self._n_step_buffer.clear()
        self._episode_done = False
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)

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


# Backwards compatibility alias for code that still expects `DQNAgent`.
DQNAgent = RainbowDQNAgent

__all__ = ["RainbowDQNAgent", "DQNAgent"]
