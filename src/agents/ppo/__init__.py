"""Pure PyTorch PPO agent."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from agents.common import Actor, Critic, compute_gae, ppo_minibatch_step, mean_update_metrics
from utils.torch_io import resolve_device


class RolloutBuffer:
    """Fixed-capacity on-policy rollout buffer with GAE computation."""

    def __init__(self, n_steps: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.n_steps = n_steps
        self.device = device
        self.obs = torch.zeros(n_steps, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, action_dim, device=device)
        self.raw_actions = torch.full_like(self.actions, float("nan"))
        self.rewards = torch.zeros(n_steps, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.terminated = torch.zeros(n_steps, device=device)
        self.truncated = torch.zeros(n_steps, device=device)
        self.final_values = torch.full((n_steps,), float("nan"), device=device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        terminated: bool,
        truncated: bool,
        raw_action: Optional[np.ndarray] = None,
        final_value: Optional[float] = None,
    ) -> None:
        i = self.ptr % self.n_steps
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.raw_actions[i] = (float("nan") if raw_action is None else
                               torch.as_tensor(raw_action, dtype=torch.float32, device=self.device))
        self.rewards[i] = float(reward)
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)
        self.terminated[i] = float(terminated)
        self.truncated[i] = float(truncated)
        self.final_values[i] = float("nan") if final_value is None else float(final_value)
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
        return compute_gae(
            self.rewards[:n], self.values[:n], self.terminated[:n], self.truncated[:n],
            next_value, gamma, gae_lambda, self.final_values[:n],
        )


class PPOAgent:
    """Proximal Policy Optimization — pure PyTorch, no SB3."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        params: Dict,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = len(self.action_low)
        self.action_contract = dict(params.get("_action_contract", {"speed_control": "direct"}))
        self.physics_contract = params.get("_physics_contract")
        self.observation_contract = params.get("_observation_contract")

        # Hyperparameters (merged from training_defaults + scenario params)
        self.lr = float(params.get("learning_rate", 3e-4))
        self.lr_schedule = str(params.get("lr_schedule", "constant"))
        if self.lr_schedule not in {"constant", "linear"}:
            raise ValueError("PPO lr_schedule must be 'constant' or 'linear'.")
        if self.lr_schedule == "linear" and "learning_rate_end" not in params:
            raise ValueError("Linear PPO lr_schedule requires learning_rate_end.")
        self.lr_end = float(params.get("learning_rate_end", self.lr))
        if not all(np.isfinite(rate) and rate > 0 for rate in (self.lr, self.lr_end)):
            raise ValueError("PPO learning rates must be finite and positive.")
        if self.lr_schedule == "constant" and self.lr_end != self.lr:
            raise ValueError("A different learning_rate_end requires lr_schedule: linear.")
        self.gamma = float(params.get("gamma", 0.99))
        self.gae_lambda = float(params.get("gae_lambda", 0.95))
        self.clip_range = float(params.get("clip_range", 0.2))
        self.ent_coef = float(params.get("ent_coef", 0.01))
        self.vf_coef = float(params.get("vf_coef", 0.5))
        self.max_grad_norm = float(params.get("max_grad_norm", 0.5))
        self.n_steps = int(params.get("n_steps", 2048))
        self.n_epochs = int(params.get("n_epochs", 10))
        self.batch_size = int(params.get("batch_size", 64))
        self.min_rollout_steps = int(params.get("min_rollout_steps", 1))
        if self.min_rollout_steps < 1:
            raise ValueError("min_rollout_steps must be positive")
        self.target_kl = params.get("target_kl")
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)
            if not np.isfinite(self.target_kl) or self.target_kl <= 0:
                raise ValueError("target_kl must be finite and positive")
        self._pending_rollouts = []
        self._pending_steps = 0

        hidden_dims: List[int] = list(
            params.get("pi_hidden_dims", params.get("hidden_dims", [64, 64]))
        )
        vf_dims: List[int] = list(
            params.get("vf_hidden_dims", params.get("hidden_dims", [64, 64]))
        )
        activation: str = str(params.get("activation", "tanh"))
        self.actor_hidden_dims = list(hidden_dims)
        self.critic_hidden_dims = list(vf_dims)
        self.activation = activation

        device_str = str(params.get("device", "cpu"))
        self.device = resolve_device([device_str])

        self.actor = Actor(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        log_std_init = float(params.get("log_std_init", 0.0))
        if not np.isfinite(log_std_init) or not self.actor.LOG_STD_MIN <= log_std_init <= self.actor.LOG_STD_MAX:
            raise ValueError("log_std_init must be within the actor's log standard deviation bounds")
        with torch.no_grad():
            self.actor.log_std.fill_(log_std_init)
        self.critic = Critic(obs_dim, vf_dims, activation).to(self.device)
        self._optim_parameters = tuple(self.actor.parameters()) + tuple(self.critic.parameters())
        self.optimizer = optim.Adam(self._optim_parameters, lr=self.lr)

        self.buffer = RolloutBuffer(self.n_steps, obs_dim, self.action_dim, self.device)

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    def set_training_progress(self, progress: float) -> None:
        """Set LR from the trainer's globally completed budget fraction.

        Evaluation never advances this schedule. In parallel training only
        the parent optimizer receives progress, not individual collectors.
        """
        if self.lr_schedule == "linear":
            fraction = float(np.clip(progress, 0.0, 1.0))
            rate = self.lr + fraction * (self.lr_end - self.lr)
            for group in self.optimizer.param_groups:
                group["lr"] = rate

    @torch.no_grad()
    def value_batch(self, observations: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return self.critic(obs).cpu().numpy()

    def value(self, observation: np.ndarray) -> float:
        return float(self.value_batch(np.asarray(observation)[None])[0])

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic evaluation action without evaluating the critic."""
        obs_t = torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32, device=self.device)
        return torch.tanh(self.actor.net(obs_t))[0].cpu().numpy()

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action from policy.

        Returns:
            (action_normalized, log_prob, value)
            action_normalized is in [-1, 1] — caller denormalizes for env.step()
        """
        actions, log_probs, values = self.act_batch(np.asarray(obs)[None], deterministic)
        return actions[0], float(log_probs[0]), float(values[0])

    @torch.no_grad()
    def act_batch(self, observations: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions, log_probs, raw_actions = self.actor.get_action(
            obs_t, deterministic=deterministic, return_raw=True
        )
        values = self.critic(obs_t)
        # One device-to-host transfer for all environments and policy outputs.
        outputs = torch.cat((actions, raw_actions, log_probs[:, None], values[:, None]), dim=1).cpu().numpy()
        self.last_raw_actions = outputs[:, self.action_dim:2 * self.action_dim].copy()
        return outputs[:, :self.action_dim], outputs[:, -2], outputs[:, -1]

    def update(self, next_value: float) -> Dict[str, float]:
        """Compute GAE and run PPO update epochs.

        Returns dict of training metrics (empty dict if buffer too small to update).
        """
        if self.buffer.size() < 1:
            return {}

        advantages, returns = self.buffer.compute_gae(
            next_value,
            self.gamma,
            self.gae_lambda,
        )
        n = self.buffer.size()
        return self._queue_update((
            self.buffer.obs[:n], self.buffer.actions[:n], self.buffer.log_probs[:n],
            advantages, returns, self.buffer.raw_actions[:n],
        ))

    def update_rollouts(self, rollouts) -> Dict[str, float]:
        """Pool independently bootstrapped worker rollouts from one frozen policy."""
        tensors = [
            torch.as_tensor(np.concatenate(parts), dtype=torch.float32, device=self.device)
            for parts in zip(*rollouts)
        ]
        return self._queue_update(tuple(tensors))

    def _queue_update(self, tensors) -> Dict[str, float]:
        # Each fragment already has its own terminal/truncation bootstrap.
        # Copy before collectors reuse their buffers; keep the actor frozen
        # until enough decisions have accumulated across episodes/workers.
        self._pending_rollouts.append(tuple(t.detach().clone() for t in tensors))
        self._pending_steps += len(tensors[0])
        if self._pending_steps < self.min_rollout_steps:
            return {}
        return self.flush_pending_update()

    def flush_pending_update(self) -> Dict[str, float]:
        if not self._pending_rollouts:
            return {}
        tensors = tuple(torch.cat(parts) for parts in zip(*self._pending_rollouts))
        self._pending_rollouts.clear()
        self._pending_steps = 0
        return self._update(*tensors)

    def _update(self, observations, actions, log_probs, advantages, returns, raw_actions=None) -> Dict[str, float]:
        # Normalize advantages — use correction=0 so std is always valid for n>=1
        adv_std = advantages.std(correction=0)
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        n = len(observations)
        metric_rows = []
        early_stop = False
        measured_kl = 0.0
        for _ in range(self.n_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                obs_b, act_b, old_lp_b = observations[idx], actions[idx], log_probs[idx]
                adv_b = advantages[idx]
                ret_b = returns[idx]
                raw_b = raw_actions[idx] if raw_actions is not None else None
                if self.target_kl is not None:
                    with torch.no_grad():
                        current_lp, _ = self.actor.evaluate_actions(obs_b, act_b, raw_b)
                        log_ratio = current_lp - old_lp_b
                        measured_kl = float((torch.expm1(log_ratio) - log_ratio).mean())
                    if not np.isfinite(measured_kl) or measured_kl > self.target_kl:
                        early_stop = True
                        break

                metric_rows.append(ppo_minibatch_step(
                    self, obs_b, obs_b, act_b, old_lp_b, adv_b, ret_b,
                    raw_actions=raw_b,
                ))
            if early_stop:
                break

        metrics = mean_update_metrics(metric_rows)
        metrics["train/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["train/rollout_steps"] = n
        metrics["train/optimizer_steps"] = len(metric_rows)
        metrics["train/kl_early_stop"] = float(early_stop)
        metrics["train/last_checked_kl"] = measured_kl
        metrics["train/action_saturation"] = float((actions.abs() >= 1 - 1e-6).float().mean())
        with torch.no_grad():
            metrics["train/action_std"] = float(self.actor.log_std.clamp(
                self.actor.LOG_STD_MIN, self.actor.LOG_STD_MAX).exp().mean())
        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "algorithm": "ppo",
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "action_contract": self.action_contract,
                "physics_contract": self.physics_contract,
                "observation_contract": self.observation_contract,
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "activation": self.activation,
            },
            path,
        )

    def load(self, path: str, *, load_optimizer: bool = True, observation_extension: Optional[str] = None) -> None:
        """Load actor/critic weights; transfer runs keep their fresh optimizer."""
        from utils.torch_io import safe_load
        ckpt = safe_load(path, map_location=self.device)
        if observation_extension not in {None, "target_frenet"}:
            raise ValueError("Unsupported PPO observation extension")
        if (observation_extension == "target_frenet" and
                ckpt.get("observation_contract") != self.observation_contract):
            from copy import deepcopy
            if load_optimizer:
                raise ValueError("Observation expansion requires a fresh optimizer")
            destination = deepcopy(self.observation_contract)
            if not isinstance(destination, dict) or not isinstance(destination.get("observation"), dict):
                raise ValueError("Observation expansion requires explicit observation contracts")
            target = destination["observation"].pop("target_frenet", {})
            if (not target.get("enabled") or destination != ckpt.get("observation_contract")
                    or self.obs_dim != ckpt.get("obs_dim", -1) + 5):
                raise ValueError("Target extension must preserve the complete pretrained observation prefix")
            # Pad only the first linear layer; every original column and every
            # other actor/critic parameter is retained exactly.
            for network in ("actor", "critic"):
                weights = ckpt[network]["net.0.weight"]
                if weights.shape[1] != ckpt["obs_dim"]:
                    raise ValueError("Checkpoint input weights disagree with observation dimensions")
                ckpt[network]["net.0.weight"] = torch.nn.functional.pad(weights, (0, 5))
            ckpt["obs_dim"] = self.obs_dim
            ckpt["observation_contract"] = self.observation_contract
        for key in ("physics_contract", "observation_contract"):
            if ckpt.get(key) != getattr(self, key):
                raise ValueError(f"Incompatible checkpoint {key}; physics/observation semantics differ")
        for key, expected in (
            ("algorithm", "ppo"), ("obs_dim", self.obs_dim),
            ("action_dim", self.action_dim),
            ("actor_hidden_dims", self.actor_hidden_dims),
            ("critic_hidden_dims", self.critic_hidden_dims),
            ("activation", self.activation),
        ):
            if key in ckpt and ckpt[key] != expected:
                raise ValueError(
                    f"Incompatible PPO checkpoint {key}: checkpoint={ckpt[key]!r}, current={expected!r}."
                )
        if ckpt.get("action_contract", {"speed_control": "direct"}) != self.action_contract:
            raise ValueError("Incompatible PPO checkpoint action contract (speed control semantics differ).")
        for key, expected in (("action_low", self.action_low), ("action_high", self.action_high)):
            if key in ckpt:
                actual = np.asarray(ckpt[key], dtype=np.float32)
                if actual.shape != expected.shape or not np.allclose(actual, expected):
                    raise ValueError(f"Incompatible PPO checkpoint {key}: action bounds differ.")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
