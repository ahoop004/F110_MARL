"""Shared buffers and utilities for PPO agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class EntropySchedule:
    initial: float
    current: float
    final: float
    decay_start: int
    decay_episodes: int
    episode_idx: int = 0

    def update(self, episodes: int) -> float:
        if self.decay_episodes <= 0:
            return self.current
        self.episode_idx += max(int(episodes), 0)
        if self.episode_idx < self.decay_start:
            return self.current
        progress = self.episode_idx - self.decay_start
        frac = min(max(progress, 0) / max(self.decay_episodes, 1), 1.0)
        self.current = self.initial * (1.0 - frac) + self.final * frac
        return self.current


class BasePPOAgent:
    """Common rollout buffer + GAE utilities shared across PPO variants."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device) -> None:
        self.obs_dim = int(cfg["obs_dim"])
        self.act_dim = int(cfg["act_dim"])

        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("lam", 0.95))
        self.clip_eps = float(cfg.get("clip_eps", 0.2))
        self.update_epochs = int(cfg.get("update_epochs", 10))
        self.minibatch_size = int(cfg.get("minibatch_size", 64))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))

        base_ent_coef = float(cfg.get("ent_coef", 0.0))
        schedule_cfg = cfg.get("ent_coef_schedule") or {}
        self.entropy = EntropySchedule(
            initial=float(schedule_cfg.get("start", base_ent_coef)),
            current=float(schedule_cfg.get("start", base_ent_coef)),
            final=float(schedule_cfg.get("final", base_ent_coef)),
            decay_start=int(schedule_cfg.get("decay_start", 0)),
            decay_episodes=max(int(schedule_cfg.get("decay_episodes", 0)), 0),
        )

        # Allow subclasses to query the unresolved schedule parameters.
        self.ent_coef_initial = self.entropy.initial
        self.ent_coef_final = self.entropy.final
        self.ent_coef_decay_start = self.entropy.decay_start
        self.ent_coef_decay_episodes = self.entropy.decay_episodes

        self._episodes_since_update = 0
        self.device = device

        self.reset_buffer()

    # ------------------------------------------------------------------
    # Rollout buffer helpers
    # ------------------------------------------------------------------
    def reset_buffer(self) -> None:
        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[np.ndarray] = []
        self.raw_act_buf: List[np.ndarray] = []
        self.rew_buf: List[float] = []
        self.done_buf: List[bool] = []
        self.logp_buf: List[float] = []
        self.val_buf: List[float] = []
        self.adv_buf = np.zeros(0, dtype=np.float32)
        self.ret_buf = np.zeros(0, dtype=np.float32)
        self._pending_bootstrap: Optional[float] = None
        self._episode_bootstrap: List[float] = []
        self._episode_boundaries: List[int] = [0]

    def store_transition(self, rew: float, done: bool) -> None:
        self.rew_buf.append(float(rew))
        self.done_buf.append(bool(done))
        if done:
            self._episodes_since_update += 1
            bootstrap = float(self._pending_bootstrap or 0.0)
            self._episode_bootstrap.append(bootstrap)
            self._pending_bootstrap = None
            self._episode_boundaries.append(len(self.rew_buf))
            self.on_episode_end()

    def on_episode_end(self) -> None:  # pragma: no cover - hook for subclasses
        return None

    # ------------------------------------------------------------------
    # Advantage / return calculation
    # ------------------------------------------------------------------
    def finish_path(self, *, normalize_advantage: bool = True) -> None:
        T = min(len(self.rew_buf), len(self.obs_buf), len(self.val_buf))
        if T == 0:
            self.adv_buf = np.zeros(0, dtype=np.float32)
            self.ret_buf = np.zeros(0, dtype=np.float32)
            return

        if len(self.done_buf) < T:
            raise ValueError(f"rollout length mismatch: rewards {T}, dones {len(self.done_buf)}")

        rewards = np.asarray(self.rew_buf[:T], dtype=np.float32)
        values = np.asarray(self.val_buf[:T], dtype=np.float32)
        if values.shape[0] < T:
            pad_val = values[-1] if values.size else 0.0
            values = np.concatenate(
                [values, np.full(T - values.shape[0], pad_val, dtype=np.float32)],
                axis=0,
            )
        else:
            values = values[:T]
        dones = np.asarray(self.done_buf[:T], dtype=np.float32)

        normalised_boundaries: List[int] = []
        seen = set()
        for boundary in self._episode_boundaries:
            b = min(max(int(boundary), 0), T)
            if b not in seen:
                normalised_boundaries.append(b)
                seen.add(b)
        if not normalised_boundaries or normalised_boundaries[0] != 0:
            normalised_boundaries.insert(0, 0)
        if normalised_boundaries[-1] != T:
            normalised_boundaries.append(T)
        self._episode_boundaries = normalised_boundaries
        while len(self._episode_bootstrap) < len(self._episode_boundaries) - 1:
            if self._pending_bootstrap is not None:
                self._episode_bootstrap.append(self._pending_bootstrap)
                self._pending_bootstrap = None
            else:
                self._episode_bootstrap.append(0.0)
        self._episode_bootstrap = self._episode_bootstrap[: len(self._episode_boundaries) - 1]

        adv = np.zeros(T, dtype=np.float32)
        ret = np.zeros(T, dtype=np.float32)

        for idx in range(len(self._episode_boundaries) - 1):
            start = self._episode_boundaries[idx]
            end = self._episode_boundaries[idx + 1]
            if end <= start:
                continue
            bootstrap_v = self._episode_bootstrap[idx] if idx < len(self._episode_bootstrap) else 0.0
            gae = 0.0
            for t in reversed(range(start, end)):
                mask = 1.0 - dones[t]
                next_value = bootstrap_v if t == end - 1 else values[t + 1]
                delta = rewards[t] + self.gamma * next_value * mask - values[t]
                gae = delta + self.gamma * self.lam * mask * gae
                adv[t] = gae
            ret[start:end] = adv[start:end] + values[start:end]

        if normalize_advantage and adv.size:
            std = adv.std()
            if std > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = np.zeros_like(adv)

        self.adv_buf = adv
        self.ret_buf = ret
        self.obs_buf = self.obs_buf[:T]
        self.act_buf = self.act_buf[:T]
        self.raw_act_buf = self.raw_act_buf[:T]
        self.logp_buf = self.logp_buf[:T]
        self.val_buf = list(values)
        self.rew_buf = self.rew_buf[:T]
        self.done_buf = self.done_buf[:T]

    # ------------------------------------------------------------------
    # Bootstrapping helpers
    # ------------------------------------------------------------------
    def record_final_value(self, obs: Any) -> None:
        self._pending_bootstrap = float(self._estimate_value(obs))

    def _estimate_value(self, obs: Any) -> float:
        """Subclasses must implement critic evaluation for a single observation."""

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Entropy schedule helpers
    # ------------------------------------------------------------------
    def apply_entropy_decay(self, episodes: int) -> None:
        self.entropy.update(episodes)

    @property
    def ent_coef(self) -> float:
        return self.entropy.current

    @ent_coef.setter
    def ent_coef(self, value: float) -> None:
        self.entropy.current = float(value)


__all__ = ["BasePPOAgent"]
