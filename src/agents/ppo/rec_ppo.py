"""Recurrent PPO agent supporting RNN/GRU/LSTM policies."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from agents.ppo.base import BasePPOAgent
from agents.common.networks import build_mlp


def _to_tensor_list(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(arrays, dtype=np.float32)


class RecurrentPPOAgent(BasePPOAgent):
    def __init__(self, cfg: Dict[str, Any]):
        device = torch.device(cfg.get("device", "cpu"))
        super().__init__(cfg, device)
        target_kl = cfg.get("target_kl")
        self.target_kl = float(target_kl) if target_kl is not None else None
        self.sequence_batch_size = int(cfg.get("sequence_batch_size", cfg.get("minibatch_size", 1)) or 1)

        action_low = np.asarray(cfg.get("action_low"), dtype=np.float32)
        action_high = np.asarray(cfg.get("action_high"), dtype=np.float32)
        if action_low.shape != (self.act_dim,) or action_high.shape != (self.act_dim,):
            raise ValueError("Recurrent PPO requires action_low/action_high vectors matching act_dim")
        self.action_low_np = action_low
        self.action_high_np = action_high
        self.action_low_t = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high_t = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.action_scale_t = self.action_high_t - self.action_low_t
        self.action_mid_t = (self.action_low_t + self.action_high_t) * 0.5
        self.action_half_range_t = self.action_scale_t * 0.5
        self.squash_eps = 1e-6

        hidden_size = int(cfg.get("rnn_hidden_size", 128))
        num_layers = int(cfg.get("rnn_layers", 1))
        dropout = float(cfg.get("rnn_dropout", 0.0)) if num_layers > 1 else 0.0
        rnn_type = str(cfg.get("rnn_type", "lstm")).lower()
        encoder_dims: Iterable[int] = cfg.get("mlp_hidden_dims", [256, 256]) or []

        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}.get(rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Expected one of rnn/gru/lstm.")

        encoder_output = encoder_dims[-1] if encoder_dims else self.obs_dim

        # Actor network -------------------------------------------------
        self.actor_encoder = build_mlp(self.obs_dim, encoder_dims, encoder_output).to(self.device)
        self.actor_rnn = rnn_cls(
            input_size=encoder_output,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        ).to(self.device)
        self.actor_mu = nn.Linear(hidden_size, self.act_dim).to(self.device)
        self.actor_log_std = nn.Parameter(torch.zeros(self.act_dim, device=self.device))

        # Critic network ------------------------------------------------
        self.critic_encoder = build_mlp(self.obs_dim, encoder_dims, encoder_output).to(self.device)
        self.critic_rnn = rnn_cls(
            input_size=encoder_output,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        ).to(self.device)
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)

        self._actor_params = list(self.actor_encoder.parameters()) + list(self.actor_rnn.parameters()) + list(self.actor_mu.parameters()) + [self.actor_log_std]
        self._critic_params = list(self.critic_encoder.parameters()) + list(self.critic_rnn.parameters()) + list(self.value_head.parameters())

        self.actor_opt = torch.optim.Adam(self._actor_params, lr=float(cfg.get("actor_lr", 3e-4)))
        self.critic_opt = torch.optim.Adam(self._critic_params, lr=float(cfg.get("critic_lr", 1e-3)))

        # Hidden state caches (updated during interaction)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.actor_hidden: Optional[HiddenState] = None
        self.critic_hidden: Optional[HiddenState] = None
        self._pending_bootstrap: Optional[float] = None

        # Rollout buffers -----------------------------------------------
        self.reset_buffer()

    # ------------------------------------------------------------------
    # Rollout interaction helpers
    # ------------------------------------------------------------------

    def reset_hidden_state(self) -> None:
        self.actor_hidden = None
        self.critic_hidden = None

    def on_episode_end(self) -> None:
        self.reset_hidden_state()

    def _init_hidden(self, batch_size: int) -> "HiddenState":
        shape = (self.num_layers, batch_size, self.hidden_size)
        if self.rnn_type == "lstm":
            h = torch.zeros(shape, device=self.device)
            c = torch.zeros(shape, device=self.device)
            return h, c
        return torch.zeros(shape, device=self.device)

    @staticmethod
    def _detach_hidden(hidden: "HiddenState") -> "HiddenState":
        if isinstance(hidden, tuple):
            return (hidden[0].detach(), hidden[1].detach())
        return hidden.detach()

    @staticmethod
    def _clone_hidden(hidden: Optional["HiddenState"]) -> Optional["HiddenState"]:
        if hidden is None:
            return None
        if isinstance(hidden, tuple):
            return (hidden[0].clone(), hidden[1].clone())
        return hidden.clone()

    def _encode_actor(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.actor_encoder(obs_batch)

    def _encode_critic(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.critic_encoder(obs_batch)

    def _actor_forward_step(self, obs_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, "HiddenState"]:
        batch = obs_tensor.unsqueeze(0)  # (1, obs_dim)
        encoded = self._encode_actor(batch)
        encoded = encoded.unsqueeze(1)  # (1, 1, feat)
        hidden = self.actor_hidden if self.actor_hidden is not None else self._init_hidden(batch_size=1)
        output, next_hidden = self.actor_rnn(encoded, hidden)
        next_hidden = self._detach_hidden(next_hidden)
        self.actor_hidden = next_hidden
        features = output[:, -1, :]
        mu = self.actor_mu(features)
        log_std = torch.clamp(self.actor_log_std, -5.0, 2.0).unsqueeze(0)
        return mu, log_std, next_hidden

    def _critic_forward_step(self, obs_tensor: torch.Tensor, *, hidden_override: Optional["HiddenState"] = None) -> Tuple[torch.Tensor, "HiddenState"]:
        batch = obs_tensor.unsqueeze(0)
        encoded = self._encode_critic(batch)
        encoded = encoded.unsqueeze(1)
        hidden_src = hidden_override if hidden_override is not None else self.critic_hidden
        hidden = hidden_src if hidden_src is not None else self._init_hidden(batch_size=1)
        output, next_hidden = self.critic_rnn(encoded, hidden)
        next_hidden = self._detach_hidden(next_hidden)
        if hidden_override is None:
            self.critic_hidden = next_hidden
        value = self.value_head(output[:, -1, :]).squeeze(-1)
        return value, next_hidden

    def _scale_action(self, squashed: torch.Tensor) -> torch.Tensor:
        return squashed.clamp(-1.0, 1.0) * self.action_half_range_t + self.action_mid_t

    def act(self, obs: np.ndarray, deterministic: bool = False, aid: Optional[str] = None) -> np.ndarray:
        """Select action (protocol-compliant interface).

        Args:
            obs: Observation
            deterministic: If True, select mean action (for eval)
            aid: Optional agent ID (legacy parameter)

        Returns:
            action: Selected action
        """
        if deterministic:
            return self.act_deterministic(obs, aid=aid)
        else:
            return self.act_stochastic(obs, aid=aid)

    def act_stochastic(self, obs: np.ndarray, aid: Optional[str] = None) -> np.ndarray:
        """Select stochastic action for training."""
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

        mu, log_std, _ = self._actor_forward_step(obs_t)
        std = log_std.exp()
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        squashed = torch.tanh(raw_action)
        scaled = self._scale_action(squashed)

        logp = dist.log_prob(raw_action).sum(dim=-1)
        logp -= torch.log(1 - squashed.pow(2) + self.squash_eps).sum(dim=-1)

        value, _ = self._critic_forward_step(obs_t)

        self.obs_buf.append(obs_np)
        self.act_buf.append(scaled.squeeze(0).detach().cpu().numpy())
        self.raw_act_buf.append(raw_action.squeeze(0).detach().cpu().numpy())
        self.logp_buf.append(float(logp.item()))
        self.val_buf.append(float(value.item()))

        return scaled.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def act_deterministic(self, obs: np.ndarray, aid: Optional[str] = None) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

        mu, _, _ = self._actor_forward_step(obs_t)
        squashed = torch.tanh(mu)
        scaled = self._scale_action(squashed)
        self._critic_forward_step(obs_t)
        return scaled.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def store(
        self,
        obs: np.ndarray,
        act: Any,
        rew: float,
        done: bool,
        terminated: bool = False,
    ) -> None:
        self.store_transition(rew, done, terminated)

    def _estimate_value(self, obs: Any) -> float:
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        critic_hidden_backup = self._clone_hidden(self.critic_hidden)
        value, _ = self._critic_forward_step(obs_t, hidden_override=critic_hidden_backup)
        return float(value.item())

    # ------------------------------------------------------------------
    # Advantage calculation & optimisation
    # ------------------------------------------------------------------

    def finish_path(self, *, normalize_advantage: bool = True) -> None:
        super().finish_path(normalize_advantage=normalize_advantage)

    def _ensure_episode_boundaries(self) -> None:
        if not self._episode_boundaries:
            self._episode_boundaries = [0]
        if self._episode_boundaries[-1] != len(self.rew_buf):
            self._episode_boundaries.append(len(self.rew_buf))
            if len(self._episode_bootstrap) < len(self._episode_boundaries) - 1:
                self._episode_bootstrap.append(0.0)

    def _prepare_sequences(self) -> List[Dict[str, torch.Tensor]]:
        self._ensure_episode_boundaries()

        adv_tensor = torch.as_tensor(self.adv_buf, dtype=torch.float32, device=self.device)

        ret_tensor = torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device)
        logp_old_tensor = torch.as_tensor(self.logp_buf, dtype=torch.float32, device=self.device)
        values_old_tensor = torch.as_tensor(np.asarray(self.val_buf, dtype=np.float32), dtype=torch.float32, device=self.device)

        obs_arr = _to_tensor_list(self.obs_buf)
        raw_arr = _to_tensor_list(self.raw_act_buf)

        episodes: List[Dict[str, torch.Tensor]] = []
        start = 0
        for boundary in self._episode_boundaries[1:]:
            end = boundary
            if end <= start:
                start = end
                continue
            obs_seq = torch.as_tensor(obs_arr[start:end], dtype=torch.float32, device=self.device)
            raw_seq = torch.as_tensor(raw_arr[start:end], dtype=torch.float32, device=self.device)
            seq_len = obs_seq.shape[0]
            logp_seq = logp_old_tensor[start : start + seq_len]
            adv_seq = adv_tensor[start : start + seq_len]
            ret_seq = ret_tensor[start : start + seq_len]
            val_seq = values_old_tensor[start : start + seq_len]
            episodes.append(
                {
                    "obs": obs_seq,
                    "raw_actions": raw_seq,
                    "logp_old": logp_seq,
                    "adv": adv_seq,
                    "ret": ret_seq,
                    "values_old": val_seq,
                }
            )
            start = end
        return episodes

    def _actor_eval_sequence(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_seq: (T, obs_dim)
        batch = obs_seq.unsqueeze(0)  # (1, T, obs_dim)
        B, T, _ = batch.shape
        flat = batch.view(B * T, -1)
        encoded = self._encode_actor(flat)
        encoded = encoded.view(B, T, -1)
        hidden = self._init_hidden(batch_size=B)
        output, _ = self.actor_rnn(encoded, hidden)
        features = output.squeeze(0)  # (T, hidden_size)
        mu = self.actor_mu(features)
        log_std = torch.clamp(self.actor_log_std, -5.0, 2.0).unsqueeze(0).expand_as(mu)
        return mu, log_std

    def _critic_eval_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        batch = obs_seq.unsqueeze(0)
        B, T, _ = batch.shape
        flat = batch.view(B * T, -1)
        encoded = self._encode_critic(flat)
        encoded = encoded.view(B, T, -1)
        hidden = self._init_hidden(batch_size=B)
        output, _ = self.critic_rnn(encoded, hidden)
        values = self.value_head(output.squeeze(0)).squeeze(-1)
        return values

    def update(self) -> Optional[Dict[str, float]]:
        if not self.rew_buf:
            return None

        self.finish_path(normalize_advantage=self.normalize_advantage)
        episodes = self._prepare_sequences()
        if not episodes:
            self.reset_buffer()
            self.reset_hidden_state()
            return None

        # Entropy schedule bookkeeping
        decay_count = max(self._episodes_since_update, len(episodes))
        self.apply_entropy_decay(decay_count)
        self._episodes_since_update = 0

        if self.episode_batch:
            sequence_batch = max(1, len(episodes))
        else:
            sequence_batch = max(1, self.sequence_batch_size)

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        stop_early = False

        for _ in range(self.update_epochs):
            random.shuffle(episodes)
            for idx in range(0, len(episodes), sequence_batch):
                batch_eps = episodes[idx : idx + sequence_batch]
                if not batch_eps:
                    continue

                total_steps = sum(ep["obs"].shape[0] for ep in batch_eps)
                if total_steps == 0:
                    continue

                policy_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
                value_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
                entropy_term = torch.zeros(1, dtype=torch.float32, device=self.device)
                kl_term = torch.zeros(1, dtype=torch.float32, device=self.device)

                for ep in batch_eps:
                    obs_seq = ep["obs"]
                    raw_actions = ep["raw_actions"]
                    logp_old = ep["logp_old"]
                    adv = ep["adv"]
                    ret = ep["ret"]
                    values_old = ep.get("values_old")

                    mu, log_std = self._actor_eval_sequence(obs_seq)
                    std = log_std.exp()
                    dist = Normal(mu, std)
                    values_pred = self._critic_eval_sequence(obs_seq)
                    policy_loss_seq, value_loss_seq, entropy_seq, kl_seq = self.compute_losses(
                        dist=dist,
                        raw_actions=raw_actions,
                        logp_old=logp_old,
                        advantages=adv,
                        returns=ret,
                        values_pred=values_pred,
                        values_old=values_old,
                        reduction="sum",
                    )

                    policy_loss += policy_loss_seq
                    value_loss += value_loss_seq
                    entropy_term += entropy_seq
                    kl_term += kl_seq

                policy_loss = policy_loss / total_steps
                value_loss = value_loss / total_steps
                entropy_mean = entropy_term / total_steps
                approx_kl = (kl_term / total_steps).abs()

                loss = policy_loss + value_loss - self.ent_coef * entropy_mean

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()

                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)

                self.actor_opt.step()
                self.critic_opt.step()

                approx_kl_value = float(approx_kl.detach().cpu().item())
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy_mean.detach().cpu().item()))
                approx_kls.append(approx_kl_value)

                # Early stopping based on KL divergence
                if self.target_kl is not None and approx_kl_value > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        stats = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

        self.reset_buffer()
        self.reset_hidden_state()
        return stats

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor_encoder": self.actor_encoder.state_dict(),
                "actor_rnn": self.actor_rnn.state_dict(),
                "actor_mu": self.actor_mu.state_dict(),
                "actor_log_std": self.actor_log_std.detach().cpu(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic_rnn": self.critic_rnn.state_dict(),
                "value_head": self.value_head.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "entropy_episode_idx": self.entropy.episode_idx,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor_encoder.load_state_dict(ckpt["actor_encoder"])
        self.actor_rnn.load_state_dict(ckpt["actor_rnn"])
        self.actor_mu.load_state_dict(ckpt["actor_mu"])
        if "actor_log_std" in ckpt:
            self.actor_log_std.data = ckpt["actor_log_std"].to(self.device)
        self.critic_encoder.load_state_dict(ckpt["critic_encoder"])
        self.critic_rnn.load_state_dict(ckpt["critic_rnn"])
        self.value_head.load_state_dict(ckpt["value_head"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        episode_idx = ckpt.get("entropy_episode_idx", ckpt.get("episode_idx", 0))
        self.entropy.episode_idx = int(episode_idx)
        self.reset_hidden_state()

HiddenState = Optional[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]
