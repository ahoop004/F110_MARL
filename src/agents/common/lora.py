"""Low-rank residuals on a frozen driving actor, shared or routed by teammate."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from agents.common.networks import Actor


def resolve_lora_config(value):
    """None preserves ordinary full fine-tuning; reject misspelled adapter options."""
    if value is None:
        return None
    allowed = {"mode", "rank", "alpha", "train_log_std"}
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError(f"lora must be a mapping with fields {sorted(allowed)}")
    mode = value.get("mode", "shared")
    if not isinstance(mode, str) or mode not in {"shared", "per_agent"}:
        raise ValueError("lora.mode must be shared or per_agent")
    rank = value.get("rank", 4)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("lora.rank must be a positive integer")
    alpha = value.get("alpha", rank)
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("lora.alpha must be finite and positive")
    train_log_std = value.get("train_log_std", True)
    if not isinstance(train_log_std, bool):
        raise ValueError("lora.train_log_std must be a boolean")
    return dict(mode=mode, rank=rank, alpha=float(alpha), train_log_std=train_log_std)


class LowRankResidual(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        if rank > min(in_features, out_features):
            raise ValueError("lora.rank cannot exceed either adapted layer dimension")
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))
        self.scale = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, inputs):
        return F.linear(F.linear(inputs, self.A), self.B) * self.scale


class LoRAActor(Actor):
    """Reuse Actor's squashed-Gaussian likelihood with explicit adapter routing.

    Every hidden linear layer has a residual; the output head and all original
    biases stay frozen. Exploration is one shared vector in both routing modes.
    No dropout or automatic weight merging changes the rollout/update policy.
    """

    def __init__(self, base: Actor, config, n_agents):
        nn.Module.__init__(self)
        self.net = base.net
        self.log_std = base.log_std
        self.register_buffer("_entropy_nodes", base._entropy_nodes, persistent=False)
        self.register_buffer("_entropy_weights", base._entropy_weights, persistent=False)
        self.config = resolve_lora_config(config)
        self.target_layers = [i for i, layer in enumerate(self.net)
                              if isinstance(layer, nn.Linear) and i != len(self.net) - 1]
        if not self.target_layers:
            raise ValueError("LoRA requires at least one actor hidden layer")
        count = n_agents if self.config["mode"] == "per_agent" else 1
        self.adapters = nn.ModuleList([
            nn.ModuleDict({str(i): LowRankResidual(
                self.net[i].in_features, self.net[i].out_features,
                self.config["rank"], self.config["alpha"],
            ) for i in self.target_layers}) for _ in range(count)
        ])
        self.net.requires_grad_(False)
        self.log_std.requires_grad_(self.config["train_log_std"])

    def base_state_dict(self):
        return {key: value for key, value in self.state_dict().items()
                if not key.startswith("adapters.")}

    def reset_adapters(self):
        for bank in self.adapters:
            for residual in bank.values():
                residual.reset_parameters()

    def _mean(self, obs, bank):
        for i, layer in enumerate(self.net):
            output = layer(obs)
            if str(i) in bank:
                output = output + bank[str(i)](obs)
            obs = output
        return obs

    def forward(self, obs, adapter_indices=None):
        if self.config["mode"] == "shared":
            if adapter_indices is not None:
                raise ValueError("Shared LoRA does not take per-agent adapter indices")
            mean = self._mean(obs, self.adapters[0])
        else:
            if (adapter_indices is None or adapter_indices.shape != (len(obs),)
                    or adapter_indices.dtype != torch.long):
                raise ValueError("Per-agent LoRA requires one integer adapter index per observation")
            if torch.any((adapter_indices < 0) | (adapter_indices >= len(self.adapters))):
                raise ValueError("Invalid LoRA adapter index")
            mean = obs.new_zeros((len(obs), self.net[-1].out_features))
            for index, bank in enumerate(self.adapters):
                rows = torch.nonzero(adapter_indices == index, as_tuple=True)[0]
                if rows.numel():
                    mean = mean.index_copy(0, rows, self._mean(obs.index_select(0, rows), bank))
        return mean, self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
