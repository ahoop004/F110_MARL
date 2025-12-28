"""Utilities coordinating federated weight averaging across parallel runs."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from f110x.trainer.base import Trainer


@dataclass
class FederatedConfig:
    enabled: bool
    interval: int
    agents: Tuple[str, ...]
    root: Path
    mode: str = "mean"
    timeout: float = 600.0
    weights: Optional[Mapping[Any, float]] = None
    checkpoint_after_sync: bool = True
    optimizer_strategy: str = "local"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], base_dir: Optional[Path] = None) -> "FederatedConfig":
        enabled = bool(data.get("enabled", False))
        interval = max(int(data.get("interval", 100) or 1), 1)
        agents_raw = data.get("agents") or []
        if isinstance(agents_raw, Mapping):
            agents_iter: Iterable[Any] = agents_raw.keys()
        elif isinstance(agents_raw, (list, tuple, set)):
            agents_iter = agents_raw
        elif agents_raw:
            agents_iter = [agents_raw]
        else:
            agents_iter = []
        agents = tuple(str(agent).strip() for agent in agents_iter if str(agent).strip())

        root_raw = data.get("root")
        if root_raw:
            root_path = Path(str(root_raw)).expanduser()
            if base_dir is not None and not root_path.is_absolute():
                root_path = (base_dir / root_path).resolve()
        else:
            root_path = (base_dir or Path.cwd()).resolve()
        root_path.mkdir(parents=True, exist_ok=True)

        mode = str(data.get("mode", data.get("strategy", "mean"))).strip().lower() or "mean"
        timeout = float(data.get("timeout", 600.0) or 600.0)

        weights_raw = data.get("weights")
        weights: Optional[Mapping[Any, float]]
        if isinstance(weights_raw, Mapping):
            weights = {key: float(value) for key, value in weights_raw.items()}
        elif isinstance(weights_raw, (list, tuple)):
            weights = {idx: float(value) for idx, value in enumerate(weights_raw)}
        else:
            weights = None

        checkpoint_after = bool(data.get("checkpoint_after_sync", True))
        optimizer_strategy = str(data.get("optimizer_strategy", "local")).strip().lower() or "local"
        if optimizer_strategy not in {"local", "average", "reset"}:
            raise ValueError(f"Unsupported optimizer strategy '{optimizer_strategy}'")

        return cls(
            enabled=enabled,
            interval=interval,
            agents=agents,
            root=root_path,
            mode=mode,
            timeout=timeout,
            weights=weights,
            checkpoint_after_sync=checkpoint_after,
            optimizer_strategy=optimizer_strategy,
        )


class FederatedAverager:
    """Coordinate shared weight averaging across distributed trainer replicas."""

    def __init__(self, config: FederatedConfig, *, client_id: int, total_clients: int, logger: Optional[Any] = None) -> None:
        if not config.enabled:
            raise ValueError("FederatedAverager requires an enabled config")
        if not config.agents:
            raise ValueError("FederatedAverager requires at least one target agent")
        self.config = config
        self.client_id = int(client_id)
        self.total_clients = max(int(total_clients), 1)
        self.logger = logger

        self._client_name = f"client_{self.client_id:02d}"

    # ------------------------------------------------------------------
    def sync(self, trainers: Mapping[str, Trainer], round_index: int) -> Optional[Dict[str, float]]:
        """Publish local weights and average across all clients."""

        config = self.config
        round_dir = config.root / f"round_{round_index:05d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        local_path = round_dir / f"{self._client_name}.pt"

        payload: Dict[str, Any] = {}
        for agent_id in config.agents:
            trainer = trainers.get(agent_id)
            if trainer is None:
                raise KeyError(f"Trainer '{agent_id}' not available for federated averaging")
            snapshot = trainer.state_dict(include_optimizer=False)
            payload[agent_id] = snapshot

        tmp_path = local_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, local_path)

        metrics: Dict[str, float] = {}
        start = time.monotonic()
        aggregated: Dict[str, List[Tuple[Any, float]]] = {}

        deadline = start + config.timeout
        while True:
            missing = []
            for idx in range(self.total_clients):
                peer_path = round_dir / f"client_{idx:02d}.pt"
                if not peer_path.exists():
                    missing.append(peer_path)
            if not missing:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Federated averaging timed out waiting for peers: {missing}")
            time.sleep(1.0)

        weights_vector = self._resolve_weights(self.total_clients)

        for idx in range(self.total_clients):
            peer_path = round_dir / f"client_{idx:02d}.pt"
            state = torch.load(peer_path, map_location="cpu")
            weight = float(weights_vector[min(idx, weights_vector.shape[0] - 1)])
            for agent_id, snapshot in state.items():
                aggregated.setdefault(agent_id, []).append((snapshot, weight))

        include_optimizer = self.config.optimizer_strategy == "average"

        for agent_id, entries in aggregated.items():
            averaged = self._average_snapshots(entries)
            trainer = trainers.get(agent_id)
            if trainer is None:
                continue
            trainer.load_state_dict(averaged, strict=False, include_optimizer=include_optimizer)

        elapsed = time.monotonic() - start
        metrics["federated/round_time"] = elapsed
        metrics["federated/round_index"] = float(round_index)
        metrics["federated/clients"] = float(self.total_clients)

        if self.logger is not None:
            try:
                self.logger.info(
                    "Federated averaging completed",
                    extra={
                        "round": round_index,
                        "elapsed": elapsed,
                        "clients": self.total_clients,
                    },
                )
            except Exception:
                pass

        return metrics

    # ------------------------------------------------------------------
    def _resolve_weights(self, count: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("No snapshots available for averaging")
        custom = self.config.weights
        if not custom:
            return torch.full((count,), 1.0 / count, dtype=torch.float32)
        vector = torch.zeros(count, dtype=torch.float32)
        ordered_items = list(custom.items()) if isinstance(custom, Mapping) else []
        total = 0.0
        for idx in range(count):
            weight: Optional[float] = None
            key_candidates = (
                idx,
                str(idx),
                f"client_{idx}",
                f"client_{idx:02d}",
            )
            for key in key_candidates:
                if key in custom:
                    weight = float(custom[key])
                    break
            if weight is None and idx < len(ordered_items):
                weight = float(ordered_items[idx][1])
            if weight is None:
                weight = 0.0
            vector[idx] = weight
            total += weight
        if total <= 0.0:
            vector.fill_(1.0 / count)
        else:
            vector /= total
        return vector

    @staticmethod
    def _average_snapshots(entries: Iterable[Tuple[Mapping[str, Any], float]]) -> Dict[str, Any]:
        per_key: Dict[str, List[Tuple[Any, float]]] = {}
        for snapshot, weight in entries:
            for key, value in snapshot.items():
                per_key.setdefault(key, []).append((value, weight))

        averaged: Dict[str, Any] = {}
        for key, values in per_key.items():
            averaged[key] = FederatedAverager._combine_values(values)
        return averaged

    @staticmethod
    def _combine_values(entries: List[Tuple[Any, float]]) -> Any:
        if not entries:
            return None

        exemplar, _ = entries[0]

        if isinstance(exemplar, torch.Tensor):
            total = torch.zeros_like(exemplar)
            for value, weight in entries:
                total = total + value.detach().clone() * weight
            return total

        if isinstance(exemplar, (float, int)):
            weighted_sum = sum(float(value) * weight for value, weight in entries)
            weight_total = sum(weight for _, weight in entries)
            if weight_total == 0.0:
                mean_value = weighted_sum
            else:
                mean_value = weighted_sum / weight_total
            if isinstance(exemplar, int):
                return int(round(mean_value))
            return float(mean_value)

        if isinstance(exemplar, dict):
            keys = set(exemplar.keys())
            for value, _ in entries:
                if set(value.keys()) != keys:
                    raise ValueError("Optimizer state dictionaries must share the same keys")
            combined: Dict[str, Any] = {}
            for key in keys:
                sub_entries = [(value[key], weight) for value, weight in entries]
                combined[key] = FederatedAverager._combine_values(sub_entries)
            return combined

        if isinstance(exemplar, (list, tuple)):
            length = len(exemplar)
            for value, _ in entries:
                if len(value) != length:
                    raise ValueError("Optimizer parameter groups must share the same length")
            aggregated_seq = [
                FederatedAverager._combine_values([(value[idx], weight) for value, weight in entries])
                for idx in range(length)
            ]
            return type(exemplar)(aggregated_seq)

        # Fallback: return exemplar (first value)
        return exemplar


__all__ = ["FederatedConfig", "FederatedAverager"]
