import tempfile
import time
from pathlib import Path

import torch

from f110x.federated.averager import FederatedConfig, FederatedAverager


class DummyTrainer:
    def __init__(self, value: float, steps: int = 0) -> None:
        self._value = torch.tensor([value], dtype=torch.float32)
        self.total_it = steps

    def state_dict(self, include_optimizer: bool = False):
        return {
            "weight": self._value.clone(),
            "total_it": int(self.total_it),
        }

    def load_state_dict(self, state, strict: bool = False, include_optimizer: bool = False):
        self._value.copy_(state["weight"])
        if "total_it" in state:
            self.total_it = int(state["total_it"])


def test_avg_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = FederatedConfig.from_mapping(
            {
                "enabled": True,
                "interval": 1,
                "agents": ["car_0"],
                "root": tmp,
                "timeout": 5,
            },
            base_dir=Path(tmp),
        )
        trainers = {"car_0": DummyTrainer(1.0, steps=7)}
        server = FederatedAverager(cfg, client_id=0, total_clients=1)
        metrics = server.sync(trainers, round_index=0)
        assert metrics is not None
        assert metrics["federated/clients"] == 1
        assert torch.allclose(trainers["car_0"]._value, torch.tensor([1.0]))
        assert trainers["car_0"].total_it == 7


def test_weight_index_mapping():
    cfg = FederatedConfig.from_mapping(
        {
            "enabled": True,
            "interval": 1,
            "agents": ["car_0", "car_1"],
            "root": Path("/tmp"),
            "weights": [0.7, 0.3],
        },
        base_dir=Path("/tmp"),
    )
    weights = cfg.weights
    assert weights is not None
    assert weights[0] == 0.7 and weights[1] == 0.3
    assert cfg.checkpoint_after_sync is True


def test_resolve_weights_with_named_clients():
    cfg = FederatedConfig.from_mapping(
        {
            "enabled": True,
            "interval": 1,
            "agents": ["car_0"],
            "root": Path("/tmp"),
            "weights": {"client_00": 0.8, "client_01": 0.2},
            "checkpoint_after_sync": False,
        },
        base_dir=Path("/tmp"),
    )
    server = FederatedAverager(cfg, client_id=0, total_clients=2)
    weights = server._resolve_weights(2)
    assert torch.allclose(weights, torch.tensor([0.8, 0.2], dtype=torch.float32))
    assert cfg.checkpoint_after_sync is False


def test_combine_optimizer_state():
    entries = [
        (
            {
                "state": {
                    0: {
                        "exp_avg": torch.tensor([1.0, 2.0]),
                        "exp_avg_sq": torch.tensor([4.0, 9.0]),
                        "step": 10,
                    }
                },
                "param_groups": [
                    {
                        "lr": 0.001,
                        "betas": (0.9, 0.999),
                    }
                ],
            },
            0.5,
        ),
        (
            {
                "state": {
                    0: {
                        "exp_avg": torch.tensor([3.0, 4.0]),
                        "exp_avg_sq": torch.tensor([16.0, 25.0]),
                        "step": 30,
                    }
                },
                "param_groups": [
                    {
                        "lr": 0.001,
                        "betas": (0.9, 0.999),
                    }
                ],
            },
            0.5,
        ),
    ]

    averaged = FederatedAverager._combine_values(entries)
    assert torch.allclose(averaged["state"][0]["exp_avg"], torch.tensor([2.0, 3.0]))
    assert averaged["state"][0]["step"] == 20
    assert averaged["param_groups"][0]["lr"] == 0.001

def test_sync_timeout(tmp_path):
    cfg = FederatedConfig.from_mapping(
        {
            "enabled": True,
            "interval": 1,
            "agents": ["car_0"],
            "root": tmp_path,
            "timeout": 0.05,
        },
        base_dir=tmp_path,
    )
    trainer = DummyTrainer(1.0)
    server = FederatedAverager(cfg, client_id=0, total_clients=2)
    start = time.monotonic()
    try:
        server.sync({"car_0": trainer}, round_index=0)
        assert False, "Expected timeout"
    except TimeoutError:
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05
