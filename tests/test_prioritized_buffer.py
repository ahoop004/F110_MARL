import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "src" / "f110x"
_PACKAGES = [
    ("f110x", _ROOT),
    ("f110x.policies", _ROOT / "policies"),
    ("f110x.policies.buffers", _ROOT / "policies" / "buffers"),
]

for name, path in _PACKAGES:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_load_module("f110x.policies.buffers.replay", _ROOT / "policies" / "buffers" / "replay.py")
prioritized_module = _load_module(
    "f110x.policies.buffers.prioritized",
    _ROOT / "policies" / "buffers" / "prioritized.py",
)

PrioritizedReplayBuffer = prioritized_module.PrioritizedReplayBuffer


def test_prioritized_beta_schedule_respects_final():
    buffer = PrioritizedReplayBuffer(
        capacity=16,
        obs_shape=(3,),
        action_shape=(2,),
        alpha=0.6,
        beta=0.4,
        beta_increment_per_sample=0.5,
        beta_final=0.7,
        store_actions=True,
        store_action_indices=False,
    )

    for i in range(8):
        obs = np.full((3,), float(i), dtype=np.float32)
        action = np.full((2,), float(i), dtype=np.float32)
        reward = float(i)
        next_obs = obs + 1.0
        buffer.add(obs, action, reward, next_obs, done=False)

    assert buffer.beta_target == pytest.approx(0.7)
    assert buffer.beta == pytest.approx(0.4)

    priorities = np.linspace(0.1, 1.0, len(buffer), dtype=np.float32)
    buffer.update_priorities(np.arange(len(buffer)), priorities)

    _ = buffer.sample(4)
    assert buffer.beta == pytest.approx(0.7)

    beta_before = buffer.beta
    batch = buffer.sample(4)
    assert buffer.beta == pytest.approx(0.7)

    current_size = len(buffer)
    scaled = np.power(np.maximum(buffer._priorities[:current_size], buffer.min_priority), buffer.alpha)
    probs = scaled / scaled.sum()
    expected_weights = np.power(current_size * probs[batch["indices"]], -beta_before)
    expected_weights /= expected_weights.max()

    np.testing.assert_allclose(
        batch["weights"].reshape(-1),
        expected_weights.astype(np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
