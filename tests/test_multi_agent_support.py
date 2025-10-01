import os
from dataclasses import dataclass

import pytest

from f110x.utils import multi_agent_support as mas


@dataclass
class DummySpec:
    role: str


@dataclass
class DummyAssignment:
    spec: DummySpec
    agent_id: str


class DummyRoster:
    def __init__(self):
        self.roles = {
            "attacker": ["car_0", "car_1"],
            "defender": "car_2",
        }
        self.assignments = [
            DummyAssignment(DummySpec("attacker"), "car_0"),
            DummyAssignment(DummySpec("attacker"), "car_1"),
            DummyAssignment(DummySpec("defender"), "car_2"),
        ]


def test_gather_role_ids_handles_lists_and_assignments():
    roster = DummyRoster()
    attackers = mas.gather_role_ids(roster, "attacker")
    assert attackers == ["car_0", "car_1"]
    defender = mas.gather_role_ids(roster, "defender")
    assert defender == ["car_2"]


def test_gather_role_ids_falls_back_to_default():
    roster = DummyRoster()
    assert mas.gather_role_ids(roster, "spotter", default=["fallback"]) == ["fallback"]


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("multi_agent_draft", True),
        ("!multi_agent_draft", False),
        ("", False),
    ],
)
def test_feature_enabled_uses_environment(monkeypatch, env_value, expected):
    monkeypatch.setenv("F110_MULTI_AGENT_FLAGS", env_value)
    mas._ENV_FLAGS.clear()
    mas._ENV_FLAGS.update(mas._parse_env_flags())
    assert mas.feature_enabled("multi_agent_draft") is expected


def test_feature_enabled_respects_source_mapping():
    flags = {"feature_flags": {"multi_agent_draft": True}}
    assert mas.feature_enabled("multi_agent_draft", source=flags) is True


def test_joint_replay_stub_round_robin_sampling():
    buffer = mas.JointReplayStub(capacity=3)
    exp = mas.JointExperience(
        attacker_ids=["car_0", "car_1"],
        defender_id="car_2",
        observations={"car_0": {}, "car_1": {}, "car_2": {}},
        actions={"car_0": 0, "car_1": 1, "car_2": None},
        rewards={"car_0": 1.0, "car_1": 0.5},
        dones={},
    )
    for _ in range(5):
        buffer.append(exp)
    assert len(buffer) == 3
    sample = buffer.sample(2)
    assert len(sample) == 2
    assert all(isinstance(item, mas.JointExperience) for item in sample)
    assert sample[0].attacker_actions() == [0, 1]
