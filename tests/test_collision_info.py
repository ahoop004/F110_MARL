from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.env.collision_state import build_step_terminations, build_truncations, update_collision_flags
from src.env.info_builder import (
    STABLE_RESET_INFO_KEYS,
    STABLE_STEP_INFO_KEYS,
    add_step_info_fields,
    add_time_limit_info,
    build_step_facts,
    build_reset_info_payloads,
    filter_info_payloads,
)
from src.env.types import AgentState, GlobalState, ProgressState


def test_collision_flags_and_global_collision_termination():
    agents = ["car_0", "car_1"]
    flags = np.zeros(2, dtype=bool)
    steps = np.full(2, -1, dtype=np.int32)

    update_collision_flags(agents, np.array([False, True]), flags, steps, elapsed_steps=7)

    assert flags.tolist() == [False, True]
    assert steps.tolist() == [-1, 7]

    terminations = build_step_terminations(
        agents,
        flags,
        lap_completion={},
        terminate_on_collision={"car_0": True, "car_1": True},
    )
    assert terminations == {"car_0": True, "car_1": True}


def test_truncations_and_step_info_fields():
    truncations, truncated = build_truncations(["car_0"], max_steps=3, elapsed_steps=2)
    assert truncated is True
    assert truncations == {"car_0": True}

    infos = {"car_0": {}}
    add_step_info_fields(
        infos,
        possible_agents=["car_0"],
        agent_target_index={"car_0": 1},
        collision_flags=np.array([False, True]),
        finish_crossed=np.array([False, True]),
        locked_velocities={"car_0": 0.5},
        lock_speed_steps=4,
        episode_step_count=2,
    )

    assert infos["car_0"]["collision"] is False
    assert infos["car_0"]["target_collision"] is True
    assert infos["car_0"]["target_finished"] is True
    assert infos["car_0"]["locked_velocity"] == 0.5
    assert infos["car_0"]["lock_speed_active"] is True


def test_filter_info_payloads_minimal_keeps_only_stable_training_facts():
    filtered = filter_info_payloads(
        {
            "car_0": {
                "collision": False,
                "target_collision": True,
                "map_bundle": "line2",
                "spawn_s": 0.5,
            }
        },
        info_level="minimal",
    )

    assert filtered == {"car_0": {"collision": False, "target_collision": True}}


def test_build_reset_info_payloads_adds_map_spawn_and_finish_fields():
    finish_crossed = np.array([True, False])
    infos = build_reset_info_payloads(
        agent_ids=["car_0", "car_1"],
        map_bundle="line2",
        spawn_mapping={"car_0": "spawn_a"},
        spawn_metadata={"spawn_source": "map_yaml"},
        finish_line_data={"enabled": True},
        finish_crossed=finish_crossed,
        agent_id_to_index={"car_0": 0, "car_1": 1},
        info_level="training",
    )

    assert infos["car_0"]["map_bundle"] == "line2"
    assert infos["car_0"]["spawn_point"] == "spawn_a"
    assert infos["car_0"]["spawn_source"] == "map_yaml"
    assert infos["car_0"]["finish_line"] is True
    assert infos["car_1"]["finish_line"] is False


def test_add_time_limit_info_only_when_truncated():
    infos = {"car_0": {}, "car_1": {}}
    add_time_limit_info(infos, truncated=False)
    assert infos == {"car_0": {}, "car_1": {}}

    add_time_limit_info(infos, truncated=True)
    assert infos == {"car_0": {"time_limit": True}, "car_1": {"time_limit": True}}


def test_stable_step_info_keys_are_a_subset_of_full_step_info():
    """Keys added by add_step_info_fields must include all stable step keys."""
    infos = {"car_0": {}}
    add_step_info_fields(
        infos,
        possible_agents=["car_0"],
        agent_target_index={"car_0": None},
        collision_flags=np.array([False]),
        finish_crossed=None,
        locked_velocities={},
        lock_speed_steps=0,
        episode_step_count=0,
    )
    add_time_limit_info(infos, truncated=True)
    present = set(infos["car_0"])
    # Every stable step key (except "centerline" which is optional) must be present.
    required = STABLE_STEP_INFO_KEYS - {"centerline"}
    missing = required - present
    assert not missing, f"Stable step info keys missing from output: {missing}"


def test_stable_reset_info_keys_documentation():
    """STABLE_RESET_INFO_KEYS is a frozenset of strings (contract sanity check)."""
    assert isinstance(STABLE_RESET_INFO_KEYS, frozenset)
    assert all(isinstance(k, str) for k in STABLE_RESET_INFO_KEYS)
    assert "map_bundle" in STABLE_RESET_INFO_KEYS
    assert "spawn_point" in STABLE_RESET_INFO_KEYS


def test_build_step_facts_captures_public_transition_payloads():
    agent_state = AgentState(
        agent_id="car_0",
        pose=np.zeros(3, dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        angular_velocity=0.0,
        collision=True,
        progress=ProgressState(),
    )
    global_state = GlobalState(
        agent_ids=("car_0",),
        vector=np.zeros(7, dtype=np.float32),
    )

    facts = build_step_facts(
        agent_ids=["car_0"],
        agent_states={"car_0": agent_state},
        global_state=global_state,
        collision_flags=np.array([True]),
        terminations={"car_0": True},
        truncations={"car_0": False},
        infos={"car_0": {"collision": True}},
    )

    assert facts.agent_states["car_0"] is agent_state
    assert facts.global_state is global_state
    assert facts.collisions == {"car_0": True}
    assert facts.terminations == {"car_0": True}
    assert facts.truncations == {"car_0": False}
    assert facts.info == {"car_0": {"collision": True}}
