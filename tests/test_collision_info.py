from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.env.collision_state import build_step_terminations, build_truncations, update_collision_flags
from src.env.info_builder import add_step_info_fields


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
