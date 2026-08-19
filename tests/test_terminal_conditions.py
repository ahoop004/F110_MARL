from __future__ import annotations

import pytest

from env.collision_state import (
    apply_episode_termination_policy,
    normalize_episode_termination_mode,
)


AGENTS = ["car_0", "car_1", "car_2"]


def test_any_agent_propagates_joint_termination() -> None:
    terminations, episode_done = apply_episode_termination_policy(
        {"car_0": False, "car_1": True, "car_2": False},
        {aid: False for aid in AGENTS},
        active_agents=AGENTS,
        possible_agents=AGENTS,
        trainable_agents=["car_0", "car_1"],
        mode="any_agent",
    )

    assert episode_done is True
    assert terminations == {aid: True for aid in AGENTS}


def test_all_agents_preserves_individual_termination() -> None:
    terminations, episode_done = apply_episode_termination_policy(
        {"car_0": False, "car_1": True, "car_2": False},
        {aid: False for aid in AGENTS},
        active_agents=AGENTS,
        possible_agents=AGENTS,
        trainable_agents=["car_0", "car_1"],
        mode="all_agents",
    )

    assert episode_done is False
    assert terminations == {"car_0": False, "car_1": True, "car_2": False}


def test_all_trainable_ends_when_last_active_trainable_finishes() -> None:
    terminations, episode_done = apply_episode_termination_policy(
        {"car_0": True, "car_1": False, "car_2": False},
        {aid: False for aid in AGENTS},
        active_agents=["car_0", "car_2"],
        possible_agents=AGENTS,
        trainable_agents=["car_0", "car_1"],
        mode="all_trainable",
    )

    assert episode_done is True
    assert terminations["car_2"] is False


def test_time_limit_always_ends_joint_episode() -> None:
    _, episode_done = apply_episode_termination_policy(
        {aid: False for aid in AGENTS},
        {"car_0": True, "car_1": True, "car_2": True},
        active_agents=AGENTS,
        possible_agents=AGENTS,
        trainable_agents=["car_0"],
        mode="all_agents",
    )

    assert episode_done is True


def test_invalid_episode_termination_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported episode_termination.mode"):
        normalize_episode_termination_mode("focal")
