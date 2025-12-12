import numpy as np
import pytest

from f110x.tasks.reward.base import RewardStep
from f110x.tasks.reward.kamikaze import KamikazeRewardStrategy


def _make_step(
    ego_collision: bool,
    target_collision: bool,
    *,
    ego_pose=(0.0, 0.0, 0.0),
    target_pose=(0.0, 0.0, 0.0),
    timestep: float = 0.01,
    ego_speed: float = 1.0,
) -> RewardStep:
    all_obs = {
        "car_0": {
            "collision": ego_collision,
            "pose": np.asarray(ego_pose, dtype=np.float32),
            "speed": ego_speed,
        },
        "car_1": {"collision": target_collision, "pose": np.asarray(target_pose, dtype=np.float32)},
    }
    return RewardStep(
        agent_id="car_0",
        obs=all_obs["car_0"],
        env_reward=0.0,
        done=False,
        info={},
        all_obs=all_obs,
        episode_index=0,
        step_index=0,
        current_time=0.0,
        timestep=timestep,
        events={},
    )


def test_kamikaze_success_even_if_ego_crashes():
    strategy = KamikazeRewardStrategy(target_agent="car_1", success_reward=10.0)

    total, components = strategy.compute(_make_step(ego_collision=True, target_collision=True))
    assert components["kamikaze_success"] == 10.0
    assert total == 10.0


def test_kamikaze_self_collision_penalty_without_target_crash():
    strategy = KamikazeRewardStrategy(
        target_agent="car_1",
        success_reward=10.0,
        self_collision_penalty=-2.0,
    )

    total, components = strategy.compute(_make_step(ego_collision=True, target_collision=False))
    assert components["self_collision_penalty"] == -2.0
    assert total == -2.0


def test_kamikaze_distance_reward_scaling():
    strategy = KamikazeRewardStrategy(
        target_agent="car_1",
        distance_reward_near=1.0,
        distance_reward_near_distance=2.0,
        distance_reward_far_distance=4.0,
    )
    step = _make_step(
        ego_collision=False,
        target_collision=False,
        ego_pose=(0.0, 0.0, 0.0),
        target_pose=(1.0, 0.0, 0.0),
        timestep=0.02,
    )
    total, components = strategy.compute(step)
    assert components["distance_reward"] == pytest.approx(0.02)
    assert total == pytest.approx(0.02)


def test_kamikaze_distance_penalty_far():
    strategy = KamikazeRewardStrategy(
        target_agent="car_1",
        distance_penalty_far=0.5,
        distance_reward_far_distance=5.0,
    )
    step = _make_step(
        ego_collision=False,
        target_collision=False,
        ego_pose=(0.0, 0.0, 0.0),
        target_pose=(6.0, 0.0, 0.0),
        timestep=0.01,
    )
    total, components = strategy.compute(step)
    assert components["distance_penalty"] == pytest.approx(-0.005)
    assert total == pytest.approx(-0.005)


def test_kamikaze_radius_bonus_once():
    strategy = KamikazeRewardStrategy(
        target_agent="car_1",
        radius_bonus_reward=7.0,
        radius_bonus_distance=0.5,
        radius_bonus_once=True,
    )
    step = _make_step(
        ego_collision=False,
        target_collision=False,
        ego_pose=(0.0, 0.0, 0.0),
        target_pose=(0.2, 0.0, 0.0),
    )
    total, components = strategy.compute(step)
    assert components["radius_bonus"] == 7.0
    assert total == 7.0

    total_second, components_second = strategy.compute(step)
    assert "radius_bonus" not in components_second
    assert total_second == 0.0


def test_kamikaze_distance_delta_reward():
    strategy = KamikazeRewardStrategy(
        target_agent="car_1",
        distance_delta_reward=2.0,
    )
    first = _make_step(
        ego_collision=False,
        target_collision=False,
        ego_pose=(0.0, 0.0, 0.0),
        target_pose=(5.0, 0.0, 0.0),
    )
    strategy.compute(first)

    second = _make_step(
        ego_collision=False,
        target_collision=False,
        ego_pose=(0.0, 0.0, 0.0),
        target_pose=(3.5, 0.0, 0.0),
    )
    total, components = strategy.compute(second)
    assert components["distance_delta_reward"] == pytest.approx(3.0)
    assert total == pytest.approx(3.0)


def test_kamikaze_idle_penalty():
    strategy = KamikazeRewardStrategy(
        target_agent="car_1",
        idle_penalty=-0.5,
        idle_penalty_steps=2,
        idle_speed_threshold=0.2,
    )
    strategy.compute(
        _make_step(
            ego_collision=False,
            target_collision=False,
            ego_speed=0.1,
        )
    )
    total, components = strategy.compute(
        _make_step(
            ego_collision=False,
            target_collision=False,
            ego_speed=0.05,
        )
    )
    assert components["idle_penalty"] == pytest.approx(-0.5)
    assert total == pytest.approx(-0.5)
