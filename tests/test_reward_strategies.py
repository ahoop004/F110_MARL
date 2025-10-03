import math
from types import SimpleNamespace

import numpy as np

from f110x.wrappers.reward import RewardRuntimeContext, RewardWrapper


class DummyEnv:
    def __init__(self, timestep: float = 0.1) -> None:
        self.timestep = timestep
        self.current_time = 0.0


def make_context(centerline=None, timestep: float = 0.1):
    env = DummyEnv(timestep=timestep)
    map_data = SimpleNamespace(centerline=centerline)
    return env, RewardRuntimeContext(env=env, map_data=map_data)


def test_gaplock_strategy_success_reward():
    env, runtime = make_context()
    config = {"mode": "gaplock", "target_crash_reward": 5.0, "success_once": True}
    wrapper = RewardWrapper(config=config, context=runtime)
    wrapper.reset(episode_index=0)

    obs = {
        "ego": {"collision": False},
        "opponent": {"collision": True, "agent_id": "opponent"},
    }
    shaped = wrapper(
        obs,
        "ego",
        0.0,
        done=False,
        info={},
        all_obs=obs,
        step_index=0,
    )
    assert math.isclose(shaped, 5.0, rel_tol=1e-6)
    components = wrapper.get_last_components("ego")
    assert math.isclose(components.get("success_reward", 0.0), 5.0, rel_tol=1e-6)


def test_progress_strategy_rewards_forward_motion():
    centerline = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)
    env, runtime = make_context(centerline=centerline, timestep=0.1)
    config = {"mode": "progress", "progress": {"progress_weight": 1.0}}
    wrapper = RewardWrapper(config=config, context=runtime)
    wrapper.reset(episode_index=0)

    obs0 = {"car": {"pose": np.array([0.0, 0.0, 0.0], dtype=np.float32)}}
    _ = wrapper(obs0, "car", 0.0, done=False, info={}, all_obs=obs0, step_index=0)

    env.current_time += env.timestep
    obs1 = {"car": {"pose": np.array([0.9, 0.0, 0.0], dtype=np.float32)}}
    reward = wrapper(obs1, "car", 0.0, done=False, info={}, all_obs=obs1, step_index=1)
    assert reward > 0.0
    components = wrapper.get_last_components("car")
    assert "progress" in components and components["progress"] > 0.0


def test_fastest_lap_strategy_rewards_lap_completion():
    env, runtime = make_context(timestep=0.1)
    config = {
        "mode": "fastest_lap",
        "fastest_lap": {"lap_bonus": 2.0, "best_bonus": 1.0},
        "step_penalty": 0.0,
    }
    wrapper = RewardWrapper(config=config, context=runtime)
    wrapper.reset(episode_index=0)

    # Initial steps with zero laps
    lap_obs = {"car": {"lap": np.array([0.0, 0.0], dtype=np.float32)}}
    _ = wrapper(lap_obs, "car", 0.0, done=False, info={}, all_obs=lap_obs, step_index=0)

    # Complete a lap
    env.current_time += env.timestep
    lap_obs_1 = {"car": {"lap": np.array([1.0, env.current_time], dtype=np.float32)}}
    reward = wrapper(lap_obs_1, "car", 0.0, done=False, info={}, all_obs=lap_obs_1, step_index=1)
    assert reward > 0.0
    components = wrapper.get_last_components("car")
    assert components.get("lap_bonus", 0.0) >= 2.0

