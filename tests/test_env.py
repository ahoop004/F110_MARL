# tests/test_env.py

import yaml
import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from f110x.envs import F110ParallelEnv

@pytest.fixture(scope="module")
def config_and_poses():
    cfg_path = "tests/resources/test_env_config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get("env", {})
    start_poses = cfg.get("start_poses", None)
    if start_poses is not None:
        start_poses = np.array(start_poses, dtype=np.float32)
    return env_cfg, start_poses

@pytest.fixture
def make_env(config_and_poses):
    env_cfg, _ = config_and_poses
    # ensure correct key names
    return F110ParallelEnv(env=env_cfg, render_mode="human")

@pytest.fixture
def poses(config_and_poses):
    _, sp = config_and_poses
    return sp

def test_reset_with_start_poses(make_env, poses):
    env = make_env
    if poses is not None:
        obs, infos = env.reset(options=poses)
    else:
        obs, infos = env.reset()
    assert isinstance(obs, dict)
    assert len(obs) == len(env.agents)
    env.close()

def test_pettingzoo_api_with_poses(make_env, poses):
    env = make_env
    if poses is not None:
        env.reset(options=poses)
    else:
        env.reset()
    parallel_api_test(env, num_cycles=5)
    env.close()

def test_smoke_run(make_env, poses):
    env = make_env
    if poses is not None:
        obs, infos = env.reset(options=poses)
    else:
        obs, infos = env.reset()
    for step in range(10):
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

        if not env.agents:
            break

        if env.render_mode == "human":
            env.render()
    env.close()
