import numpy as np
from pettingzoo.test import parallel_api_test
from f110x.envs.f110ParallelEnv import F110ParallelEnv


def make_env():
    # minimal config: 2 agents, default map
    return F110ParallelEnv(
        env={
            "seed": 42,
            "n_agents": 2,
            "map": "/home/aaron/F110_MARL/tests/levine.png",
            "map_dir": "/home/aaron/F110_MARL/tests/",
            "max_steps": 50,
        },
        render_mode="human",  # or "rgb_array"
    )


def test_api():
    env = make_env()
    parallel_api_test(env, num_cycles=5)  # built-in PettingZoo compliance test
    print("✅ parallel_api_test passed")


def smoke_run():
    env = make_env()
    obs, infos = env.reset()
    for step in range(10):
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        print(f"Step {step}: rewards={rewards}")
        if env.render_mode == "human":
            env.render()
        if not env.agents:
            break
    env.close()


if __name__ == "__main__":
    test_api()
    smoke_run()
