# test_f110_simple.py

import numpy as np
import torch
from src.f110x.envs.f110ParallelEnv import F110Env

def run_test(env, mode: str, steps: int = 100, fps: int = 30):
    """
    mode: 'human', 'rgb_array', or 'none'
    steps: how many env.step() calls
    fps: for video saving if needed
    """
    # set render_mode
    env.render_mode = None if mode == "none" else mode

    obs, infos = env.reset()

    frames = []
    for t in range(steps):
        # sample random actions for all agents
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}

        obs, rewards, terminations, truncations, infos = env.step(actions)

        if mode == "human":
            env.render()
        elif mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if all(terminations.get(a, False) or truncations.get(a, False) for a in env.possible_agents):
            break

    env.close()

    if mode == "rgb_array" and len(frames) > 0:
        import imageio.v3 as iio
        iio.imwrite("test_rollout.mp4", np.stack(frames, 0), fps=fps)
        print(f"Saved video: test_rollout.mp4, {len(frames)} frames.")
    else:
        print("Finished rollout (no video).")


if __name__ == "__main__":
    # constants
    NUM_AGENTS = 4
    STEPS = 200

    # test no rendering
    env = F110Env(num_agents=NUM_AGENTS, max_steps=STEPS, render_mode="none")
    run_test(env, mode="none", steps=STEPS)

    # test rgb_array (headless)
    env = F110Env(num_agents=NUM_AGENTS, max_steps=STEPS, render_mode="rgb_array")
    run_test(env, mode="rgb_array", steps=STEPS, fps=30)

    # test human (opening a window)
    env = F110Env(num_agents=NUM_AGENTS, max_steps=STEPS, render_mode="human")
    run_test(env, mode="human", steps=STEPS)

    print("All modes done.")
