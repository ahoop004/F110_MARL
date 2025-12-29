#!/usr/bin/env python3
"""Profile F110_MARL performance to identify bottlenecks.

This script profiles key subsystems:
1. Environment step (physics, collision, observations)
2. Agent action selection
3. Reward computation
4. Training updates (if enabled)
"""

import cProfile
import pstats
import io
import time
import numpy as np
from pathlib import Path

# Setup paths
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env.f110ParallelEnv import F110ParallelEnv
from agents.ftg import FTGAgent
from rewards.builder import build_reward_strategy

def profile_environment_steps(num_steps=1000):
    """Profile environment stepping with multiple agents."""
    print(f"\n{'='*70}")
    print(f"PROFILING: Environment Steps ({num_steps} steps)")
    print(f"{'='*70}")

    # Create environment
    env = F110ParallelEnv(
        num_agents=4,
        map_name="Example",
        max_episode_steps=1000,
        render_mode=None,  # No rendering for profiling
    )

    # Create simple FTG agents
    agents = {
        agent_id: FTGAgent({"max_speed": 8.0, "min_speed": 2.0})
        for agent_id in env.possible_agents
    }

    # Reset environment
    obs, info = env.reset()

    # Profile stepping
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.perf_counter()
    for step in range(num_steps):
        actions = {
            agent_id: agents[agent_id].get_action(None, obs[agent_id])
            for agent_id in env.agents
        }
        obs, rewards, dones, truncated, info = env.step(actions)

        if dones.get("__all__", False):
            obs, info = env.reset()

    elapsed = time.perf_counter() - start_time
    profiler.disable()

    # Print results
    print(f"\n✓ Completed {num_steps} steps in {elapsed:.2f}s")
    print(f"  Average: {elapsed/num_steps*1000:.2f}ms per step")
    print(f"  Throughput: {num_steps/elapsed:.1f} steps/sec")

    # Show top time consumers
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions

    print("\nTop 30 functions by cumulative time:")
    print(s.getvalue())

    return profiler

def profile_ftg_agent(num_calls=10000):
    """Profile FTG agent action selection."""
    print(f"\n{'='*70}")
    print(f"PROFILING: FTG Agent ({num_calls} action selections)")
    print(f"{'='*70}")

    agent = FTGAgent({
        "max_speed": 8.0,
        "min_speed": 2.0,
        "bubble_radius": 2,
        "max_steer": 0.32,
    })

    # Create mock observation
    obs = {
        "scans": np.random.uniform(0.5, 10.0, size=1080).astype(np.float32),
        "velocity": np.array([5.0, 0.0], dtype=np.float32),
    }

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.perf_counter()
    for _ in range(num_calls):
        action = agent.get_action(None, obs)

    elapsed = time.perf_counter() - start_time
    profiler.disable()

    print(f"\n✓ Completed {num_calls} calls in {elapsed:.2f}s")
    print(f"  Average: {elapsed/num_calls*1000:.3f}ms per call")
    print(f"  Throughput: {num_calls/elapsed:.1f} calls/sec")

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())

    return profiler

def profile_reward_computation(num_calls=10000):
    """Profile reward computation."""
    print(f"\n{'='*70}")
    print(f"PROFILING: Reward Computation ({num_calls} calls)")
    print(f"{'='*70}")

    # Build reward strategy
    reward_config = {
        "preset": "gaplock_simple",
    }
    reward_strategy = build_reward_strategy(reward_config, agent_id="agent_0")

    # Create mock state
    state = {
        "pose": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "velocity": np.array([5.0, 0.0], dtype=np.float32),
        "progress": 0.5,
        "lap_time": 10.0,
    }

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.perf_counter()
    for _ in range(num_calls):
        reward = reward_strategy.compute(state, state)

    elapsed = time.perf_counter() - start_time
    profiler.disable()

    print(f"\n✓ Completed {num_calls} calls in {elapsed:.2f}s")
    print(f"  Average: {elapsed/num_calls*1000:.3f}ms per call")
    print(f"  Throughput: {num_calls/elapsed:.1f} calls/sec")

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())

    return profiler

def profile_observation_wrapper(num_calls=10000):
    """Profile observation wrapper."""
    print(f"\n{'='*70}")
    print(f"PROFILING: Observation Wrapper ({num_calls} calls)")
    print(f"{'='*70}")

    from wrappers.observation import ObsWrapper
    from core.observations import get_observation_config

    # Get config
    obs_config = get_observation_config(preset="gaplock")

    # Create wrapper (no centerline for simpler profiling)
    wrapper = ObsWrapper(
        max_scan=30.0,
        normalize=True,
        lidar_beams=1080,
        components=obs_config.get("components", []),
        centerline=None,
    )

    # Create mock observations
    obs_dict = {
        "agent_0": {
            "scans": np.random.uniform(0.5, 10.0, size=1080).astype(np.float32),
            "velocity": np.array([5.0, 0.0], dtype=np.float32),
            "pose_x": 0.0,
            "pose_y": 0.0,
            "pose_theta": 0.0,
        }
    }

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.perf_counter()
    for _ in range(num_calls):
        obs_vec = wrapper(obs_dict, "agent_0")

    elapsed = time.perf_counter() - start_time
    profiler.disable()

    print(f"\n✓ Completed {num_calls} calls in {elapsed:.2f}s")
    print(f"  Average: {elapsed/num_calls*1000:.3f}ms per call")
    print(f"  Throughput: {num_calls/elapsed:.1f} calls/sec")

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())

    return profiler

def main():
    """Run all profiling tests."""
    print(f"\n{'#'*70}")
    print("F110_MARL Performance Profiling")
    print(f"{'#'*70}")

    try:
        # Profile each subsystem (skip full environment for now - needs valid map)
        # profile_environment_steps(num_steps=500)
        profile_ftg_agent(num_calls=5000)
        profile_reward_computation(num_calls=10000)
        profile_observation_wrapper(num_calls=5000)

        print(f"\n{'#'*70}")
        print("Profiling Complete!")
        print(f"{'#'*70}\n")

        print("Summary:")
        print("  - Check cumulative times to identify bottlenecks")
        print("  - Focus on functions called frequently in hot paths")
        print("  - Consider Numba JIT for numerical loops")
        print("  - Pre-allocate arrays where possible")

    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
