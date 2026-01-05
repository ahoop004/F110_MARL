#!/usr/bin/env python3
"""Quick diagnostic to check if SB3 rewards are being computed correctly."""

import sys
sys.path.insert(0, '/home/aaron/F110_MARL/src')

from src.core.setup import create_training_setup
from src.core.config import load_and_expand_scenario, resolve_observation_preset
from src.baselines.sb3_wrapper import SB3SingleAgentWrapper
import numpy as np

# Load scenario
scenario = load_and_expand_scenario('scenarios/sac.yaml')
env, agents, reward_strategies = create_training_setup(scenario)

# Get SB3 agent config
sb3_agent_id = 'car_0'
sb3_agent_cfg = scenario['agents'][sb3_agent_id]
observation_preset = 'gaplock'  # resolve_observation_preset(sb3_agent_cfg)
target_id = sb3_agent_cfg.get('target_id', 'car_1')
reward_strategy = reward_strategies.get(sb3_agent_id)

print(f"\n{'='*60}")
print(f"SB3 REWARD DIAGNOSTIC")
print(f"{'='*60}")
print(f"SB3 Agent: {sb3_agent_id}")
print(f"Reward Strategy: {type(reward_strategy).__name__ if reward_strategy else 'None'}")
print(f"Observation Preset: {observation_preset}")
print(f"Target ID: {target_id}")

# Create wrapper
wrapped_env = SB3SingleAgentWrapper(
    env,
    agent_id=sb3_agent_id,
    obs_dim=126,  # gaplock preset
    observation_preset=observation_preset,
    target_id=target_id,
    reward_strategy=reward_strategy,
)

# Set FTG defender
other_agents = {aid: agent for aid, agent in agents.items() if aid != sb3_agent_id}
wrapped_env.set_other_agents(other_agents)

print(f"\nOther agents: {list(other_agents.keys())}")

# Run a few steps
print(f"\n{'='*60}")
print("Running 10 episodes to check rewards...")
print(f"{'='*60}\n")

for ep in range(10):
    obs, info = wrapped_env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < 100:
        # Random action
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    outcome = info.get('outcome', 'unknown')
    success = info.get('is_success', False)

    print(f"Ep {ep+1:2d}: reward={total_reward:7.2f}, steps={steps:4d}, outcome={outcome:15s}, success={success}")

print(f"\n{'='*60}")
print("If all rewards are 0.0, there's a bug in reward computation!")
print("If rewards are non-zero but success=False, the task is hard/policy is random")
print(f"{'='*60}\n")
