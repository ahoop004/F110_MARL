#!/usr/bin/env python3
"""Debug script to check terminal reward computation during training.

Add this to your training loop to see what's happening when episodes end.
"""

def debug_terminal_reward(step_info, reward_components, total_reward, episode_steps, max_steps):
    """Add this to your training loop when an episode ends (done=True).

    Example usage in enhanced_training.py:

        if done[agent_id]:
            from debug_terminal_rewards import debug_terminal_reward
            debug_terminal_reward(
                step_info=reward_info,
                reward_components=components,
                total_reward=total_reward,
                episode_steps=episode_steps,
                max_steps=self.env.max_steps
            )
    """
    print("\n" + "="*80)
    print("EPISODE END DEBUG")
    print("="*80)
    print(f"Episode steps: {episode_steps}/{max_steps}")
    print(f"Total reward: {total_reward:.2f}")
    print()

    print("step_info keys:")
    print(f"  done: {step_info.get('done', 'NOT SET')}")
    print(f"  truncated: {step_info.get('truncated', 'NOT SET')}")
    print()

    print("info dict contents:")
    info = step_info.get('info', {})
    for key, value in sorted(info.items()):
        print(f"  {key}: {value}")
    print()

    print("Reward components:")
    for key, value in sorted(reward_components.items()):
        if 'terminal' in key:
            print(f"  {key}: {value} <<<< TERMINAL REWARD")
        else:
            print(f"  {key}: {value}")
    print()

    # Check for terminal rewards
    has_terminal = any('terminal' in k for k in reward_components.keys())
    if not has_terminal:
        print("⚠️  WARNING: NO TERMINAL REWARD COMPONENT FOUND!")
        print("Checking why...")
        print(f"  done flag: {step_info.get('done')}")
        print(f"  truncated flag: {step_info.get('truncated')}")
        print(f"  info.truncated: {info.get('truncated')}")
        print(f"  Expected truncation: {episode_steps >= max_steps}")
    else:
        terminal_keys = [k for k in reward_components.keys() if 'terminal' in k]
        print(f"✓ Terminal reward found: {terminal_keys}")

    print("="*80)
    print()


if __name__ == "__main__":
    # Test with sample data
    print("Testing debug_terminal_reward function...")

    # Simulate timeout
    step_info = {
        'done': True,
        'truncated': True,
        'info': {
            'collision': False,
            'target_collision': False,
        },
        'obs': {'velocity': [0.5, 0.0]},
        'target_obs': {},
        'timestep': 0.01,
    }

    reward_components = {
        'distance/near': 0.12,
        'speed/bonus': 0.025,
        'terminal/timeout': -90.0,
    }

    debug_terminal_reward(
        step_info=step_info,
        reward_components=reward_components,
        total_reward=-89.85,
        episode_steps=5000,
        max_steps=5000
    )
