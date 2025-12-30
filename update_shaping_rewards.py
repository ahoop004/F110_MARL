#!/usr/bin/env python3
"""Update shaping rewards in all scenario files to prevent timeout farming."""

import yaml
from pathlib import Path

# New reduced values (80% reduction from previous)
REWARD_UPDATES = {
    'pressure': {
        'bonus': 0.004,
        'streak_bonus': 0.004,
    },
    'distance': {
        'reward_near': 0.004,
        'penalty_far': 0.002,
    },
    'forcing': {
        'pinch_pockets': {'weight': 0.004},
        'clearance': {'weight': 0.003},
        'turn': {'weight': 0.006},
    },
    'heading': {
        'coefficient': 0.001,
    },
    'speed': {
        'bonus_coef': 0.0005,
    },
    'step_reward': -0.001,
}

def update_scenario(filepath: Path):
    """Update a single scenario file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Find attacker agent (the one with custom rewards)
    updated = False
    for agent_id, agent_config in data.get('agents', {}).items():
        if 'reward' not in agent_config:
            continue

        reward_config = agent_config['reward']
        overrides = reward_config.get('overrides', {})

        # Update pressure
        if 'pressure' in overrides:
            if 'bonus' in REWARD_UPDATES['pressure']:
                overrides['pressure']['bonus'] = REWARD_UPDATES['pressure']['bonus']
                updated = True
            if 'streak_bonus' in REWARD_UPDATES['pressure']:
                overrides['pressure']['streak_bonus'] = REWARD_UPDATES['pressure']['streak_bonus']
                updated = True

        # Update distance
        if 'distance' in overrides:
            if 'reward_near' in REWARD_UPDATES['distance']:
                overrides['distance']['reward_near'] = REWARD_UPDATES['distance']['reward_near']
                updated = True
            if 'penalty_far' in REWARD_UPDATES['distance']:
                overrides['distance']['penalty_far'] = REWARD_UPDATES['distance']['penalty_far']
                updated = True

        # Update forcing
        if 'forcing' in overrides:
            for component, values in REWARD_UPDATES['forcing'].items():
                if component in overrides['forcing']:
                    for key, val in values.items():
                        overrides['forcing'][component][key] = val
                        updated = True

        # Update heading
        if 'heading' in overrides and 'coefficient' in REWARD_UPDATES['heading']:
            overrides['heading']['coefficient'] = REWARD_UPDATES['heading']['coefficient']
            updated = True

        # Update speed
        if 'speed' in overrides and 'bonus_coef' in REWARD_UPDATES['speed']:
            overrides['speed']['bonus_coef'] = REWARD_UPDATES['speed']['bonus_coef']
            updated = True

        # Update step_reward (top-level in overrides)
        if 'step_reward' in overrides:
            overrides['step_reward'] = REWARD_UPDATES['step_reward']
            updated = True

    if updated:
        # Write back
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)
        print(f"✓ Updated {filepath.name}")
        return True
    else:
        print(f"- Skipped {filepath.name} (no reward config found)")
        return False

def main():
    """Update all scenario files."""
    scenario_dir = Path(__file__).parent / 'scenarios' / 'v2'

    scenario_files = [
        'gaplock_sac_easier.yaml',
        'gaplock_td3.yaml',
        'gaplock_sac.yaml',
        'gaplock_ppo_easier.yaml',
        'gaplock_ppo.yaml',
        'gaplock_rainbow_easier.yaml',
        'gaplock_rainbow.yaml',
    ]

    print("Updating shaping rewards to prevent timeout farming...\n")

    updated_count = 0
    for filename in scenario_files:
        filepath = scenario_dir / filename
        if filepath.exists():
            if update_scenario(filepath):
                updated_count += 1
        else:
            print(f"✗ File not found: {filename}")

    print(f"\nUpdated {updated_count} scenario files.")
    print("\nNew reward structure:")
    print("  - Timeout (2500 steps): ~60 shaping - 2.5 step - 100 terminal = -42.5 (NEGATIVE!)")
    print("  - Success (500 steps): ~12 shaping - 0.5 step + 200 terminal = +211.5")
    print("  - Success is now 250 points better than timeout!")

if __name__ == '__main__':
    main()
