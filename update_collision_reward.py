#!/usr/bin/env python3
"""Update collision reward to 0.0 (neutral) in all scenarios."""

import yaml
from pathlib import Path

def update_scenario(filepath: Path):
    """Update collision reward in a scenario file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    updated = False
    for agent_id, agent_config in data.get('agents', {}).items():
        if 'reward' not in agent_config:
            continue

        reward_config = agent_config['reward']
        overrides = reward_config.get('overrides', {})

        if 'terminal' in overrides:
            terminal = overrides['terminal']
            if 'collision' in terminal:
                old_value = terminal['collision']
                terminal['collision'] = 0.0
                updated = True
                print(f"✓ {filepath.name}: collision {old_value} → 0.0")

    if updated:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)
        return True
    return False

def main():
    """Update all scenario files."""
    scenario_dir = Path('scenarios/v2')

    scenario_files = [
        'gaplock_td3_easier.yaml',
        'gaplock_sac_easier.yaml',
        'gaplock_ppo_easier.yaml',
        'gaplock_rainbow_easier.yaml',
        'gaplock_td3.yaml',
        'gaplock_sac.yaml',
        'gaplock_ppo.yaml',
        'gaplock_rainbow.yaml',
    ]

    print("Updating collision reward to 0.0 (neutral)...\n")

    for filename in scenario_files:
        filepath = scenario_dir / filename
        if filepath.exists():
            update_scenario(filepath)

    print("\n" + "="*60)
    print("Rationale:")
    print("="*60)
    print("Mutual collision (both crash) is now NEUTRAL (0.0)")
    print("- Not as bad as self-crash (-20)")
    print("- Not as good as success (+200)")
    print("- Encourages aggressive pursuit without fear of mutual crashes")
    print("="*60)

if __name__ == '__main__':
    main()
