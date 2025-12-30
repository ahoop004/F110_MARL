#!/usr/bin/env python3
"""Update hyperparameters in all 'easier' scenarios for better convergence."""

import yaml
from pathlib import Path
from typing import Dict, Any

def update_sac_easier():
    """Update SAC easier scenario."""
    filepath = Path('scenarios/v2/gaplock_sac_easier.yaml')
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    params = data['agents']['car_0']['params']

    # Update hyperparameters
    params['gamma'] = 0.97  # Lower discount
    params['hidden_dims'] = [512, 512]  # More capacity
    params['success_buffer_ratio'] = 0.3  # Learn from successes
    params['learning_starts'] = 20000  # More initial data

    # Update curriculum
    curriculum = data['environment']['spawn_curriculum']
    curriculum['activation_samples'] = 100
    curriculum['min_episode'] = 200
    curriculum['enable_patience'] = 10
    curriculum['disable_patience'] = 4
    curriculum['cooldown'] = 40

    curriculum['stages'][0]['enable_rate'] = 0.70
    curriculum['stages'][1]['enable_rate'] = 0.75
    curriculum['stages'][1]['disable_rate'] = 0.55
    curriculum['stages'][2]['enable_rate'] = 0.80
    curriculum['stages'][2]['disable_rate'] = 0.60
    curriculum['stages'][3]['disable_rate'] = 0.65

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"✓ Updated {filepath.name}")
    print(f"  - gamma: 0.99 → 0.97")
    print(f"  - hidden_dims: [256, 256] → [512, 512]")
    print(f"  - success_buffer_ratio: 0.1 → 0.3")
    print(f"  - learning_starts: 10000 → 20000")
    print(f"  - Curriculum made more conservative")

def update_ppo_easier():
    """Update PPO easier scenario."""
    filepath = Path('scenarios/v2/gaplock_ppo_easier.yaml')
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    params = data['agents']['car_0']['params']

    # Update hyperparameters (PPO-specific)
    params['gamma'] = 0.97  # Lower discount
    params['hidden_dims'] = [512, 512]  # More capacity
    # PPO doesn't have learning_starts or success_buffer

    # Update curriculum
    curriculum = data['environment']['spawn_curriculum']
    curriculum['activation_samples'] = 100
    curriculum['min_episode'] = 200
    curriculum['enable_patience'] = 10
    curriculum['disable_patience'] = 4
    curriculum['cooldown'] = 40

    curriculum['stages'][0]['enable_rate'] = 0.70
    curriculum['stages'][1]['enable_rate'] = 0.75
    curriculum['stages'][1]['disable_rate'] = 0.55
    curriculum['stages'][2]['enable_rate'] = 0.80
    curriculum['stages'][2]['disable_rate'] = 0.60
    curriculum['stages'][3]['disable_rate'] = 0.65

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"\n✓ Updated {filepath.name}")
    print(f"  - gamma: 0.99 → 0.97")
    print(f"  - hidden_dims: [256, 256] → [512, 512]")
    print(f"  - Curriculum made more conservative")

def update_rainbow_easier():
    """Update Rainbow DQN easier scenario."""
    filepath = Path('scenarios/v2/gaplock_rainbow_easier.yaml')
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    params = data['agents']['car_0']['params']

    # Update hyperparameters (Rainbow-specific)
    params['gamma'] = 0.97  # Lower discount
    params['hidden_dims'] = [512, 512]  # More capacity
    if 'learning_starts' in params:
        params['learning_starts'] = 20000

    # Update curriculum
    curriculum = data['environment']['spawn_curriculum']
    curriculum['activation_samples'] = 100
    curriculum['min_episode'] = 200
    curriculum['enable_patience'] = 10
    curriculum['disable_patience'] = 4
    curriculum['cooldown'] = 40

    curriculum['stages'][0]['enable_rate'] = 0.70
    curriculum['stages'][1]['enable_rate'] = 0.75
    curriculum['stages'][1]['disable_rate'] = 0.55
    curriculum['stages'][2]['enable_rate'] = 0.80
    curriculum['stages'][2]['disable_rate'] = 0.60
    curriculum['stages'][3]['disable_rate'] = 0.65

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"\n✓ Updated {filepath.name}")
    print(f"  - gamma: 0.99 → 0.97")
    print(f"  - hidden_dims: [256, 256] → [512, 512]")
    if 'learning_starts' in params:
        print(f"  - learning_starts: 10000 → 20000")
    print(f"  - Curriculum made more conservative")

if __name__ == '__main__':
    print("Updating hyperparameters in 'easier' scenarios...\n")

    try:
        update_sac_easier()
    except Exception as e:
        print(f"✗ Failed to update SAC: {e}")

    try:
        update_ppo_easier()
    except Exception as e:
        print(f"✗ Failed to update PPO: {e}")

    try:
        update_rainbow_easier()
    except Exception as e:
        print(f"✗ Failed to update Rainbow: {e}")

    print("\n" + "="*60)
    print("Summary of Changes:")
    print("="*60)
    print("1. Discount factor: γ = 0.99 → 0.97 (faster credit assignment)")
    print("2. Network capacity: [256,256] → [512,512] (more capacity)")
    print("3. Success buffer ratio: 0.1 → 0.3 (learn from successes)")
    print("4. Learning starts: 10k → 20k (more initial data)")
    print("5. Curriculum: More conservative stage progression")
    print("="*60)
