#!/usr/bin/env python3
"""Verify HER and optimizations are correctly configured."""

import yaml
from pathlib import Path

def check_scenario(filepath: Path):
    """Check a scenario file for all optimizations."""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath.name}")
    print('='*60)

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Check agent config
    for agent_id, agent_config in data.get('agents', {}).items():
        if agent_config.get('role') != 'attacker':
            continue

        algo = agent_config.get('algorithm', 'unknown')
        params = agent_config.get('params', {})
        reward = agent_config.get('reward', {})

        print(f"\n{agent_id} ({algo}):")
        print("-" * 40)

        # Check hyperparameters
        gamma = params.get('gamma')
        hidden_dims = params.get('hidden_dims', [])

        print(f"✓ gamma: {gamma} {'(optimized!)' if gamma and gamma <= 0.97 else '(not optimized)'}")
        print(f"✓ hidden_dims: {hidden_dims} {'(optimized!)' if len(hidden_dims) >= 2 and hidden_dims[0] >= 512 else '(not optimized)'}")

        # Algo-specific checks
        if algo == 'td3':
            exploration_noise = params.get('exploration_noise', 0)
            target_noise = params.get('target_noise', 0)
            success_ratio = params.get('success_buffer_ratio', 0)
            learning_starts = params.get('learning_starts', 0)

            print(f"✓ exploration_noise: {exploration_noise} {'(optimized!)' if exploration_noise >= 0.2 else ''}")
            print(f"✓ target_noise: {target_noise} {'(optimized!)' if target_noise <= 0.1 else ''}")
            print(f"✓ success_buffer_ratio: {success_ratio} {'(optimized!)' if success_ratio >= 0.3 else ''}")
            print(f"✓ learning_starts: {learning_starts} {'(optimized!)' if learning_starts >= 20000 else ''}")

        elif algo == 'sac':
            warmup_steps = params.get('warmup_steps', 0)
            success_ratio = params.get('success_buffer_ratio', 0)

            print(f"✓ warmup_steps: {warmup_steps} {'(optimized!)' if warmup_steps >= 20000 else ''}")
            print(f"✓ success_buffer_ratio: {success_ratio} {'(optimized!)' if success_ratio >= 0.3 else ''}")

        elif algo == 'ppo':
            lam = params.get('lam', 0)
            ent_coef = params.get('ent_coef', 0)
            batch_size = params.get('batch_size', 0)
            update_epochs = params.get('update_epochs', 0)

            print(f"✓ lam (GAE): {lam} {'(optimized!)' if lam <= 0.90 else ''}")
            print(f"✓ ent_coef: {ent_coef} {'(optimized!)' if ent_coef >= 0.05 else ''}")
            print(f"✓ batch_size: {batch_size} {'(optimized!)' if batch_size >= 4096 else ''}")
            print(f"✓ update_epochs: {update_epochs} {'(optimized!)' if update_epochs >= 20 else ''}")

        # Check reward config
        overrides = reward.get('overrides', {})

        # Check penalties disabled
        penalties = overrides.get('penalties', {})
        penalties_enabled = penalties.get('enabled', True)
        print(f"\n{'✓' if not penalties_enabled else '✗'} Penalties disabled: {not penalties_enabled}")

        # Check terminal rewards
        terminal = overrides.get('terminal', {})
        timeout = terminal.get('timeout', 0)
        target_crash = terminal.get('target_crash', 0)
        print(f"{'✓' if timeout <= -100 else '✗'} Timeout penalty: {timeout} (should be -100)")
        print(f"{'✓' if target_crash >= 200 else '✗'} Success reward: {target_crash} (should be 200)")

        # Check shaping rewards
        pressure = overrides.get('pressure', {})
        pressure_bonus = pressure.get('bonus', 0)
        print(f"{'✓' if pressure_bonus <= 0.004 else '✗'} Pressure bonus: {pressure_bonus} (should be ≤0.004)")

        distance = overrides.get('distance', {})
        reward_near = distance.get('reward_near', 0)
        print(f"{'✓' if reward_near <= 0.004 else '✗'} Distance reward: {reward_near} (should be ≤0.004)")

def check_her_implementation():
    """Check if HER is implemented in agents."""
    print(f"\n{'='*60}")
    print("Checking HER Implementation")
    print('='*60)

    # Check TD3
    td3_path = Path('src/agents/td3/td3.py')
    if td3_path.exists():
        with open(td3_path, 'r') as f:
            td3_code = f.read()

        has_her_method = 'def store_hindsight_transition' in td3_code
        has_distance_param = 'distance_to_target: float' in td3_code
        has_her_bonus = 'her_bonus' in td3_code

        print(f"\nTD3 Agent:")
        print(f"{'✓' if has_her_method else '✗'} HER method exists")
        print(f"{'✓' if has_distance_param else '✗'} Distance parameter exists")
        print(f"{'✓' if has_her_bonus else '✗'} HER bonus logic exists")

    # Check SAC
    sac_path = Path('src/agents/sac/sac.py')
    if sac_path.exists():
        with open(sac_path, 'r') as f:
            sac_code = f.read()

        has_her_method = 'def store_hindsight_transition' in sac_code
        has_distance_param = 'distance_to_target: float' in sac_code
        has_her_bonus = 'her_bonus' in sac_code

        print(f"\nSAC Agent:")
        print(f"{'✓' if has_her_method else '✗'} HER method exists")
        print(f"{'✓' if has_distance_param else '✗'} Distance parameter exists")
        print(f"{'✓' if has_her_bonus else '✗'} HER bonus logic exists")

    # Check training loop
    training_path = Path('src/core/enhanced_training.py')
    if training_path.exists():
        with open(training_path, 'r') as f:
            training_code = f.read()

        has_her_call = 'store_hindsight_transition' in training_code
        has_distance_calc = '_calculate_distance_to_target' in training_code

        print(f"\nTraining Loop:")
        print(f"{'✓' if has_her_call else '✗'} HER integration exists")
        print(f"{'✓' if has_distance_calc else '✗'} Distance calculation exists")

def main():
    """Run all checks."""
    print("="*60)
    print("HER + OPTIMIZATIONS VERIFICATION")
    print("="*60)

    # Check HER implementation
    check_her_implementation()

    # Check easier scenarios
    easier_scenarios = [
        'scenarios/v2/gaplock_td3_easier.yaml',
        'scenarios/v2/gaplock_sac_easier.yaml',
        'scenarios/v2/gaplock_ppo_easier.yaml',
    ]

    for scenario_path in easier_scenarios:
        filepath = Path(scenario_path)
        if filepath.exists():
            check_scenario(filepath)
        else:
            print(f"\n✗ File not found: {scenario_path}")

    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print('='*60)
    print("\nNext steps:")
    print("1. Run: python3 run_v2.py scenarios/v2/gaplock_td3_easier.yaml")
    print("2. Watch for:")
    print("   - No idle penalties in reward components")
    print("   - Agent velocity > 0.1 m/s")
    print("   - Success buffer filling up")
    print("3. Monitor W&B for success rate climbing")

if __name__ == '__main__':
    main()
