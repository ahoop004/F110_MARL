#!/usr/bin/env python3
"""Debug script to verify WandB sweep parameters are applied correctly to SB3 model."""

import sys
import yaml
import argparse

# Mock WandB config to simulate sweep parameters
class MockWandBConfig:
    def __init__(self, params):
        self._params = params

    def __getitem__(self, key):
        return self._params.get(key)

    def __iter__(self):
        return iter(self._params)

    def __bool__(self):
        return len(self._params) > 0

def set_nested_value(d: dict, path: str, value):
    """Set a nested dictionary value using dot notation."""
    keys = path.split('.')
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value

def main():
    parser = argparse.ArgumentParser(description='Debug sweep parameter application')
    parser.add_argument('--scenario', default='scenarios/sac.yaml', help='Scenario file')
    parser.add_argument('--algo', default='sac', help='Algorithm')
    args = parser.parse_args()

    # Load scenario
    with open(args.scenario) as f:
        scenario = yaml.safe_load(f)

    # Print original params
    print("="*60)
    print("ORIGINAL SCENARIO PARAMS")
    print("="*60)
    sb3_agent_id = 'car_0'
    original_params = scenario['agents'][sb3_agent_id].get('params', {})
    print(f"Agent ID: {sb3_agent_id}")
    for key, value in original_params.items():
        print(f"  {key}: {value}")

    # Simulate sweep parameters (from sac_sweep.yaml)
    sweep_params = {
        'learning_rate': 0.0001,  # Different from scenario (0.0003)
        'gamma': 0.99,             # Different from scenario (0.995)
        'tau': 0.01,               # Different from scenario (0.005)
        'batch_size': 128,         # Different from scenario (256)
        'hidden_dims': [512, 512], # Different from scenario ([256, 256])
    }

    # Simulate WandB config
    import wandb
    wandb.config = MockWandBConfig(sweep_params)

    # Apply sweep params (same logic as run_sb3_baseline.py:425-443)
    print("\n" + "="*60)
    print("APPLYING SWEEP PARAMETERS")
    print("="*60)
    sweep_params_applied = {}

    if wandb.config:
        wandb_params = {k: wandb.config[k] for k in wandb.config}
        for key, value in wandb_params.items():
            if key.startswith('_') or key in [
                'method', 'metric', 'program', 'algorithm', 'scenario'
            ]:
                continue
            if key == 'episodes':
                scenario.setdefault('experiment', {})['episodes'] = value
                sweep_params_applied[key] = value
                continue
            if key == 'seed':
                scenario.setdefault('experiment', {})['seed'] = value
                sweep_params_applied[key] = value
                continue

            override_path = key if '.' in key else f"agents.{sb3_agent_id}.params.{key}"
            set_nested_value(scenario, override_path, value)
            sweep_params_applied[key] = value

        if sweep_params_applied:
            print(f"Applied {len(sweep_params_applied)} sweep parameter(s):")
            for key, value in sweep_params_applied.items():
                print(f"  {key} = {value}")

    # Print modified params
    print("\n" + "="*60)
    print("MODIFIED SCENARIO PARAMS (after sweep)")
    print("="*60)
    modified_params = scenario['agents'][sb3_agent_id].get('params', {})
    print(f"Agent ID: {sb3_agent_id}")
    for key, value in modified_params.items():
        print(f"  {key}: {value}")

    # Check what would be passed to create_sb3_agent
    print("\n" + "="*60)
    print("PARAMS PASSED TO create_sb3_agent()")
    print("="*60)
    agent_params = scenario['agents'][sb3_agent_id].get('params', {})
    for key, value in agent_params.items():
        print(f"  {key}: {value}")

    # Verify each parameter is correct
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    all_correct = True
    for key, expected_value in sweep_params.items():
        actual_value = agent_params.get(key)
        match = actual_value == expected_value
        status = "✓" if match else "✗"
        print(f"{status} {key}: expected={expected_value}, actual={actual_value}")
        if not match:
            all_correct = False

    if all_correct:
        print("\n✓ All sweep parameters applied correctly!")
        return 0
    else:
        print("\n✗ Some sweep parameters NOT applied correctly!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
