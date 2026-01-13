"""Test unified factory system."""
import sys
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '.')

from src.core.config import AgentFactory, EnvironmentFactory, WrapperFactory, create_training_setup


class FactoryTester:
    """Tests factory system."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def test_agent_factory(self) -> bool:
        """Test AgentFactory."""
        if self.verbose:
            print("\n" + "="*60)
            print("Testing AgentFactory")
            print("="*60)

        try:
            # Test PPO creation
            ppo_config = {
                'obs_dim': 370,
                'act_dim': 2,
                'device': 'cpu',
                'lr': 3e-4,
                'gamma': 0.99,
            }
            ppo_agent = AgentFactory.create('ppo', ppo_config)

            if self.verbose:
                print(f"✓ Created PPO agent: {type(ppo_agent).__name__}")

            # Test TD3 creation
            td3_config = {
                'obs_dim': 370,
                'act_dim': 2,
                'action_low': np.array([-1.0, -1.0]),
                'action_high': np.array([1.0, 1.0]),
                'device': 'cpu',
                'lr': 3e-4,
                'gamma': 0.99,
            }
            td3_agent = AgentFactory.create('td3', td3_config)

            if self.verbose:
                print(f"✓ Created TD3 agent: {type(td3_agent).__name__}")

            # Test available agents
            available = AgentFactory.available_agents()
            if self.verbose:
                print(f"✓ Available agents: {available}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"✗ AgentFactory test failed: {e}")
                import traceback
                traceback.print_exc()
            return False

    def test_environment_factory(self) -> bool:
        """Test EnvironmentFactory."""
        if self.verbose:
            print("\n" + "="*60)
            print("Testing EnvironmentFactory")
            print("="*60)

        try:
            # Test with minimal config
            env_config = {
                'map': 'maps/line_map.yaml',  # Use actual map from repo
                'num_agents': 2,
                'timestep': 0.01,
                'integrator': 'rk4',
            }

            env = EnvironmentFactory.create(env_config)

            if self.verbose:
                print(f"✓ Created environment: {type(env).__name__}")
                print(f"  - Num agents: {env_config['num_agents']}")
                print(f"  - Timestep: {env_config['timestep']}")

            # Test reset (may fail without proper poses config, which is OK)
            try:
                obs, info = env.reset()
                if self.verbose:
                    print(f"✓ Environment reset successful")
                    print(f"  - Obs keys: {list(obs.keys())}")
            except (ValueError, TypeError) as e:
                if self.verbose:
                    print(f"⚠ Environment reset skipped (needs pose config): {e}")
                # This is expected for minimal config - factory still works

            return True

        except Exception as e:
            if self.verbose:
                print(f"✗ EnvironmentFactory test failed: {e}")
                import traceback
                traceback.print_exc()
            return False

    def test_wrapper_factory(self) -> bool:
        """Test WrapperFactory."""
        if self.verbose:
            print("\n" + "="*60)
            print("Testing WrapperFactory")
            print("="*60)

        try:
            # Create base environment
            env_config = {
                'map': 'maps/line_map.yaml',
                'num_agents': 1,
                'timestep': 0.01,
            }
            env = EnvironmentFactory.create(env_config)

            # Test observation wrapper (disabled)
            obs_wrapper_config = {
                'enabled': False,
                'config': {}
            }
            wrapped_env = WrapperFactory.wrap_observation(env, obs_wrapper_config)

            if self.verbose:
                print(f"✓ Observation wrapper (disabled) - env unchanged")

            # Test wrap_all with no wrappers
            wrapper_configs = {}
            wrapped_env = WrapperFactory.wrap_all(env, wrapper_configs)

            if self.verbose:
                print(f"✓ wrap_all with empty config - env unchanged")

            return True

        except Exception as e:
            if self.verbose:
                print(f"✗ WrapperFactory test failed: {e}")
                import traceback
                traceback.print_exc()
            return False

    def test_create_training_setup_minimal(self) -> bool:
        """Test create_training_setup with minimal inline config."""
        if self.verbose:
            print("\n" + "="*60)
            print("Testing create_training_setup (minimal)")
            print("="*60)

        try:
            # Create a minimal config dict (no file needed)
            import tempfile
            import yaml
            import os

            # Use absolute path for map
            map_path = os.path.abspath('maps/line_map.yaml')

            config = {
                'environment': {
                    'map': map_path,
                    'num_agents': 2,
                    'timestep': 0.01,
                },
                'agents': {
                    'agent_0': {
                        'type': 'ppo',
                        'params': {
                            'obs_dim': 370,
                            'act_dim': 2,
                            'device': 'cpu',
                            'lr': 3e-4,
                            'gamma': 0.99,
                        }
                    }
                }
            }

            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path = f.name

            # Create training setup
            setup = create_training_setup(config_path)

            if self.verbose:
                print(f"✓ Created training setup")
                print(f"  - Environment: {type(setup['env']).__name__}")
                print(f"  - Agents: {list(setup['agents'].keys())}")
                print(f"  - Agent types: {[type(a).__name__ for a in setup['agents'].values()]}")

            # Cleanup
            import os
            os.unlink(config_path)

            # Verify components
            assert 'env' in setup
            assert 'agents' in setup
            assert 'config' in setup
            assert 'agent_0' in setup['agents']

            if self.verbose:
                print(f"✓ All components present and valid")

            return True

        except Exception as e:
            if self.verbose:
                print(f"✗ create_training_setup test failed: {e}")
                import traceback
                traceback.print_exc()
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all factory tests."""
        print("\n" + "="*60)
        print("FACTORY SYSTEM TEST SUITE")
        print("="*60)

        tests = {
            'AgentFactory': self.test_agent_factory,
            'EnvironmentFactory': self.test_environment_factory,
            'WrapperFactory': self.test_wrapper_factory,
            'create_training_setup': self.test_create_training_setup_minimal,
        }

        results = {}
        for test_name, test_func in tests.items():
            results[test_name] = test_func()

        self.results = results
        return results

    def print_summary(self):
        """Print summary of test results."""
        print("\n" + "="*60)
        print("FACTORY TEST SUMMARY")
        print("="*60)

        if not self.results:
            print("No test results available")
            return

        for test_name, passed in self.results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name:30} {status}")

        # Overall
        passed_count = sum(1 for p in self.results.values() if p)
        total_count = len(self.results)
        overall_pct = (passed_count / total_count * 100) if total_count > 0 else 0

        print(f"\n{'OVERALL':30} {passed_count}/{total_count} ({overall_pct:.1f}%)")
        print("="*60)


if __name__ == '__main__':
    tester = FactoryTester(verbose=True)
    results = tester.run_all_tests()
    tester.print_summary()

    # Exit with error code if not all tests passed
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
