"""Test script to verify agents conform to the Agent protocol.

This script checks that all RL agents implement the required protocol methods
without needing wrapper classes.
"""
import sys
import numpy as np
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, '.')

from v2.core.protocol import Agent, OnPolicyAgent, OffPolicyAgent


class ProtocolTester:
    """Tests agent protocol compliance."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def test_agent(self, agent_name: str, agent_class: type, config: Dict[str, Any]) -> Dict[str, bool]:
        """Test a single agent for protocol compliance.

        Args:
            agent_name: Name of agent (e.g., "PPO")
            agent_class: Agent class to test
            config: Configuration dict for agent

        Returns:
            results: Dictionary of test results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Testing {agent_name} Agent Protocol Compliance")
            print(f"{'='*60}")

        results = {}

        try:
            # 1. Test instantiation
            agent = agent_class(config)
            results['instantiation'] = True
            if self.verbose:
                print(f"✓ Instantiation successful")

            # 2. Test protocol conformance
            is_agent = isinstance(agent, Agent)
            results['agent_protocol'] = is_agent
            if self.verbose:
                print(f"{'✓' if is_agent else '✗'} Implements Agent protocol: {is_agent}")

            # Check on-policy vs off-policy
            is_on_policy = isinstance(agent, OnPolicyAgent)
            is_off_policy = isinstance(agent, OffPolicyAgent)
            results['on_policy'] = is_on_policy
            results['off_policy'] = is_off_policy
            if self.verbose:
                print(f"  - On-policy: {is_on_policy}")
                print(f"  - Off-policy: {is_off_policy}")

            # 3. Test act() method
            obs = np.random.randn(config['obs_dim']).astype(np.float32)
            try:
                action = agent.act(obs, deterministic=False)
                results['act_stochastic'] = True
                if self.verbose:
                    print(f"✓ act(obs, deterministic=False) works")
                    print(f"  - Action shape: {np.array(action).shape}")
            except Exception as e:
                results['act_stochastic'] = False
                if self.verbose:
                    print(f"✗ act(obs, deterministic=False) failed: {e}")

            try:
                action = agent.act(obs, deterministic=True)
                results['act_deterministic'] = True
                if self.verbose:
                    print(f"✓ act(obs, deterministic=True) works")
            except Exception as e:
                results['act_deterministic'] = False
                if self.verbose:
                    print(f"✗ act(obs, deterministic=True) failed: {e}")

            # 4. Test store methods
            if is_on_policy:
                # Test store() for on-policy
                try:
                    agent.store(obs, action, 1.0, False, terminated=False)
                    results['store'] = True
                    if self.verbose:
                        print(f"✓ store(obs, action, reward, done, terminated) works")
                except Exception as e:
                    results['store'] = False
                    if self.verbose:
                        print(f"✗ store() failed: {e}")

                # Test finish_path() for on-policy
                try:
                    agent.finish_path()
                    results['finish_path'] = True
                    if self.verbose:
                        print(f"✓ finish_path() works")
                except Exception as e:
                    results['finish_path'] = False
                    if self.verbose:
                        print(f"✗ finish_path() failed: {e}")

            if is_off_policy:
                # Test store_transition() for off-policy
                try:
                    next_obs = np.random.randn(config['obs_dim']).astype(np.float32)
                    agent.store_transition(obs, action, 1.0, next_obs, False)
                    results['store_transition'] = True
                    if self.verbose:
                        print(f"✓ store_transition(obs, action, reward, next_obs, done) works")
                except Exception as e:
                    results['store_transition'] = False
                    if self.verbose:
                        print(f"✗ store_transition() failed: {e}")

            # 5. Test update() method
            try:
                stats = agent.update()
                results['update'] = True
                if self.verbose:
                    print(f"✓ update() works")
                    if stats:
                        print(f"  - Returned stats: {list(stats.keys())}")
                    else:
                        print(f"  - Returned None (expected if not enough data)")
            except Exception as e:
                results['update'] = False
                if self.verbose:
                    print(f"✗ update() failed: {e}")

            # 6. Test save/load methods
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                checkpoint_path = f.name

            try:
                agent.save(checkpoint_path)
                results['save'] = True
                if self.verbose:
                    print(f"✓ save(path) works")
            except Exception as e:
                results['save'] = False
                if self.verbose:
                    print(f"✗ save() failed: {e}")

            try:
                agent.load(checkpoint_path)
                results['load'] = True
                if self.verbose:
                    print(f"✓ load(path) works")
            except Exception as e:
                results['load'] = False
                if self.verbose:
                    print(f"✗ load() failed: {e}")

            # Clean up
            import os
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        except Exception as e:
            results['instantiation'] = False
            if self.verbose:
                print(f"✗ Failed to instantiate agent: {e}")
                import traceback
                traceback.print_exc()

        # Calculate compliance score
        passed = sum(1 for v in results.values() if v is True)
        total = len(results)
        compliance_pct = (passed / total * 100) if total > 0 else 0

        if self.verbose:
            print(f"\n{agent_name} Compliance Score: {passed}/{total} ({compliance_pct:.1f}%)")

        return results

    def test_all_agents(self) -> Dict[str, Dict[str, bool]]:
        """Test all registered agents."""
        from v2.core.config import AgentFactory

        print("\n" + "="*60)
        print("AGENT PROTOCOL COMPLIANCE TEST SUITE")
        print("="*60)
        print(f"\nTesting {len(AgentFactory.available_agents())} agents")
        print(f"Available: {AgentFactory.available_agents()}")

        # Test configurations for each agent type
        test_configs = {
            'ppo': {
                'obs_dim': 370,
                'act_dim': 2,
                'device': 'cpu',
                'lr': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'n_epochs': 10,
                'batch_size': 64,
                'max_buffer_size': 2048,
                'squash_tanh': True,
            },
            'rec_ppo': {
                'obs_dim': 370,
                'act_dim': 2,
                'action_low': np.array([-1.0, -1.0]),
                'action_high': np.array([1.0, 1.0]),
                'device': 'cpu',
                'lr': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'n_epochs': 10,
                'batch_size': 64,
                'max_buffer_size': 2048,
                'squash_tanh': True,
                'hidden_size': 128,
                'rnn_type': 'lstm',
                'num_rnn_layers': 1,
            },
            'td3': {
                'obs_dim': 370,
                'act_dim': 2,
                'action_low': np.array([-1.0, -1.0]),
                'action_high': np.array([1.0, 1.0]),
                'device': 'cpu',
                'lr': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'policy_delay': 2,
                'exploration_noise': 0.1,
                'target_noise': 0.2,
                'noise_clip': 0.5,
                'buffer_size': 100000,
                'warmup_steps': 100,
                'batch_size': 256,
            },
            'sac': {
                'obs_dim': 370,
                'act_dim': 2,
                'action_low': np.array([-1.0, -1.0]),
                'action_high': np.array([1.0, 1.0]),
                'device': 'cpu',
                'lr': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'auto_entropy_tuning': True,
                'buffer_size': 100000,
                'warmup_steps': 100,
                'batch_size': 256,
            },
            'dqn': {
                'obs_dim': 370,
                'action_set': np.array([[1.0, 0.0], [1.0, 0.5], [1.0, -0.5], [0.5, 0.0], [0.0, 0.0]]),
                'device': 'cpu',
                'lr': 1e-3,
                'gamma': 0.99,
                'epsilon_start': 0.9,
                'epsilon_end': 0.05,
                'epsilon_decay': 10000,
                'target_update_interval': 100,
                'buffer_size': 100000,
                'warmup_steps': 100,
                'batch_size': 128,
            },
            'rainbow': {
                'obs_dim': 370,
                'action_set': np.array([[1.0, 0.0], [1.0, 0.5], [1.0, -0.5], [0.5, 0.0], [0.0, 0.0]]),
                'device': 'cpu',
                'lr': 1e-3,
                'gamma': 0.99,
                'target_update_interval': 100,
                'buffer_size': 100000,
                'warmup_steps': 100,
                'batch_size': 128,
                'n_step': 3,
                'v_min': -10.0,
                'v_max': 10.0,
                'n_atoms': 51,
                'alpha': 0.6,
                'beta_start': 0.4,
                'beta_frames': 100000,
            },
        }

        results = {}

        # Test each unique agent type (skip aliases)
        tested_agents = set()
        for agent_type in AgentFactory.available_agents():
            # Skip aliases
            if agent_type in ['recurrent_ppo', 'rainbow_dqn']:
                continue

            if agent_type not in tested_agents:
                config = test_configs.get(agent_type)
                if config:
                    try:
                        agent_class = AgentFactory._registry[agent_type]
                        results[agent_type] = self.test_agent(agent_type.upper(), agent_class, config)
                        tested_agents.add(agent_type)
                    except Exception as e:
                        print(f"\n✗ Failed to test {agent_type}: {e}")
                        import traceback
                        traceback.print_exc()

        self.results = results
        return results

    def print_summary(self):
        """Print summary of all test results."""
        print("\n" + "="*60)
        print("PROTOCOL COMPLIANCE SUMMARY")
        print("="*60)

        if not self.results:
            print("No test results available")
            return

        for agent_name, agent_results in self.results.items():
            passed = sum(1 for v in agent_results.values() if v is True)
            total = len(agent_results)
            pct = (passed / total * 100) if total > 0 else 0

            status = "✓ PASS" if pct == 100 else "⚠ PARTIAL" if pct >= 70 else "✗ FAIL"
            print(f"{agent_name.upper():12} {status:10} {passed:2}/{total:2} ({pct:5.1f}%)")

        # Overall stats
        all_passed = sum(sum(1 for v in r.values() if v) for r in self.results.values())
        all_total = sum(len(r) for r in self.results.values())
        overall_pct = (all_passed / all_total * 100) if all_total > 0 else 0

        print(f"\n{'OVERALL':12} {'':10} {all_passed:2}/{all_total:2} ({overall_pct:5.1f}%)")
        print("="*60)


if __name__ == '__main__':
    tester = ProtocolTester(verbose=True)
    results = tester.test_all_agents()
    tester.print_summary()

    # Exit with error code if not all tests passed
    all_passed = all(all(r.values()) for r in results.values())
    sys.exit(0 if all_passed else 1)
