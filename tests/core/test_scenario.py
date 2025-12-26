"""Tests for scenario configuration system."""

import pytest
import tempfile
import yaml
from pathlib import Path

from v2.core.scenario import (
    ScenarioError,
    load_scenario,
    expand_reward_preset,
    expand_observation_preset,
    expand_agent_config,
    expand_scenario,
    validate_scenario,
    resolve_target_ids,
    load_and_expand_scenario,
)


@pytest.fixture
def temp_scenario_file():
    """Create a temporary scenario file."""
    def _create_scenario(content: dict) -> str:
        """Helper to create temp file with content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(content, f)
            return f.name
    return _create_scenario


@pytest.fixture
def valid_scenario():
    """Valid minimal scenario."""
    return {
        'experiment': {
            'name': 'test_experiment',
            'episodes': 100,
        },
        'environment': {
            'map': 'maps/test.yaml',
            'num_agents': 2,
        },
        'agents': {
            'car_0': {
                'algorithm': 'ppo',
                'observation': {'preset': 'gaplock'},
                'reward': {'preset': 'gaplock_simple'},
            },
            'car_1': {
                'algorithm': 'ftg',
            },
        },
    }


class TestLoadScenario:
    """Test scenario loading from YAML."""

    def test_load_valid_scenario(self, temp_scenario_file, valid_scenario):
        """Test loading valid scenario file."""
        path = temp_scenario_file(valid_scenario)
        scenario = load_scenario(path)

        assert scenario['experiment']['name'] == 'test_experiment'
        assert scenario['environment']['map'] == 'maps/test.yaml'
        assert 'car_0' in scenario['agents']

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(ScenarioError, match="not found"):
            load_scenario('nonexistent.yaml')

    def test_load_invalid_yaml(self, temp_scenario_file):
        """Test loading invalid YAML raises error."""
        # Create file with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            path = f.name

        with pytest.raises(ScenarioError, match="Invalid YAML"):
            load_scenario(path)

    def test_load_non_dict_yaml(self, temp_scenario_file):
        """Test loading non-dict YAML raises error."""
        path = temp_scenario_file(['list', 'instead', 'of', 'dict'])

        with pytest.raises(ScenarioError, match="must be a YAML dictionary"):
            load_scenario(path)


class TestExpandRewardPreset:
    """Test reward preset expansion."""

    def test_expand_preset_without_overrides(self):
        """Test expanding preset without overrides."""
        config = {'preset': 'gaplock_simple'}
        expanded = expand_reward_preset(config)

        assert 'terminal' in expanded
        assert 'pressure' in expanded
        assert expanded['terminal']['target_crash'] == 60.0

    def test_expand_preset_with_overrides(self):
        """Test expanding preset with overrides."""
        config = {
            'preset': 'gaplock_simple',
            'overrides': {
                'terminal': {'target_crash': 100.0},
            },
        }
        expanded = expand_reward_preset(config)

        assert expanded['terminal']['target_crash'] == 100.0
        # Other values should be unchanged
        assert expanded['terminal']['self_crash'] == -90.0

    def test_expand_no_preset(self):
        """Test expanding config without preset."""
        config = {
            'terminal': {'target_crash': 50.0},
            'pressure': {'enabled': True},
        }
        expanded = expand_reward_preset(config)

        # Should return deep copy of original
        assert expanded['terminal']['target_crash'] == 50.0

    def test_expand_unknown_preset(self):
        """Test expanding unknown preset raises error."""
        config = {'preset': 'unknown_preset'}

        with pytest.raises(ScenarioError, match="Reward preset error"):
            expand_reward_preset(config)


class TestExpandObservationPreset:
    """Test observation preset expansion."""

    def test_expand_preset_without_overrides(self):
        """Test expanding preset without overrides."""
        config = {'preset': 'gaplock'}
        expanded = expand_observation_preset(config)

        assert expanded['lidar']['beams'] == 720
        assert expanded['lidar']['max_range'] == 12.0

    def test_expand_preset_with_overrides(self):
        """Test expanding preset with overrides."""
        config = {
            'preset': 'gaplock',
            'overrides': {
                'lidar': {'beams': 360},
            },
        }
        expanded = expand_observation_preset(config)

        assert expanded['lidar']['beams'] == 360
        assert expanded['lidar']['max_range'] == 12.0  # Unchanged

    def test_expand_no_preset(self):
        """Test expanding config without preset."""
        config = {
            'lidar': {'enabled': True, 'beams': 500},
        }
        expanded = expand_observation_preset(config)

        assert expanded['lidar']['beams'] == 500

    def test_expand_unknown_preset(self):
        """Test expanding unknown preset raises error."""
        config = {'preset': 'unknown_preset'}

        with pytest.raises(ScenarioError, match="Observation preset error"):
            expand_observation_preset(config)


class TestExpandAgentConfig:
    """Test agent config expansion."""

    def test_expand_agent_with_presets(self):
        """Test expanding agent config with presets."""
        config = {
            'algorithm': 'ppo',
            'observation': {'preset': 'gaplock'},
            'reward': {'preset': 'gaplock_simple'},
        }

        expanded = expand_agent_config(config)

        assert expanded['algorithm'] == 'ppo'
        assert expanded['observation']['lidar']['beams'] == 720
        assert expanded['reward']['terminal']['target_crash'] == 60.0

    def test_expand_agent_with_overrides(self):
        """Test expanding agent config with overrides."""
        config = {
            'algorithm': 'ppo',
            'observation': {
                'preset': 'gaplock',
                'overrides': {'lidar': {'beams': 360}},
            },
            'reward': {
                'preset': 'gaplock_simple',
                'overrides': {'terminal': {'target_crash': 100.0}},
            },
        }

        expanded = expand_agent_config(config)

        assert expanded['observation']['lidar']['beams'] == 360
        assert expanded['reward']['terminal']['target_crash'] == 100.0

    def test_expand_agent_without_presets(self):
        """Test expanding agent config without presets."""
        config = {
            'algorithm': 'ppo',
            'params': {'lr': 0.0005},
        }

        expanded = expand_agent_config(config)

        assert expanded['algorithm'] == 'ppo'
        assert expanded['params']['lr'] == 0.0005


class TestExpandScenario:
    """Test scenario expansion."""

    def test_expand_scenario_with_presets(self, valid_scenario):
        """Test expanding complete scenario."""
        expanded = expand_scenario(valid_scenario)

        # Check that presets were expanded
        assert 'lidar' in expanded['agents']['car_0']['observation']
        assert 'terminal' in expanded['agents']['car_0']['reward']

    def test_expand_scenario_preserves_other_fields(self, valid_scenario):
        """Test that expansion preserves non-preset fields."""
        expanded = expand_scenario(valid_scenario)

        assert expanded['experiment']['name'] == 'test_experiment'
        assert expanded['environment']['map'] == 'maps/test.yaml'

    def test_expand_scenario_without_agents(self):
        """Test expanding scenario without agents."""
        scenario = {
            'experiment': {'name': 'test'},
            'environment': {'map': 'test.yaml'},
        }

        expanded = expand_scenario(scenario)

        assert expanded['experiment']['name'] == 'test'


class TestValidateScenario:
    """Test scenario validation."""

    def test_validate_valid_scenario(self, valid_scenario):
        """Test validating valid scenario."""
        # Should not raise
        validate_scenario(valid_scenario)

    def test_validate_missing_experiment(self, valid_scenario):
        """Test validating scenario without experiment."""
        del valid_scenario['experiment']

        with pytest.raises(ScenarioError, match="must have 'experiment'"):
            validate_scenario(valid_scenario)

    def test_validate_missing_environment(self, valid_scenario):
        """Test validating scenario without environment."""
        del valid_scenario['environment']

        with pytest.raises(ScenarioError, match="must have 'environment'"):
            validate_scenario(valid_scenario)

    def test_validate_missing_agents(self, valid_scenario):
        """Test validating scenario without agents."""
        del valid_scenario['agents']

        with pytest.raises(ScenarioError, match="must have 'agents'"):
            validate_scenario(valid_scenario)

    def test_validate_missing_experiment_name(self, valid_scenario):
        """Test validating scenario without experiment name."""
        del valid_scenario['experiment']['name']

        with pytest.raises(ScenarioError, match="must have 'name'"):
            validate_scenario(valid_scenario)

    def test_validate_missing_environment_map(self, valid_scenario):
        """Test validating scenario without environment map."""
        del valid_scenario['environment']['map']

        with pytest.raises(ScenarioError, match="must have 'map'"):
            validate_scenario(valid_scenario)

    def test_validate_agents_not_dict(self, valid_scenario):
        """Test validating scenario with non-dict agents."""
        valid_scenario['agents'] = ['car_0', 'car_1']

        with pytest.raises(ScenarioError, match="must be a dictionary"):
            validate_scenario(valid_scenario)

    def test_validate_empty_agents(self, valid_scenario):
        """Test validating scenario with no agents."""
        valid_scenario['agents'] = {}

        with pytest.raises(ScenarioError, match="at least one agent"):
            validate_scenario(valid_scenario)

    def test_validate_agent_without_algorithm_or_role(self, valid_scenario):
        """Test validating agent without algorithm or role."""
        valid_scenario['agents']['car_0'] = {'params': {'lr': 0.0005}}

        with pytest.raises(ScenarioError, match="must have 'algorithm'"):
            validate_scenario(valid_scenario)


class TestResolveTargetIds:
    """Test target ID resolution."""

    def test_resolve_attacker_defender(self):
        """Test resolving target ID for attacker/defender."""
        scenario = {
            'agents': {
                'car_0': {'role': 'attacker', 'algorithm': 'ppo'},
                'car_1': {'role': 'defender', 'algorithm': 'ftg'},
            }
        }

        resolved = resolve_target_ids(scenario)

        assert resolved['agents']['car_0']['target_id'] == 'car_1'

    def test_resolve_multiple_attackers(self):
        """Test resolving multiple attackers."""
        scenario = {
            'agents': {
                'car_0': {'role': 'attacker', 'algorithm': 'ppo'},
                'car_1': {'role': 'attacker', 'algorithm': 'td3'},
                'car_2': {'role': 'defender', 'algorithm': 'ftg'},
            }
        }

        resolved = resolve_target_ids(scenario)

        # Both attackers should target first defender
        assert resolved['agents']['car_0']['target_id'] == 'car_2'
        assert resolved['agents']['car_1']['target_id'] == 'car_2'

    def test_resolve_no_roles(self):
        """Test resolving when no roles specified."""
        scenario = {
            'agents': {
                'car_0': {'algorithm': 'ppo'},
                'car_1': {'algorithm': 'ftg'},
            }
        }

        resolved = resolve_target_ids(scenario)

        # Should not add target_id if no roles
        assert 'target_id' not in resolved['agents']['car_0']

    def test_resolve_preserves_explicit_target_id(self):
        """Test that explicit target_id is preserved."""
        scenario = {
            'agents': {
                'car_0': {'role': 'attacker', 'algorithm': 'ppo', 'target_id': 'car_2'},
                'car_1': {'role': 'defender', 'algorithm': 'ftg'},
                'car_2': {'role': 'defender', 'algorithm': 'ftg'},
            }
        }

        resolved = resolve_target_ids(scenario)

        # Should preserve explicit target_id
        assert resolved['agents']['car_0']['target_id'] == 'car_2'


class TestLoadAndExpandScenario:
    """Test complete scenario loading and expansion."""

    def test_load_and_expand_valid_scenario(self, temp_scenario_file, valid_scenario):
        """Test complete load and expand flow."""
        path = temp_scenario_file(valid_scenario)
        scenario = load_and_expand_scenario(path)

        # Check expansion happened
        assert 'lidar' in scenario['agents']['car_0']['observation']
        assert 'terminal' in scenario['agents']['car_0']['reward']

    def test_load_and_expand_with_roles(self, temp_scenario_file):
        """Test load and expand with role resolution."""
        scenario = {
            'experiment': {'name': 'test'},
            'environment': {'map': 'test.yaml'},
            'agents': {
                'car_0': {'role': 'attacker', 'algorithm': 'ppo'},
                'car_1': {'role': 'defender', 'algorithm': 'ftg'},
            },
        }

        path = temp_scenario_file(scenario)
        expanded = load_and_expand_scenario(path)

        # Check target_id was resolved
        assert expanded['agents']['car_0']['target_id'] == 'car_1'

    def test_load_and_expand_validation_error(self, temp_scenario_file):
        """Test that validation errors are raised."""
        invalid_scenario = {
            'experiment': {'name': 'test'},
            # Missing environment
            'agents': {'car_0': {'algorithm': 'ppo'}},
        }

        path = temp_scenario_file(invalid_scenario)

        with pytest.raises(ScenarioError, match="must have 'environment'"):
            load_and_expand_scenario(path)

    def test_load_and_expand_skip_validation(self, temp_scenario_file):
        """Test skipping validation."""
        invalid_scenario = {
            'experiment': {'name': 'test'},
            # Missing environment
            'agents': {'car_0': {'algorithm': 'ppo'}},
        }

        path = temp_scenario_file(invalid_scenario)

        # Should not raise with validate=False
        scenario = load_and_expand_scenario(path, validate=False)
        assert scenario['experiment']['name'] == 'test'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
