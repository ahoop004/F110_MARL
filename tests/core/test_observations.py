"""Tests for observation configuration system."""

import pytest
from src.core.observations import (
    compute_obs_dim,
    load_observation_preset,
    merge_observation_config,
    get_observation_config,
    OBSERVATION_PRESETS,
    GAPLOCK_OBS,
    MINIMAL_OBS,
    FULL_OBS,
)


class TestComputeObsDim:
    """Test observation dimension computation."""

    def test_gaplock_dims(self):
        """Test gaplock configuration gives 738 dims."""
        dim = compute_obs_dim(GAPLOCK_OBS)
        assert dim == 738

    def test_gaplock_breakdown(self):
        """Test gaplock dimension breakdown."""
        config = GAPLOCK_OBS.copy()

        # LiDAR only
        lidar_only = {
            'lidar': config['lidar'],
            'ego_state': {'pose': False, 'velocity': False},
            'target_state': {'enabled': False},
            'relative_pose': {'enabled': False},
        }
        assert compute_obs_dim(lidar_only) == 720

        # Ego state only
        ego_only = {
            'lidar': {'enabled': False},
            'ego_state': {'pose': True, 'velocity': True},
            'target_state': {'enabled': False},
            'relative_pose': {'enabled': False},
        }
        assert compute_obs_dim(ego_only) == 7  # 4 pose + 3 velocity

        # Target state only
        target_only = {
            'lidar': {'enabled': False},
            'ego_state': {'pose': False, 'velocity': False},
            'target_state': {'enabled': True, 'pose': True, 'velocity': True},
            'relative_pose': {'enabled': False},
        }
        assert compute_obs_dim(target_only) == 7

        # Relative pose only
        relative_only = {
            'lidar': {'enabled': False},
            'ego_state': {'pose': False, 'velocity': False},
            'target_state': {'enabled': False},
            'relative_pose': {'enabled': True},
        }
        assert compute_obs_dim(relative_only) == 4

    def test_minimal_dims(self):
        """Test minimal configuration gives 115 dims."""
        dim = compute_obs_dim(MINIMAL_OBS)
        assert dim == 115  # 108 LiDAR + 7 ego state

    def test_full_dims(self):
        """Test full configuration gives 1098 dims."""
        dim = compute_obs_dim(FULL_OBS)
        assert dim == 1098  # 1080 LiDAR + 7 ego + 7 target + 4 relative

    def test_custom_lidar_beams(self):
        """Test custom LiDAR beam count."""
        config = {
            'lidar': {'enabled': True, 'beams': 360},
            'ego_state': {'pose': True, 'velocity': True},
            'target_state': {'enabled': False},
            'relative_pose': {'enabled': False},
        }
        assert compute_obs_dim(config) == 367  # 360 + 7

    def test_no_lidar(self):
        """Test configuration without LiDAR."""
        config = {
            'lidar': {'enabled': False},
            'ego_state': {'pose': True, 'velocity': True},
            'target_state': {'enabled': False},
            'relative_pose': {'enabled': False},
        }
        assert compute_obs_dim(config) == 7

    def test_pose_only(self):
        """Test ego pose without velocity."""
        config = {
            'lidar': {'enabled': False},
            'ego_state': {'pose': True, 'velocity': False},
            'target_state': {'enabled': False},
            'relative_pose': {'enabled': False},
        }
        assert compute_obs_dim(config) == 4

    def test_velocity_only(self):
        """Test ego velocity without pose."""
        config = {
            'lidar': {'enabled': False},
            'ego_state': {'pose': False, 'velocity': True},
            'target_state': {'enabled': False},
            'relative_pose': {'enabled': False},
        }
        assert compute_obs_dim(config) == 3


class TestPresets:
    """Test observation presets."""

    def test_preset_registry(self):
        """Test that all presets are registered."""
        assert 'gaplock' in OBSERVATION_PRESETS
        assert 'minimal' in OBSERVATION_PRESETS
        assert 'full' in OBSERVATION_PRESETS

    def test_gaplock_preset_structure(self):
        """Test gaplock preset structure."""
        preset = GAPLOCK_OBS

        assert preset['lidar']['enabled'] is True
        assert preset['lidar']['beams'] == 720
        assert preset['lidar']['max_range'] == 12.0
        assert preset['ego_state']['pose'] is True
        assert preset['ego_state']['velocity'] is True
        assert preset['target_state']['enabled'] is True
        assert preset['relative_pose']['enabled'] is True

    def test_minimal_preset_structure(self):
        """Test minimal preset structure."""
        preset = MINIMAL_OBS

        assert preset['lidar']['beams'] == 108
        assert preset['target_state']['enabled'] is False
        assert preset['relative_pose']['enabled'] is False

    def test_full_preset_structure(self):
        """Test full preset structure."""
        preset = FULL_OBS

        assert preset['lidar']['beams'] == 1080
        assert preset['target_state']['enabled'] is True
        assert preset['relative_pose']['enabled'] is True


class TestLoadPreset:
    """Test preset loading."""

    def test_load_gaplock(self):
        """Test loading gaplock preset."""
        config = load_observation_preset('gaplock')
        assert config['lidar']['beams'] == 720

    def test_load_minimal(self):
        """Test loading minimal preset."""
        config = load_observation_preset('minimal')
        assert config['lidar']['beams'] == 108

    def test_load_full(self):
        """Test loading full preset."""
        config = load_observation_preset('full')
        assert config['lidar']['beams'] == 1080

    def test_load_unknown_preset(self):
        """Test loading unknown preset raises error."""
        with pytest.raises(ValueError, match="Unknown observation preset"):
            load_observation_preset('unknown')

    def test_preset_independence(self):
        """Test that loaded presets are independent copies."""
        config1 = load_observation_preset('gaplock')
        config2 = load_observation_preset('gaplock')

        # Modify first config
        config1['lidar']['beams'] = 360

        # Second config should be unchanged
        assert config2['lidar']['beams'] == 720

    def test_preset_deep_copy(self):
        """Test that nested dicts are deep copied."""
        config1 = load_observation_preset('gaplock')
        config2 = load_observation_preset('gaplock')

        # Modify nested value
        config1['lidar']['max_range'] = 20.0

        # Should not affect config2
        assert config2['lidar']['max_range'] == 12.0


class TestMergeConfig:
    """Test configuration merging."""

    def test_merge_simple_override(self):
        """Test merging simple override."""
        preset = load_observation_preset('gaplock')
        overrides = {'lidar': {'beams': 360}}

        config = merge_observation_config(preset, overrides)

        assert config['lidar']['beams'] == 360
        assert config['lidar']['max_range'] == 12.0  # Unchanged

    def test_merge_multiple_overrides(self):
        """Test merging multiple overrides."""
        preset = load_observation_preset('gaplock')
        overrides = {
            'lidar': {'beams': 360, 'max_range': 15.0},
            'normalization': {'enabled': False},
        }

        config = merge_observation_config(preset, overrides)

        assert config['lidar']['beams'] == 360
        assert config['lidar']['max_range'] == 15.0
        assert config['normalization']['enabled'] is False

    def test_merge_nested_override(self):
        """Test merging deeply nested override."""
        preset = load_observation_preset('gaplock')
        overrides = {
            'target_state': {
                'enabled': True,
                'velocity': False,  # Override nested value
            }
        }

        config = merge_observation_config(preset, overrides)

        assert config['target_state']['enabled'] is True
        assert config['target_state']['velocity'] is False
        assert config['target_state']['pose'] is True  # Unchanged

    def test_merge_preserves_preset(self):
        """Test that merging doesn't modify original preset."""
        preset = load_observation_preset('gaplock')
        original_beams = preset['lidar']['beams']

        overrides = {'lidar': {'beams': 360}}
        merge_observation_config(preset, overrides)

        # Original preset should be unchanged
        assert preset['lidar']['beams'] == original_beams


class TestGetObservationConfig:
    """Test get_observation_config function."""

    def test_get_from_preset(self):
        """Test getting config from preset name."""
        config = get_observation_config(preset='gaplock')
        assert config['lidar']['beams'] == 720

    def test_get_from_preset_with_overrides(self):
        """Test getting config from preset with overrides."""
        config = get_observation_config(
            preset='gaplock',
            overrides={'lidar': {'beams': 360}},
        )
        assert config['lidar']['beams'] == 360

    def test_get_from_config_dict(self):
        """Test getting config from complete dict."""
        custom_config = {
            'lidar': {'enabled': True, 'beams': 500},
            'ego_state': {'pose': True, 'velocity': True},
        }

        config = get_observation_config(config=custom_config)
        assert config['lidar']['beams'] == 500

    def test_get_config_priority(self):
        """Test that config dict takes priority over preset."""
        custom_config = {'lidar': {'beams': 500}}

        # If config provided, preset should be ignored
        config = get_observation_config(
            preset='gaplock',
            config=custom_config,
        )
        assert config['lidar']['beams'] == 500

    def test_get_no_preset_or_config_raises(self):
        """Test that missing both preset and config raises error."""
        with pytest.raises(ValueError, match="Must provide either"):
            get_observation_config()

    def test_get_config_independence(self):
        """Test that returned config is independent copy."""
        config1 = get_observation_config(preset='gaplock')
        config2 = get_observation_config(preset='gaplock')

        config1['lidar']['beams'] = 360

        assert config2['lidar']['beams'] == 720


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
