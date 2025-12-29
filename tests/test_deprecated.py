"""Tests for deprecated functionality (old reward system removal)."""

import pytest
import warnings


def test_old_reward_module_removed():
    """Verify src/tasks/reward module has been removed."""
    with pytest.raises(ModuleNotFoundError):
        from tasks.reward import RewardStrategy  # noqa: F401


def test_reward_wrapper_removed():
    """Verify RewardWrapper has been removed from wrappers."""
    with pytest.raises(ModuleNotFoundError):
        from wrappers.reward import RewardWrapper  # noqa: F401


def test_reward_runtime_context_removed():
    """Verify RewardRuntimeContext has been removed."""
    with pytest.raises(ModuleNotFoundError):
        from wrappers.reward import RewardRuntimeContext  # noqa: F401


def test_wrappers_init_no_reward_exports():
    """Verify wrappers.__init__ no longer exports reward classes."""
    from wrappers import __all__

    assert "RewardWrapper" not in __all__
    assert "RewardRuntimeContext" not in __all__
    assert "ObsWrapper" in __all__  # Should still export this


def test_wrapper_factory_no_wrap_reward_method():
    """Verify WrapperFactory no longer has wrap_reward method."""
    from core.config import WrapperFactory

    assert not hasattr(WrapperFactory, "wrap_reward")


def test_wrapper_factory_has_other_methods():
    """Verify WrapperFactory still has other wrapper methods."""
    from core.config import WrapperFactory

    assert hasattr(WrapperFactory, "wrap_observation")
    assert hasattr(WrapperFactory, "wrap_action")
    assert hasattr(WrapperFactory, "wrap_all")


def test_wrapper_factory_reward_config_warns():
    """Verify WrapperFactory.wrap_all warns on reward config."""
    from core.config import WrapperFactory
    from unittest.mock import MagicMock

    env = MagicMock()
    wrapper_configs = {"reward": {"enabled": True}}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = WrapperFactory.wrap_all(env, wrapper_configs)

        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
        assert "removed" in str(w[0].message).lower()

        # Should still return env (not fail)
        assert result is env


def test_tasks_module_is_deprecated():
    """Verify src/tasks module exists but is marked deprecated."""
    import tasks

    # Module should exist
    assert tasks is not None

    # But should have empty __all__ (nothing exported)
    assert hasattr(tasks, "__all__")
    assert tasks.__all__ == []


def test_new_reward_system_works():
    """Verify new reward system is still functional."""
    from rewards import build_reward_strategy

    # Should be able to import and call
    assert callable(build_reward_strategy)


def test_new_reward_presets_available():
    """Verify new reward presets are available."""
    from rewards import load_preset

    # Should be able to load presets
    preset = load_preset("gaplock_full")
    assert preset is not None
    assert isinstance(preset, dict)


def test_reward_components_available():
    """Verify reward components are still available."""
    from rewards.gaplock import GaplockReward
    from rewards.base import RewardComponent, RewardStrategy

    # Should be able to import protocols and implementations
    assert GaplockReward is not None
    assert RewardComponent is not None
    assert RewardStrategy is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
