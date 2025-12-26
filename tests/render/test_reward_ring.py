"""Tests for reward ring extension."""
import pytest
import numpy as np

from v2.render import EnvRenderer, RewardRingExtension


def test_reward_ring_initialization():
    """Test reward ring extension initializes correctly."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)

    assert ring.renderer == renderer
    assert ring._enabled is False
    assert ring._target_agent is None

    renderer.close()


def test_reward_ring_configure():
    """Test configuring reward ring parameters."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)

    ring.configure(
        enabled=True,
        target_agent='car_1',
        inner_radius=1.5,
        outer_radius=3.0,
        preferred_radius=2.0,
    )

    assert ring._enabled is True
    assert ring._target_agent == 'car_1'
    assert ring._inner_radius == 1.5
    assert ring._outer_radius == 3.0
    assert ring._preferred_radius == 2.0
    assert ring._angles is not None
    assert ring._cos_vals is not None
    assert ring._sin_vals is not None

    renderer.close()


def test_reward_ring_disable():
    """Test disabling reward ring."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)

    # Enable first
    ring.configure(enabled=True, target_agent='car_1', outer_radius=3.0)
    assert ring._enabled is True

    # Disable
    ring.configure(enabled=False)
    assert ring._enabled is False

    renderer.close()


def test_reward_ring_update_with_agent():
    """Test reward ring updates when target agent exists."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)
    renderer.add_extension(ring)

    ring.configure(
        enabled=True,
        target_agent='car_1',
        inner_radius=1.0,
        outer_radius=3.0,
    )

    render_obs = {
        'car_0': {'poses_x': 0.0, 'poses_y': 0.0, 'poses_theta': 0.0},
        'car_1': {'poses_x': 5.0, 'poses_y': 10.0, 'poses_theta': 0.5},
    }

    ring.update(render_obs)

    # Geometry should be created
    assert ring._fill_vlist is not None
    assert ring._outer_border_vlist is not None

    renderer.close()


def test_reward_ring_update_without_target():
    """Test reward ring handles missing target agent gracefully."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)

    ring.configure(
        enabled=True,
        target_agent='car_999',  # Non-existent agent
        outer_radius=3.0,
    )

    render_obs = {
        'car_0': {'poses_x': 0.0, 'poses_y': 0.0, 'poses_theta': 0.0},
    }

    # Should not crash
    ring.update(render_obs)

    # Geometry should not be created
    assert ring._fill_vlist is None

    renderer.close()


def test_reward_ring_with_renderer():
    """Test reward ring integration with renderer."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)
    renderer.add_extension(ring)

    ring.configure(
        enabled=True,
        target_agent='car_0',
        inner_radius=1.0,
        outer_radius=2.5,
        preferred_radius=1.8,
        segments=48,
    )

    render_obs = {
        'car_0': {'poses_x': 2.0, 'poses_y': 3.0, 'poses_theta': 0.0, 'scans': np.ones(720) * 5.0},
    }

    renderer.update_obs(render_obs)

    # Ring should have created geometry
    assert ring._fill_vlist is not None
    assert ring._outer_border_vlist is not None
    assert ring._preferred_vlist is not None  # Should exist since preferred_radius > 0

    renderer.close()


def test_reward_ring_custom_colors():
    """Test reward ring with custom colors."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)

    custom_fill = (0.5, 0.8, 0.3, 0.4)
    custom_border = (1.0, 0.0, 0.0, 1.0)
    custom_pref = (0.0, 0.0, 1.0, 1.0)

    ring.configure(
        enabled=True,
        target_agent='car_0',
        outer_radius=3.0,
        fill_color=custom_fill,
        border_color=custom_border,
        preferred_color=custom_pref,
    )

    assert ring._fill_color == custom_fill
    assert ring._border_color == custom_border
    assert ring._preferred_color == custom_pref

    renderer.close()


def test_reward_ring_cleanup():
    """Test reward ring cleanup removes geometry."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)
    renderer.add_extension(ring)

    ring.configure(
        enabled=True,
        target_agent='car_0',
        inner_radius=1.0,
        outer_radius=3.0,
    )

    render_obs = {'car_0': {'poses_x': 0.0, 'poses_y': 0.0, 'poses_theta': 0.0}}
    ring.update(render_obs)

    assert ring._fill_vlist is not None

    # Cleanup
    ring.cleanup()

    assert ring._fill_vlist is None
    assert ring._outer_border_vlist is None
    assert ring._inner_border_vlist is None

    renderer.close()


def test_reward_ring_zero_inner_radius():
    """Test reward ring with zero inner radius (only outer circle)."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)
    renderer.add_extension(ring)

    ring.configure(
        enabled=True,
        target_agent='car_0',
        inner_radius=0.0,  # Zero inner radius
        outer_radius=3.0,
    )

    render_obs = {'car_0': {'poses_x': 1.0, 'poses_y': 1.0, 'poses_theta': 0.0}}
    ring.update(render_obs)

    # Fill should go from center to outer edge
    assert ring._fill_vlist is not None
    assert ring._inner_radius == 0.0

    renderer.close()


def test_reward_ring_multiple_updates():
    """Test reward ring updates correctly across multiple frames."""
    renderer = EnvRenderer(800, 600)
    ring = RewardRingExtension(renderer)
    renderer.add_extension(ring)

    ring.configure(
        enabled=True,
        target_agent='car_0',
        outer_radius=3.0,
    )

    # Update with different positions
    for x in [0.0, 5.0, 10.0]:
        render_obs = {'car_0': {'poses_x': x, 'poses_y': 0.0, 'poses_theta': 0.0}}
        ring.update(render_obs)
        # Should continue to work without errors

    assert ring._fill_vlist is not None

    renderer.close()
