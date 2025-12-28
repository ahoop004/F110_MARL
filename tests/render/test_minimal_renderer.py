"""Tests for minimal v2 renderer."""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from src.render import EnvRenderer, MinimalHUD, RenderExtension


def test_renderer_initialization():
    """Test renderer creates without errors."""
    renderer = EnvRenderer(800, 600, lidar_fov=4.7, max_range=30.0)
    assert renderer.width == 800
    assert renderer.height == 600
    assert renderer.lidar_fov == 4.7
    assert renderer.max_range == 30.0
    assert renderer.zoom_level == 1.2
    assert renderer._follow_enabled is True
    assert len(renderer._extensions) == 0
    renderer.close()


def test_renderer_with_hud_extension():
    """Test adding HUD extension."""
    renderer = EnvRenderer(800, 600)
    hud = MinimalHUD(renderer)
    renderer.add_extension(hud)
    assert len(renderer._extensions) == 1
    hud.configure(enabled=True)
    assert hud._enabled is True

    # Update with mock data
    render_obs = {
        'car_0': {
            'poses_x': 0.0,
            'poses_y': 0.0,
            'poses_theta': 0.0,
            'scans': np.zeros(720),
        }
    }
    renderer.update_obs(render_obs)
    assert hud._agent_count == 1
    assert hud._camera_mode == "follow"
    renderer.close()


def test_extension_enable_disable():
    """Test extension enable/disable functionality."""
    renderer = EnvRenderer(800, 600)
    hud = MinimalHUD(renderer)
    renderer.add_extension(hud)

    # Enable
    hud.configure(enabled=True)
    assert hud._enabled is True

    # Disable
    hud.configure(enabled=False)
    assert hud._enabled is False

    renderer.close()


def test_camera_zoom():
    """Test camera zoom controls."""
    renderer = EnvRenderer(800, 600)

    initial_zoom = renderer.zoom_level
    initial_width = renderer.zoomed_width

    # Zoom in
    renderer._apply_zoom(1.5, 0.5, 0.5)
    assert renderer.zoom_level > initial_zoom
    assert renderer.zoomed_width > initial_width

    # Zoom out
    renderer._apply_zoom(0.5, 0.5, 0.5)
    assert renderer.zoom_level < initial_zoom * 1.5

    # Test zoom limits (should not go below 0.01 or above 10.0)
    renderer.zoom_level = 9.5
    renderer._apply_zoom(2.0, 0.5, 0.5)  # Would go to 19.0, should be clamped
    assert renderer.zoom_level == 9.5  # Should not change

    renderer.close()


def test_camera_pan():
    """Test camera pan controls."""
    renderer = EnvRenderer(800, 600)

    initial_left = renderer.left
    initial_bottom = renderer.bottom

    # Pan right (positive dx)
    renderer._pan_view(10, 0)
    assert renderer.left < initial_left  # Panned right (left bound moves left)

    # Pan up (positive dy)
    renderer._pan_view(0, 10)
    assert renderer.bottom < initial_bottom  # Panned up (bottom bound moves down)

    renderer.close()


def test_map_loading(tmp_path):
    """Test map rendering from occupancy grid."""
    # Create minimal test map
    img_size = 100
    img = Image.new('L', (img_size, img_size), 0)  # Black (occupied)
    img_path = tmp_path / "test.png"
    img.save(img_path)

    # Map metadata
    meta = {
        'resolution': 0.05,
        'origin': [0.0, 0.0, 0.0],
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
        'negate': 0,
    }

    renderer = EnvRenderer(800, 600)
    renderer.update_map(
        str(img_path.with_suffix("")),
        ".png",
        map_meta=meta
    )

    assert renderer.map_vlist is not None
    assert renderer._map_vertex_count > 0
    assert renderer.map_points is not None
    assert renderer.map_points.shape[0] > 0

    renderer.close()


def test_map_with_negate_flag(tmp_path):
    """Test map loading with negate=1 flag."""
    img_size = 50
    img = Image.new('L', (img_size, img_size), 255)  # White
    img_path = tmp_path / "test_negate.png"
    img.save(img_path)

    meta = {
        'resolution': 0.1,
        'origin': [-1.0, -1.0, 0.0],
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
        'negate': 1,  # Inverted semantics
    }

    renderer = EnvRenderer(800, 600)
    renderer.update_map(
        str(img_path.with_suffix("")),
        ".png",
        map_meta=meta
    )

    assert renderer.map_vlist is not None
    renderer.close()


def test_single_agent_rendering():
    """Test rendering single agent."""
    renderer = EnvRenderer(800, 600)

    render_obs = {
        'car_0': {
            'poses_x': 1.0,
            'poses_y': 2.0,
            'poses_theta': 0.5,
            'scans': np.ones(720) * 5.0,
        },
    }

    renderer.update_obs(render_obs)

    # Check car rendering
    assert len(renderer.cars_vlist) == 1
    assert 'car_0' in renderer.cars_vlist
    assert 'car_0' in renderer.agent_infos

    # Check LiDAR rendering
    assert len(renderer.scan_hits_vlist) == 1
    assert 'car_0' in renderer.scan_hits_vlist

    # Check agent info cached
    assert renderer.agent_infos['car_0']['poses_x'] == 1.0
    assert renderer.agent_infos['car_0']['poses_y'] == 2.0

    # Check camera target
    assert renderer._camera_target == 'car_0'

    renderer.close()


def test_multi_agent_rendering():
    """Test rendering multiple agents."""
    renderer = EnvRenderer(800, 600)

    render_obs = {
        'car_0': {
            'poses_x': 1.0, 'poses_y': 2.0, 'poses_theta': 0.5,
            'scans': np.ones(720) * 5.0,
        },
        'car_1': {
            'poses_x': 3.0, 'poses_y': 4.0, 'poses_theta': 1.0,
            'scans': np.ones(720) * 8.0,
        },
    }

    renderer.update_obs(render_obs)

    assert len(renderer.cars_vlist) == 2
    assert len(renderer.scan_hits_vlist) == 2
    assert 'car_0' in renderer.agent_infos
    assert 'car_1' in renderer.agent_infos
    assert len(renderer.agent_ids) == 2

    # First agent should have different color (CAR_LEARNER)
    # Second agent should have CAR_OTHER color

    renderer.close()


def test_agent_without_lidar():
    """Test rendering agent without LiDAR scans."""
    renderer = EnvRenderer(800, 600)

    render_obs = {
        'car_0': {
            'poses_x': 1.0,
            'poses_y': 2.0,
            'poses_theta': 0.5,
            # No 'scans' field
        },
    }

    renderer.update_obs(render_obs)

    # Car should render
    assert 'car_0' in renderer.cars_vlist
    # But no LiDAR scans
    assert 'car_0' not in renderer.scan_hits_vlist

    renderer.close()


def test_lidar_buffer_caching():
    """Test that LiDAR buffers are cached and reused."""
    renderer = EnvRenderer(800, 600, lidar_fov=4.7)

    # First update creates buffer
    render_obs = {
        'car_0': {
            'poses_x': 0.0,
            'poses_y': 0.0,
            'poses_theta': 0.0,
            'scans': np.ones(720) * 5.0,
        },
    }
    renderer.update_obs(render_obs)

    # Buffer should be cached
    key = (720, 4.7)
    assert key in renderer._lidar_buffer_cache

    # Second update reuses buffer
    render_obs['car_0']['poses_x'] = 1.0
    renderer.update_obs(render_obs)

    # Same buffer should still be there
    assert key in renderer._lidar_buffer_cache

    renderer.close()


def test_camera_follow_mode():
    """Test camera follow functionality."""
    renderer = EnvRenderer(800, 600)

    render_obs = {
        'car_0': {
            'poses_x': 10.0,
            'poses_y': 20.0,
            'poses_theta': 0.0,
            'scans': np.ones(720) * 5.0,
        },
    }

    # Enable follow mode (default)
    renderer._follow_enabled = True
    renderer.update_obs(render_obs)

    # Camera should center on agent
    cx = 10.0 * renderer.render_scale
    cy = 20.0 * renderer.render_scale
    expected_left = cx - renderer.zoomed_width / 2
    expected_bottom = cy - renderer.zoomed_height / 2

    assert abs(renderer.left - expected_left) < 1e-3
    assert abs(renderer.bottom - expected_bottom) < 1e-3

    renderer.close()


def test_reset_state():
    """Test reset_state cleans up all agent data."""
    renderer = EnvRenderer(800, 600)

    # Add some agents
    render_obs = {
        'car_0': {'poses_x': 0.0, 'poses_y': 0.0, 'poses_theta': 0.0, 'scans': np.ones(720)},
        'car_1': {'poses_x': 1.0, 'poses_y': 1.0, 'poses_theta': 0.5, 'scans': np.ones(720)},
    }
    renderer.update_obs(render_obs)

    assert len(renderer.cars_vlist) == 2
    assert len(renderer.scan_hits_vlist) == 2

    # Reset
    renderer.reset_state()

    # All should be cleared
    assert len(renderer.cars_vlist) == 0
    assert len(renderer.scan_hits_vlist) == 0
    assert len(renderer.agent_infos) == 0
    assert len(renderer.agent_ids) == 0
    assert renderer._camera_target is None

    renderer.close()


def test_window_resize():
    """Test window resize updates camera bounds."""
    renderer = EnvRenderer(800, 600)

    initial_width = renderer.zoomed_width
    initial_height = renderer.zoomed_height

    # Resize window
    renderer.on_resize(1000, 800)

    # Zoomed dimensions should update
    assert renderer.zoomed_width != initial_width
    assert renderer.zoomed_height != initial_height

    # Bounds should be recalculated
    assert renderer.right - renderer.left == renderer.zoomed_width
    assert renderer.top - renderer.bottom == renderer.zoomed_height

    renderer.close()


def test_custom_extension():
    """Test creating and using custom extension."""
    renderer = EnvRenderer(800, 600)

    class TestExtension(RenderExtension):
        def __init__(self, renderer):
            super().__init__(renderer)
            self.update_count = 0

        def update(self, render_obs):
            if self._enabled:
                self.update_count += 1

    ext = TestExtension(renderer)
    renderer.add_extension(ext)
    ext.configure(enabled=True)

    render_obs = {'car_0': {'poses_x': 0.0, 'poses_y': 0.0, 'poses_theta': 0.0}}
    renderer.update_obs(render_obs)

    assert ext.update_count == 1

    # Disable and update again
    ext.configure(enabled=False)
    renderer.update_obs(render_obs)

    # Should not increment
    assert ext.update_count == 1

    renderer.close()


def test_mouse_interactions():
    """Test mouse drag and scroll handlers."""
    renderer = EnvRenderer(800, 600)

    # Test mouse drag (should disable follow mode)
    renderer._follow_enabled = True
    renderer.on_mouse_drag(400, 300, 10, 10, None, None)
    assert renderer._follow_enabled is False

    # Test mouse scroll (zoom)
    initial_zoom = renderer.zoom_level
    renderer.on_mouse_scroll(400, 300, 0, 1)  # Scroll up (zoom in)
    assert renderer.zoom_level != initial_zoom

    renderer.close()
