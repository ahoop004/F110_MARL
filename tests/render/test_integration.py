"""Integration tests for v2 renderer with F110ParallelEnv."""
import pytest
import os
import numpy as np

# Set headless mode for testing
os.environ["PYGLET_HEADLESS"] = "1"

from v2.env.f110ParallelEnv import F110ParallelEnv
from v2.render import EnvRenderer, MinimalHUD


def test_renderer_with_parallel_env():
    """Test that renderer works with F110ParallelEnv."""
    # Create environment with rendering enabled
    env_config = {
        "map": "maps/line2.yaml",
        "num_agents": 2,
        "max_steps": 5000,
        "timestep": 0.01,
        "render_mode": "rgb_array",  # Use rgb_array for testing (works headless)
    }

    env = F110ParallelEnv(**env_config)

    # Reset environment
    obs, info = env.reset()

    assert env.renderer is None  # Not created until first render() call

    # Render first frame
    frame = env.render()

    # Renderer should be created
    assert env.renderer is not None
    assert isinstance(env.renderer, EnvRenderer)

    # Frame should be returned for rgb_array mode
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (400, 1000, 3)  # WINDOW_H x WINDOW_W x RGB

    # Take a step
    actions = {aid: env.action_space(aid).sample() for aid in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Render again
    frame2 = env.render()
    assert frame2 is not None

    env.close()


def test_renderer_with_hud_extension():
    """Test adding HUD extension to environment renderer."""
    env_config = {
        "map": "maps/line2.yaml",
        "num_agents": 2,
        "render_mode": "rgb_array",
    }

    env = F110ParallelEnv(**env_config)
    env.reset()

    # Trigger renderer creation
    env.render()

    # Add HUD extension
    hud = MinimalHUD(env.renderer)
    env.renderer.add_extension(hud)
    hud.configure(enabled=True)

    # Take a step and render
    actions = {aid: env.action_space(aid).sample() for aid in env.agents}
    env.step(actions)
    frame = env.render()

    assert frame is not None
    assert hud._agent_count == len(env.agents)

    env.close()


def test_renderer_stubs_dont_crash():
    """Test that renderer stubs handle v1 API calls gracefully."""
    env_config = {
        "map": "maps/line2.yaml",
        "num_agents": 2,
        "render_mode": "rgb_array",
    }

    env = F110ParallelEnv(**env_config)
    env.reset()
    env.render()

    # These are called by the environment but should not crash
    # (they're stubs in the minimal renderer)
    env.renderer.update_metrics(phase="test", metrics={"foo": 1.0})
    env.renderer.update_ticker(["Line 1", "Line 2"])
    env.renderer.update_reward_heatmap(None)
    env.renderer.update_reward_overlays([])
    env.renderer.configure_reward_ring(enabled=False)

    # Render should still work
    frame = env.render()
    assert frame is not None

    env.close()


def test_multi_episode_rendering():
    """Test renderer across multiple episodes."""
    env_config = {
        "map": "maps/line2.yaml",
        "num_agents": 2,
        "render_mode": "rgb_array",
    }

    env = F110ParallelEnv(**env_config)

    for episode in range(3):
        env.reset()
        env.render()

        for step in range(10):
            actions = {aid: env.action_space(aid).sample() for aid in env.agents}
            env.step(actions)
            frame = env.render()
            assert frame is not None

    env.close()


@pytest.mark.skipif(os.environ.get("PYGLET_HEADLESS") == "1", reason="Requires display for human mode")
def test_renderer_human_mode():
    """Test renderer in human mode (requires display)."""
    env_config = {
        "map": "maps/line2.yaml",
        "num_agents": 2,
        "render_mode": "human",
    }

    env = F110ParallelEnv(**env_config)
    env.reset()

    # In human mode, render() returns None
    result = env.render()
    assert result is None

    # But renderer should be created
    assert env.renderer is not None

    env.close()


def test_renderer_camera_follow():
    """Test camera follow mode with actual agents."""
    env_config = {
        "map": "maps/line2.yaml",
        "num_agents": 2,
        "render_mode": "rgb_array",
    }

    env = F110ParallelEnv(**env_config)
    env.reset()
    env.render()

    # Camera should follow first agent by default
    assert env.renderer._follow_enabled is True
    assert env.renderer._camera_target is not None

    # Take steps and verify camera updates
    for _ in range(5):
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}
        env.step(actions)
        env.render()

        # Camera should still be following
        assert env.renderer._camera_target in env.agents

    env.close()


def test_renderer_map_loading():
    """Test renderer properly loads map from environment."""
    env_config = {
        "map": "maps/line2.yaml",
        "num_agents": 2,
        "render_mode": "rgb_array",
    }

    env = F110ParallelEnv(**env_config)
    env.reset()
    env.render()

    # Map should be loaded
    assert env.renderer.map_vlist is not None
    assert env.renderer._map_vertex_count > 0
    assert env.renderer.map_points is not None

    env.close()
