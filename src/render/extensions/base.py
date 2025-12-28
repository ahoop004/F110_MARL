"""Base class for optional renderer extensions."""
from typing import Any, Dict
import pyglet


class RenderExtension:
    """Base class for optional visualization features.

    Extensions can add custom overlays like reward zones, heatmaps,
    or telemetry displays without cluttering the core renderer.

    Usage:
        class MyExtension(RenderExtension):
            def configure(self, enabled=True, **kwargs):
                super().configure(enabled, **kwargs)
                # Configure extension-specific parameters

            def update(self, render_obs):
                # Update extension state based on new observations
                pass

            def draw_geometry(self, batch, shader_group):
                # Add visualization geometry to rendering batch
                pass
    """

    def __init__(self, renderer):
        """Initialize extension with reference to parent renderer.

        Args:
            renderer: EnvRenderer instance
        """
        self.renderer = renderer
        self._enabled = False

    def configure(self, enabled: bool = True, **kwargs) -> None:
        """Configure extension parameters.

        Args:
            enabled: Enable/disable this extension
            **kwargs: Extension-specific configuration
        """
        self._enabled = enabled
        if not enabled:
            self.cleanup()

    def update(self, render_obs: Dict[str, Any]) -> None:
        """Update extension based on new observation data.

        Called every frame in renderer.update_obs().

        Args:
            render_obs: Dict mapping agent_id -> observation dict
                Example: {'car_0': {'poses_x': 1.0, 'poses_y': 2.0, ...}}
        """
        if not self._enabled:
            return

    def draw_geometry(self, batch: pyglet.graphics.Batch,
                     shader_group: pyglet.graphics.ShaderGroup) -> None:
        """Add geometry to rendering batch.

        Called during on_draw() to render extension visualizations.

        Args:
            batch: Pyglet rendering batch
            shader_group: Shader group for rendering
        """
        if not self._enabled:
            return

    def cleanup(self) -> None:
        """Clean up resources (vertex lists, buffers, etc.)."""
        pass
