"""Minimal HUD overlay for basic renderer info."""
import pyglet
from .base import RenderExtension


class MinimalHUD(RenderExtension):
    """Displays basic info: agent count, camera mode, FPS.

    Usage:
        renderer = EnvRenderer(800, 600)
        hud = MinimalHUD(renderer)
        renderer.add_extension(hud)
        hud.configure(enabled=True)
    """

    def __init__(self, renderer):
        super().__init__(renderer)
        self._agent_count = 0
        self._camera_mode = "follow"

        # Create labels
        h = renderer.height
        self.hud_label = pyglet.text.Label(
            '', font_size=14, x=10, y=h-10,
            anchor_x='left', anchor_y='top',
            color=(255, 255, 255, 255),
            multiline=True, width=400
        )
        self.fps_display = pyglet.window.FPSDisplay(renderer)

    def configure(self, enabled: bool = True, **kwargs):
        """Configure HUD display.

        Args:
            enabled: Enable/disable HUD
            **kwargs: Unused (for extensibility)
        """
        super().configure(enabled, **kwargs)
        if enabled:
            self._update_label()

    def update(self, render_obs):
        """Update HUD with current agent count and camera mode.

        Args:
            render_obs: Dict mapping agent_id -> observation dict
        """
        if not self._enabled:
            return
        self._agent_count = len(render_obs)
        self._camera_mode = "free" if not self.renderer._follow_enabled else "follow"
        self._update_label()

    def _update_label(self):
        """Update HUD label text."""
        self.hud_label.text = f"Agents: {self._agent_count}\nCamera: {self._camera_mode}"

    def draw_geometry(self, batch, shader_group):
        """Draw HUD labels and FPS display.

        Args:
            batch: Rendering batch (unused, HUD draws directly)
            shader_group: Shader group (unused)
        """
        if not self._enabled:
            return
        self.hud_label.draw()
        self.fps_display.draw()

    def cleanup(self):
        """Clean up HUD resources."""
        pass
