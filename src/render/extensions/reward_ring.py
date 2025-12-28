"""Reward ring extension for visualizing pressure zones.

Draws concentric circles around a target agent to visualize reward zones
(e.g., pressure distance thresholds in gaplock tasks).
"""
import numpy as np
from array import array
import pyglet
from typing import Optional

from .base import RenderExtension


# Default colors
REWARD_RING_FILL_COLOR = (120/255, 220/255, 120/255, 96/255)
REWARD_RING_BORDER_COLOR = (56/255, 182/255, 86/255, 220/255)
REWARD_RING_PREFERRED_COLOR = (1.0, 1.0, 1.0, 1.0)


class RewardRingExtension(RenderExtension):
    """Visualizes circular reward zones around a target agent.

    Usage:
        renderer = EnvRenderer(800, 600)
        ring = RewardRingExtension(renderer)
        ring.configure(
            enabled=True,
            target_agent='car_1',
            inner_radius=1.0,
            outer_radius=3.0,
            preferred_radius=2.0  # optional
        )
        renderer.add_extension(ring)
    """

    def __init__(self, renderer):
        super().__init__(renderer)
        self._target_agent = None
        self._inner_radius = 0.0
        self._outer_radius = 0.0
        self._preferred_radius = 0.0
        self._segments = 96
        self._fill_color = REWARD_RING_FILL_COLOR
        self._border_color = REWARD_RING_BORDER_COLOR
        self._preferred_color = REWARD_RING_PREFERRED_COLOR

        # Geometry
        self._fill_vlist = None
        self._outer_border_vlist = None
        self._inner_border_vlist = None
        self._preferred_vlist = None
        self._fill_positions = None
        self._fill_view = None
        self._outer_positions = None
        self._outer_view = None
        self._inner_positions = None
        self._inner_view = None
        self._pref_positions = None
        self._pref_view = None

        # Precomputed trigonometry
        self._angles = None
        self._cos_vals = None
        self._sin_vals = None

    def configure(
        self,
        enabled: bool = True,
        target_agent: Optional[str] = None,
        inner_radius: float = 1.0,
        outer_radius: float = 3.0,
        preferred_radius: float = 0.0,
        segments: int = 96,
        fill_color: Optional[tuple] = None,
        border_color: Optional[tuple] = None,
        preferred_color: Optional[tuple] = None,
    ):
        """Configure reward ring visualization.

        Args:
            enabled: Enable/disable the ring
            target_agent: Agent ID to center the ring on
            inner_radius: Inner circle radius in meters
            outer_radius: Outer circle radius in meters
            preferred_radius: Optional preferred radius circle in meters (0 = disabled)
            segments: Number of circle segments (higher = smoother)
            fill_color: Fill zone color (R,G,B,A) normalized [0-1]
            border_color: Border circle color (R,G,B,A)
            preferred_color: Preferred radius circle color (R,G,B,A)
        """
        super().configure(enabled=enabled)

        if not enabled:
            self.cleanup()
            return

        self._target_agent = target_agent
        self._inner_radius = max(float(inner_radius), 0.0)
        self._outer_radius = max(float(outer_radius), self._inner_radius)
        self._preferred_radius = max(float(preferred_radius), 0.0)
        self._segments = max(int(segments), 8)
        self._fill_color = tuple(fill_color) if fill_color else REWARD_RING_FILL_COLOR
        self._border_color = tuple(border_color) if border_color else REWARD_RING_BORDER_COLOR
        self._preferred_color = tuple(preferred_color) if preferred_color else REWARD_RING_PREFERRED_COLOR

        # Precompute circle angles
        self._angles = np.linspace(0.0, 2.0 * np.pi, self._segments, endpoint=False, dtype=np.float32)
        self._cos_vals = np.cos(self._angles)
        self._sin_vals = np.sin(self._angles)

        # Force geometry rebuild
        self.cleanup()

    def update(self, render_obs):
        """Update ring position based on target agent location."""
        if not self._enabled or not self._target_agent:
            return

        # Check if target agent exists
        if self._target_agent not in render_obs:
            return

        # Get target position
        agent_state = render_obs[self._target_agent]
        tx = float(agent_state.get('poses_x', 0.0))
        ty = float(agent_state.get('poses_y', 0.0))

        # Convert to render coordinates
        cx = tx * self.renderer.render_scale
        cy = ty * self.renderer.render_scale

        # Create geometry if needed
        if self._fill_vlist is None:
            self._create_geometry()

        # Update positions
        self._update_fill_ring(cx, cy)
        self._update_outer_border(cx, cy)
        if self._inner_radius > 0:
            self._update_inner_border(cx, cy)
        if self._preferred_radius > 0:
            self._update_preferred_border(cx, cy)

    def _create_geometry(self):
        """Create vertex lists for ring geometry."""
        n = self._segments

        # Fill zone (triangle strip between inner and outer circles)
        fill_verts = 2 * (n + 1)  # Pairs of inner/outer vertices + closing
        self._fill_positions = array('f', [0.0] * (2 * fill_verts))
        self._fill_view = np.frombuffer(self._fill_positions, dtype=np.float32)
        fill_colors = list(self._fill_color) * fill_verts

        self._fill_vlist = self.renderer.shader.vertex_list(
            fill_verts,
            pyglet.gl.GL_TRIANGLE_STRIP,
            batch=self.renderer.batch,
            group=self.renderer.shader_group,
            position=('f/dynamic', self._fill_positions),
            color=('f', fill_colors),
        )

        # Outer border (line loop)
        self._outer_positions = array('f', [0.0] * (2 * n))
        self._outer_view = np.frombuffer(self._outer_positions, dtype=np.float32)
        border_colors = list(self._border_color) * n

        self._outer_border_vlist = self.renderer.shader.vertex_list(
            n,
            pyglet.gl.GL_LINE_LOOP,
            batch=self.renderer.batch,
            group=self.renderer.shader_group,
            position=('f/dynamic', self._outer_positions),
            color=('f', border_colors),
        )

        # Inner border (if inner_radius > 0)
        if self._inner_radius > 0:
            self._inner_positions = array('f', [0.0] * (2 * n))
            self._inner_view = np.frombuffer(self._inner_positions, dtype=np.float32)

            self._inner_border_vlist = self.renderer.shader.vertex_list(
                n,
                pyglet.gl.GL_LINE_LOOP,
                batch=self.renderer.batch,
                group=self.renderer.shader_group,
                position=('f/dynamic', self._inner_positions),
                color=('f', border_colors),
            )

        # Preferred radius circle (if preferred_radius > 0)
        if self._preferred_radius > 0:
            self._pref_positions = array('f', [0.0] * (2 * n))
            self._pref_view = np.frombuffer(self._pref_positions, dtype=np.float32)
            pref_colors = list(self._preferred_color) * n

            self._preferred_vlist = self.renderer.shader.vertex_list(
                n,
                pyglet.gl.GL_LINE_LOOP,
                batch=self.renderer.batch,
                group=self.renderer.shader_group,
                position=('f/dynamic', self._pref_positions),
                color=('f', pref_colors),
            )

    def _update_fill_ring(self, cx: float, cy: float):
        """Update fill zone triangle strip."""
        if self._fill_view is None:
            return

        n = self._segments
        scale = self.renderer.render_scale

        # Interleaved inner and outer vertices
        for i in range(n + 1):
            idx = i % n
            # Inner vertex
            self._fill_view[4*i] = cx + self._inner_radius * scale * self._cos_vals[idx]
            self._fill_view[4*i + 1] = cy + self._inner_radius * scale * self._sin_vals[idx]
            # Outer vertex
            self._fill_view[4*i + 2] = cx + self._outer_radius * scale * self._cos_vals[idx]
            self._fill_view[4*i + 3] = cy + self._outer_radius * scale * self._sin_vals[idx]

    def _update_outer_border(self, cx: float, cy: float):
        """Update outer border circle."""
        if self._outer_view is None:
            return

        scale = self.renderer.render_scale
        radius_px = self._outer_radius * scale

        self._outer_view[0::2] = cx + radius_px * self._cos_vals
        self._outer_view[1::2] = cy + radius_px * self._sin_vals

    def _update_inner_border(self, cx: float, cy: float):
        """Update inner border circle."""
        if self._inner_view is None or self._inner_radius <= 0:
            return

        scale = self.renderer.render_scale
        radius_px = self._inner_radius * scale

        self._inner_view[0::2] = cx + radius_px * self._cos_vals
        self._inner_view[1::2] = cy + radius_px * self._sin_vals

    def _update_preferred_border(self, cx: float, cy: float):
        """Update preferred radius circle."""
        if self._pref_view is None or self._preferred_radius <= 0:
            return

        scale = self.renderer.render_scale
        radius_px = self._preferred_radius * scale

        self._pref_view[0::2] = cx + radius_px * self._cos_vals
        self._pref_view[1::2] = cy + radius_px * self._sin_vals

    def draw_geometry(self, batch, shader_group):
        """Geometry is already in batch, nothing to do here."""
        pass

    def cleanup(self):
        """Clean up vertex lists."""
        for vlist in [self._fill_vlist, self._outer_border_vlist,
                      self._inner_border_vlist, self._preferred_vlist]:
            if vlist is not None:
                try:
                    vlist.delete()
                except Exception:
                    pass

        self._fill_vlist = None
        self._outer_border_vlist = None
        self._inner_border_vlist = None
        self._preferred_vlist = None
        self._fill_positions = None
        self._fill_view = None
        self._outer_positions = None
        self._outer_view = None
        self._inner_positions = None
        self._inner_view = None
        self._pref_positions = None
        self._pref_view = None
