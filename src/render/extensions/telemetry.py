"""Enhanced telemetry HUD for detailed training visualization."""
import pyglet
import pyglet.shapes
import numpy as np
from .base import RenderExtension


class TelemetryHUD(RenderExtension):
    """Enhanced telemetry display with multiple detail levels.

    Display modes:
    - 0: Off
    - 1: Minimal (episode, step, FPS)
    - 2: Basic (+ rewards, collisions)
    - 3: Detailed (+ reward components)
    - 4: Full (+ observation snapshot)

    Keyboard controls:
    - T: Cycle through modes
    - 1-9: Focus on specific agent

    Usage:
        renderer = EnvRenderer(800, 600)
        telemetry = TelemetryHUD(renderer)
        renderer.add_extension(telemetry)
        telemetry.configure(enabled=True, mode=2)
    """

    MODE_OFF = 0
    MODE_MINIMAL = 1
    MODE_BASIC = 2
    MODE_DETAILED = 3
    MODE_FULL = 4

    def __init__(self, renderer):
        super().__init__(renderer)
        self._mode = self.MODE_BASIC
        self._focused_agent = None  # None = show all, or specific agent_id

        # Episode/step tracking
        self._episode = 0
        self._step = 0

        # Reward tracking per agent
        self._rewards = {}  # agent_id -> current reward
        self._episode_rewards = {}  # agent_id -> cumulative reward
        self._reward_components = {}  # agent_id -> dict of component rewards

        # Collision status
        self._collisions = {}  # agent_id -> bool

        # Observation snapshot (for full mode)
        self._obs_snapshot = {}  # agent_id -> key observation values

        # Create labels
        h = renderer.height
        w = renderer.width

        # Main HUD panel (top-left)
        self.hud_label = pyglet.text.Label(
            '', font_size=12, x=10, y=h-10,
            anchor_x='left', anchor_y='top',
            color=(255, 255, 255, 255),
            multiline=True, width=500
        )

        # Agent-specific panel (top-right)
        self.agent_label = pyglet.text.Label(
            '', font_size=11, x=w-10, y=h-10,
            anchor_x='right', anchor_y='top',
            color=(200, 200, 255, 255),
            multiline=True, width=400
        )

        # FPS display (bottom-right)
        self.fps_display = pyglet.window.FPSDisplay(renderer)
        self.fps_display.label.x = w - 10
        self.fps_display.label.y = 10
        self.fps_display.label.anchor_x = 'right'

        # Mode indicator (bottom-left)
        self.mode_label = pyglet.text.Label(
            '', font_size=10, x=10, y=10,
            anchor_x='left', anchor_y='bottom',
            color=(150, 150, 150, 255)
        )

        # Colored agent legend swatches (one rectangle per agent)
        # Built and positioned dynamically in _rebuild_swatches()
        self._swatches: list = []           # pyglet.shapes.Rectangle objects
        self._swatch_batch = pyglet.graphics.Batch()
        self._SWATCH_W = 10
        self._SWATCH_H = 10
        self._SWATCH_X = 10                 # left edge of swatch column
        self._LINE_H = 14                   # px per text line (font_size=12 → ~14px)

    def configure(self, enabled: bool = True, mode: int = MODE_BASIC, **kwargs):
        """Configure telemetry display.

        Args:
            enabled: Enable/disable telemetry
            mode: Display detail level (0-4)
            **kwargs: Additional options
        """
        super().configure(enabled, **kwargs)
        self._mode = mode if enabled else self.MODE_OFF
        self._update_mode_label()

    def set_mode(self, mode: int):
        """Set display mode.

        Args:
            mode: Display detail level (0-4)
        """
        self._mode = max(0, min(4, mode))
        self._enabled = (self._mode > 0)
        self._update_mode_label()

    def cycle_mode(self):
        """Cycle to next display mode."""
        self.set_mode((self._mode + 1) % 5)

    def set_focused_agent(self, agent_id: str = None):
        """Focus on specific agent or show all.

        Args:
            agent_id: Agent to focus on, or None for all agents
        """
        self._focused_agent = agent_id

    def update_episode_info(self, episode: int, step: int):
        """Update episode and step counters.

        Args:
            episode: Current episode number
            step: Current step in episode
        """
        self._episode = episode
        self._step = step

    def update_rewards(self, agent_id: str, reward: float,
                      components: dict = None, reset: bool = False):
        """Update reward information for an agent.

        Args:
            agent_id: Agent identifier
            reward: Current step reward
            components: Dict of reward component values (optional)
            reset: Whether to reset cumulative reward (new episode)
        """
        self._rewards[agent_id] = reward

        if reset or agent_id not in self._episode_rewards:
            self._episode_rewards[agent_id] = 0.0
        self._episode_rewards[agent_id] += reward

        if components is not None:
            self._reward_components[agent_id] = components

    def update_collision_status(self, agent_id: str, collision: bool):
        """Update collision status for an agent.

        Args:
            agent_id: Agent identifier
            collision: Whether agent is in collision
        """
        self._collisions[agent_id] = collision

    def update(self, render_obs, **kwargs):
        """Update telemetry with current state.

        Args:
            render_obs: Dict mapping agent_id -> observation dict
            **kwargs: Additional data (rewards, collisions, etc.)
        """
        if not self._enabled:
            return

        # Extract observation snapshot for full mode
        if self._mode >= self.MODE_FULL:
            for agent_id, obs in render_obs.items():
                snapshot = {}
                # Extract key observation values
                if isinstance(obs, dict):
                    # Position
                    snapshot['x'] = obs.get('poses_x', 0.0)
                    snapshot['y'] = obs.get('poses_y', 0.0)
                    snapshot['theta'] = obs.get('poses_theta', 0.0)

                    # Velocity
                    snapshot['vx'] = obs.get('linear_vels_x', 0.0)
                    snapshot['vy'] = obs.get('linear_vels_y', 0.0)

                    # LiDAR stats
                    scans = obs.get('scans')
                    if scans is not None:
                        scans = np.asarray(scans)
                        snapshot['lidar_min'] = float(np.min(scans))
                        snapshot['lidar_mean'] = float(np.mean(scans))

                self._obs_snapshot[agent_id] = snapshot

        self._update_labels()

    def _update_mode_label(self):
        """Update mode indicator label."""
        mode_names = ['OFF', 'MINIMAL', 'BASIC', 'DETAILED', 'FULL']
        mode_name = mode_names[self._mode] if 0 <= self._mode < len(mode_names) else 'UNKNOWN'
        focus = f" (Focus: {self._focused_agent})" if self._focused_agent else " (All)"
        self.mode_label.text = f"[T] Telemetry: {mode_name}{focus}"

    def _agent_color_rgba255(self, agent_id: str) -> tuple:
        """Return (R, G, B, A) in 0-255 range for agent_id from renderer palette."""
        from render.renderer import _agent_default_color
        renderer = self.renderer
        color_f = getattr(renderer, "_agent_colors", {}).get(agent_id)
        if color_f is None:
            color_f = _agent_default_color(agent_id)
        return tuple(int(c * 255) for c in color_f)

    def _update_labels(self):
        """Update all HUD labels based on current mode."""
        if self._mode == self.MODE_OFF:
            self.hud_label.text = ""
            self.agent_label.text = ""
            self._swatches.clear()
            return

        # Build main HUD text
        lines = []

        # Episode info (all modes)
        lines.append(f"Episode: {self._episode}")
        lines.append(f"Step: {self._step}")

        # Track which lines correspond to agent names (for swatch positions)
        agent_line_indices: list = []  # list of (line_index, agent_id)

        if self._mode >= self.MODE_BASIC:
            # Reward summary
            if self._rewards:
                lines.append("")
                lines.append("Rewards")
                for agent_id in sorted(self._rewards.keys()):
                    if self._focused_agent and agent_id != self._focused_agent:
                        continue
                    current = self._rewards.get(agent_id, 0.0)
                    cumulative = self._episode_rewards.get(agent_id, 0.0)
                    collision = self._collisions.get(agent_id, False)
                    collision_marker = " !" if collision else ""
                    agent_line_indices.append((len(lines), agent_id))
                    lines.append(f"  {agent_id}: {current:+.3f} (Σ{cumulative:+.1f}){collision_marker}")

        if self._mode >= self.MODE_DETAILED:
            if self._reward_components:
                lines.append("")
                lines.append("Components")
                for agent_id in sorted(self._reward_components.keys()):
                    if self._focused_agent and agent_id != self._focused_agent:
                        continue
                    agent_line_indices.append((len(lines), agent_id))
                    lines.append(f"  {agent_id}:")
                    components = self._reward_components[agent_id]
                    for comp_name, comp_value in sorted(components.items()):
                        lines.append(f"    {comp_name}: {comp_value:+.3f}")

        self.hud_label.text = "\n".join(lines)

        # Build agent-specific panel (only in FULL mode)
        if self._mode >= self.MODE_FULL and self._obs_snapshot:
            agent_lines = ["Obs Snapshot"]
            for agent_id in sorted(self._obs_snapshot.keys()):
                if self._focused_agent and agent_id != self._focused_agent:
                    continue
                snapshot = self._obs_snapshot[agent_id]
                agent_lines.append(f"{agent_id}:")
                agent_lines.append(f"  ({snapshot.get('x', 0):.1f}, {snapshot.get('y', 0):.1f})")
                agent_lines.append(f"  θ={snapshot.get('theta', 0):.2f}  v=({snapshot.get('vx', 0):.1f},{snapshot.get('vy', 0):.1f})")
                if 'lidar_min' in snapshot:
                    agent_lines.append(f"  lidar min={snapshot['lidar_min']:.2f}m")
            self.agent_label.text = "\n".join(agent_lines)
        else:
            self.agent_label.text = ""

        # Rebuild colored swatches next to each agent name line
        self._rebuild_swatches(agent_line_indices, len(lines))

    def _rebuild_swatches(self, agent_line_indices: list, total_lines: int) -> None:
        """Draw small colored rectangles next to agent-name lines in the HUD."""
        self._swatches.clear()
        if not agent_line_indices:
            return

        h = self.renderer.height
        # Top of HUD text in window coords (label anchor is top-left at y=h-10)
        text_top_y = h - 10
        sw = self._SWATCH_W
        sh = self._SWATCH_H
        lh = self._LINE_H
        x = self._SWATCH_X

        for line_idx, agent_id in agent_line_indices:
            # Y of the top of this line (lines counted from top, 0-indexed)
            line_y = text_top_y - line_idx * lh
            swatch_y = line_y - sh  # bottom-left of the swatch square

            r, g, b, a = self._agent_color_rgba255(agent_id)
            rect = pyglet.shapes.Rectangle(
                x=x, y=swatch_y,
                width=sw, height=sh,
                color=(r, g, b),
                batch=self._swatch_batch,
            )
            rect.opacity = a
            self._swatches.append(rect)

    def draw_geometry(self, batch, shader_group):
        """Draw telemetry labels and colored agent swatches."""
        if not self._enabled:
            return

        self._swatch_batch.draw()
        self.hud_label.draw()
        self.agent_label.draw()
        self.fps_display.draw()
        self.mode_label.draw()

    def cleanup(self):
        """Clean up telemetry resources."""
        self._rewards.clear()
        self._episode_rewards.clear()
        self._reward_components.clear()
        self._collisions.clear()
        self._obs_snapshot.clear()
        self._swatches.clear()
