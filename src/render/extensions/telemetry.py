"""Enhanced telemetry HUD for detailed training visualization."""
import pyglet
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

    def _update_labels(self):
        """Update all HUD labels based on current mode."""
        if self._mode == self.MODE_OFF:
            self.hud_label.text = ""
            self.agent_label.text = ""
            return

        # Build main HUD text
        lines = []

        # Episode info (all modes)
        lines.append(f"Episode: {self._episode}")
        lines.append(f"Step: {self._step}")

        if self._mode >= self.MODE_BASIC:
            # Reward summary
            if self._rewards:
                lines.append("")
                lines.append("=== Rewards ===")
                for agent_id in sorted(self._rewards.keys()):
                    if self._focused_agent and agent_id != self._focused_agent:
                        continue

                    current = self._rewards.get(agent_id, 0.0)
                    cumulative = self._episode_rewards.get(agent_id, 0.0)
                    collision = self._collisions.get(agent_id, False)

                    collision_marker = " [COLLISION]" if collision else ""
                    lines.append(f"{agent_id}: {current:+.3f} (Σ {cumulative:+.1f}){collision_marker}")

        if self._mode >= self.MODE_DETAILED:
            # Reward components
            if self._reward_components:
                lines.append("")
                lines.append("=== Reward Components ===")
                for agent_id in sorted(self._reward_components.keys()):
                    if self._focused_agent and agent_id != self._focused_agent:
                        continue

                    lines.append(f"{agent_id}:")
                    components = self._reward_components[agent_id]
                    for comp_name, comp_value in sorted(components.items()):
                        lines.append(f"  {comp_name}: {comp_value:+.3f}")

        self.hud_label.text = "\n".join(lines)

        # Build agent-specific panel (only in FULL mode)
        if self._mode >= self.MODE_FULL and self._obs_snapshot:
            agent_lines = []
            agent_lines.append("=== Observation Snapshot ===")

            for agent_id in sorted(self._obs_snapshot.keys()):
                if self._focused_agent and agent_id != self._focused_agent:
                    continue

                snapshot = self._obs_snapshot[agent_id]
                agent_lines.append(f"{agent_id}:")
                agent_lines.append(f"  Pos: ({snapshot.get('x', 0):.2f}, {snapshot.get('y', 0):.2f})")
                agent_lines.append(f"  Theta: {snapshot.get('theta', 0):.2f} rad")
                agent_lines.append(f"  Vel: ({snapshot.get('vx', 0):.2f}, {snapshot.get('vy', 0):.2f})")

                if 'lidar_min' in snapshot:
                    agent_lines.append(f"  LiDAR: min={snapshot['lidar_min']:.2f}m, mean={snapshot['lidar_mean']:.2f}m")

            self.agent_label.text = "\n".join(agent_lines)
        else:
            self.agent_label.text = ""

    def draw_geometry(self, batch, shader_group):
        """Draw telemetry labels.

        Args:
            batch: Rendering batch (unused, labels draw directly)
            shader_group: Shader group (unused)
        """
        if not self._enabled:
            return

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
