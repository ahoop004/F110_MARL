"""Reward heatmap visualization for spatial reward fields."""
import pyglet
import numpy as np
from array import array
from .base import RenderExtension


class RewardHeatmap(RenderExtension):
    """Visualizes spatial reward field as a 2D heatmap.

    Shows what reward the agent would receive at different positions,
    useful for understanding the reward landscape and agent behavior.

    Keyboard controls:
    - H: Toggle heatmap display

    Usage:
        renderer = EnvRenderer(800, 600)
        heatmap = RewardHeatmap(renderer)
        renderer.add_extension(heatmap)
        heatmap.configure(
            enabled=True,
            target_agent='car_1',
            attacker_agent='car_0',
            extent_m=6.0,
            cell_size_m=0.25,
            alpha=0.22
        )
    """

    def __init__(self, renderer):
        super().__init__(renderer)
        self._target_agent = None
        self._attacker_agent = None

        # Heatmap parameters
        self._extent_m = 6.0  # Spatial extent in meters (half-width)
        self._cell_size_m = 0.25  # Cell size in meters
        self._alpha = 0.22  # Transparency

        # Grid dimensions
        self._grid_cells = 0
        self._grid_size = 0

        # Reward strategy reference (for computing actual rewards)
        self._reward_strategy = None

        # Agent positions and observations
        self._target_x = 0.0
        self._target_y = 0.0
        self._target_theta = 0.0
        self._target_obs = None
        self._attacker_x = 0.0
        self._attacker_y = 0.0
        self._attacker_theta = 0.0

        # Vertex list for heatmap cells
        self._heatmap_vlist = None
        self._positions_array = None
        self._positions_view = None
        self._colors_array = None
        self._colors_view = None

        # Update tracking
        self._update_counter = 0
        self._update_frequency = 5  # Update every N frames
        self._params_printed = False  # Track if we've printed parameters

    def configure(self, enabled: bool = True,
                 target_agent: str = None,
                 attacker_agent: str = None,
                 reward_strategy = None,
                 extent_m: float = 6.0,
                 cell_size_m: float = 0.25,
                 alpha: float = 0.22,
                 update_frequency: int = 5,
                 **kwargs):
        """Configure reward heatmap visualization.

        Args:
            enabled: Enable/disable heatmap
            target_agent: Target agent ID
            attacker_agent: Attacker agent ID
            reward_strategy: Reward strategy to query for actual reward values
            extent_m: Heatmap spatial extent in meters (half-width)
            cell_size_m: Cell size in meters
            alpha: Heatmap transparency (0-1)
            update_frequency: Update heatmap every N frames (1 = every frame)
            **kwargs: Additional options (backward compatibility)
        """
        super().configure(enabled, **kwargs)

        self._target_agent = target_agent
        self._attacker_agent = attacker_agent
        self._reward_strategy = reward_strategy
        self._extent_m = extent_m
        self._cell_size_m = cell_size_m
        self._alpha = alpha
        self._update_frequency = max(1, update_frequency)

        # Calculate grid dimensions
        self._grid_cells = int(2.0 * self._extent_m / self._cell_size_m)
        self._grid_size = self._grid_cells * self._grid_cells

        # Print reward parameters if reward strategy is provided
        if reward_strategy is not None and enabled and not self._params_printed:
            self._print_reward_parameters()
            self._params_printed = True

        if enabled:
            self._create_heatmap_geometry()

    def _print_reward_parameters(self):
        """Print reward strategy parameters for debugging."""
        if self._reward_strategy is None:
            return

        print("\n" + "="*60)
        print("HEATMAP REWARD PARAMETERS")
        print("="*60)

        # Get the composer and its components
        try:
            composer = self._reward_strategy.composer
            components = composer.components if hasattr(composer, 'components') else []

            print(f"\nActive reward components: {len(components)}")
            for comp in components:
                comp_name = comp.__class__.__name__
                print(f"\n  • {comp_name}")

                # Extract parameters from each component
                if comp_name == 'ForcingReward':
                    if hasattr(comp, 'enabled') and comp.enabled:
                        print(f"    - Forcing enabled: {comp.enabled}")
                        if hasattr(comp, 'pinch_enabled') and comp.pinch_enabled:
                            print(f"    - Pinch pockets:")
                            print(f"      • anchor_forward: {comp.pinch_anchor_forward:.3f}m")
                            print(f"      • anchor_lateral: {comp.pinch_anchor_lateral:.3f}m")
                            print(f"      • sigma: {comp.pinch_sigma:.3f}")
                            print(f"      • weight: {comp.pinch_weight:.3f}")

                elif comp_name == 'DistanceReward':
                    if hasattr(comp, 'enabled') and comp.enabled:
                        print(f"    - Distance shaping enabled")
                        if hasattr(comp, 'near_distance'):
                            print(f"      • near_distance: {comp.near_distance:.3f}m")
                            print(f"      • far_distance: {comp.far_distance:.3f}m")
                            print(f"      • reward_near: {comp.reward_near:.3f}")
                            print(f"      • penalty_far: {comp.penalty_far:.3f}")

                elif comp_name == 'HeadingReward':
                    if hasattr(comp, 'enabled') and comp.enabled:
                        print(f"    - Heading alignment enabled")
                        if hasattr(comp, 'coefficient'):
                            print(f"      • coefficient: {comp.coefficient:.3f}")

        except Exception as e:
            print(f"Could not extract parameters: {e}")

        print("\n" + "="*60 + "\n")

    def _create_heatmap_geometry(self):
        """Create vertex list for heatmap grid."""
        # Clean up existing geometry
        if self._heatmap_vlist is not None:
            self._heatmap_vlist.delete()

        # Each cell is a quad (4 vertices)
        num_vertices = self._grid_size * 4

        # Create position and color arrays
        self._positions_array = array('f', [0.0] * (2 * num_vertices))
        self._positions_view = np.frombuffer(self._positions_array, dtype=np.float32)

        self._colors_array = array('f', [0.0] * (4 * num_vertices))
        self._colors_view = np.frombuffer(self._colors_array, dtype=np.float32).reshape((num_vertices, 4))

        # Create vertex list (using quads)
        self._heatmap_vlist = self.renderer.shader.vertex_list(
            num_vertices,
            pyglet.gl.GL_QUADS,
            batch=self.renderer.batch,
            group=self.renderer.shader_group,
            position=('f/dynamic', self._positions_array),
            color=('f/dynamic', self._colors_array)
        )

        # Initialize cell positions (these don't change)
        scale = self.renderer.render_scale
        cell_size_px = self._cell_size_m * scale

        idx = 0
        for row in range(self._grid_cells):
            for col in range(self._grid_cells):
                # Cell corners in local grid coordinates
                x0 = -self._extent_m + col * self._cell_size_m
                y0 = -self._extent_m + row * self._cell_size_m
                x1 = x0 + self._cell_size_m
                y1 = y0 + self._cell_size_m

                # Store as quad vertices (will be offset by target position in update)
                # Bottom-left
                self._positions_view[idx*8 + 0] = x0 * scale
                self._positions_view[idx*8 + 1] = y0 * scale
                # Bottom-right
                self._positions_view[idx*8 + 2] = x1 * scale
                self._positions_view[idx*8 + 3] = y0 * scale
                # Top-right
                self._positions_view[idx*8 + 4] = x1 * scale
                self._positions_view[idx*8 + 5] = y1 * scale
                # Top-left
                self._positions_view[idx*8 + 6] = x0 * scale
                self._positions_view[idx*8 + 7] = y1 * scale

                idx += 1

    def update(self, render_obs, **kwargs):
        """Update heatmap based on agent positions.

        Args:
            render_obs: Dict mapping agent_id -> observation dict
            **kwargs: Additional data
        """
        if not self._enabled or self._reward_strategy is None:
            return

        # Create geometry if needed (e.g., when toggled on after initialization)
        if self._heatmap_vlist is None:
            self._create_heatmap_geometry()

        # Throttle updates for performance
        self._update_counter += 1
        if self._update_counter < self._update_frequency:
            return
        self._update_counter = 0

        # Get target agent observation (full obs for reward calculation)
        if self._target_agent and self._target_agent in render_obs:
            obs = render_obs[self._target_agent]
            self._target_x = float(obs.get('poses_x', 0.0))
            self._target_y = float(obs.get('poses_y', 0.0))
            self._target_theta = float(obs.get('poses_theta', 0.0))

            # Store full observation for reward calculation
            self._target_obs = {
                'pose': np.array([self._target_x, self._target_y, self._target_theta]),
                'scans': obs.get('scans', np.zeros(720)),
            }
        else:
            return

        # Get attacker position (for reference)
        if self._attacker_agent and self._attacker_agent in render_obs:
            obs = render_obs[self._attacker_agent]
            self._attacker_x = float(obs.get('poses_x', 0.0))
            self._attacker_y = float(obs.get('poses_y', 0.0))
            self._attacker_theta = float(obs.get('poses_theta', 0.0))

        # Update heatmap
        self._update_heatmap_colors()
        self._update_heatmap_positions()

    def _update_heatmap_colors(self):
        """Update heatmap cell colors based on reward values."""
        if self._colors_view is None or self._target_obs is None:
            return

        # Collect all rewards for normalization
        rewards = np.zeros(self._grid_size)
        idx = 0

        for row in range(self._grid_cells):
            for col in range(self._grid_cells):
                # Cell center in world coordinates
                x = -self._extent_m + (col + 0.5) * self._cell_size_m + self._target_x
                y = -self._extent_m + (row + 0.5) * self._cell_size_m + self._target_y

                # Query actual reward strategy for this position
                reward = self._query_reward_at_position(x, y)
                rewards[idx] = reward
                idx += 1

        # Normalize rewards to colors
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)

        idx = 0
        for row in range(self._grid_cells):
            for col in range(self._grid_cells):
                # Map reward to color
                color = self._reward_to_color(rewards[idx], min_reward, max_reward)

                # Set color for all 4 vertices of this quad
                for i in range(4):
                    self._colors_view[idx*4 + i] = color

                idx += 1

        # Update vertex list colors
        if self._heatmap_vlist is not None:
            try:
                self._heatmap_vlist.color[:] = self._colors_array
            except Exception:
                pass

    def _query_reward_at_position(self, x: float, y: float) -> float:
        """Query reward strategy for attacker at given position.

        Args:
            x: World X coordinate
            y: World Y coordinate

        Returns:
            Spatial reward component (pinch, distance, heading)
        """
        # Construct mock observation with attacker at this position
        # Heading: face toward target for consistent evaluation
        dx = self._target_x - x
        dy = self._target_y - y
        theta = np.arctan2(dy, dx)

        attacker_obs = {
            'pose': np.array([x, y, theta]),
            'scans': np.zeros(720),  # Dummy LiDAR (not used for spatial rewards)
        }

        # Construct step_info for reward calculation
        step_info = {
            'obs': attacker_obs,
            'target_obs': self._target_obs,
            'done': False,
            'truncated': False,
            'info': {},
            'timestep': 0.01,
        }

        # Query reward strategy
        _, components = self._reward_strategy.compute(step_info)

        # Extract spatial components (pinch pockets, distance, heading)
        spatial_reward = 0.0
        for key, value in components.items():
            # Include spatial reward components
            if any(kw in key for kw in ['forcing/pinch', 'distance', 'heading']):
                spatial_reward += value

        return spatial_reward

    def _reward_to_color(self, reward: float, min_reward: float, max_reward: float) -> tuple:
        """Convert reward value to RGBA color.

        Args:
            reward: Reward value
            min_reward: Minimum reward in dataset
            max_reward: Maximum reward in dataset

        Returns:
            RGBA tuple (0-1 range)
        """
        # Normalize reward to 0-1 range
        if max_reward > min_reward:
            normalized = (reward - min_reward) / (max_reward - min_reward)
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            normalized = 0.5

        # Color map: red (low) -> yellow (mid) -> green (high)
        if normalized < 0.5:
            # Red to yellow
            t = normalized * 2.0
            r = 1.0
            g = t
            b = 0.0
        else:
            # Yellow to green
            t = (normalized - 0.5) * 2.0
            r = 1.0 - t
            g = 1.0
            b = 0.0

        return (r, g, b, self._alpha)

    def _update_heatmap_positions(self):
        """Update heatmap cell positions centered on target."""
        if self._heatmap_vlist is None or self._positions_view is None:
            return

        # Offset all cells by target position
        scale = self.renderer.render_scale
        offset_x = self._target_x * scale
        offset_y = self._target_y * scale

        # Update positions in the array buffer
        idx = 0
        for row in range(self._grid_cells):
            for col in range(self._grid_cells):
                # Cell corners in local grid coordinates
                x0 = -self._extent_m + col * self._cell_size_m
                y0 = -self._extent_m + row * self._cell_size_m
                x1 = x0 + self._cell_size_m
                y1 = y0 + self._cell_size_m

                # Convert to render coordinates and offset by target position
                # Bottom-left
                self._positions_view[idx*8 + 0] = (x0 * scale) + offset_x
                self._positions_view[idx*8 + 1] = (y0 * scale) + offset_y
                # Bottom-right
                self._positions_view[idx*8 + 2] = (x1 * scale) + offset_x
                self._positions_view[idx*8 + 3] = (y0 * scale) + offset_y
                # Top-right
                self._positions_view[idx*8 + 4] = (x1 * scale) + offset_x
                self._positions_view[idx*8 + 5] = (y1 * scale) + offset_y
                # Top-left
                self._positions_view[idx*8 + 6] = (x0 * scale) + offset_x
                self._positions_view[idx*8 + 7] = (y1 * scale) + offset_y

                idx += 1

        # Flush to vertex list
        try:
            self._heatmap_vlist.position[:] = self._positions_array
        except Exception:
            pass

    def draw_geometry(self, batch, shader_group):
        """Draw heatmap (geometry already in batch).

        Args:
            batch: Rendering batch (heatmap already added)
            shader_group: Shader group (unused)
        """
        # Heatmap is drawn automatically via batch
        pass

    def cleanup(self):
        """Clean up heatmap resources."""
        if self._heatmap_vlist is not None:
            self._heatmap_vlist.delete()
            self._heatmap_vlist = None

        self._positions_array = None
        self._positions_view = None
        self._colors_array = None
        self._colors_view = None
