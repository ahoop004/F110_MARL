"""Helper functions for setting up visualization extensions."""

from typing import Optional, Dict, Any
from render import TelemetryHUD, RewardRingExtension, RewardHeatmap


def setup_default_extensions(
    env,
    telemetry: bool = True,
    telemetry_mode: int = 2,  # BASIC mode
    reward_ring: bool = True,
    reward_ring_config: Optional[Dict[str, Any]] = None,
    reward_heatmap: bool = False,
    heatmap_config: Optional[Dict[str, Any]] = None,
):
    """Add visualization extensions to environment renderer.

    This function should be called after creating the environment but before
    training starts. It uses the render callback system to add extensions
    when the renderer is first created.

    Args:
        env: F110ParallelEnv instance
        telemetry: Enable telemetry HUD (default: True)
        telemetry_mode: Initial telemetry mode (0-4, default: 2=BASIC)
        reward_ring: Enable reward ring visualization (default: True)
        reward_ring_config: Config dict for reward ring (optional)
        reward_heatmap: Enable reward heatmap (default: False, expensive)
        heatmap_config: Config dict for heatmap (optional)

    Example:
        env = F110ParallelEnv(...)
        setup_default_extensions(
            env,
            telemetry=True,
            telemetry_mode=TelemetryHUD.MODE_DETAILED,
            reward_ring=True,
            reward_heatmap=False
        )
    """
    extensions_to_add = []

    # Configure telemetry
    if telemetry:
        def add_telemetry(renderer):
            # Check if already added
            for ext in renderer._extensions:
                if ext.__class__.__name__ == 'TelemetryHUD':
                    return

            telem = TelemetryHUD(renderer)
            telem.configure(enabled=True, mode=telemetry_mode)
            renderer.add_extension(telem)
            print(f"✓ Added TelemetryHUD (mode: {telemetry_mode})")

        extensions_to_add.append(add_telemetry)

    # Configure reward ring
    if reward_ring:
        def add_ring(renderer):
            # Check if already added
            for ext in renderer._extensions:
                if ext.__class__.__name__ == 'RewardRingExtension':
                    return

            config = reward_ring_config or {}
            # Default config from gaplock
            default_config = {
                'enabled': True,
                'target_agent': 'car_1',
                'inner_radius': 1.0,
                'outer_radius': 2.5,
                'preferred_radius': 1.5,
            }
            default_config.update(config)

            ring = RewardRingExtension(renderer)
            ring.configure(**default_config)
            renderer.add_extension(ring)
            print(f"✓ Added RewardRing (target: {default_config['target_agent']})")

        extensions_to_add.append(add_ring)

    # Configure heatmap
    if reward_heatmap:
        def add_heatmap(renderer):
            # Check if already added
            for ext in renderer._extensions:
                if ext.__class__.__name__ == 'RewardHeatmap':
                    return

            config = heatmap_config or {}
            default_config = {
                'enabled': True,
                'target_agent': 'car_1',
                'attacker_agent': 'car_0',
                'extent_m': 6.0,
                'cell_size_m': 0.25,
                'alpha': 0.22,
                'near_distance': 1.0,
                'far_distance': 2.5,
                'reward_near': 0.12,
                'penalty_far': 0.08,
                'update_frequency': 5,
            }
            default_config.update(config)

            hmap = RewardHeatmap(renderer)
            hmap.configure(**default_config)
            renderer.add_extension(hmap)
            print(f"✓ Added RewardHeatmap (extent: {default_config['extent_m']}m)")

        extensions_to_add.append(add_heatmap)

    # Add all extensions via callbacks
    for callback in extensions_to_add:
        env.add_render_callback(callback)

    print(f"ℹ Configured {len(extensions_to_add)} visualization extensions")
    print("  Press T to cycle telemetry, R for ring, H for heatmap")


def setup_gaplock_extensions(env, heatmap: bool = False):
    """Shortcut for gaplock task visualization.

    Args:
        env: F110ParallelEnv instance
        heatmap: Enable expensive heatmap (default: False)
    """
    from rewards.presets import load_preset

    # Load gaplock reward config for proper parameters
    try:
        reward_config = load_preset('gaplock_full')
        dist_config = reward_config['distance']

        ring_config = {
            'target_agent': 'car_1',
            'inner_radius': dist_config['near_distance'],
            'outer_radius': dist_config['far_distance'],
            'preferred_radius': 1.5,
        }

        heatmap_config = {
            'target_agent': 'car_1',
            'attacker_agent': 'car_0',
            'near_distance': dist_config['near_distance'],
            'far_distance': dist_config['far_distance'],
            'reward_near': dist_config['reward_near'],
            'penalty_far': dist_config['penalty_far'],
        }
    except Exception:
        # Fallback to defaults
        ring_config = None
        heatmap_config = None

    setup_default_extensions(
        env,
        telemetry=True,
        telemetry_mode=TelemetryHUD.MODE_BASIC,
        reward_ring=True,
        reward_ring_config=ring_config,
        reward_heatmap=heatmap,
        heatmap_config=heatmap_config,
    )
