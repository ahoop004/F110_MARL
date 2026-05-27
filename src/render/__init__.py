"""V2 rendering system - minimal core with optional extensions.

Imports are lazy so non-render helpers can be imported on headless systems
without initializing pyglet.
"""

__all__ = [
    'EnvRenderer',
    'get_default_shader',
    'RenderExtension',
    'MinimalHUD',
    'RewardRingExtension',
    'TelemetryHUD',
    'RewardHeatmap',
]


def __getattr__(name: str):
    if name == "get_default_shader":
        from .shader import get_default_shader
        return get_default_shader
    if name == "EnvRenderer":
        from .renderer import EnvRenderer
        return EnvRenderer
    if name == "RenderExtension":
        from .extensions.base import RenderExtension
        return RenderExtension
    if name == "MinimalHUD":
        from .extensions.hud import MinimalHUD
        return MinimalHUD
    if name == "RewardRingExtension":
        from .extensions.reward_ring import RewardRingExtension
        return RewardRingExtension
    if name == "TelemetryHUD":
        from .extensions.telemetry import TelemetryHUD
        return TelemetryHUD
    if name == "RewardHeatmap":
        from .extensions.heatmap import RewardHeatmap
        return RewardHeatmap
    raise AttributeError(f"module 'render' has no attribute {name!r}")
