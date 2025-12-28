"""V2 rendering system - minimal core with optional extensions."""
from .shader import get_default_shader
from .renderer import EnvRenderer

# Optional extensions
from .extensions.base import RenderExtension
from .extensions.hud import MinimalHUD
from .extensions.reward_ring import RewardRingExtension
from .extensions.telemetry import TelemetryHUD
from .extensions.heatmap import RewardHeatmap

__all__ = [
    'EnvRenderer',
    'get_default_shader',
    'RenderExtension',
    'MinimalHUD',
    'RewardRingExtension',
    'TelemetryHUD',
    'RewardHeatmap',
]
