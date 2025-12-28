"""Reward task registry and built-in strategies."""

from .base import PerAgentStateMixin, RewardRuntimeContext, RewardStep, RewardStrategy
from .components import RewardAccumulator
from .registry import (
    RewardTaskConfig,
    RewardTaskFactory,
    RewardTaskRegistry,
    RewardTaskSpec,
    migrate_reward_config,
    register_reward_task,
    resolve_reward_task,
    reward_task_registry,
)
from .presets import REWARD_FEATURE_PRESETS

# Ensure built-in tasks are registered
from . import composite as _composite  # noqa: F401
from . import fastest_lap as _fastest_lap  # noqa: F401
from . import gaplock as _gaplock  # noqa: F401
from . import kamikaze as _kamikaze  # noqa: F401
from . import progress as _progress  # noqa: F401

from .fastest_lap import FASTEST_LAP_PARAM_KEYS, FastestLapRewardStrategy
from .gaplock import GAPLOCK_PARAM_KEYS, GaplockRewardStrategy
from .kamikaze import KAMIKAZE_PARAM_KEYS, KamikazeRewardStrategy
from .progress import PROGRESS_PARAM_KEYS, ProgressRewardStrategy

__all__ = [
    "FASTEST_LAP_PARAM_KEYS",
    "GAPLOCK_PARAM_KEYS",
    "KAMIKAZE_PARAM_KEYS",
    "PROGRESS_PARAM_KEYS",
    "FastestLapRewardStrategy",
    "GaplockRewardStrategy",
    "KamikazeRewardStrategy",
    "ProgressRewardStrategy",
    "PerAgentStateMixin",
    "RewardAccumulator",
    "RewardRuntimeContext",
    "RewardStep",
    "RewardStrategy",
    "RewardTaskConfig",
    "RewardTaskFactory",
    "RewardTaskRegistry",
    "RewardTaskSpec",
    "migrate_reward_config",
    "register_reward_task",
    "resolve_reward_task",
    "reward_task_registry",
    "REWARD_FEATURE_PRESETS",
]
