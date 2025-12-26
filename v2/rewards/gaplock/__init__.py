"""Gaplock adversarial task rewards.

This module implements the reward system for the gaplock task, where an
attacker agent tries to force a defender (FTG baseline) to crash without
crashing itself.
"""

from v2.rewards.gaplock.gaplock import GaplockReward
from v2.rewards.gaplock.terminal import TerminalReward
from v2.rewards.gaplock.pressure import PressureReward
from v2.rewards.gaplock.distance import DistanceReward
from v2.rewards.gaplock.heading import HeadingReward
from v2.rewards.gaplock.speed import SpeedReward
from v2.rewards.gaplock.forcing import ForcingReward
from v2.rewards.gaplock.penalties import BehaviorPenalties

__all__ = [
    'GaplockReward',
    'TerminalReward',
    'PressureReward',
    'DistanceReward',
    'HeadingReward',
    'SpeedReward',
    'ForcingReward',
    'BehaviorPenalties',
]
