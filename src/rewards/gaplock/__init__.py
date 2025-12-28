"""Gaplock adversarial task rewards.

This module implements the reward system for the gaplock task, where an
attacker agent tries to force a defender (FTG baseline) to crash without
crashing itself.
"""

from rewards.gaplock.gaplock import GaplockReward
from rewards.gaplock.terminal import TerminalReward
from rewards.gaplock.pressure import PressureReward
from rewards.gaplock.distance import DistanceReward
from rewards.gaplock.heading import HeadingReward
from rewards.gaplock.speed import SpeedReward
from rewards.gaplock.forcing import ForcingReward
from rewards.gaplock.penalties import BehaviorPenalties
from rewards.gaplock.step_penalty import StepPenalty

__all__ = [
    'GaplockReward',
    'TerminalReward',
    'PressureReward',
    'DistanceReward',
    'HeadingReward',
    'SpeedReward',
    'ForcingReward',
    'BehaviorPenalties',
    'StepPenalty',
]
