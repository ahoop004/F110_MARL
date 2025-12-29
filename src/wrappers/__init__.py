"""Public wrapper utilities."""

from wrappers.observation import ObsWrapper

__all__ = ["ObsWrapper"]

# DEPRECATED: RewardWrapper and RewardRuntimeContext have been removed.
# The old task-based reward system (src/tasks/reward/) has been removed.
#
# Use the new component-based reward system instead:
#   from rewards import build_reward_strategy
#
# See docs/REWARD_SYSTEM_REMOVAL.md for migration instructions.
