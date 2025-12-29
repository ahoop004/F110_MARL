"""Scenario task registries (DEPRECATED)."""

# DEPRECATED: The old task-based reward system has been removed.
#
# The src/tasks/reward/ module has been completely removed in favor of
# the newer component-based reward system in src/rewards/.
#
# Migration:
#   OLD: from tasks.reward import resolve_reward_task
#   NEW: from rewards import build_reward_strategy
#
#   OLD config format:
#     reward:
#       task: gaplock
#       params: {...}
#
#   NEW config format:
#     reward:
#       preset: gaplock_full
#       overrides: {...}
#
# See docs/REWARD_SYSTEM_REMOVAL.md for detailed migration guide.

__all__ = []
