"""Episode outcome classification for gaplock task.

Defines the 6 mutually exclusive episode outcomes and provides
logic to determine which outcome occurred based on episode info.
"""

from enum import Enum
from typing import Dict, Any


class EpisodeOutcome(Enum):
    """Episode outcome types for gaplock adversarial task.

    From the attacker's perspective:
    - TARGET_CRASH: Success - target crashed into wall
    - SELF_CRASH: Failure - attacker crashed into wall
    - COLLISION: Failure - both cars collided
    - TIMEOUT: Failure - episode exceeded max steps
    - IDLE_STOP: Failure - attacker was idle too long
    - TARGET_FINISH: Failure - target reached finish line

    These outcomes are mutually exclusive with the following priority:
    1. TARGET_FINISH (highest priority)
    2. COLLISION (both crashed)
    3. TARGET_CRASH (attacker success)
    4. SELF_CRASH (attacker solo crash)
    5. IDLE_STOP (idle timeout)
    6. TIMEOUT (step limit)
    """
    TARGET_CRASH = "target_crash"
    SELF_CRASH = "self_crash"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    IDLE_STOP = "idle_stop"
    TARGET_FINISH = "target_finish"

    def is_success(self) -> bool:
        """Check if outcome represents attacker success."""
        return self == EpisodeOutcome.TARGET_CRASH

    def is_failure(self) -> bool:
        """Check if outcome represents attacker failure."""
        return not self.is_success()


def determine_outcome(info: Dict[str, Any], truncated: bool = False) -> EpisodeOutcome:
    """Determine episode outcome from environment info.

    Args:
        info: Episode info dict from environment, should contain:
            - collision: bool (attacker crashed)
            - target_collision: bool (target crashed)
            - target_finished: bool (target crossed finish line)
            - idle_truncation: bool (attacker was idle too long)
        truncated: Whether episode was truncated (timeout)

    Returns:
        EpisodeOutcome enum value

    Priority order (from highest to lowest):
    1. TARGET_FINISH - Target crossed finish line
    2. COLLISION - Both cars crashed
    3. TARGET_CRASH - Only target crashed (attacker success!)
    4. SELF_CRASH - Only attacker crashed
    5. IDLE_STOP - Attacker was idle too long
    6. TIMEOUT - Episode exceeded max steps

    Example:
        >>> info = {'collision': False, 'target_collision': True}
        >>> determine_outcome(info)
        <EpisodeOutcome.TARGET_CRASH: 'target_crash'>

        >>> info = {'collision': True, 'target_collision': True}
        >>> determine_outcome(info)
        <EpisodeOutcome.COLLISION: 'collision'>
    """
    # Check for target finish (highest priority)
    if info.get('target_finished', False):
        return EpisodeOutcome.TARGET_FINISH

    # Check collision states
    attacker_crashed = info.get('collision', False)
    target_crashed = info.get('target_collision', False)

    # Both crashed = collision (attacker failure)
    if attacker_crashed and target_crashed:
        return EpisodeOutcome.COLLISION

    # Only target crashed = success!
    if target_crashed:
        return EpisodeOutcome.TARGET_CRASH

    # Only attacker crashed = failure
    if attacker_crashed:
        return EpisodeOutcome.SELF_CRASH

    # Check idle truncation
    if info.get('idle_truncation', False):
        return EpisodeOutcome.IDLE_STOP

    # If truncated but none of the above, assume timeout
    if truncated:
        return EpisodeOutcome.TIMEOUT

    # Default to timeout if episode ended without clear outcome
    return EpisodeOutcome.TIMEOUT


__all__ = ['EpisodeOutcome', 'determine_outcome']
