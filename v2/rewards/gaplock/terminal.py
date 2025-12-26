"""Terminal rewards for gaplock task.

Handles episode-ending outcomes with appropriate rewards/penalties.
"""

from typing import Dict


class TerminalReward:
    """Terminal rewards for episode outcomes.

    Provides rewards/penalties based on how the episode ends:
    - target_crash: Target crashed, attacker survived (SUCCESS)
    - self_crash: Attacker crashed alone (FAILURE)
    - collision: Both crashed (FAILURE)
    - timeout: Max steps reached (FAILURE)
    - idle_stop: Attacker stopped moving (FAILURE)
    - target_finish: Target crossed finish line (FAILURE)

    Only fires on done=True steps.
    """

    def __init__(self, config: dict):
        """Initialize terminal rewards.

        Args:
            config: Dict with keys:
                - target_crash: Reward when target crashes (default: 60.0)
                - self_crash: Penalty when attacker crashes (default: -90.0)
                - collision: Penalty for mutual collision (default: -90.0)
                - timeout: Penalty for timeout (default: -20.0)
                - idle_stop: Penalty for idle truncation (default: -5.0)
                - target_finish: Penalty when target finishes (default: -20.0)
        """
        self.target_crash = float(config.get('target_crash', 60.0))
        self.self_crash = float(config.get('self_crash', -90.0))
        self.collision = float(config.get('collision', -90.0))
        self.timeout = float(config.get('timeout', -20.0))
        self.idle_stop = float(config.get('idle_stop', -5.0))
        self.target_finish = float(config.get('target_finish', -20.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute terminal reward if episode is done.

        Args:
            step_info: Must contain:
                - done: Whether episode ended
                - info: Info dict from environment with outcome info

        Returns:
            Dict with terminal reward (empty if not done)
        """
        if not step_info.get('done', False):
            return {}

        info = step_info.get('info', {})

        # Determine outcome from info dict
        # Priority: finish > collision > crashes > idle > timeout

        # Check if target finished (highest priority success for target)
        if info.get('target_finished', False) or info.get('car_1/finished', False):
            return {'terminal/target_finish': self.target_finish}

        # Check collision state
        attacker_crashed = info.get('collision', False) or info.get('car_0/collision', False)
        target_crashed = info.get('target_collision', False) or info.get('car_1/collision', False)

        if attacker_crashed and target_crashed:
            # Both crashed - mutual collision
            return {'terminal/collision': self.collision}

        if target_crashed:
            # Target crashed, attacker survived - SUCCESS!
            return {'terminal/success': self.target_crash}

        if attacker_crashed:
            # Attacker crashed alone
            return {'terminal/self_crash': self.self_crash}

        # Check for idle truncation
        if info.get('idle_triggered', False) or info.get('idle_truncation', False):
            return {'terminal/idle_stop': self.idle_stop}

        # Default to timeout if episode ended but no specific outcome detected
        if step_info.get('truncated', False) or info.get('truncated', False):
            return {'terminal/timeout': self.timeout}

        # Episode done but no clear outcome - treat as timeout
        return {'terminal/timeout': self.timeout}


__all__ = ['TerminalReward']
