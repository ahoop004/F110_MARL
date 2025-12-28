"""Step penalty component for gaplock task.

Applies a constant reward/penalty every timestep to encourage faster completion.
"""

from typing import Dict


class StepPenalty:
    """Constant per-step penalty to encourage faster episode completion.

    Typically configured as a small negative value (e.g., -0.01) to
    incentivize the agent to complete the task quickly.

    Example:
        >>> penalty = StepPenalty({'step_reward': -0.01})
        >>> rewards = penalty.compute({})
        >>> rewards['penalties/step']
        -0.01
    """

    def __init__(self, config: dict):
        """Initialize step penalty.

        Args:
            config: Dict with keys:
                - step_reward: Reward value per step (default: 0.0)
                  Use negative values for penalty (e.g., -0.01)
                - enabled: Whether enabled (default: True if step_reward != 0)
        """
        self.step_reward = float(config.get('step_reward', 0.0))
        self.enabled = bool(config.get('enabled', self.step_reward != 0.0))

    def reset(self) -> None:
        """Reset component for new episode (no-op for stateless component)."""
        pass

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute step penalty.

        Args:
            step_info: Step information (not used for this component)

        Returns:
            Dict with single key 'penalties/step' containing the step reward value
        """
        if not self.enabled or self.step_reward == 0.0:
            return {}

        return {'penalties/step': self.step_reward}


__all__ = ['StepPenalty']
