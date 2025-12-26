"""Helper utilities for integration tests."""
from typing import Any, Dict
import numpy as np
from v2.wrappers.observation import ObsWrapper


class SimpleObservationWrapper:
    """Simple gym-style wrapper that flattens dict observations to arrays.

    This wrapper transforms the F110ParallelEnv's dict observations
    (with keys like 'scans', 'poses_x', etc.) into flat numpy arrays
    that agents can process.
    """

    def __init__(self, env: Any):
        """Wrap environment with observation transformation.

        Args:
            env: F110ParallelEnv instance
        """
        self.env = env
        # Create ObsWrapper with default configuration
        self.obs_wrapper = ObsWrapper(
            max_scan=30.0,
            normalize=True,
            components=['lidar', 'pose', 'velocity']  # Basic components
        )

    def reset(self, **kwargs):
        """Reset environment and transform observations."""
        obs_dict, info = self.env.reset(**kwargs)
        # Transform dict observations to flat arrays for each agent
        flat_obs = {}
        for agent_id in obs_dict.keys():
            flat_obs[agent_id] = self.obs_wrapper(obs_dict, agent_id)
        return flat_obs, info

    def step(self, actions: Dict[str, np.ndarray]):
        """Step environment and transform observations."""
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
        # Transform dict observations to flat arrays for each agent
        flat_obs = {}
        for agent_id in obs_dict.keys():
            flat_obs[agent_id] = self.obs_wrapper(obs_dict, agent_id)
        return flat_obs, rewards, terminations, truncations, infos

    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)
