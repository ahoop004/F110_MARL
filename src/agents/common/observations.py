"""Per-learner observations with padding confined to batched policy/storage I/O."""
from collections.abc import Mapping

import numpy as np


def pack_observations(agent_ids, observations, obs_dims, width):
    if not isinstance(observations, Mapping) and len(observations) != len(agent_ids):
        raise ValueError("Observation batch shape must match the agent IDs")
    rows = np.zeros((len(agent_ids), width), dtype=np.float32)
    for i, aid in enumerate(agent_ids):
        value = observations[aid] if isinstance(observations, Mapping) else observations[i]
        value = np.asarray(value, dtype=np.float32)
        size = obs_dims[aid]
        if value.shape == (width,) and size < width:
            if np.any(value[size:] != 0):
                raise ValueError(f"Nonzero observation padding for {aid}")
        elif value.shape != (size,):
            raise ValueError(f"Observation for {aid} must have shape ({size},), got {value.shape}")
        rows[i, :size] = value[:size]
    return rows
