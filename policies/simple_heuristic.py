import numpy as np
def simple_heuristic(action_space,obs):
    # Example: small forward velocity, no steer
    return np.array([0.0, 1.0], dtype=np.float32)