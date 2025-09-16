import numpy as np

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=0.01):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.zeros(self.size, dtype=np.float32)

    def reset(self):
        """Reset state to zero (or mu). Call at the start of each episode."""
        self.state = np.zeros(self.size, dtype=np.float32)

    def sample(self):
        """Get the next noise vector."""
        dx = self.theta * (self.mu - self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state = self.state + dx
        return self.state
