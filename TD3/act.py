import numpy as np
from typing import Optional

class ActionWrapper:
    """
    Minimal direct mapper: a in [-1,1]^2 -> [steer, speed] in env units.
    No rates, no smoothing, no helpers.
    """

    def __init__(
        self,
        steer_min: float,
        steer_max: float,
        vel_min: float,
        vel_max: float,
        *,
        clip: bool = True,
        allow_reverse: bool = False,
    ):
        if not (steer_max > steer_min): raise ValueError("steer_max must be > steer_min")
        if not (vel_max > vel_min):     raise ValueError("vel_max must be > vel_min")
        self.steer_min = float(steer_min)
        self.steer_max = float(steer_max)
        self.vel_min   = float(vel_min)
        self.vel_max   = float(vel_max)
        self.clip = bool(clip)
        self.allow_reverse = bool(allow_reverse)

        # Precompute centers/halfspans
        self._steer_c = 0.5 * (self.steer_min + self.steer_max)
        self._steer_h = 0.5 * (self.steer_max - self.steer_min)
        self._vel_c   = 0.5 * (self.vel_min   + self.vel_max)
        self._vel_h   = 0.5 * (self.vel_max   - self.vel_min)

    def reset(self) -> None:
        pass  # no state

    def build(self, action, *, can_choose: Optional[bool] = None):
        """Return np.array([steer, speed], dtype=float32)."""
        a = self._to_np(action)
        if a.ndim == 2 and a.shape == (1, 2):
            a = a[0]
        if a.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("Action contains NaN/Inf")

        steer = self._steer_c + self._steer_h * float(a[0])
        speed = self._vel_c   + self._vel_h   * float(a[1])

        if self.clip:
            steer = float(np.clip(steer, self.steer_min, self.steer_max))
            vmin = self.vel_min if self.allow_reverse else max(0.0, self.vel_min)
            speed = float(np.clip(speed, vmin, self.vel_max))

        return np.array([steer, speed], dtype=np.float32)

    @staticmethod
    def _to_np(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(x, dtype=np.float32)
