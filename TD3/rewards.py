import math
import numpy as np
from collections import deque

class RewardWrapper:
    """
    Goal: make the opp crash while keeping ego safe & stable.

    Terms:
      + crash_opp_bonus   (ONLY if opp newly crashes AND ego was recently close)
      - crash_ego_penalty (once, when ego crashes)
      + progress_gain * forward_progress_m
      - reverse_gain  * backward_progress_m
      - alive_cost
      - spin_penalty (large yaw change while barely moving)
      - steer_speed_lambda * |steer_norm| * |speed_norm|     [stability]
      - lambda_dsteer * |Δ steer_norm| (between steps)       [smoothness]
    """

    def __init__(
        # Core weights
        self,
        alive_cost: float = 0.002,
        crash_ego_penalty: float = -60.0,   # start mild; raise to -120 once stable
        crash_opp_bonus: float = 100.0,
        progress_gain: float = 10.0,
        reverse_gain: float = 5.0,

        # Stagnation detector
        no_progress_window: int = 10,
        no_progress_eps: float = 0.15,
        no_progress_penalty: float = -3.0,

        # Spin deterrent
        spin_yaw_thresh: float = 0.20,  # rad/step
        spin_move_eps: float = 0.01,    # m/step
        spin_penalty: float = -0.5,

        # Causal gate for opp crash
        cause_window: int = 100,         # steps to look back
        cause_dist_thresh: float = 12.5, # meters (min recent ego–opp distance)

        # Stability terms (normalized action space: [-1,1]^2)
        steer_speed_lambda: float = 0.05,  # |steer|*|speed| cost
        lambda_dsteer: float = 0.01,       # |Δ steer| cost between steps
    ):
        # Store weights
        self.alive_cost = float(alive_cost)
        self.crash_ego_penalty = float(crash_ego_penalty)
        self.crash_opp_bonus = float(crash_opp_bonus)
        self.progress_gain = float(progress_gain)
        self.reverse_gain = float(reverse_gain)

        self.no_progress_window = int(no_progress_window)
        self.no_progress_eps = float(no_progress_eps)
        self.no_progress_penalty = float(no_progress_penalty)

        self.spin_yaw_thresh = float(spin_yaw_thresh)
        self.spin_move_eps = float(spin_move_eps)
        self.spin_penalty = float(spin_penalty)

        # self.cause_window = int(cause_window)
        # self.cause_dist_thresh = float(cause_dist_thresh)

        self.steer_speed_lambda = float(steer_speed_lambda)
        self.lambda_dsteer = float(lambda_dsteer)

        # Internal state
        self._prev_pose = None                 # (x, y, theta)
        self._prev_collisions = None          # np.array([ego, opp])
        self._prev_act = None                 # last normalized action
        self._ds_hist = deque(maxlen=self.no_progress_window)
        # self._dist_hist = deque(maxlen=self.cause_window)  # recent ego–opp dists
        self.opp_crashed_now = False

    # ---- Public API ----
    def reset(self, observations: dict) -> None:
        ego = int(observations["ego_idx"])
        x = float(observations["poses_x"][ego])
        y = float(observations["poses_y"][ego])
        th = float(observations["poses_theta"][ego])
        self._prev_pose = (x, y, th)

        try:
            c = np.asarray(observations.get("collisions", None), dtype=np.int8)
            if c.ndim == 0:
                c = np.array([int(c)], dtype=np.int8)
            self._prev_collisions = c.copy()
        except Exception:
            self._prev_collisions = None

        self._prev_act = None
        self._ds_hist.clear()
        # self._dist_hist.clear()
        # seed distance history if possible
        try:
            opp = 1 - ego
            xo = float(observations["poses_x"][opp])
            yo = float(observations["poses_y"][opp])
            self._dist_hist.append(self._euclid(x, y, xo, yo))
        except Exception:
            pass

        self.opp_crashed_now = False

    def compute(self, observations: dict, action=None) -> float:
        ego = int(observations["ego_idx"]); opp = 1 - ego
        curr_col = np.asarray(observations["collisions"], dtype=np.int8)
        ego_crash = bool(curr_col[ego])
        new_opp_crash = (self._prev_collisions is not None and curr_col[opp] == 1 and self._prev_collisions[opp] == 0)

        r = 1.0 if (new_opp_crash and not ego_crash) else 0.0
        self.opp_crashed_now = bool(new_opp_crash and not ego_crash)
        
        if ego_crash:
            r = -0.7 

        # update prev collisions
        self._prev_collisions = curr_col.copy()
        return r

    # ---- helpers ----
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Return signed smallest angle difference a-b in [-pi, pi]."""
        d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return d

    @staticmethod
    def _euclid(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)
