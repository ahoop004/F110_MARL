import numpy as np
import math

class ObservationWrapper:

    def __init__(self, lidar_max_range: float, bounds: dict, vel_max: float = 20.0, lidar_reduce_factor: int = 6):
        # Basic checks
        if lidar_max_range <= 0:
            raise ValueError("lidar_max_range must be > 0")
        self.lidar_max_range = float(lidar_max_range)

        self.x_min = float(bounds["x_min"]); self.x_max = float(bounds["x_max"])
        self.y_min = float(bounds["y_min"]); self.y_max = float(bounds["y_max"])
        if self.x_max <= self.x_min:
            raise ValueError("Invalid x bounds: x_max must be > x_min")
        if self.y_max <= self.y_min:
            raise ValueError("Invalid y bounds: y_max must be > y_min")

        self._x_span = self.x_max - self.x_min
        self._y_span = self.y_max - self.y_min

        if vel_max <= 0:
            raise ValueError("vel_max must be > 0")
        self.vel_max = float(vel_max)

        if lidar_reduce_factor < 1:
            raise ValueError("lidar_reduce_factor must be >= 1")
        self.lidar_reduce_factor = int(lidar_reduce_factor)

    def _reduce_lidar(self, scan: np.ndarray) -> np.ndarray:
        # Fixed subsample: every k-th beam
        if self.lidar_reduce_factor <= 1:
            return scan.astype(np.float32, copy=False)
        return scan[::self.lidar_reduce_factor].astype(np.float32, copy=False)

    def build(self, observations: dict) -> np.ndarray:

        ego = int(observations["ego_idx"])

        scans      = np.asarray(observations["scans"], dtype=np.float32)
        poses_x    = np.asarray(observations["poses_x"], dtype=np.float32)
        poses_y    = np.asarray(observations["poses_y"], dtype=np.float32)
        poses_th   = np.asarray(observations["poses_theta"], dtype=np.float32)
        collisions = np.asarray(observations["collisions"], dtype=np.float32)

        if scans.ndim != 2:
            raise ValueError("scans must be (N_agents, N_beams)")
        num_agents, n_beams = scans.shape
        if n_beams <= 0:
            raise ValueError("scans has zero beams")
        if num_agents != 2:
            raise ValueError(f"ObservationWrapper assumes exactly 2 agents, got {num_agents}")
        if not (0 <= ego < num_agents):
            raise IndexError("ego_idx out of range")

        # Ego/opp indices (2-agent assumption)
        opp = 1 - ego

        # LiDAR (ego) → [0,1], then reduce
        lidar_raw = np.clip(scans[ego] / self.lidar_max_range, 0.0, 1.0)
        lidar = self._reduce_lidar(lidar_raw)

        # Ego kinematics
        ex = float(poses_x[ego]); ey = float(poses_y[ego]); et = float(poses_th[ego])
        ego_crashed = float(collisions[ego])

        # Opponent kinematics
        ox = float(poses_x[opp]); oy = float(poses_y[opp]); ot = float(poses_th[opp])
        opp_crashed = float(collisions[opp])

        # Normalize positions to [-1,1]
        ex_n = np.clip(2.0 * (ex - self.x_min) / self._x_span - 1.0, -1.0, 1.0)
        ey_n = np.clip(2.0 * (ey - self.y_min) / self._y_span - 1.0, -1.0, 1.0)
        ox_n = np.clip(2.0 * (ox - self.x_min) / self._x_span - 1.0, -1.0, 1.0)
        oy_n = np.clip(2.0 * (oy - self.y_min) / self._y_span - 1.0, -1.0, 1.0)

        # Normalize angles to [-1,1] via /pi
        et_n = np.clip(et / math.pi, -1.0, 1.0)
        ot_n = np.clip(ot / math.pi, -1.0, 1.0)

        tail = np.array(
            [ex_n, ey_n, et_n, ego_crashed,
             ox_n, oy_n, ot_n,  opp_crashed],
            dtype=np.float32,
        )
        return np.concatenate([lidar, tail], axis=0).astype(np.float32, copy=False)
