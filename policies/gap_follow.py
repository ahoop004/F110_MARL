import numpy as np

class FollowTheGapPolicy:
    def __init__(self,
                 max_distance=30.0,   # actual sensor range (m)
                 window_size=5,
                 bubble_radius=10,
                 max_steer=0.4,
                 min_speed=2.0,
                 max_speed=6.0,
                 steering_gain=1.0,
                 fov=np.deg2rad(270),
                 normalized=False):
        self.max_distance = max_distance
        self.window_size = window_size
        self.bubble_radius = bubble_radius
        self.max_steer = max_steer
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.steering_gain = steering_gain
        self.fov = fov
        self.normalized = normalized

    def preprocess_lidar(self, ranges):
        """Smooth LiDAR with moving average + clip to max_distance."""
        N = len(ranges)
        half = self.window_size // 2
        proc = []
        for i in range(N):
            start = max(0, i - half)
            end = min(N - 1, i + half)
            avg = np.mean(np.clip(ranges[start:end+1], 0, self.max_distance))
            proc.append(avg)
        return np.array(proc)

    def create_bubble(self, proc):
        """Zero out a bubble around the closest obstacle."""
        closest = np.argmin(proc)
        start = max(0, closest - self.bubble_radius)
        end = min(len(proc) - 1, closest + self.bubble_radius)
        proc[start:end+1] = 0
        return proc

    def find_max_gap(self, proc):
        """Find the largest contiguous nonzero gap."""
        gaps, start = [], None
        for i, v in enumerate(proc > 2.5):
            if v and start is None:
                start = i
            elif not v and start is not None:
                gaps.append((start, i-1))
                start = None
        if start is not None:
            gaps.append((start, len(proc)-1))
        if not gaps:
            return 0, len(proc)-1
        return max(gaps, key=lambda g: g[1]-g[0])

    def best_point_midgap(self, gap):
        """Return the midpoint of the widest gap."""
        return (gap[0] + gap[1]) // 2

    def get_action(self, action_space, obs: dict):
        scan = np.array(obs["scans"])
        if self.normalized:
            scan = scan * self.max_distance

        N = len(scan)
        center_idx = N // 2

        # 1. Base gap-following
        proc = self.preprocess_lidar(scan)
        proc = self.create_bubble(proc)
        gap = self.find_max_gap(proc)
        best = self.best_point_midgap(gap)
        offset = (best - center_idx) / center_idx
        steering = offset * self.steering_gain * self.max_steer

        # 2. Sector-based danger weighting
        left_min = np.min(scan[:center_idx]) if center_idx > 0 else np.inf
        right_min = np.min(scan[center_idx:]) if center_idx < N else np.inf
        min_scan = float(np.min(scan))

        # Dynamic panic scaling
        panic_factor = 1.0
        if min_scan < 2.0:
            panic_factor = 2.0
        if min_scan < 1.0:
            panic_factor = 3.0

        steering *= panic_factor

        # Kick away from closest side
        if left_min < right_min:
            steering += 0.5 * self.max_steer
        elif right_min < left_min:
            steering -= 0.5 * self.max_steer

        # Override when extremely close: pure evasive
        if min_scan < 1.0:
            if left_min < right_min:
                steering = +self.max_steer
            else:
                steering = -self.max_steer

        steering = np.clip(steering, -self.max_steer, self.max_steer)

        # 3. Speed schedule
        free_ahead = scan[center_idx]
        if min_scan < 2.0:
            speed = self.min_speed
        elif free_ahead > 6.0:
            speed = self.max_speed
        elif free_ahead > 2.0:
            speed = 0.7 * self.max_speed
        else:
            speed = self.min_speed

        action = np.array([steering, speed], dtype=np.float32)
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        return action
