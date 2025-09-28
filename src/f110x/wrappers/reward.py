import numpy as np

class RewardWrapper:
    def __init__(
        self,
        mode="basic",
        alive_bonus=0.02,
        forward_scale=0.15,
        reverse_penalty=1.5,
        lateral_penalty=0.05,
        target_distance_scale=0.1,
        target_wall_bonus=0.5,
        self_wall_penalty=0.3,
        herd_bonus=250.0,
        ego_collision_penalty=-40.0,
        opponent_collision_bonus=250.0,
        spin_thresh=np.pi / 6,
        spin_penalty=1.0,
    ):
        """Rich reward shaping for mixed pursuit/herding."""

        self.mode = mode
        self.alive_bonus = alive_bonus
        self.forward_scale = forward_scale
        self.reverse_penalty = reverse_penalty
        self.lateral_penalty = lateral_penalty
        self.target_distance_scale = target_distance_scale
        self.target_wall_bonus = target_wall_bonus
        self.self_wall_penalty = self_wall_penalty
        self.herd_bonus = herd_bonus
        self.ego_collision_penalty = ego_collision_penalty
        self.opponent_collision_bonus = opponent_collision_bonus
        self.spin_thresh = spin_thresh
        self.spin_penalty = spin_penalty

        self.prev_positions = {}
        self.prev_target_dist = {}
        self.opponent_crash_reward_given = set()

    def reset(self):
        self.prev_positions.clear()
        self.prev_target_dist.clear()
        self.opponent_crash_reward_given.clear()

    def _select_target_obs(self, agent_id, all_obs):
        if not all_obs:
            return None
        for other_id, other_obs in all_obs.items():
            if other_id != agent_id:
                return other_obs
        return None

    def __call__(self, obs, agent_id, reward, done, info, *, all_obs=None):
        ego_obs = obs[agent_id]
        pose = ego_obs["pose"]
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])

        prev_pos = self.prev_positions.get(agent_id, (x, y))
        dx, dy = x - prev_pos[0], y - prev_pos[1]
        self.prev_positions[agent_id] = (x, y)

        heading = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        disp = np.array([dx, dy], dtype=np.float32)
        forward_step = float(np.dot(heading, disp))
        lateral_step = float(np.cross(heading, disp))

        shaped = reward

        shaped += self.alive_bonus
        if forward_step > 0:
            shaped += self.forward_scale * forward_step
        else:
            shaped += self.reverse_penalty * forward_step  # negative value

        shaped -= self.lateral_penalty * abs(lateral_step)

        if ego_obs.get("collision", False):
            shaped += self.ego_collision_penalty

        if "velocity" in ego_obs:
            vx, vy = map(float, ego_obs["velocity"])
            speed = np.hypot(vx, vy)
            if speed < 0.5:
                shaped -= 0.2

        if "angular_velocity" in ego_obs:
            omega = abs(float(ego_obs["angular_velocity"]))
            if omega > self.spin_thresh and np.linalg.norm(disp) < 0.05:
                shaped -= self.spin_penalty

        if "scans" in ego_obs:
            self_min_scan = float(np.min(ego_obs["scans"]))
            shaped -= self.self_wall_penalty * max(0.0, 0.5 - self_min_scan)

        target_obs = self._select_target_obs(agent_id, all_obs) if all_obs else None
        if target_obs and "pose" in target_obs:
            tx, ty, _ = target_obs["pose"]
            target_dist = float(np.hypot(tx - x, ty - y))
            prev_dist = self.prev_target_dist.get(agent_id, target_dist)
            shaped += self.target_distance_scale * (prev_dist - target_dist)
            self.prev_target_dist[agent_id] = target_dist

            if "scans" in target_obs:
                tgt_min_scan = float(np.min(target_obs["scans"]))
                shaped += self.target_wall_bonus * max(0.0, 0.5 - tgt_min_scan)

            ego_crashed = ego_obs.get("collision", False)
            target_crashed = target_obs.get("collision", False)
            if ego_crashed and target_crashed:
                shaped = 0.5 * self.ego_collision_penalty
            elif target_crashed and not ego_crashed:
                key = (agent_id, target_obs.get("agent_id", "target"))
                if key not in self.opponent_crash_reward_given:
                    shaped += self.herd_bonus + self.opponent_collision_bonus
                    self.opponent_crash_reward_given.add(key)

        if done and ego_obs.get("collision", False):
            shaped += self.ego_collision_penalty

        return shaped
