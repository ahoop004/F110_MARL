import numpy as np
from typing import Dict

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
        self._last_components: Dict[str, Dict[str, float]] = {}

    def reset(self):
        self.prev_positions.clear()
        self.prev_target_dist.clear()
        self.opponent_crash_reward_given.clear()
        self._last_components.clear()

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

        shaped = float(reward)
        components: Dict[str, float] = {"env_reward": float(reward)}

        shaped += self.alive_bonus
        components["alive_bonus"] = components.get("alive_bonus", 0.0) + self.alive_bonus
        if forward_step > 0:
            forward_reward = self.forward_scale * forward_step
            shaped += forward_reward
            components["forward_progress"] = components.get("forward_progress", 0.0) + forward_reward
        else:
            reverse_penalty = self.reverse_penalty * forward_step  # negative value
            shaped += reverse_penalty
            components["reverse_penalty"] = components.get("reverse_penalty", 0.0) + reverse_penalty

        lateral_penalty = -self.lateral_penalty * abs(lateral_step)
        shaped += lateral_penalty
        components["lateral_penalty"] = components.get("lateral_penalty", 0.0) + lateral_penalty

        if ego_obs.get("collision", False):
            collision_penalty = self.ego_collision_penalty
            shaped += collision_penalty
            components["ego_collision_penalty"] = components.get("ego_collision_penalty", 0.0) + collision_penalty

        if "velocity" in ego_obs:
            vx, vy = map(float, ego_obs["velocity"])
            speed = np.hypot(vx, vy)
            if speed < 0.5:
                slow_penalty = -0.0002
                shaped += slow_penalty
                components["slow_penalty"] = components.get("slow_penalty", 0.0) + slow_penalty

        if "angular_velocity" in ego_obs:
            omega = abs(float(ego_obs["angular_velocity"]))
            if omega > self.spin_thresh and np.linalg.norm(disp) < 0.05:
                spin_penalty = -self.spin_penalty
                shaped += spin_penalty
                components["spin_penalty"] = components.get("spin_penalty", 0.0) + spin_penalty

        if "scans" in ego_obs:
            self_min_scan = float(np.min(ego_obs["scans"]))
            wall_penalty = -self.self_wall_penalty * max(0.0, 0.5 - self_min_scan)
            shaped += wall_penalty
            components["self_wall_penalty"] = components.get("self_wall_penalty", 0.0) + wall_penalty

        target_obs = self._select_target_obs(agent_id, all_obs) if all_obs else None
        if target_obs and "pose" in target_obs:
            tx, ty, _ = target_obs["pose"]
            target_dist = float(np.hypot(tx - x, ty - y))
            prev_dist = self.prev_target_dist.get(agent_id, target_dist)
            distance_delta = self.target_distance_scale * (prev_dist - target_dist)
            shaped += distance_delta
            components["target_distance_delta"] = components.get("target_distance_delta", 0.0) + distance_delta
            self.prev_target_dist[agent_id] = target_dist

            if "scans" in target_obs:
                tgt_min_scan = float(np.min(target_obs["scans"]))
                target_wall_term = self.target_wall_bonus * max(0.0, 0.5 - tgt_min_scan)
                shaped += target_wall_term
                components["target_wall_bonus"] = components.get("target_wall_bonus", 0.0) + target_wall_term

            ego_crashed = ego_obs.get("collision", False)
            target_crashed = target_obs.get("collision", False)
            if ego_crashed and target_crashed:
                shaped = 0.5 * self.ego_collision_penalty
                components = {"collision_split_penalty": shaped}
            elif target_crashed and not ego_crashed:
                key = (agent_id, target_obs.get("agent_id", "target"))
                if key not in self.opponent_crash_reward_given:
                    herd_reward = self.herd_bonus + self.opponent_collision_bonus
                    shaped += herd_reward
                    components["herd_bonus"] = components.get("herd_bonus", 0.0) + herd_reward
                    self.opponent_crash_reward_given.add(key)

        if done and ego_obs.get("collision", False):
            terminal_collision = self.ego_collision_penalty
            shaped += terminal_collision
            components["ego_collision_penalty"] = components.get("ego_collision_penalty", 0.0) + terminal_collision

        self._last_components[agent_id] = components
        return shaped

    def get_last_components(self, agent_id: str) -> Dict[str, float]:
        return dict(self._last_components.get(agent_id, {}))
