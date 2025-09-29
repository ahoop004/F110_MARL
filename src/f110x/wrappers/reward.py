import numpy as np
from typing import Dict

class RewardWrapper:
    def __init__(
        self,
        mode="basic",
        alive_bonus=0.0004,
        forward_scale=0.0015,
        reverse_penalty=0.0015,
        lateral_penalty=0.0,
        target_distance_scale=0.0006,
        target_wall_bonus=0.0,
        self_wall_penalty=0.0,
        herd_bonus=0.0,
        ego_collision_penalty=-1.0,
        opponent_collision_bonus=0.0,
        target_crash_reward=0.5,
        herd_position_radius=3.5,
        herd_position_slack=1.0,
        herd_position_scale=0.0,
        herd_angle_scale=0.0,
        herd_angle_power=2.0,
        slow_speed_threshold=0.5,
        slow_speed_penalty=-0.00012,
        truncation_penalty=-0.5,
        time_penalty=0.0,
        progress_threshold=0.05,
        progress_patience=80,
        progress_penalty=0.0,
        # spin_thresh=np.pi / 6,
        spin_thresh=0.6,            # rad/s threshold on filtered yaw rate
        spin_penalty=0.0,           # k in quadratic term
        spin_speed_gate=0.3,        # m/s max speed for spin to count
        spin_dwell_steps=5,         # consecutive steps before penalizing
        spin_step_cap=0.0,          # max |per-step| penalty
        spin_episode_cap=0.0,       # max |per-episode| penalty
        spin_grace_steps=20,        # steps after reset with no spin penalty
        spin_alpha=0.2,             # EMA smoothing for yaw rate
        dt=0.01
    ):
        """Rich reward shaping for mixed pursuit/herding."""

        self.mode = mode
        self.alive_bonus = float(alive_bonus)
        self.forward_scale = float(forward_scale)
        self.reverse_penalty = float(reverse_penalty)
        self.lateral_penalty = float(lateral_penalty)
        self.target_distance_scale = float(target_distance_scale)
        self.target_wall_bonus = float(target_wall_bonus)
        self.self_wall_penalty = float(self_wall_penalty)
        self.herd_bonus = float(herd_bonus)
        self.ego_collision_penalty = float(ego_collision_penalty)
        self.opponent_collision_bonus = float(opponent_collision_bonus)
        self.herd_position_radius = float(herd_position_radius)
        self.herd_position_slack = max(float(herd_position_slack), 1e-6)
        self.herd_position_scale = float(herd_position_scale)
        self.herd_angle_scale = float(herd_angle_scale)
        self.herd_angle_power = float(herd_angle_power)
        self.slow_speed_threshold = float(slow_speed_threshold)
        self.slow_speed_penalty = float(slow_speed_penalty)
        self.target_crash_reward = float(target_crash_reward)
        self.truncation_penalty = float(truncation_penalty)
        self.time_penalty = float(time_penalty)
        self.progress_threshold = float(progress_threshold)
        self.progress_patience = max(int(progress_patience), 1)
        self.progress_penalty = float(progress_penalty)
        self.spin_thresh = float(spin_thresh)
        self.spin_penalty = float(spin_penalty)

        self.spin_speed_gate = float(spin_speed_gate)
        self.spin_dwell_steps = int(spin_dwell_steps)
        self.spin_step_cap = float(spin_step_cap)
        self.spin_episode_cap = float(spin_episode_cap)
        self.spin_grace_steps = int(spin_grace_steps)
        self.spin_alpha = float(spin_alpha)
        self.dt = float(dt)

        self.prev_positions = {}
        self.prev_target_dist = {}
        self.opponent_crash_reward_given = set()
        self._last_components: Dict[str, Dict[str, float]] = {}
        self._spin_state: Dict[str, Dict[str, float]] = {}
        self._progress_state: Dict[str, Dict[str, int]] = {}

    def reset(self):
        self.prev_positions.clear()
        self.prev_target_dist.clear()
        self.opponent_crash_reward_given.clear()
        self._last_components.clear()
        self._spin_state.clear()
        self._progress_state.clear()
    
    def _spin_ctx(self, aid):
        ctx = self._spin_state.get(aid)
        if ctx is None:
            ctx = {"ema_omega": 0.0, "dwell": 0, "ep_accum": 0.0, "age": 0}
            self._spin_state[aid] = ctx
        return ctx

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
        if self.time_penalty:
            shaped += self.time_penalty
            components["time_penalty"] = components.get("time_penalty", 0.0) + self.time_penalty
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
            if speed < self.slow_speed_threshold:
                slow_penalty = self.slow_speed_penalty
                shaped += slow_penalty
                components["slow_penalty"] = components.get("slow_penalty", 0.0) + slow_penalty

        if "scans" in ego_obs:
            self_min_scan = float(np.min(ego_obs["scans"]))
            wall_penalty = -self.self_wall_penalty * max(0.0, 0.5 - self_min_scan)
            shaped += wall_penalty
            components["self_wall_penalty"] = components.get("self_wall_penalty", 0.0) + wall_penalty

        target_obs = self._select_target_obs(agent_id, all_obs) if all_obs else None

        ctx = self._spin_ctx(agent_id)
        ctx["age"] += 1

        # derive speed and raw yaw rate
        vx, vy = (0.0, 0.0)
        if "velocity" in ego_obs:
            vx, vy = map(float, ego_obs["velocity"])
        speed = float(np.hypot(vx, vy))

        omega_raw = float(abs(ego_obs.get("angular_velocity", 0.0)))
        # EMA smoothing
        ctx["ema_omega"] = (1.0 - self.spin_alpha) * ctx["ema_omega"] + self.spin_alpha * omega_raw
        omega_f = ctx["ema_omega"]

        # grace window
        if ctx["age"] <= self.spin_grace_steps:
            spin_term = 0.0
        else:
            # gate: only when slow AND above threshold
            if (speed < self.spin_speed_gate) and (omega_f > self.spin_thresh):
                ctx["dwell"] += 1
            else:
                ctx["dwell"] = 0

            if ctx["dwell"] >= self.spin_dwell_steps and ctx["ep_accum"] > -self.spin_episode_cap:
                # quadratic excess above threshold, time-weighted
                excess = omega_f - self.spin_thresh
                step_pen = - self.spin_penalty * (excess * excess) * self.dt
                # per-step cap
                step_pen = max(step_pen, -self.spin_step_cap)

                prev_accum = ctx["ep_accum"]
                capped_accum = max(prev_accum + step_pen, -self.spin_episode_cap)
                spin_term = capped_accum - prev_accum
                ctx["ep_accum"] = capped_accum
            else:
                spin_term = 0.0

        shaped += spin_term
        if spin_term:
            components["spin_penalty"] = components.get("spin_penalty", 0.0) + float(spin_term)

        if target_obs and "pose" in target_obs:
            pose_t = target_obs["pose"]
            tx = float(pose_t[0])
            ty = float(pose_t[1])
            ttheta = float(pose_t[2]) if len(pose_t) > 2 else 0.0

            to_agent_vec = np.array([x - tx, y - ty], dtype=np.float32)
            target_dist = float(np.linalg.norm(to_agent_vec))
            had_prev = agent_id in self.prev_target_dist
            prev_dist = self.prev_target_dist.get(agent_id, target_dist)
            distance_delta = self.target_distance_scale * (prev_dist - target_dist)
            shaped += distance_delta
            components["target_distance_delta"] = components.get("target_distance_delta", 0.0) + distance_delta
            self.prev_target_dist[agent_id] = target_dist

            progress_ctx = self._progress_state.setdefault(agent_id, {"stall": 0})
            if had_prev:
                improvement = prev_dist - target_dist
                if improvement >= self.progress_threshold:
                    progress_ctx["stall"] = 0
                else:
                    progress_ctx["stall"] += 1
                    if self.progress_penalty and progress_ctx["stall"] >= self.progress_patience:
                        shaped += self.progress_penalty
                        components["progress_penalty"] = components.get("progress_penalty", 0.0) + self.progress_penalty
                        progress_ctx["stall"] = 0
            else:
                progress_ctx["stall"] = 0

            if "scans" in target_obs:
                tgt_min_scan = float(np.min(target_obs["scans"]))
                target_wall_term = self.target_wall_bonus * max(0.0, 0.5 - tgt_min_scan)
                shaped += target_wall_term
                components["target_wall_bonus"] = components.get("target_wall_bonus", 0.0) + target_wall_term

            herd_pos_bonus = 0.0
            if target_dist > 1e-6:
                radial_err = (target_dist - self.herd_position_radius) / self.herd_position_slack
                radial_bonus = float(np.exp(-0.5 * radial_err * radial_err) * self.herd_position_scale)

                heading_vec = np.array([np.cos(ttheta), np.sin(ttheta)], dtype=np.float32)
                heading_norm = float(np.linalg.norm(heading_vec))
                if heading_norm < 1e-6 and "velocity" in target_obs:
                    target_vel = np.asarray(target_obs["velocity"], dtype=np.float32)
                    if target_vel.shape[0] >= 2:
                        heading_vec = target_vel[:2]
                        heading_norm = float(np.linalg.norm(heading_vec))

                align = 0.0
                if heading_norm > 1e-6:
                    bearing = to_agent_vec / target_dist
                    target_heading_unit = heading_vec / heading_norm
                    align = max(0.0, float(np.dot(bearing, -target_heading_unit)))

                angle_bonus = (align ** self.herd_angle_power) * self.herd_angle_scale if align > 0.0 else 0.0
                herd_pos_bonus = radial_bonus + float(angle_bonus)
                if herd_pos_bonus > 0.0 and (
                    ego_obs.get("collision", False) or target_obs.get("collision", False)
                ):
                    herd_pos_bonus = 0.0

            if herd_pos_bonus > 0.0:
                shaped += herd_pos_bonus
                components["herd_position_bonus"] = components.get("herd_position_bonus", 0.0) + herd_pos_bonus

            ego_crashed = ego_obs.get("collision", False)
            target_crashed = target_obs.get("collision", False)
            if ego_crashed and target_crashed:
                progress_ctx["stall"] = 0
                shaped = 0.5 * self.ego_collision_penalty
                components = {"collision_split_penalty": shaped}
            elif target_crashed and not ego_crashed:
                key = (agent_id, target_obs.get("agent_id", "target"))
                if key not in self.opponent_crash_reward_given:
                    progress_ctx["stall"] = 0
                    herd_reward = self.herd_bonus + self.opponent_collision_bonus
                    total_bonus = herd_reward + self.target_crash_reward
                    shaped += total_bonus
                    if herd_reward:
                        components["herd_bonus"] = components.get("herd_bonus", 0.0) + herd_reward
                    if self.target_crash_reward:
                        components["target_crash_reward"] = (
                            components.get("target_crash_reward", 0.0) + self.target_crash_reward
                        )
                    self.opponent_crash_reward_given.add(key)

        if done and ego_obs.get("collision", False):
            terminal_collision = self.ego_collision_penalty
            shaped += terminal_collision
            components["ego_collision_penalty"] = components.get("ego_collision_penalty", 0.0) + terminal_collision

        self._last_components[agent_id] = components
        return shaped

    def get_last_components(self, agent_id: str) -> Dict[str, float]:
        return dict(self._last_components.get(agent_id, {}))
