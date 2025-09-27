import numpy as np

class RewardWrapper:
    def __init__(self,
                 mode="basic",
                 collision_penalty=-150.0,   # much harsher ego crash penalty
                 alive_bonus=0.01,
                 progress_scale=0.1,         # de-emphasize raw forward motion
                 spin_penalty=0.55,
                 spin_thresh=np.pi/6,
                 pursuit_scale=0.05,
                 herd_bonus=150.0,            # big reward if target crashes
                 reverse_penalty=1.0,
                 speed_scale=0.05,
                 safe_dist=0.5,              # meters from wall for safety shaping
                 wall_bonus=0.2):            # reward if target is near a wall
        """
        mode: "basic", "pursuit", "adversarial"
        """
        self.mode = mode
        self.collision_penalty = collision_penalty
        self.alive_bonus = alive_bonus
        self.progress_scale = progress_scale
        self.spin_penalty = spin_penalty
        self.spin_thresh = spin_thresh
        self.pursuit_scale = pursuit_scale
        self.herd_bonus = herd_bonus
        self.reverse_penalty = reverse_penalty
        self.speed_scale = speed_scale
        self.safe_dist = safe_dist
        self.wall_bonus = wall_bonus

        self.prev_positions = {}
        self.prev_thetas = {}
        self.prev_target_dists = {}

    def reset(self):
        self.prev_positions.clear()
        self.prev_thetas.clear()
        self.prev_target_dists.clear()

    def __call__(self, obs, agent_id, reward, done, info):
        ego_obs = obs[agent_id]
        x, y, theta = ego_obs["pose"]

        # --- displacement ---
        prev = self.prev_positions.get(agent_id, (x, y))
        dx, dy = x - prev[0], y - prev[1]
        dist = np.sqrt(dx * dx + dy * dy)

        prev_theta = self.prev_thetas.get(agent_id, theta)
        dtheta = abs(theta - prev_theta)

        self.prev_positions[agent_id] = (x, y)
        self.prev_thetas[agent_id] = theta

        shaped = reward + self.alive_bonus

        # --- forward progress ---
        heading = np.array([np.cos(theta), np.sin(theta)])
        disp = np.array([dx, dy])
        forward_proj = np.dot(heading, disp)

        if forward_proj > 0:
            shaped += self.progress_scale * forward_proj
        else:
            shaped -= self.reverse_penalty * abs(forward_proj)

        # --- forward speed bonus ---
        if "velocity" in ego_obs:
            vx, vy = ego_obs["velocity"]
            forward_speed = np.dot(heading, np.array([vx, vy]))
            if forward_speed > 0:
                shaped += self.speed_scale * forward_speed

        # --- survival penalties ---
        if ego_obs["collision"]:
    # already penalized, don't give pursuit reward
            shaped += 0.0
        

        if dist < 0.05 and dtheta > self.spin_thresh:
            shaped -= self.spin_penalty

        # --- near-wall penalty for ego ---
        if "scans" in ego_obs:
            min_scan = float(np.min(ego_obs["scans"]))
            if min_scan < self.safe_dist:
                shaped -= 0.1 * (self.safe_dist - min_scan)

        # --- pursuit shaping ---
        if self.mode in ("pursuit", "adversarial") and "target_pose" in ego_obs:
            if not ego_obs["collision"]:
                tx, ty, _ = ego_obs["target_pose"]
                target_dist = np.sqrt((tx - x) ** 2 + (ty - y) ** 2)
                prev_dist = self.prev_target_dists.get(agent_id, target_dist)
                shaped += self.pursuit_scale * (prev_dist - target_dist)
                self.prev_target_dists[agent_id] = target_dist

            # --- target-near-wall bonus ---
            tgt_obs = obs.get("car_1")  # assumes ego=car_0, target=car_1
            if tgt_obs and "scans" in tgt_obs:
                tgt_min_scan = float(np.min(tgt_obs["scans"]))
                if tgt_min_scan < self.safe_dist:
                    shaped += self.wall_bonus * (self.safe_dist - tgt_min_scan)

        # --- adversarial bonus ---
        if self.mode == "adversarial":
            if ego_obs.get("target_collision", False):
                shaped += self.herd_bonus

        return shaped
