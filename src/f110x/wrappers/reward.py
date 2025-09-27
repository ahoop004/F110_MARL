import numpy as np

class RewardWrapper:
    def __init__(self,
                 mode="basic",            # "basic", "herding", "pursuit"
                 collision_penalty=-5.0,
                 alive_bonus=0.1,
                 progress_scale=1.0,
                 spin_penalty=0.5,
                 spin_thresh=np.pi/6,
                 herd_bonus=10.0,
                 pursuit_scale=0.1):
        self.mode = mode
        self.collision_penalty = collision_penalty
        self.alive_bonus = alive_bonus
        self.progress_scale = progress_scale
        self.spin_penalty = spin_penalty
        self.spin_thresh = spin_thresh
        self.herd_bonus = herd_bonus
        self.pursuit_scale = pursuit_scale
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

        # progress
        prev = self.prev_positions.get(agent_id, (x, y))
        dx, dy = x - prev[0], y - prev[1]
        dist = np.sqrt(dx * dx + dy * dy)

        # spin detection
        prev_theta = self.prev_thetas.get(agent_id, theta)
        dtheta = abs(theta - prev_theta)

        # update trackers
        self.prev_positions[agent_id] = (x, y)
        self.prev_thetas[agent_id] = theta

        shaped = reward + self.alive_bonus + self.progress_scale * dist

        # universal collision penalty
        if ego_obs["collision"]:
            shaped += self.collision_penalty

        # discourage spinning
        if dist < 0.05 and dtheta > self.spin_thresh:
            shaped -= self.spin_penalty

        # pursuit mode: reward reducing distance to target
        if self.mode in ("pursuit", "herding"):
            if "target_pose" in ego_obs:
                tx, ty, _ = ego_obs["target_pose"]
                target_dist = np.sqrt((tx - x) ** 2 + (ty - y) ** 2)
                prev_dist = self.prev_target_dists.get(agent_id, target_dist)
                shaped += self.pursuit_scale * (prev_dist - target_dist)
                self.prev_target_dists[agent_id] = target_dist

        # herding: big bonus when target collides
        if self.mode == "herding":
            if "target_collision" in ego_obs and ego_obs["target_collision"]:
                shaped += self.herd_bonus

        return shaped
