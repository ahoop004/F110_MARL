import numpy as np

class ObsWrapper:
    def __init__(self, max_scan=30.0, normalize=True):
        self.max_scan = max_scan
        self.normalize = normalize

    def __call__(self, obs, ego_id, target_id=None):
        """
        obs: per-agent obs dict from F110ParallelEnv (obs[aid] = {...})
        ego_id: agent id string (e.g. "car_0")
        target_id: optional agent id. If None and only 2 agents, auto-pick the other.
        """
        if target_id is None:
            agent_ids = list(obs.keys())
            if len(agent_ids) != 2:
                raise ValueError("Auto target selection only works for 2 agents")
            target_id = [a for a in agent_ids if a != ego_id][0]

        ego_obs = obs[ego_id]
        tgt_obs = obs[target_id]

        # LiDAR
        scan = np.array(ego_obs["scans"], dtype=np.float32)
        if self.normalize:
            scan = scan / self.max_scan

        # Ego features
        pose = np.array(ego_obs["pose"], dtype=np.float32)  # [x, y, theta]
        ego = np.concatenate([pose, [float(ego_obs["collision"])]], dtype=np.float32)

        # Target features (if available in obs)
        if "target_pose" in ego_obs and "target_collision" in ego_obs:
            target = np.concatenate([
                np.array(ego_obs["target_pose"], dtype=np.float32),
                [float(ego_obs["target_collision"])]
            ], dtype=np.float32)
        else:
            # fallback: build from target agent's own obs
            tgt_pose = np.array(tgt_obs["pose"], dtype=np.float32)
            target = np.concatenate([tgt_pose, [float(tgt_obs["collision"])]], dtype=np.float32)

        return np.concatenate([scan, ego, target])
