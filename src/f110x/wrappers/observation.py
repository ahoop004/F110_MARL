from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from f110x.wrappers.common import downsample_lidar, to_numpy


class ObsWrapper:
    """Canonical observation adapter for policy networks."""

    def __init__(self, max_scan: float = 30.0, normalize: bool = True, lidar_beams: Optional[int] = None):
        self.max_scan = float(max_scan)
        self.normalize = bool(normalize)
        if lidar_beams is not None:
            beams = int(lidar_beams)
            self.lidar_beams = beams if beams > 0 else None
        else:
            self.lidar_beams = None

    def __call__(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        target_id: Optional[str] = None,
    ) -> np.ndarray:
        """Project raw env observations to a flat feature vector."""

        target_key = self._resolve_target_id(obs, ego_id, target_id)

        ego_obs = obs[ego_id]
        target_obs = obs[target_key] if target_key is not None else None

        scan = self._prepare_scan(ego_obs)
        ego_features = self._prepare_ego_features(ego_obs)
        target_features = self._prepare_target_features(ego_obs, target_obs)

        return np.concatenate([scan, ego_features, target_features])

    # ------------------------------------------------------------------
    def _resolve_target_id(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        target_id: Optional[str],
    ) -> Optional[str]:
        if target_id is not None:
            return target_id

        agent_ids = list(obs.keys())
        if len(agent_ids) == 2:
            return agent_ids[0] if agent_ids[1] == ego_id else agent_ids[1]

        raise ValueError("ObsWrapper requires explicit target_id when more than two agents are present")

    def _prepare_scan(self, ego_obs: Dict[str, Any]) -> np.ndarray:
        if "scans" not in ego_obs:
            raise KeyError("ObsWrapper expects a 'scans' entry in the ego observation")

        scan = downsample_lidar(ego_obs["scans"], self.lidar_beams)
        if self.normalize and self.max_scan > 0.0:
            scan = scan / self.max_scan
        return scan.astype(np.float32, copy=False)

    def _prepare_ego_features(self, ego_obs: Dict[str, Any]) -> np.ndarray:
        pose = to_numpy(ego_obs.get("pose", ()), flatten=True)
        if pose.size < 3:
            raise ValueError("ObsWrapper requires ego pose with at least (x, y, theta)")

        x, y, theta = map(float, pose[:3])
        if self.max_scan > 0.0:
            x /= self.max_scan
            y /= self.max_scan
        orientation = np.sin(theta)
        collision = float(ego_obs.get("collision", False))
        return np.array([x, y, orientation, collision], dtype=np.float32)

    def _prepare_target_features(
        self,
        ego_obs: Dict[str, Any],
        target_obs: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        if "target_pose" in ego_obs and "target_collision" in ego_obs:
            pose = to_numpy(ego_obs["target_pose"], flatten=True)
            if pose.size < 3:
                raise ValueError("ObsWrapper target_pose must have at least 3 elements")
            collision = float(ego_obs.get("target_collision", False))
            return np.concatenate([pose[:3].astype(np.float32, copy=False), np.array([collision], dtype=np.float32)])

        if target_obs is None:
            raise ValueError("ObsWrapper could not resolve a target observation")

        pose = to_numpy(target_obs.get("pose", ()), flatten=True)
        if pose.size < 3:
            raise ValueError("Target observation requires pose with at least (x, y, theta)")
        collision = float(target_obs.get("collision", False))
        return np.concatenate([pose[:3].astype(np.float32, copy=False), np.array([collision], dtype=np.float32)])
