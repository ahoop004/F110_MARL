"""ObservationComposer — assembles components into a flat numpy array."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from wrappers.observations.base import ObservationComponent
from wrappers.observations.lidar import LidarComponent
from wrappers.observations.ego_state import EgoStateComponent
from wrappers.observations.target_state import TargetStateComponent
from wrappers.observations.relative_pose import RelativePoseComponent
from wrappers.observations.progress import ProgressComponent
from wrappers.observations.prev_action import PrevActionComponent


class ObservationComposer:
    """Concatenates enabled ObservationComponents into a flat float32 array.

    Built from a config dict (loaded from configs/observations/<policy>.yaml)
    and the env config (for lidar_beams, lidar_range).
    """

    def __init__(self, components: List[ObservationComponent]) -> None:
        self._components = components

    @property
    def obs_dim(self) -> int:
        return sum(c.dim for c in self._components)

    @property
    def components(self) -> List[ObservationComponent]:
        return self._components

    def wrap(self, raw_obs: Dict, info: Optional[Dict] = None) -> np.ndarray:
        info = info or {}
        parts = [c.compute(raw_obs, info) for c in self._components]
        return np.concatenate(parts).astype(np.float32)

    def reset(self) -> None:
        for c in self._components:
            if hasattr(c, "reset"):
                c.reset()

    def update_prev_action(self, action: np.ndarray) -> None:
        for c in self._components:
            if isinstance(c, PrevActionComponent):
                c.update(action)

    @classmethod
    def from_config(
        cls,
        obs_config: Dict,
        env_config: Dict,
        action_dim: int = 2,
    ) -> "ObservationComposer":
        """Build from a parsed observation config dict and env config.

        obs_config: the 'observation:' block from the YAML file
        env_config: the 'environment:' block (provides lidar_beams, lidar_range)
        """
        n_beams = int(env_config.get("lidar_beams", 108))
        lidar_range = float(env_config.get("lidar_range", 10.0))

        obs = obs_config.get("observation", obs_config)
        components: List[ObservationComponent] = []

        lidar_cfg = obs.get("lidar", {})
        if lidar_cfg.get("enabled", False):
            components.append(
                LidarComponent(
                    n_beams=n_beams,
                    lidar_range=lidar_range,
                    normalize=bool(lidar_cfg.get("normalize", True)),
                )
            )

        ego_cfg = obs.get("ego_state", {})
        if ego_cfg.get("enabled", False):
            components.append(
                EgoStateComponent(
                    include_velocity=bool(ego_cfg.get("include_velocity", True)),
                    include_pose=bool(ego_cfg.get("include_pose", False)),
                )
            )

        tgt_cfg = obs.get("target_state", {})
        if tgt_cfg.get("enabled", False):
            components.append(TargetStateComponent())

        rel_cfg = obs.get("relative_pose", {})
        if rel_cfg.get("enabled", False):
            components.append(RelativePoseComponent())

        prog_cfg = obs.get("progress", {})
        if prog_cfg.get("enabled", False):
            components.append(ProgressComponent())

        pa_cfg = obs.get("prev_action", {})
        if pa_cfg.get("enabled", False):
            components.append(PrevActionComponent(action_dim=action_dim))

        if not components:
            raise ValueError("ObservationComposer: no components enabled in obs config.")

        return cls(components)

    @classmethod
    def from_file(
        cls,
        path: str,
        env_config: Dict,
        action_dim: int = 2,
    ) -> "ObservationComposer":
        """Load from a YAML observation config file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Observation config not found: {path}")
        with open(p) as f:
            obs_config = yaml.safe_load(f) or {}
        return cls.from_config(obs_config, env_config, action_dim=action_dim)
