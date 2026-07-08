"""ObservationComposer — assembles components into a flat numpy array."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from wrappers.observations.base import ObservationComponent
from wrappers.observations.lidar import LidarComponent
from wrappers.observations.ego_state import EgoStateComponent
from wrappers.observations.centerline_ego_state import CenterlineEgoStateComponent
from wrappers.observations.target_state import TargetStateComponent
from wrappers.observations.relative_pose import RelativePoseComponent
from wrappers.observations.progress import ProgressComponent
from wrappers.observations.prev_action import PrevActionComponent


class ObservationComposer:
    """Concatenates enabled ObservationComponents into a flat float32 array.

    Built from a config dict (loaded from configs/observations/<policy>.yaml)
    and the env config (for lidar_beams, lidar_range).

    Each :meth:`wrap` call writes component outputs directly into a
    pre-allocated internal buffer via :meth:`~ObservationComponent.compute_into`,
    then returns a copy.  This eliminates the intermediate per-component array
    allocations and the final :func:`numpy.concatenate` call.
    """

    def __init__(self, components: List[ObservationComponent]) -> None:
        self._components = components
        # Pre-allocate assembly buffer and slice boundaries once.
        self._obs_dim: int = sum(c.dim for c in components)
        self._buf = np.zeros(self._obs_dim, dtype=np.float32)
        offsets = []
        offset = 0
        for c in components:
            offsets.append(offset)
            offset += c.dim
        self._offsets: List[int] = offsets

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def components(self) -> List[ObservationComponent]:
        return self._components

    def wrap(self, raw_obs: Dict, info: Optional[Dict] = None) -> np.ndarray:
        """Assemble and return a fresh float32 observation vector.

        Each component writes directly into a pre-allocated internal buffer
        (zero intermediate allocations for components that implement
        :meth:`~ObservationComponent.compute_into`).  Returns ``buf.copy()``
        so the caller owns the data independently.
        """
        info = info or {}
        buf = self._buf
        for c, start in zip(self._components, self._offsets):
            c.compute_into(raw_obs, info, buf[start : start + c.dim])
        return buf.copy()

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

        cl_ego_cfg = obs.get("centerline_ego_state", {})
        if cl_ego_cfg.get("enabled", False):
            components.append(CenterlineEgoStateComponent())

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
