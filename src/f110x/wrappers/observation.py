from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from f110x.wrappers.common import downsample_lidar, to_numpy
from f110x.utils.centerline import (
    CenterlineProjection,
    progress_from_spacing,
    project_to_centerline,
)


ComponentBuilder = Callable[
    [
        "ObsWrapper",
        Dict[str, Dict[str, Any]],
        str,
        "ComponentSpec",
        Optional[str],
    ],
    np.ndarray,
]


@dataclass(frozen=True)
class ComponentSpec:
    """Normalised configuration for a single observation component."""

    type: str
    params: Dict[str, Any]
    target_agent: Optional[str]
    component_id: Optional[str]
    enabled: bool = True


class ObsWrapper:
    """Composable observation adapter driven by a component registry."""

    def __init__(
        self,
        max_scan: float = 30.0,
        normalize: bool = True,
        lidar_beams: Optional[int] = None,
        *,
        components: Optional[Iterable[Any]] = None,
        centerline: Optional[np.ndarray] = None,
        centerline_features: bool = False,
        centerline_normalize: bool = True,
        legacy_target_agent: Optional[str] = None,
    ):
        self.max_scan = float(max_scan)
        self.normalize = bool(normalize)
        if lidar_beams is not None:
            beams = int(lidar_beams)
            self.lidar_beams = beams if beams > 0 else None
        else:
            self.lidar_beams = None

        self.centerline_points = None if centerline is None else np.asarray(centerline, dtype=np.float32)
        self.centerline_features_enabled = bool(centerline_features) and self.centerline_points is not None
        self.centerline_normalize = bool(centerline_normalize)
        self._centerline_last_index: Dict[str, Optional[int]] = {}

        self._default_target_agent = legacy_target_agent
        self._component_specs: List[ComponentSpec] = []

        self._component_builders: Dict[str, ComponentBuilder] = {
            "lidar": self._component_lidar,
            "ego_pose": self._component_ego_pose,
            "pose": self._component_pose,
            "velocity": self._component_velocity,
            "collision": self._component_collision,
            "lap": self._component_lap,
            "target_pose": self._component_target_pose,
            "relative_pose": self._component_relative_pose,
            "distance": self._component_distance,
            "centerline": self._component_centerline,
        }

        self._compile_components(components)

    # ------------------------------------------------------------------
    def __call__(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        target_id: Optional[str] = None,
    ) -> np.ndarray:
        """Project raw env observations to a flat feature vector."""

        ego_obs = obs.get(ego_id)
        if ego_obs is None:
            raise KeyError(f"Observation for agent '{ego_id}' missing")

        pieces: List[np.ndarray] = []
        for spec in self._component_specs:
            if not spec.enabled:
                continue

            resolved_target = spec.target_agent or target_id or self._default_target_agent
            component = self._dispatch_component(obs, ego_id, spec, resolved_target)
            if component is None:
                continue
            pieces.append(component)

        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces)

    # ------------------------------------------------------------------
    def _compile_components(self, raw_components: Optional[Iterable[Any]]) -> None:
        if raw_components is None:
            self._component_specs = self._default_legacy_components()
            return

        compiled: List[ComponentSpec] = []
        for entry in raw_components:
            spec = self._parse_component(entry)
            if spec is not None:
                compiled.append(spec)
        self._component_specs = compiled

    def _parse_component(self, entry: Any) -> Optional[ComponentSpec]:
        if entry is None:
            return None

        if isinstance(entry, str):
            component_type = entry.strip().lower()
            params: Dict[str, Any] = {}
            target_agent = None
            component_id = None
            enabled = True
        elif isinstance(entry, dict):
            component_type = str(entry.get("type") or entry.get("name") or "").strip().lower()
            if not component_type:
                raise ValueError("Observation component requires a 'type' field")
            params = dict(entry.get("params", {}) or {})
            component_id_raw = entry.get("id") or entry.get("component_id")
            component_id = str(component_id_raw) if component_id_raw is not None else None
            enabled_val = entry.get("enabled")
            enabled = bool(enabled_val) if enabled_val is not None else True
            target_spec = entry.get("target")
            if isinstance(target_spec, dict):
                target_agent_raw = target_spec.get("agent") or target_spec.get("agent_id")
                target_agent = str(target_agent_raw) if target_agent_raw is not None else None
            elif target_spec is None:
                target_agent = None
            else:
                target_agent = str(target_spec)
        else:
            raise TypeError(f"Unsupported observation component definition: {type(entry)!r}")

        if component_type not in self._component_builders:
            available = ", ".join(sorted(self._component_builders))
            raise KeyError(f"Unknown observation component '{component_type}'. Available: {available}")

        return ComponentSpec(
            type=component_type,
            params=params,
            target_agent=target_agent,
            component_id=component_id,
            enabled=enabled,
        )

    def _default_legacy_components(self) -> List[ComponentSpec]:
        specs: List[ComponentSpec] = []

        specs.append(
            ComponentSpec(
                type="lidar",
                params={
                    "beams": self.lidar_beams,
                    "normalize": self.normalize,
                    "max_range": self.max_scan,
                },
                target_agent=None,
                component_id="lidar",
            )
        )

        specs.append(
            ComponentSpec(
                type="ego_pose",
                params={"normalize_xy": self.max_scan, "angle_mode": "sin"},
                target_agent=None,
                component_id="ego_pose",
            )
        )

        specs.append(
            ComponentSpec(
                type="collision",
                params={},
                target_agent=None,
                component_id="ego_collision",
            )
        )

        specs.append(
            ComponentSpec(
                type="target_pose",
                params={},
                target_agent=None,
                component_id="target_pose",
            )
        )

        specs.append(
            ComponentSpec(
                type="collision",
                params={},
                target_agent=None,
                component_id="target_collision",
            )
        )

        if self.centerline_features_enabled:
            specs.append(
                ComponentSpec(
                    type="centerline",
                    params={},
                    target_agent=None,
                    component_id="centerline",
                )
            )

        return specs

    def _dispatch_component(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        target_id: Optional[str],
    ) -> Optional[np.ndarray]:
        builder = self._component_builders.get(spec.type)
        if builder is None:
            return None

        ego_obs = obs.get(ego_id)
        if ego_obs is None:
            raise KeyError(f"Ego observation for agent '{ego_id}' missing")

        if spec.target_agent is not None:
            # Explicit target specified at component level
            if spec.target_agent not in obs:
                raise KeyError(f"Target observation for agent '{spec.target_agent}' missing")
            target_id = spec.target_agent
        elif target_id is not None and target_id not in obs:
            raise KeyError(f"Target observation for agent '{target_id}' missing")

        return builder(obs, ego_id, spec, target_id)

    # ------------------------------------------------------------------
    def _centerline_feature_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        include_lateral = bool(params.get("include_lateral", True))
        include_longitudinal = bool(params.get("include_longitudinal", True))
        include_progress = bool(params.get("include_progress", True))
        include_heading_raw = bool(params.get("include_heading_raw", False))

        raw_mode = params.get("angle_mode", "raw")
        heading_mode = str(raw_mode).lower() if raw_mode is not None else "raw"
        if heading_mode in {"none", "off", "false"}:
            heading_mode = "none"

        include_waypoint = bool(params.get("include_waypoint", False))
        waypoint_mode = str(params.get("waypoint_mode", "relative")).lower()
        if waypoint_mode not in {"relative", "absolute"}:
            waypoint_mode = "relative"

        waypoint_offsets: Tuple[float, ...] = ()
        waypoint_dim = 0
        waypoint_include_position = bool(params.get("waypoint_include_position", True))
        waypoint_include_distance = bool(params.get("waypoint_include_distance", False))
        per_waypoint_dim = 0
        if waypoint_include_position:
            per_waypoint_dim += 2
        if waypoint_include_distance:
            per_waypoint_dim += 1
        waypoint_distance_scale = params.get("waypoint_distance_scale")
        if waypoint_distance_scale is not None:
            try:
                waypoint_distance_scale = float(waypoint_distance_scale)
            except (TypeError, ValueError):
                waypoint_distance_scale = None
        waypoint_count_limit_raw = params.get("waypoint_spacing_count")
        if waypoint_count_limit_raw is None:
            waypoint_count_limit_raw = params.get("waypoint_count")
        try:
            waypoint_count_limit = None if waypoint_count_limit_raw is None else int(waypoint_count_limit_raw)
        except (TypeError, ValueError):
            waypoint_count_limit = None
        if waypoint_count_limit is not None and waypoint_count_limit <= 0:
            waypoint_count_limit = None
        if include_waypoint:
            offsets: List[float] = []
            raw_offsets = params.get("waypoint_progress")
            if raw_offsets is not None:
                if isinstance(raw_offsets, (float, int)):
                    raw_offsets = [raw_offsets]
                for value in raw_offsets:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    offsets.append(numeric)
            step_value = params.get("waypoint_progress_step")
            try:
                step_numeric = float(step_value)
            except (TypeError, ValueError):
                step_numeric = 0.0
            if step_numeric > 0.0:
                offsets.append(step_numeric)

            spacing_value = params.get("waypoint_spacing")
            try:
                spacing_numeric = float(spacing_value)
            except (TypeError, ValueError):
                spacing_numeric = 0.0
            if spacing_numeric > 0.0 and self.centerline_points is not None:
                spacing_offsets = progress_from_spacing(self.centerline_points, spacing_numeric)
                if spacing_offsets:
                    offsets.extend(spacing_offsets)

            normalized: List[float] = []
            for value in offsets:
                if not np.isfinite(value):
                    continue
                if value <= 0.0:
                    continue
                frac = float(value % 1.0)
                if frac <= 0.0:
                    continue
                normalized.append(frac)
            if normalized:
                unique = tuple(sorted(set(normalized)))
                if waypoint_count_limit is not None and waypoint_count_limit < len(unique):
                    unique = unique[:waypoint_count_limit]
                waypoint_offsets = unique
                pair_count = len(waypoint_offsets)
                waypoint_dim = per_waypoint_dim * pair_count if per_waypoint_dim > 0 else 0
            else:
                pair_count = 1
                waypoint_dim = per_waypoint_dim if per_waypoint_dim > 0 else 0
        else:
            pair_count = 0

        if heading_mode == "sin_cos":
            heading_dim = 2
        elif heading_mode in {"sin", "cos"}:
            heading_dim = 1
        elif heading_mode == "none":
            heading_dim = 0
        else:
            heading_mode = "raw"
            heading_dim = 1
            include_heading_raw = False  # raw already present

        if heading_mode == "none":
            include_heading_raw = False

        size = 0
        if include_lateral:
            size += 1
        if include_longitudinal:
            size += 1
        size += heading_dim
        if include_heading_raw:
            size += 1
        if include_progress:
            size += 1
        size += waypoint_dim

        return {
            "include_lateral": include_lateral,
            "include_longitudinal": include_longitudinal,
            "include_progress": include_progress,
            "include_heading_raw": include_heading_raw,
            "heading_mode": heading_mode,
            "include_waypoint": include_waypoint,
            "waypoint_mode": waypoint_mode,
            "waypoint_lookahead": int(max(1, int(params.get("waypoint_lookahead", 1)))) if include_waypoint else 0,
            "waypoint_scale": float(params.get("waypoint_scale", self.max_scan)) if include_waypoint else 1.0,
            "waypoint_progress_offsets": waypoint_offsets,
            "waypoint_feature_dim": per_waypoint_dim,
            "waypoint_include_position": waypoint_include_position,
            "waypoint_include_distance": waypoint_include_distance,
            "waypoint_distance_scale": waypoint_distance_scale,
            "size": size,
        }

    def _prepare_centerline_features(
        self,
        ego_id: str,
        ego_obs: Dict[str, Any],
        params: Dict[str, Any],
    ) -> np.ndarray:
        plan = self._centerline_feature_plan(params)
        size = max(plan["size"], 1)
        if self.centerline_points is None or self.centerline_points.size == 0:
            return np.zeros(size, dtype=np.float32)

        pose = to_numpy(ego_obs.get("pose", ()), flatten=True)
        if pose.size < 3:
            return np.zeros(size, dtype=np.float32)

        x, y, theta = map(float, pose[:3])
        last_index = self._centerline_last_index.get(ego_id)
        projection: CenterlineProjection
        try:
            projection = project_to_centerline(
                self.centerline_points,
                np.array([x, y], dtype=np.float32),
                theta,
                last_index=last_index,
            )
        except ValueError:
            return np.zeros(size, dtype=np.float32)

        self._centerline_last_index[ego_id] = projection.index

        features: List[float] = []

        if plan["include_lateral"]:
            lateral = float(projection.lateral_error)
            lateral_scale = params.get("lateral_scale")
            if lateral_scale is None and self.centerline_normalize and self.max_scan > 0.0:
                lateral_scale = self.max_scan
            if lateral_scale:
                scale = float(lateral_scale)
                if scale != 0.0:
                    lateral /= scale
            features.append(lateral)

        if plan["include_longitudinal"]:
            longitudinal = float(projection.longitudinal_error)
            longitudinal_scale = params.get("longitudinal_scale")
            if longitudinal_scale is None and self.centerline_normalize and self.max_scan > 0.0:
                longitudinal_scale = self.max_scan
            if longitudinal_scale:
                scale = float(longitudinal_scale)
                if scale != 0.0:
                    longitudinal /= scale
            features.append(longitudinal)

        heading_value = float(projection.heading_error)
        heading_mode = plan["heading_mode"]
        if heading_mode == "sin_cos":
            features.extend([float(np.sin(heading_value)), float(np.cos(heading_value))])
        elif heading_mode == "sin":
            features.append(float(np.sin(heading_value)))
        elif heading_mode == "cos":
            features.append(float(np.cos(heading_value)))
        elif heading_mode == "raw":
            features.append(heading_value)

        if plan["include_heading_raw"]:
            features.append(heading_value)

        if plan["include_progress"]:
            progress_val = float(projection.progress)
            progress_scale = params.get("progress_scale")
            if progress_scale:
                scale = float(progress_scale)
                if scale != 0.0:
                    progress_val /= scale
            features.append(progress_val)

        if plan["include_waypoint"]:
            points = self.centerline_points[:, :2]
            offsets: Tuple[float, ...] = plan.get("waypoint_progress_offsets", ())
            pair_count = len(offsets) if offsets else 1
            per_waypoint_dim = int(plan.get("waypoint_feature_dim", 2))

            def encode_point(point_xy: np.ndarray) -> Tuple[float, ...]:
                payload: List[float] = []
                raw_dx = float(point_xy[0] - x)
                raw_dy = float(point_xy[1] - y)
                waypoint_scale = plan.get("waypoint_scale", 0.0)
                if plan.get("waypoint_include_position", True) and per_waypoint_dim:
                    if plan["waypoint_mode"] == "absolute":
                        wx = float(point_xy[0])
                        wy = float(point_xy[1])
                        if waypoint_scale:
                            scale = float(waypoint_scale)
                            if scale != 0.0:
                                wx /= scale
                                wy /= scale
                    else:
                        wx = raw_dx
                        wy = raw_dy
                        if waypoint_scale:
                            scale = float(waypoint_scale)
                            if scale != 0.0:
                                wx /= scale
                                wy /= scale
                    payload.extend([wx, wy])
                if plan.get("waypoint_include_distance", False):
                    dist = float(np.hypot(raw_dx, raw_dy))
                    distance_scale = plan.get("waypoint_distance_scale")
                    if distance_scale is None:
                        distance_scale = plan.get("waypoint_scale")
                    if distance_scale:
                        scale = float(distance_scale)
                        if scale != 0.0:
                            dist /= scale
                    payload.append(dist)
                return tuple(payload)

            if points.size == 0:
                zero_count = per_waypoint_dim * pair_count
                if zero_count > 0:
                    features.extend([0.0] * zero_count)
            else:
                total_points = points.shape[0]
                if offsets:
                    denom = max(total_points - 1, 1)
                    base_progress = float(projection.progress)
                    for offset in offsets:
                        target_progress = (base_progress + float(offset)) % 1.0
                        scaled = target_progress * denom
                        target_index = int(np.floor(scaled + 0.5)) % total_points
                        target_point = points[target_index]
                        features.extend(encode_point(target_point))
                else:
                    next_index = (projection.index + plan["waypoint_lookahead"]) % total_points
                    target_point = points[next_index]
                    features.extend(encode_point(target_point))

        result = np.asarray(features, dtype=np.float32)
        if result.size < size:
            padded = np.zeros(size, dtype=np.float32)
            padded[: result.size] = result
            return padded
        return result

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------
    def _component_lidar(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        _: Optional[str],
    ) -> np.ndarray:
        ego_obs = obs[ego_id]
        scan = ego_obs.get("scans")
        if scan is None:
            scan = ego_obs.get("lidar")
        if scan is None:
            raise KeyError("Observation is missing 'scans'/'lidar' entry for lidar component")

        beams = spec.params.get("beams")
        beams_val = int(beams) if beams is not None else self.lidar_beams
        lidar = downsample_lidar(scan, beams_val)

        max_range = float(spec.params.get("max_range", self.max_scan))
        if spec.params.get("normalize", self.normalize) and max_range > 0.0:
            lidar = lidar / max_range

        clip_val = spec.params.get("clip")
        if clip_val is not None:
            lidar = np.clip(lidar, 0.0, float(clip_val))

        return lidar.astype(np.float32, copy=False)

    def _component_ego_pose(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        _: Optional[str],
    ) -> np.ndarray:
        return self._component_pose(obs, ego_id, spec, None)

    def _component_pose(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        target_id: Optional[str],
    ) -> np.ndarray:
        agent_id = target_id or ego_id
        agent_obs = obs.get(agent_id)
        if agent_obs is None:
            raise KeyError(f"Observation is missing pose for agent '{agent_id}'")

        pose = to_numpy(agent_obs.get("pose", ()), flatten=True)
        if pose.size < 3:
            raise ValueError(f"Pose observation for '{agent_id}' must contain at least 3 values")

        x, y, theta = map(float, pose[:3])
        pieces: List[float] = []

        normalize_xy = spec.params.get("normalize_xy")
        if normalize_xy is None and spec.type == "ego_pose":
            normalize_xy = self.max_scan
        if normalize_xy:
            scale = float(normalize_xy)
            if scale != 0.0:
                x /= scale
                y /= scale

        if spec.params.get("include_xy", True):
            pieces.extend([x, y])

        angle_mode = spec.params.get("angle_mode", "raw")
        if angle_mode == "sin":
            pieces.append(float(np.sin(theta)))
        elif angle_mode == "sin_cos":
            pieces.extend([float(np.sin(theta)), float(np.cos(theta))])
        elif angle_mode == "cos":
            pieces.append(float(np.cos(theta)))
        elif angle_mode == "raw":
            pieces.append(theta)
        elif angle_mode is None:
            pass
        else:
            raise ValueError(f"Unsupported angle_mode '{angle_mode}' for pose component")

        if spec.params.get("include_theta", False) and angle_mode != "raw":
            pieces.append(theta)

        return np.array(pieces, dtype=np.float32)

    def _component_velocity(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        target_id: Optional[str],
    ) -> np.ndarray:
        agent_id = target_id or ego_id
        agent_obs = obs.get(agent_id, {})
        velocity = agent_obs.get("velocity")
        if velocity is None:
            raise KeyError(f"Velocity data missing for agent '{agent_id}'")

        arr = to_numpy(velocity, flatten=True)
        if arr.size < 2:
            raise ValueError(f"Velocity for '{agent_id}' must contain at least two elements")

        vx, vy = float(arr[0]), float(arr[1])
        pieces: List[float] = [vx, vy]

        if spec.params.get("include_speed"):
            pieces.append(float(np.linalg.norm([vx, vy])))

        normalize = spec.params.get("normalize")
        if normalize:
            scale = float(normalize)
            if scale != 0.0:
                pieces = [value / scale for value in pieces]

        return np.asarray(pieces, dtype=np.float32)

    def _component_collision(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        _: ComponentSpec,
        target_id: Optional[str],
    ) -> np.ndarray:
        agent_id = target_id or ego_id
        agent_obs = obs.get(agent_id, {})
        value = float(agent_obs.get("collision", agent_obs.get("target_collision", 0.0)))
        return np.array([value], dtype=np.float32)

    def _component_lap(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        target_id: Optional[str],
    ) -> np.ndarray:
        agent_id = target_id or ego_id
        agent_obs = obs.get(agent_id, {})
        payload = agent_obs.get("lap")
        if payload is None:
            raise KeyError(f"Lap information missing for agent '{agent_id}'")

        arr = to_numpy(payload, flatten=True)
        if arr.size < 2:
            raise ValueError("Lap component expects at least count and time")

        values = [float(arr[0]), float(arr[1])]
        normalize_time = spec.params.get("normalize_time")
        if normalize_time:
            scale = float(normalize_time)
            if scale != 0:
                values[1] /= scale

        return np.asarray(values, dtype=np.float32)

    def _component_target_pose(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        target_id: Optional[str],
    ) -> np.ndarray:
        if target_id is None:
            raise ValueError("target_pose component requires a target agent")
        return self._component_pose(obs, ego_id, spec, target_id)

    def _component_relative_pose(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        target_id: Optional[str],
    ) -> np.ndarray:
        if target_id is None:
            raise ValueError("relative_pose component requires a target agent")

        ego_pose = to_numpy(obs[ego_id].get("pose", ()), flatten=True)
        target_pose = to_numpy(obs[target_id].get("pose", ()), flatten=True)
        if ego_pose.size < 3 or target_pose.size < 3:
            raise ValueError("relative_pose requires pose entries with at least three values")

        dx = float(target_pose[0] - ego_pose[0])
        dy = float(target_pose[1] - ego_pose[1])
        dtheta = float(target_pose[2] - ego_pose[2])

        normalize_xy = spec.params.get("normalize_xy")
        if normalize_xy:
            scale = float(normalize_xy)
            if scale != 0.0:
                dx /= scale
                dy /= scale

        angle_mode = spec.params.get("angle_mode", "raw")
        if angle_mode == "sin":
            angle_terms = [float(np.sin(dtheta))]
        elif angle_mode == "sin_cos":
            angle_terms = [float(np.sin(dtheta)), float(np.cos(dtheta))]
        elif angle_mode == "raw":
            angle_terms = [dtheta]
        elif angle_mode is None:
            angle_terms = []
        else:
            raise ValueError(f"Unsupported angle_mode '{angle_mode}' for relative_pose component")

        pieces = [dx, dy]
        pieces.extend(angle_terms)
        return np.asarray(pieces, dtype=np.float32)

    def _component_distance(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        target_id: Optional[str],
    ) -> np.ndarray:
        if target_id is None:
            raise ValueError("distance component requires a target agent")

        ego_pose = to_numpy(obs[ego_id].get("pose", ()), flatten=True)
        target_pose = to_numpy(obs[target_id].get("pose", ()), flatten=True)
        if ego_pose.size < 2 or target_pose.size < 2:
            raise ValueError("distance component requires pose entries with x/y")

        dx = float(target_pose[0] - ego_pose[0])
        dy = float(target_pose[1] - ego_pose[1])
        dist = float(np.linalg.norm([dx, dy]))

        normalize = spec.params.get("normalize")
        if normalize:
            scale = float(normalize)
            if scale != 0.0:
                dist /= scale

        return np.array([dist], dtype=np.float32)

    def _component_centerline(
        self,
        obs: Dict[str, Dict[str, Any]],
        ego_id: str,
        spec: ComponentSpec,
        __: Optional[str],
    ) -> np.ndarray:
        ego_obs = obs.get(ego_id, {})
        return self._prepare_centerline_features(ego_id, ego_obs, spec.params)


__all__ = ["ObsWrapper"]
