"""Centerline and finish-line state helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.utils.centerline import (
    centerline_heading,
    progress_from_spacing,
    project_to_centerline,
)
from src.env.collision_state import RaceLifecycle


@dataclass
class CenterlineRuntimeState:
    """Mutable centerline/render-feature state owned outside the env core."""

    render_auto: bool = True
    feature_auto: bool = True
    autoload_auto: bool = True
    render_enabled: bool = False
    feature_requested: bool = False
    features_enabled: bool = False
    render_progress: Tuple[float, ...] = field(default_factory=tuple)
    render_spacing: float = 0.0
    render_connect: bool = True
    points: Optional[np.ndarray] = None
    path: Optional[Path] = None
    render_points: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CenterlineRuntimeState":
        render_user_override = bool(cfg.pop("_centerline_render_user_override", False))
        feature_user_override = bool(cfg.pop("_centerline_features_user_override", False))
        autoload_user_override = bool(cfg.pop("_centerline_autoload_user_override", False))

        render_cfg_value = cfg.get("centerline_render")
        features_cfg_value = cfg.get("centerline_features")

        render_cfg = render_cfg_value if render_user_override else None
        features_cfg = features_cfg_value if feature_user_override else None

        render_progress = normalize_progress_fractions(cfg.get("centerline_render_progress"))
        spacing_value = cfg.get("centerline_render_spacing")
        try:
            render_spacing = max(float(spacing_value), 0.0)
        except (TypeError, ValueError):
            render_spacing = 0.0

        raw_connect = cfg.get("centerline_render_connect")
        if raw_connect is None:
            render_connect = not bool(render_progress)
        else:
            render_connect = bool(raw_connect)

        feature_requested = bool(features_cfg) if features_cfg is not None else False
        return cls(
            render_auto=not render_user_override,
            feature_auto=not feature_user_override,
            autoload_auto=not autoload_user_override,
            render_enabled=bool(render_cfg) if render_cfg is not None else False,
            feature_requested=feature_requested,
            features_enabled=feature_requested,
            render_progress=render_progress,
            render_spacing=render_spacing,
            render_connect=render_connect,
        )

    def build_render_points(self) -> Optional[np.ndarray]:
        return build_render_centerline_points(
            self.points,
            self.render_progress,
            self.render_spacing,
        )

    def set_centerline(self, centerline: Optional[np.ndarray], *, path: Optional[Path] = None) -> None:
        if centerline is not None:
            array = np.asarray(centerline, dtype=np.float32)
            array.setflags(write=False)
        else:
            array = None
        self.points = array
        self.path = path.resolve() if path is not None else None
        self.features_enabled = self.feature_requested and array is not None
        self.render_points = self.build_render_points()

    def register_usage(self, *, require_render: bool = False, require_features: bool = False) -> bool:
        changed = False
        if require_features and self.feature_auto and not self.features_enabled:
            self.features_enabled = True
            self.feature_requested = True
            changed = True
        if require_render and self.render_auto and not self.render_enabled:
            self.render_enabled = True
            changed = True
        if changed:
            self.render_points = self.build_render_points()
        return changed


def apply_centerline_to_renderer(renderer: Any, state: CenterlineRuntimeState) -> None:
    if renderer is None:
        return
    if state.render_enabled:
        renderer.update_centerline(state.render_points, connect=state.render_connect)
    else:
        renderer.update_centerline(None)


# ---------------------------------------------------------------------------
# Per-episode progress facts derived from centerline projection
# ---------------------------------------------------------------------------

_HALF_LAP = 0.5  # wrap-around guard for progress delta


class CenterlineProgressTracker:
    """Tracks per-step centerline projection facts for all agents.

    Call :meth:`reset` on each episode reset and :meth:`update` each step
    when centerline features are enabled.  The per-agent nearest-waypoint
    index is carried across steps so that :func:`project_to_centerline` can
    use an efficient windowed search rather than a full O(N) scan.

    The dict returned by :meth:`update` is keyed by agent_id and contains:

    ``progress``
        Normalised arc-length position in [0, 1].
    ``progress_delta``
        Signed progress change from the previous step, wrap-corrected at the
        start/finish crossing.
    ``d``
        Cross-track (lateral) deviation in metres (positive = left of track).
    ``vs``
        Speed component along the track tangent (positive = forward).
    ``vd``
        Speed component perpendicular to the track tangent.
    ``heading_error``
        Ego heading minus track tangent, normalised to [-pi, pi].
    ``wrong_way``
        True when |heading_error| > *wrong_way_threshold* (default pi/2).
    """

    def __init__(
        self,
        agent_ids: Sequence[str],
        *,
        search_window: int = 50,
        wrong_way_threshold: float = math.pi / 2,
    ) -> None:
        self._agent_ids: List[str] = list(agent_ids)
        self.search_window = int(search_window)
        self.wrong_way_threshold = float(wrong_way_threshold)
        self._last_indices: Dict[str, int] = {}
        self._prev_progress: Dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        """Reset tracking state at the start of a new episode."""
        for aid in self._agent_ids:
            self._last_indices[aid] = -1
            self._prev_progress[aid] = -1.0  # sentinel: no previous value yet

    def update(
        self,
        centerline: np.ndarray,
        poses_x: np.ndarray,
        poses_y: np.ndarray,
        poses_theta: np.ndarray,
        linear_vels_x: np.ndarray,
        linear_vels_y: np.ndarray,
        agent_index: Mapping[str, int],
    ) -> Dict[str, Dict[str, float]]:
        """Project all agents onto *centerline* and return per-agent facts."""
        result: Dict[str, Dict[str, float]] = {}
        if centerline is None or centerline.ndim != 2 or centerline.shape[0] < 2:
            return result

        for aid in self._agent_ids:
            idx = agent_index.get(aid)
            if idx is None:
                continue

            pos = np.array(
                [float(poses_x[idx]), float(poses_y[idx])], dtype=np.float32
            )
            heading = float(poses_theta[idx])
            vx = float(linear_vels_x[idx])
            vy = float(linear_vels_y[idx])

            last_idx = self._last_indices.get(aid, -1)
            proj = project_to_centerline(
                centerline,
                pos,
                heading,
                last_index=last_idx if last_idx >= 0 else None,
                search_window=self.search_window,
            )
            self._last_indices[aid] = proj.index

            # Speed projections along and perpendicular to the track tangent.
            tangent_theta = centerline_heading(centerline, proj.index)
            cos_t = math.cos(tangent_theta)
            sin_t = math.sin(tangent_theta)
            vs = vx * cos_t + vy * sin_t       # forward
            vd = -vx * sin_t + vy * cos_t      # lateral

            # Progress delta, wrap-corrected at the start/finish line.
            prev = self._prev_progress.get(aid, -1.0)
            if prev < 0.0:
                # First step of the episode — no meaningful delta yet.
                delta = 0.0
            else:
                delta = proj.progress - prev
                if delta > _HALF_LAP:
                    delta -= 1.0
                elif delta < -_HALF_LAP:
                    delta += 1.0
            self._prev_progress[aid] = proj.progress

            wrong_way = abs(proj.heading_error) > self.wrong_way_threshold

            result[aid] = {
                "progress": proj.progress,
                "progress_delta": delta,
                "d": proj.lateral_error,
                "vs": vs,
                "vd": vd,
                "heading_error": proj.heading_error,
                "wrong_way": wrong_way,
            }
        return result


def normalize_progress_fractions(raw: Optional[Any]) -> Tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (float, int)):
        candidates: List[Any] = [raw]
    elif isinstance(raw, str):
        candidates = [raw]
    else:
        try:
            candidates = list(raw)  # type: ignore[arg-type]
        except TypeError:
            candidates = [raw]

    fractions: List[float] = []
    for value in candidates:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        frac = numeric % 1.0
        if frac <= 0.0:
            continue
        fractions.append(frac)
    if not fractions:
        return ()
    return tuple(sorted(set(fractions)))


def build_render_centerline_points(
    centerline: Optional[np.ndarray],
    progress_fractions: Sequence[float],
    spacing: float,
) -> Optional[np.ndarray]:
    if centerline is None:
        return None
    if centerline.ndim != 2 or centerline.shape[0] == 0:
        return centerline

    fractions: List[float] = list(progress_fractions)
    try:
        spacing_value = float(spacing)
    except (TypeError, ValueError):
        spacing_value = 0.0
    if spacing_value > 0.0:
        spacing_fracs = progress_from_spacing(centerline, spacing_value)
        if spacing_fracs:
            fractions.extend(spacing_fracs)
    if not fractions:
        return centerline

    unique = sorted(set(frac for frac in fractions if 0.0 < frac < 1.0))
    if not unique:
        return centerline
    denom = max(centerline.shape[0] - 1, 1)
    indices = set()
    for frac in unique:
        idx = int(round(frac * denom)) % centerline.shape[0]
        indices.add(idx)
    if not indices:
        return centerline
    ordered = sorted(indices)
    return centerline[ordered]


def parse_finish_line(cfg: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(cfg, Mapping):
        return None
    start_raw = cfg.get("start")
    end_raw = cfg.get("end")
    if start_raw is None or end_raw is None:
        return None
    try:
        start = np.asarray(start_raw, dtype=np.float32).reshape(-1)
        end = np.asarray(end_raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if start.size < 2 or end.size < 2:
        return None
    start = start[:2]
    end = end[:2]
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length <= 1e-6:
        return None
    segment_unit = segment / length
    length_sq = float(length * length)
    tolerance = _float_with_default(
        cfg.get("tolerance", cfg.get("thickness", cfg.get("width", 1.0))),
        1.0,
    )
    tolerance = max(tolerance, 1e-3)
    padding = max(_float_with_default(cfg.get("padding", 0.5), 0.5), 0.0)
    hysteresis = max(_float_with_default(cfg.get("hysteresis", 0.5), 0.5), 1e-3)

    direction_unit: Optional[np.ndarray] = None
    dir_vec = cfg.get("direction")
    if dir_vec is not None:
        try:
            dir_arr = np.asarray(dir_vec, dtype=np.float32).reshape(-1)
            if dir_arr.size >= 2:
                norm = float(np.linalg.norm(dir_arr[:2]))
                if norm > 0.0:
                    direction_unit = dir_arr[:2] / norm
        except (TypeError, ValueError):
            direction_unit = None

    min_speed = max(
        _float_with_default(cfg.get("trigger_speed", cfg.get("min_speed", 0.0)), 0.0),
        0.0,
    )
    return {
        "start": start,
        "end": end,
        "segment": segment,
        "segment_unit": segment_unit,
        "segment_length": length,
        "segment_length_sq": length_sq,
        "tolerance": tolerance,
        "padding": padding,
        "hysteresis": hysteresis,
        "direction": direction_unit,
        "min_speed": min_speed,
    }


def resolve_finish_line_config(
    explicit: Optional[Mapping[str, Any]],
    map_metadata: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Resolve scenario override first, then the map's shared annotation."""
    if isinstance(explicit, Mapping):
        return explicit
    metadata = map_metadata or {}
    annotations = metadata.get("annotations", {})
    if isinstance(annotations, Mapping):
        annotated = annotations.get("finish_line")
        if isinstance(annotated, Mapping):
            return annotated
    top_level = metadata.get("finish_line")
    return top_level if isinstance(top_level, Mapping) else None


def validate_finish_line(
    cfg: Optional[Mapping[str, Any]],
    *,
    centerline: Optional[np.ndarray] = None,
    spawn_poses: Optional[np.ndarray] = None,
    proximity_tolerance: float = 5.0,
) -> Dict[str, Any]:
    """Parse finish geometry and reject ambiguous or misplaced annotations."""
    parsed = parse_finish_line(cfg)
    if parsed is None:
        raise ValueError("finish_line requires two distinct start/end points")
    if parsed["direction"] is None:
        raise ValueError("finish_line.direction must be a non-zero 2D vector")

    midpoint = (parsed["start"] + parsed["end"]) * 0.5
    centerline_distance: Optional[float] = None
    if centerline is not None:
        points = np.asarray(centerline, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] and points.shape[1] >= 2:
            centerline_distance = float(
                np.min(np.linalg.norm(points[:, :2] - midpoint, axis=1))
            )
    if (
        centerline_distance is not None
        and centerline_distance > float(proximity_tolerance)
    ):
        raise ValueError("finish_line is not close to the centerline")
    return parsed


class LapTracker:
    """Shared-line, forward-only multi-lap tracker for every physical agent."""

    def __init__(
        self,
        agent_ids: Sequence[str],
        finish_line: Mapping[str, Any],
        lifecycle: RaceLifecycle,
    ) -> None:
        self.agent_ids = tuple(str(agent_id) for agent_id in agent_ids)
        self.finish_line = dict(finish_line)
        self.lifecycle = lifecycle
        self._previous = np.zeros(len(self.agent_ids), dtype=np.float32)
        self._armed = np.zeros(len(self.agent_ids), dtype=bool)

        segment = np.asarray(self.finish_line["segment"], dtype=np.float32)
        normal = np.array([-segment[1], segment[0]], dtype=np.float32)
        direction = np.asarray(self.finish_line["direction"], dtype=np.float32)
        orientation = float(np.dot(direction, normal))
        if abs(orientation) <= 1e-6:
            raise ValueError("finish_line.direction must cross, not run along, the line")
        self._forward_sign = 1.0 if orientation > 0.0 else -1.0

    def reset(self, poses_x: np.ndarray, poses_y: np.ndarray) -> None:
        self.lifecycle.reset()
        hysteresis = float(self.finish_line["hysteresis"])
        for idx, _agent_id in enumerate(self.agent_ids):
            point = np.array([poses_x[idx], poses_y[idx]], dtype=np.float32)
            oriented = self._oriented_distance(point)
            self._previous[idx] = oriented
            # Spawning on or beyond the completed side cannot count immediately.
            self._armed[idx] = oriented <= -hysteresis

    def update(
        self,
        poses_x: np.ndarray,
        poses_y: np.ndarray,
        linear_vels_x: np.ndarray,
        linear_vels_y: np.ndarray,
        *,
        step: int,
    ) -> Dict[str, bool]:
        """Return per-agent crossing events for this simulator step."""
        self.lifecycle.begin_step()
        crossings = {agent_id: False for agent_id in self.agent_ids}
        hysteresis = float(self.finish_line["hysteresis"])
        direction = np.asarray(self.finish_line["direction"], dtype=np.float32)
        min_speed = float(self.finish_line["min_speed"])

        for idx, agent_id in enumerate(self.agent_ids):
            point = np.array([poses_x[idx], poses_y[idx]], dtype=np.float32)
            current = self._oriented_distance(point)
            previous = float(self._previous[idx])
            self._previous[idx] = current
            record = self.lifecycle.records[agent_id]
            if not record.is_active:
                continue
            if not self._armed[idx]:
                if current <= -hysteresis:
                    self._armed[idx] = True
                continue
            if not (previous < 0.0 <= current):
                continue

            # Simulator ``linear_vels_x`` is longitudinal/body-frame speed;
            # the oriented sign change supplies the world-frame direction.
            if float(linear_vels_x[idx]) < min_speed:
                continue
            if not self._crossing_within_segment(point, previous, current):
                continue

            crossings[agent_id] = True
            self._armed[idx] = False
            self.lifecycle.record_lap_crossing(agent_id, step=step)
        return crossings

    def _oriented_distance(self, point: np.ndarray) -> float:
        return self._forward_sign * signed_distance_to_finish(self.finish_line, point)

    def _crossing_within_segment(
        self,
        current_point: np.ndarray,
        previous_distance: float,
        current_distance: float,
    ) -> bool:
        denom = current_distance - previous_distance
        if abs(denom) <= 1e-9:
            return False
        velocity_fraction = -previous_distance / denom
        # Recover the prior point from this step's displacement direction is
        # unavailable here; using the current point is safe at simulator-scale
        # steps and the configured endpoint padding covers the small offset.
        rel = current_point - self.finish_line["start"]
        projection = float(np.dot(rel, self.finish_line["segment"]) / self.finish_line["segment_length_sq"])
        padding = float(self.finish_line["padding"])
        return 0.0 <= velocity_fraction <= 1.0 and -padding <= projection <= 1.0 + padding


def reset_finish_line_tracking(
    finish_line_data: Optional[Mapping[str, Any]],
    finish_signed_prev: Optional[np.ndarray],
    finish_crossed: Optional[np.ndarray],
    poses_x: np.ndarray,
    poses_y: np.ndarray,
    n_agents: int,
) -> None:
    if finish_line_data is None or finish_signed_prev is None or finish_crossed is None:
        return
    finish_crossed.fill(False)
    for idx in range(n_agents):
        point = np.array([poses_x[idx], poses_y[idx]], dtype=np.float32)
        finish_signed_prev[idx] = signed_distance_to_finish(finish_line_data, point)


def signed_distance_to_finish(
    finish_line_data: Optional[Mapping[str, Any]],
    point: np.ndarray,
) -> float:
    if finish_line_data is None:
        return 0.0
    rel = point - finish_line_data["start"]
    seg_unit = finish_line_data["segment_unit"]
    cross = seg_unit[0] * rel[1] - seg_unit[1] * rel[0]
    return float(cross)


def update_finish_line_progress(
    finish_line_data: Optional[Mapping[str, Any]],
    finish_signed_prev: Optional[np.ndarray],
    finish_crossed: Optional[np.ndarray],
    possible_agents: Sequence[str],
    poses_x: np.ndarray,
    poses_y: np.ndarray,
    linear_vels_x: np.ndarray,
    linear_vels_y: np.ndarray,
) -> Dict[str, bool]:
    if (
        finish_line_data is None
        or finish_signed_prev is None
        or finish_crossed is None
        or not possible_agents
    ):
        return {}

    completed: Dict[str, bool] = {}
    segment = finish_line_data["segment"]
    len_sq = finish_line_data["segment_length_sq"]
    tolerance = finish_line_data["tolerance"]
    padding = finish_line_data["padding"]
    direction = finish_line_data["direction"]
    min_speed = finish_line_data["min_speed"]
    for idx, agent_id in enumerate(possible_agents):
        if finish_crossed[idx]:
            continue
        point = np.array([poses_x[idx], poses_y[idx]], dtype=np.float32)
        rel = point - finish_line_data["start"]
        proj = float(np.dot(rel, segment) / len_sq)
        if proj < -padding or proj > 1.0 + padding:
            finish_signed_prev[idx] = signed_distance_to_finish(finish_line_data, point)
            continue
        curr_signed = signed_distance_to_finish(finish_line_data, point)
        prev_signed = finish_signed_prev[idx]
        finish_signed_prev[idx] = curr_signed
        sign_switch = (prev_signed <= 0.0 < curr_signed) or (prev_signed >= 0.0 > curr_signed)
        if not sign_switch:
            continue
        if abs(curr_signed) > tolerance and abs(prev_signed) > tolerance:
            continue
        if direction is not None:
            vel = np.array([linear_vels_x[idx], linear_vels_y[idx]], dtype=np.float32)
            if float(np.dot(vel, direction)) < min_speed:
                continue
        finish_crossed[idx] = True
        completed[str(agent_id)] = True
    return completed


def inject_finish_line_info(
    finish_line_data: Optional[Mapping[str, Any]],
    finish_crossed: Optional[np.ndarray],
    agent_id_to_index: Mapping[str, int],
    infos: Mapping[str, Dict[str, Any]],
) -> None:
    if finish_line_data is None or finish_crossed is None:
        return
    for agent_id, idx in agent_id_to_index.items():
        payload = infos.get(agent_id)
        if payload is None:
            continue
        payload["finish_line"] = bool(finish_crossed[idx])


def _float_with_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
