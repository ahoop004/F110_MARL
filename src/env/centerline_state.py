"""Centerline and finish-line state helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.utils.centerline import progress_from_spacing


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
        "direction": direction_unit,
        "min_speed": min_speed,
    }


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
