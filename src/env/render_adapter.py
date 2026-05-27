"""Render adapter — helpers for building render observations and flushing
RenderRuntimeState to an EnvRenderer each frame.

Extracting this logic out of ``F110ParallelEnv`` keeps the env coordinator
free of renderer-internal details.  All public functions take their
dependencies as explicit arguments so they are testable in isolation.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.render.render_state import RenderRuntimeState

# Sector/radial helpers shared with wrappers
from wrappers.common import _SECTOR_NAMES, _radial_gain, _sector_from_angle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-agent relative snapshot
# ---------------------------------------------------------------------------

def compute_relative_snapshot(
    agent_id: str,
    *,
    agent_index: Mapping[str, int],
    agent_target_index: Mapping[str, Optional[int]],
    poses_x: np.ndarray,
    poses_y: np.ndarray,
    poses_theta: np.ndarray,
    reward_ring_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compute relative position/sector/ring facts for *agent_id* vs its target.

    Returns ``None`` when *agent_id* has no configured target or the index
    is out of range.
    """
    agent_idx = agent_index.get(agent_id)
    target_idx = agent_target_index.get(agent_id)
    if agent_idx is None or target_idx is None:
        return None
    n = len(poses_x)
    if not (0 <= agent_idx < n) or not (0 <= target_idx < n):
        return None

    agent_x = float(poses_x[agent_idx])
    agent_y = float(poses_y[agent_idx])
    agent_heading = float(poses_theta[agent_idx]) if agent_idx < len(poses_theta) else 0.0

    target_x = float(poses_x[target_idx])
    target_y = float(poses_y[target_idx])

    dx = target_x - agent_x
    dy = target_y - agent_y
    distance = float(math.hypot(dx, dy))

    if dx == 0.0 and dy == 0.0:
        sector = "front"
    else:
        rel_angle = math.degrees(math.atan2(dy, dx) - agent_heading)
        sector = _sector_from_angle(rel_angle)

    cfg = reward_ring_config or {}
    preferred = float(cfg.get("preferred_radius", 0.0) or 0.0)
    inner = float(cfg.get("inner_tolerance", 0.0) or 0.0)
    outer = float(cfg.get("outer_tolerance", 0.0) or 0.0)
    falloff = str(cfg.get("falloff", "linear") or "linear").lower()

    gain = _radial_gain(distance, preferred, inner, outer, falloff)
    sector_active = gain > 0.0

    if preferred > 0.0 or inner > 0.0 or outer > 0.0:
        lower = max(preferred - inner, 0.0)
        upper = preferred + outer
        in_ring = float(lower) <= distance <= float(upper)
    else:
        in_ring = sector_active

    weights_raw = cfg.get("weights")
    sector_weight = 0.0
    if isinstance(weights_raw, dict):
        try:
            sector_weight = float(weights_raw.get(sector, 0.0))
        except (TypeError, ValueError):
            sector_weight = 0.0
    reward_sector = sector_active and sector_weight > 0.0

    sector_code = "".join(w[0].upper() for w in sector.split("_")) if sector else "--"

    return {
        "distance": distance,
        "sector": sector,
        "sector_code": sector_code,
        "sector_active": bool(sector_active),
        "in_ring": bool(in_ring),
        "sector_weight": sector_weight,
        "reward_sector": bool(reward_sector),
    }


# ---------------------------------------------------------------------------
# Render observation assembly
# ---------------------------------------------------------------------------

def build_render_observations(
    active_agents: Sequence[str],
    obs: Optional[Dict[str, Dict[str, Any]]],
    *,
    agent_index: Mapping[str, int],
    agent_target_index: Mapping[str, Optional[int]],
    poses_x: np.ndarray,
    poses_y: np.ndarray,
    poses_theta: np.ndarray,
    linear_vels_x: np.ndarray,
    linear_vels_y: np.ndarray,
    lap_times: np.ndarray,
    lap_counts: np.ndarray,
    collisions: np.ndarray,
    render_state: RenderRuntimeState,
) -> Dict[str, Dict[str, Any]]:
    """Build per-agent render observation dicts from current physics state.

    Called each step when ``_collect_render_data`` is True.  The result is
    stored on ``render_state.render_obs`` by the env coordinator.
    """
    render_obs: Dict[str, Dict[str, Any]] = {}

    for aid in active_agents:
        idx = agent_index.get(aid)
        if idx is None:
            continue

        entry: Dict[str, Any] = {
            "poses_x": float(poses_x[idx]),
            "poses_y": float(poses_y[idx]),
            "poses_theta": float(poses_theta[idx]),
            "pose": np.array(
                [float(poses_x[idx]), float(poses_y[idx]), float(poses_theta[idx])],
                dtype=np.float32,
            ),
            "linear_vels_x": float(linear_vels_x[idx]),
            "linear_vels_y": float(linear_vels_y[idx]),
            "lap_time": float(lap_times[idx]),
            "lap_count": int(lap_counts[idx]),
            "collision": bool(collisions[idx]),
        }

        agent_obs = obs.get(aid) if obs is not None else None
        if agent_obs is not None:
            scan_entry = agent_obs.get("scans")
            if scan_entry is not None:
                entry["scans"] = scan_entry
                render_state.lidar_skip[aid] = render_state.lidar_skip_default

            target_collision_val = agent_obs.get("target_collision")
            if target_collision_val is not None:
                try:
                    entry["target_collision"] = bool(target_collision_val)
                except Exception:
                    entry["target_collision"] = False

            components: Dict[str, Any] = {}
            for key, value in agent_obs.items():
                if key in {"scans", "lidar", "state"}:
                    continue
                if key in entry and key not in {"collision", "target_collision"}:
                    continue
                cleaned: Any
                if key == "relative_sector":
                    arr = np.asarray(value, dtype=np.float32).flatten()
                    sector_flags: Dict[str, bool] = {}
                    for i, name in enumerate(_SECTOR_NAMES):
                        flag = bool(arr[i] > 0.5) if i < arr.size else False
                        sector_flags[name] = flag
                    cleaned = sector_flags
                elif isinstance(value, np.ndarray):
                    cleaned = value.astype(np.float32, copy=False).flatten()
                    if cleaned.size > 6:
                        cleaned = cleaned[:6]
                    cleaned = cleaned.tolist()
                elif isinstance(value, (np.bool_, bool)):
                    cleaned = bool(value)
                elif isinstance(value, (np.floating, np.integer)):
                    cleaned = float(value)
                else:
                    cleaned = value
                components[key] = cleaned
            if components:
                entry["obs_components"] = components

        snapshot = compute_relative_snapshot(
            aid,
            agent_index=agent_index,
            agent_target_index=agent_target_index,
            poses_x=poses_x,
            poses_y=poses_y,
            poses_theta=poses_theta,
            reward_ring_config=render_state.reward_ring_config,
        )
        if snapshot is not None:
            entry["relative"] = snapshot

        wrapped_vector = render_state.wrapped_obs.get(aid)
        if wrapped_vector is not None:
            entry["wrapped_obs"] = np.asarray(wrapped_vector, dtype=np.float32)
            entry["wrapped_skip"] = int(
                render_state.lidar_skip.get(aid, render_state.lidar_skip_default)
            )

        render_obs[aid] = entry

    return render_obs


# ---------------------------------------------------------------------------
# Renderer-state flush
# ---------------------------------------------------------------------------

def flush_render_state(
    renderer: Any,
    rs: RenderRuntimeState,
    *,
    _logger: Optional[logging.Logger] = None,
) -> None:
    """Apply all dirty state in *rs* to *renderer* in one pass.

    Each sub-apply only touches the renderer if the corresponding dirty flag
    is set, so calling this every frame is cheap when nothing has changed.
    """
    if renderer is None:
        return
    _apply_reward_ring(renderer, rs)
    _apply_metrics(renderer, rs)
    _apply_ticker(renderer, rs)
    _apply_obs(renderer, rs)
    _apply_heatmap(renderer, rs)
    _apply_overlays(renderer, rs)
    _apply_callbacks(renderer, rs, _logger=_logger)


def _apply_reward_ring(renderer: Any, rs: RenderRuntimeState) -> None:
    if rs.reward_ring_dirty:
        cfg = rs.reward_ring_config
        if cfg:
            payload: Dict[str, Any] = {
                "enabled": True,
                "preferred_radius": cfg["preferred_radius"],
                "inner_tolerance": cfg["inner_tolerance"],
                "outer_tolerance": cfg["outer_tolerance"],
                "segments": cfg.get("segments", 96),
                "marker_radius": cfg.get("marker_radius", 0.0),
                "marker_segments": cfg.get("marker_segments", 12),
                "offsets_only": cfg.get("offsets_only", False),
            }
            for key in ("fill_color", "border_color", "preferred_color", "marker_color"):
                if key in cfg:
                    payload[key] = cfg[key]
            if "offsets" in cfg:
                payload["offsets"] = cfg["offsets"]
            renderer.configure_reward_ring(**payload)
        else:
            renderer.configure_reward_ring(enabled=False)
        rs.reward_ring_dirty = False

    if rs.reward_ring_target_dirty:
        renderer.set_reward_ring_target(rs.reward_ring_target)
        rs.reward_ring_target_dirty = False

    if rs.reward_ring_marker_dirty:
        target_id = rs.reward_ring_target
        if target_id:
            states = rs.reward_ring_marker_states.get(target_id)
            try:
                renderer.set_reward_ring_marker_state(target_id, states)
            except Exception:
                pass
        rs.reward_ring_marker_dirty = False


def _apply_metrics(renderer: Any, rs: RenderRuntimeState) -> None:
    if rs.metrics_payload is not None and rs.metrics_dirty:
        payload = rs.metrics_payload
        renderer.update_metrics(
            phase=payload.get("phase", ""),
            metrics=payload.get("metrics", {}),
            step=payload.get("step"),
            timestamp=payload.get("timestamp"),
        )
        rs.metrics_dirty = False


def _apply_ticker(renderer: Any, rs: RenderRuntimeState) -> None:
    if rs.ticker_dirty:
        renderer.update_ticker(list(rs.ticker)[:16])
        rs.ticker_dirty = False


def _apply_obs(renderer: Any, rs: RenderRuntimeState) -> None:
    if rs.render_obs:
        renderer.update_obs(rs.render_obs)


def _apply_heatmap(renderer: Any, rs: RenderRuntimeState) -> None:
    if rs.reward_heatmap_enabled:
        if rs.reward_heatmap_dirty or (
            not rs.reward_heatmap_applied and rs.reward_heatmap_payload is not None
        ):
            try:
                renderer.update_reward_heatmap(
                    rs.reward_heatmap_payload,
                    alpha=rs.reward_heatmap_alpha,
                    value_scale=rs.reward_heatmap_value_scale,
                    extent_m=rs.reward_heatmap_extent_m,
                    cell_size_m=rs.reward_heatmap_cell_size_m,
                )
                rs.reward_heatmap_applied = rs.reward_heatmap_payload is not None
            except Exception:
                pass
            rs.reward_heatmap_dirty = False
    elif rs.reward_heatmap_applied:
        try:
            renderer.update_reward_heatmap(
                None,
                alpha=rs.reward_heatmap_alpha,
                value_scale=rs.reward_heatmap_value_scale,
                extent_m=rs.reward_heatmap_extent_m,
                cell_size_m=rs.reward_heatmap_cell_size_m,
            )
        except Exception:
            pass
        rs.reward_heatmap_applied = False


def _apply_overlays(renderer: Any, rs: RenderRuntimeState) -> None:
    if rs.reward_overlay_enabled:
        if rs.reward_overlay_dirty or rs.reward_overlays:
            try:
                renderer.update_reward_overlays(
                    rs.reward_overlays,
                    alpha=rs.reward_overlay_alpha,
                    value_scale=rs.reward_overlay_value_scale,
                    segments=rs.reward_overlay_segments,
                )
                rs.reward_overlay_applied = bool(rs.reward_overlays)
            except Exception:
                pass
            rs.reward_overlay_dirty = False
    elif rs.reward_overlay_applied:
        try:
            renderer.update_reward_overlays(
                [],
                alpha=rs.reward_overlay_alpha,
                value_scale=rs.reward_overlay_value_scale,
                segments=rs.reward_overlay_segments,
            )
        except Exception:
            pass
        rs.reward_overlay_applied = False


def _apply_callbacks(
    renderer: Any,
    rs: RenderRuntimeState,
    *,
    _logger: Optional[logging.Logger] = None,
) -> None:
    _log = _logger or logger
    for callback in list(rs.callbacks):
        try:
            callback(renderer)
        except Exception:
            _log.exception("Render callback failed")
