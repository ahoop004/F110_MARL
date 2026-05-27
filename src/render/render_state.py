"""Render-only state helpers for F110ParallelEnv."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class OverlayConfig:
    enabled: bool
    alpha: float
    value_scale: float
    segments: int


@dataclass(frozen=True)
class HeatmapConfig:
    enabled: bool
    alpha: float
    value_scale: float
    extent_m: float
    cell_size_m: float


@dataclass
class RenderRuntimeState:
    """Mutable render-only payloads owned outside the env core."""

    render_obs: Optional[Dict[str, Dict[str, Any]]] = None
    metrics_payload: Optional[Dict[str, Any]] = None
    metrics_dirty: bool = False
    ticker: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    ticker_dirty: bool = False
    wrapped_obs: Dict[str, np.ndarray] = field(default_factory=dict)
    lidar_skip_default: int = 0
    lidar_skip: Dict[str, int] = field(default_factory=dict)
    callbacks: list[Callable[[Any], None]] = field(default_factory=list)
    reward_ring_config: Optional[Dict[str, Any]] = None
    reward_ring_focus_agent: Optional[str] = None
    reward_ring_target: Optional[str] = None
    reward_ring_dirty: bool = False
    reward_ring_target_dirty: bool = False
    reward_ring_marker_states: Dict[str, List[bool]] = field(default_factory=dict)
    reward_ring_marker_dirty: bool = False
    reward_overlays: List[Dict[str, Any]] = field(default_factory=list)
    reward_overlay_dirty: bool = False
    reward_overlay_enabled: bool = False
    reward_overlay_applied: bool = False
    reward_overlay_alpha: float = 0.25
    reward_overlay_value_scale: float = 1.0
    reward_overlay_segments: int = 48
    reward_heatmap_payload: Optional[Dict[str, Any]] = None
    reward_heatmap_dirty: bool = False
    reward_heatmap_enabled: bool = False
    reward_heatmap_applied: bool = False
    reward_heatmap_alpha: float = 0.22
    reward_heatmap_value_scale: float = 1.0
    reward_heatmap_extent_m: float = 6.0
    reward_heatmap_cell_size_m: float = 0.25

    def reset_episode(self) -> None:
        self.ticker.clear()
        self.ticker_dirty = True
        self.wrapped_obs.clear()

    def set_wrapped_observations(self, wrapped: Mapping[str, np.ndarray]) -> None:
        if not wrapped:
            return
        store: Dict[str, np.ndarray] = {}
        for agent_id, vector in wrapped.items():
            if vector is None:
                continue
            try:
                array = np.asarray(vector, dtype=np.float32)
            except Exception:
                continue
            store[str(agent_id)] = array.copy()
        if store:
            self.wrapped_obs = store

    def append_ticker_line(self, line: str) -> None:
        self.ticker.appendleft(str(line))
        self.ticker_dirty = True

    def add_callback(self, callback: Callable[[Any], None]) -> None:
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def apply_overlay_config(self, config: OverlayConfig) -> None:
        self.reward_overlay_enabled = config.enabled
        self.reward_overlay_alpha = config.alpha
        self.reward_overlay_value_scale = config.value_scale
        self.reward_overlay_segments = config.segments

    def apply_heatmap_config(self, config: HeatmapConfig) -> None:
        self.reward_heatmap_enabled = config.enabled
        self.reward_heatmap_alpha = config.alpha
        self.reward_heatmap_value_scale = config.value_scale
        self.reward_heatmap_extent_m = config.extent_m
        self.reward_heatmap_cell_size_m = config.cell_size_m

    def reset_renderer_payloads(self) -> None:
        self.reward_ring_dirty = True
        self.reward_ring_target_dirty = True
        self.reward_overlay_dirty = True
        self.reward_overlay_applied = False
        self.reward_heatmap_payload = None
        self.reward_heatmap_dirty = True
        self.reward_heatmap_applied = False

    def configure_reward_ring(self, config: Optional[Mapping[str, Any]], *, agent_id: Optional[str] = None) -> None:
        if config is None:
            self.reward_ring_config = None
            self.reward_ring_focus_agent = None
            self.reward_ring_target = None
            self.reward_ring_marker_states.clear()
            self.reward_ring_dirty = True
            self.reward_ring_target_dirty = True
            self.reward_ring_marker_dirty = True
            return

        stored = normalize_reward_ring_config(config)
        if self.reward_ring_config != stored or self.reward_ring_focus_agent != agent_id:
            self.reward_ring_config = stored
            self.reward_ring_focus_agent = agent_id
            self.reward_ring_marker_states.clear()
            self.reward_ring_dirty = True
            self.reward_ring_target_dirty = True
            self.reward_ring_marker_dirty = True

    def update_reward_ring_target(self, agent_id: str, target_id: Optional[str]) -> None:
        if self.reward_ring_config is None:
            return

        focus = self.reward_ring_focus_agent
        if focus is not None and agent_id != focus:
            return

        normalized = str(target_id) if target_id else None
        if self.reward_ring_target != normalized:
            self.reward_ring_target = normalized
            self.reward_ring_target_dirty = True
            self.reward_ring_marker_dirty = True

        if normalized:
            pending = self.reward_ring_marker_states.get(agent_id)
            if pending is not None:
                self.reward_ring_marker_states[normalized] = list(pending)
                if agent_id != normalized:
                    self.reward_ring_marker_states.pop(agent_id, None)
                self.reward_ring_marker_dirty = True

    def update_reward_ring_markers(self, agent_id: str, states: Optional[Sequence[bool]]) -> None:
        if self.reward_ring_config is None:
            return
        focus = self.reward_ring_focus_agent
        if focus is not None and agent_id != focus:
            return
        target_key = self.reward_ring_target or agent_id
        if states is None:
            if target_key in self.reward_ring_marker_states:
                self.reward_ring_marker_states.pop(target_key, None)
                self.reward_ring_marker_dirty = True
        else:
            snapshot = [bool(s) for s in states]
            if self.reward_ring_marker_states.get(target_key) != snapshot:
                self.reward_ring_marker_states[target_key] = snapshot
                self.reward_ring_marker_dirty = True

    def update_reward_overlays(
        self,
        overlays: Optional[Sequence[Mapping[str, Any]]],
        *,
        enabled: Optional[bool] = None,
        alpha: Optional[float] = None,
        value_scale: Optional[float] = None,
        segments: Optional[int] = None,
    ) -> None:
        if enabled is not None:
            enabled_val = coerce_bool_flag(enabled, default=self.reward_overlay_enabled)
            if enabled_val != self.reward_overlay_enabled:
                self.reward_overlay_enabled = enabled_val
                self.reward_overlay_dirty = True
        if overlays is None:
            if self.reward_overlays:
                self.reward_overlays = []
                self.reward_overlay_dirty = True
        else:
            self.reward_overlays = [dict(entry) for entry in overlays if isinstance(entry, Mapping)]
            self.reward_overlay_dirty = True

        if alpha is not None:
            alpha_val = _bounded_float(alpha, default=self.reward_overlay_alpha, low=0.0, high=1.0)
            self.reward_overlay_alpha = alpha_val
            self.reward_overlay_dirty = True

        if value_scale is not None:
            scale_val = _positive_float(value_scale, default=self.reward_overlay_value_scale)
            self.reward_overlay_value_scale = scale_val
            self.reward_overlay_dirty = True

        if segments is not None:
            try:
                seg_val = int(segments)
            except (TypeError, ValueError):
                seg_val = self.reward_overlay_segments
            seg_val = max(seg_val, 8)
            if seg_val != self.reward_overlay_segments:
                self.reward_overlay_segments = seg_val
                self.reward_overlay_dirty = True

    def update_reward_heatmap(
        self,
        heatmap: Optional[Mapping[str, Any]],
        *,
        enabled: Optional[bool] = None,
        alpha: Optional[float] = None,
        value_scale: Optional[float] = None,
        extent_m: Optional[float] = None,
        cell_size_m: Optional[float] = None,
    ) -> None:
        if enabled is not None:
            enabled_val = coerce_bool_flag(enabled, default=self.reward_heatmap_enabled)
            if enabled_val != self.reward_heatmap_enabled:
                self.reward_heatmap_enabled = enabled_val
                self.reward_heatmap_dirty = True

        if heatmap is None:
            if self.reward_heatmap_payload is not None:
                self.reward_heatmap_payload = None
                self.reward_heatmap_dirty = True
        elif isinstance(heatmap, Mapping):
            try:
                payload = dict(heatmap)
            except Exception:
                payload = None
            if payload is not None and payload != self.reward_heatmap_payload:
                self.reward_heatmap_payload = payload
                self.reward_heatmap_dirty = True

        if alpha is not None:
            alpha_val = _bounded_float(alpha, default=self.reward_heatmap_alpha, low=0.0, high=1.0)
            if alpha_val != self.reward_heatmap_alpha:
                self.reward_heatmap_alpha = alpha_val
                self.reward_heatmap_dirty = True

        if value_scale is not None:
            scale_val = _positive_float(value_scale, default=self.reward_heatmap_value_scale)
            if scale_val != self.reward_heatmap_value_scale:
                self.reward_heatmap_value_scale = scale_val
                self.reward_heatmap_dirty = True

        if extent_m is not None:
            extent_val = _positive_float(extent_m, default=self.reward_heatmap_extent_m)
            if extent_val != self.reward_heatmap_extent_m:
                self.reward_heatmap_extent_m = extent_val
                self.reward_heatmap_dirty = True

        if cell_size_m is not None:
            cell_val = _positive_float(cell_size_m, default=self.reward_heatmap_cell_size_m)
            if cell_val != self.reward_heatmap_cell_size_m:
                self.reward_heatmap_cell_size_m = cell_val
                self.reward_heatmap_dirty = True


def normalize_reward_ring_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    stored: Dict[str, Any] = {
        "preferred_radius": max(float(config.get("preferred_radius", 0.0)), 0.0),
        "inner_tolerance": max(float(config.get("inner_tolerance", 0.0)), 0.0),
        "outer_tolerance": max(float(config.get("outer_tolerance", 0.0)), 0.0),
        "segments": max(int(config.get("segments", 96) or 96), 8),
        "marker_radius": max(float(config.get("marker_radius", 0.0)), 0.0),
        "marker_segments": max(int(config.get("marker_segments", 12) or 12), 4),
        "offsets_only": bool(config.get("offsets_only", False)),
    }
    for key in ("fill_color", "border_color", "preferred_color"):
        if key in config and isinstance(config[key], (list, tuple)):
            stored[key] = tuple(float(component) for component in config[key])
    falloff_val = config.get("falloff")
    if falloff_val is not None:
        stored["falloff"] = str(falloff_val).lower()
    marker_color_val = config.get("marker_color")
    if isinstance(marker_color_val, (list, tuple)):
        stored["marker_color"] = tuple(float(component) for component in marker_color_val)
    offsets_val = config.get("offsets")
    if offsets_val:
        cleaned_offsets: List[Tuple[float, float]] = []
        for entry in offsets_val:
            if entry is None:
                continue
            try:
                pair = tuple(float(v) for v in entry)  # type: ignore[arg-type]
            except Exception:
                continue
            if len(pair) >= 2:
                cleaned_offsets.append((pair[0], pair[1]))
        if cleaned_offsets:
            stored["offsets"] = cleaned_offsets
    marker_color_active_val = config.get("marker_color_active")
    if isinstance(marker_color_active_val, (list, tuple)):
        stored["marker_color_active"] = tuple(float(component) for component in marker_color_active_val)
    weights_val = config.get("weights")
    if isinstance(weights_val, Mapping):
        safe_weights: Dict[str, float] = {}
        for name, value in weights_val.items():
            if value is None:
                continue
            try:
                safe_weights[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        if safe_weights:
            stored["weights"] = safe_weights
    return stored


def coerce_bool_flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "on"}:
            return True
        if lowered in {"false", "no", "n", "0", "off"}:
            return False
        return default
    return bool(value)


def parse_overlay_config(cfg: Mapping[str, Any]) -> OverlayConfig:
    overlay_cfg = cfg.get("reward_overlay")
    if isinstance(overlay_cfg, Mapping):
        enabled_raw = overlay_cfg.get("enabled", cfg.get("reward_overlay_enabled", False))
        alpha_raw = overlay_cfg.get("alpha", cfg.get("reward_overlay_alpha", 0.25))
        scale_raw = overlay_cfg.get(
            "value_scale",
            overlay_cfg.get("scale", cfg.get("reward_overlay_value_scale", 1.0)),
        )
        segments_raw = overlay_cfg.get("segments", cfg.get("reward_overlay_segments", 48))
    else:
        enabled_raw = cfg.get("reward_overlay_enabled", False)
        alpha_raw = cfg.get("reward_overlay_alpha", 0.25)
        scale_raw = cfg.get("reward_overlay_value_scale", 1.0)
        segments_raw = cfg.get("reward_overlay_segments", 48)

    alpha = _bounded_float(alpha_raw, default=0.25, low=0.0, high=1.0)
    value_scale = _positive_float(scale_raw, default=1.0)
    try:
        segments = int(segments_raw)
    except (TypeError, ValueError):
        segments = 48
    return OverlayConfig(
        enabled=coerce_bool_flag(enabled_raw, default=False),
        alpha=alpha,
        value_scale=value_scale,
        segments=max(segments, 8),
    )


def parse_heatmap_config(cfg: Mapping[str, Any]) -> HeatmapConfig:
    heatmap_cfg = cfg.get("reward_heatmap")
    if isinstance(heatmap_cfg, Mapping):
        enabled_raw = heatmap_cfg.get("enabled", cfg.get("reward_heatmap_enabled", False))
        alpha_raw = heatmap_cfg.get("alpha", cfg.get("reward_heatmap_alpha", 0.22))
        scale_raw = heatmap_cfg.get(
            "value_scale",
            heatmap_cfg.get("scale", cfg.get("reward_heatmap_value_scale", 1.0)),
        )
        extent_raw = heatmap_cfg.get(
            "extent_m",
            heatmap_cfg.get("extent", cfg.get("reward_heatmap_extent_m", cfg.get("reward_heatmap_extent", 6.0))),
        )
        cell_raw = heatmap_cfg.get(
            "cell_size_m",
            heatmap_cfg.get(
                "cell_size",
                cfg.get("reward_heatmap_cell_size_m", cfg.get("reward_heatmap_cell_size", 0.25)),
            ),
        )
    else:
        enabled_raw = cfg.get("reward_heatmap_enabled", False)
        alpha_raw = cfg.get("reward_heatmap_alpha", 0.22)
        scale_raw = cfg.get("reward_heatmap_value_scale", 1.0)
        extent_raw = cfg.get("reward_heatmap_extent_m", cfg.get("reward_heatmap_extent", 6.0))
        cell_raw = cfg.get("reward_heatmap_cell_size_m", cfg.get("reward_heatmap_cell_size", 0.25))

    return HeatmapConfig(
        enabled=coerce_bool_flag(enabled_raw, default=False),
        alpha=_bounded_float(alpha_raw, default=0.22, low=0.0, high=1.0),
        value_scale=_positive_float(scale_raw, default=1.0),
        extent_m=_positive_float(extent_raw, default=6.0),
        cell_size_m=_positive_float(cell_raw, default=0.25),
    )


def _bounded_float(value: Any, *, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return float(min(max(parsed, low), high))


def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed <= 0.0 or not np.isfinite(parsed):
        return float(default)
    return float(parsed)
