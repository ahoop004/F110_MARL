"""Coordinate-based spawn curriculum for run_v2 / EnhancedTrainingLoop.

Provides progressive difficulty by expanding the attacker's (and target's)
allowed spawn region over training, driven by rolling success rate.

Five independent axes:
  spawn_x_curriculum        — attacker x position
  spawn_y_curriculum        — attacker y position
  mirrored_y_curriculum     — reflects attacker y about target y (sequenced after y phase)
  target_spawn_x_curriculum — target x position
  target_spawn_y_curriculum — target y position

Each positional axis supports two modes:
  linear         — expands proportionally to episode count
  success_adaptive — expands/contracts based on rolling success rate via
                     AdaptiveScalarController (same as SB3SingleAgentWrapper)

mirrored_y_curriculum is stateless — it deterministically reflects whatever
y was sampled by spawn_y_curriculum about the target's y coordinate.  It has
no controller and does not call observe().  Use set_stage_mode("mirrored_y",
"active"/"off") or the curriculum orchestrator's order list to sequence it
after the y phase completes.

Usage in EnhancedTrainingLoop:
    curriculum = CoordinateSpawnCurriculum(
        spawn_x_curriculum=env_config.get("spawn_x_curriculum"),
        spawn_y_curriculum=env_config.get("spawn_y_curriculum"),
        mirrored_y_curriculum=env_config.get("mirrored_y_curriculum"),
        target_spawn_x_curriculum=env_config.get("target_spawn_x_curriculum"),
        target_spawn_y_curriculum=env_config.get("target_spawn_y_curriculum"),
        agent_id="car_0",
        target_id="car_1",
    )

    # Before env.reset():
    options = curriculum.sample(env, episode=ep_num)

    # After episode ends:
    curriculum.observe(success=True)

    # For logging:
    meta = curriculum.get_metadata()
"""

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from src.core.adaptive_controller import AdaptiveScalarController


class CoordinateSpawnCurriculum:
    """Coordinate-based spawn curriculum supporting all four spawn axes.

    Mirrors the behaviour of spawn_x/y_curriculum and
    target_spawn_x/y_curriculum from SB3SingleAgentWrapper so that
    run_v2.py / EnhancedTrainingLoop can use the same gaplock_adaptive.yaml
    config without modification.
    """

    def __init__(
        self,
        spawn_x_curriculum: Optional[Dict[str, Any]] = None,
        spawn_y_curriculum: Optional[Dict[str, Any]] = None,
        mirrored_y_curriculum: Optional[Dict[str, Any]] = None,
        target_spawn_x_curriculum: Optional[Dict[str, Any]] = None,
        target_spawn_y_curriculum: Optional[Dict[str, Any]] = None,
        agent_id: str = "car_0",
        target_id: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.target_id = target_id

        self.spawn_x_curriculum = dict(spawn_x_curriculum) if isinstance(spawn_x_curriculum, dict) else None
        self.spawn_y_curriculum = dict(spawn_y_curriculum) if isinstance(spawn_y_curriculum, dict) else None
        self.mirrored_y_curriculum = dict(mirrored_y_curriculum) if isinstance(mirrored_y_curriculum, dict) else None
        self.target_spawn_x_curriculum = dict(target_spawn_x_curriculum) if isinstance(target_spawn_x_curriculum, dict) else None
        self.target_spawn_y_curriculum = dict(target_spawn_y_curriculum) if isinstance(target_spawn_y_curriculum, dict) else None

        self._local_rng: Optional[np.random.Generator] = None

        # AdaptiveScalarControllers — created lazily on first sample
        self._spawn_x_controller: Optional[AdaptiveScalarController] = None
        self._spawn_x_context: Dict[str, float] = {}
        self._spawn_y_controller: Optional[AdaptiveScalarController] = None
        self._spawn_y_context: Dict[str, float] = {}
        self._target_spawn_x_controller: Optional[AdaptiveScalarController] = None
        self._target_spawn_x_context: Dict[str, float] = {}
        self._target_spawn_y_controller: Optional[AdaptiveScalarController] = None
        self._target_spawn_y_context: Dict[str, float] = {}

        # Mirrored-y rolling success tracker (graded stage)
        _my_cfg = self.mirrored_y_curriculum or {}
        _my_window = max(1, int(_my_cfg.get("window_episodes", 200)))
        self._mirrored_y_window: int = _my_window
        self._mirrored_y_sr_upper: float = float(_my_cfg.get("sr_upper_bound", 0.90))
        self._mirrored_y_successes: Deque[int] = deque(maxlen=_my_window)

        # Stage modes: "active" or "off"
        self._spawn_x_stage_mode = "active" if self.spawn_x_curriculum and self.spawn_x_curriculum.get("enabled", False) else "off"
        self._spawn_y_stage_mode = "active" if self.spawn_y_curriculum and self.spawn_y_curriculum.get("enabled", False) else "off"
        self._mirrored_y_stage_mode = "active" if self.mirrored_y_curriculum and self.mirrored_y_curriculum.get("enabled", False) else "off"
        self._target_spawn_x_stage_mode = "active" if self.target_spawn_x_curriculum and self.target_spawn_x_curriculum.get("enabled", False) else "off"
        self._target_spawn_y_stage_mode = "active" if self.target_spawn_y_curriculum and self.target_spawn_y_curriculum.get("enabled", False) else "off"

        # Last sampled metadata (set by sample(), read by get_metadata())
        self._last_meta: Dict[str, Any] = {}

    def is_active(self) -> bool:
        """Return True if at least one axis is active or frozen."""
        return any(m in {"active", "frozen"} for m in [
            self._spawn_x_stage_mode,
            self._spawn_y_stage_mode,
            self._mirrored_y_stage_mode,
            self._target_spawn_x_stage_mode,
            self._target_spawn_y_stage_mode,
        ])

    def sample(self, env: Any, episode: int) -> Optional[Dict[str, Any]]:
        """Sample poses for this episode and return env.reset() options.

        Args:
            env: The underlying PettingZoo environment (needs start_poses and
                 possible_agents attributes).
            episode: Current episode number.

        Returns:
            Dict with key ``"poses"`` (np.ndarray shape [n_agents, 3]) suitable
            for passing as ``options`` to ``env.reset()``, or None if no axis
            is active.
        """
        if not self.is_active():
            return None

        sampled: Optional[Dict[str, Any]] = None
        if self.spawn_y_curriculum:
            sampled = self._sample_spawn_y(env, episode, base=sampled)
        if self.spawn_x_curriculum:
            sampled = self._sample_spawn_x(env, episode, base=sampled)
        if self.target_spawn_y_curriculum:
            sampled = self._sample_target_spawn_y(env, episode, base=sampled)
        if self.target_spawn_x_curriculum:
            sampled = self._sample_target_spawn_x(env, episode, base=sampled)

        # Mirrored-y stage: reflect attacker y about target y (applied after all other axes).
        # Stays active when frozen so the flip persists through later stages.
        if self._mirrored_y_stage_mode in {"active", "frozen"} and sampled is not None:
            sampled = self._apply_mirrored_y(env, sampled)

        if sampled is not None:
            self._last_meta = {k: v for k, v in sampled.items() if k != "options"}
        else:
            self._last_meta = {}

        return sampled.get("options") if sampled is not None else None

    def observe(self, success: bool) -> None:
        """Feed episode outcome to all active controllers."""
        if self._spawn_y_stage_mode == "active" and self._spawn_y_controller is not None:
            self._spawn_y_controller.observe(bool(success))
        if self._spawn_x_stage_mode == "active" and self._spawn_x_controller is not None:
            self._spawn_x_controller.observe(bool(success))
        if self._target_spawn_y_stage_mode == "active" and self._target_spawn_y_controller is not None:
            self._target_spawn_y_controller.observe(bool(success))
        if self._target_spawn_x_stage_mode == "active" and self._target_spawn_x_controller is not None:
            self._target_spawn_x_controller.observe(bool(success))
        if self._mirrored_y_stage_mode == "active":
            self._mirrored_y_successes.append(1 if success else 0)

    def mirrored_y_progress(self) -> Optional[float]:
        """Rolling SR progress for the mirrored-y stage (0.0 → 1.0).

        Returns None if the stage is not configured, 0.0 if the window
        is not yet full, and sr / sr_upper_bound (clamped to 1.0) once
        the window has enough data.
        """
        if not self.mirrored_y_curriculum:
            return None
        if not self._mirrored_y_successes:
            return 0.0
        sr = float(sum(self._mirrored_y_successes) / len(self._mirrored_y_successes))
        return min(sr / max(self._mirrored_y_sr_upper, 1e-9), 1.0)

    def get_metadata(self) -> Dict[str, Any]:
        """Return last spawn sampling metadata (spawn_x, spawn_y, etc.)."""
        meta = dict(self._last_meta)
        if self._mirrored_y_stage_mode in {"active", "frozen"} and self._mirrored_y_successes:
            sr = float(sum(self._mirrored_y_successes) / len(self._mirrored_y_successes))
            meta.setdefault("mirrored_y_curriculum", {})
            meta["mirrored_y_curriculum"]["mirrored_y_window_sr"] = sr
            meta["mirrored_y_curriculum"]["mirrored_y_progress"] = self.mirrored_y_progress()
        return meta

    def set_stage_mode(self, axis: str, mode: str) -> None:
        """Allow external orchestrator to freeze/unfreeze an axis.

        Args:
            axis: One of "spawn_x", "spawn_y", "mirrored_y",
                  "target_spawn_x", "target_spawn_y".
            mode: "active" or "off".
        """
        if axis == "spawn_x":
            self._spawn_x_stage_mode = mode
        elif axis == "spawn_y":
            self._spawn_y_stage_mode = mode
        elif axis == "mirrored_y":
            self._mirrored_y_stage_mode = mode
        elif axis == "target_spawn_x":
            self._target_spawn_x_stage_mode = mode
        elif axis == "target_spawn_y":
            self._target_spawn_y_stage_mode = mode

    # ── Private helpers ──────────────────────────────────────────────────────

    def _rng(self, env: Any) -> np.random.Generator:
        rng = getattr(env, "rng", None)
        if rng is not None:
            return rng
        if self._local_rng is None:
            self._local_rng = np.random.default_rng()
        return self._local_rng

    def _get_base_poses(self, env: Any, base: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        if isinstance(base, dict):
            opts = base.get("options")
            if isinstance(opts, dict):
                existing = opts.get("poses")
                if existing is not None:
                    return np.asarray(existing, dtype=np.float32)
        raw = getattr(env, "start_poses", None)
        if raw is None:
            return None
        poses = np.asarray(raw, dtype=np.float32).copy()
        if poses.ndim != 2 or poses.shape[1] < 3:
            return None
        return poses

    def _agent_index(self, env: Any, agent_id: str) -> int:
        agents = list(getattr(env, "possible_agents", []))
        try:
            return agents.index(agent_id)
        except ValueError:
            return -1

    def _make_controller(self, cfg: Dict[str, Any], initial: float, bound: float, upper: float) -> AdaptiveScalarController:
        window = max(1, int(cfg.get("window_episodes", 100)))
        update_every = max(1, int(cfg.get("update_every_episodes", window)))
        sr_low = float(cfg.get("sr_lower_bound", 0.85))
        sr_high = float(cfg.get("sr_upper_bound", 0.95))
        regress_enabled = bool(cfg.get("regression_enabled", False))
        regress_trigger = float(cfg.get("regression_sr_trigger", max(0.0, sr_low - 0.20)))
        regress_floor = float(cfg.get("regression_sr_floor", max(0.0, regress_trigger - 0.20)))
        regress_patience = max(1, int(cfg.get("regression_patience_updates", 1)))
        regress_cooldown = max(0, int(cfg.get("regression_cooldown_updates", 0)))

        inc_min_raw = float(cfg.get("increment_min", -0.001))
        inc_max_raw = float(cfg.get("increment_max", -0.01))
        inc_small = -abs(inc_min_raw)
        inc_large = -abs(inc_max_raw)
        if abs(inc_large) < abs(inc_small):
            inc_small, inc_large = inc_large, inc_small

        ri_min = float(cfg.get("regression_increment_min", abs(inc_min_raw)))
        ri_max = float(cfg.get("regression_increment_max", abs(inc_max_raw)))
        ri_small = abs(ri_min)
        ri_large = abs(ri_max)
        if ri_large < ri_small:
            ri_small, ri_large = ri_large, ri_small

        return AdaptiveScalarController(
            initial_value=initial,
            value_min=bound,
            value_max=upper,
            window_size=window,
            update_every=update_every,
            advance_sr_low=sr_low,
            advance_sr_high=sr_high,
            advance_step_min=inc_small,
            advance_step_max=inc_large,
            regression_enabled=regress_enabled,
            regression_sr_trigger=regress_trigger,
            regression_sr_floor=regress_floor,
            regression_step_min=ri_small,
            regression_step_max=ri_large,
            regression_patience_updates=regress_patience,
            regression_cooldown_updates=regress_cooldown,
            progress_start=initial,
            progress_end=bound,
        )

    # ── Y axis ───────────────────────────────────────────────────────────────

    def _sample_spawn_y(self, env: Any, episode: int, base: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cfg = self.spawn_y_curriculum
        if self._spawn_y_stage_mode == "off" or not cfg:
            return base
        poses = self._get_base_poses(env, base)
        if poses is None:
            return base
        idx = self._agent_index(env, self.agent_id)
        if idx < 0 or idx >= poses.shape[0]:
            return base

        mode = str(cfg.get("mode", "linear")).strip().lower()
        if mode == "success_adaptive":
            low, high, progress, extra = self._resolve_y_bounds(cfg, poses, idx)
        else:
            y_start = float(cfg.get("y_start", poses[idx, 1]))
            y_final = float(cfg.get("y_final", -1.0))
            ramp = int(cfg.get("ramp_episodes", 1000))
            progress = min(max(float(episode) / max(ramp, 1), 0.0), 1.0)
            y_min = y_start + (y_final - y_start) * progress
            low, high = min(y_start, y_min), max(y_start, y_min)
            extra = {}

        sampled_y = float(self._rng(env).uniform(low, high))
        poses[idx, 1] = sampled_y

        result = dict(base) if isinstance(base, dict) else {}
        result["options"] = {"poses": poses}
        result["spawn_y_curriculum"] = {"spawn_y_mode": mode, "spawn_y": sampled_y,
                                         "spawn_y_min": float(low), "spawn_y_max": float(high),
                                         "spawn_y_progress": float(progress), **extra}
        return result

    def _resolve_y_bounds(self, cfg: Dict[str, Any], poses: np.ndarray, idx: int) -> Tuple[float, float, float, Dict]:
        if self._spawn_y_controller is None:
            y_ref = float(poses[idx, 1])
            y_upper_cfg = float(cfg.get("y_upper", cfg.get("y_start", y_ref)))
            y_lower_start_cfg = float(cfg.get("y_lower_start", cfg.get("y_start", y_ref)))
            y_lower_bound_cfg = float(cfg.get("y_lower_bound", cfg.get("y_final", -1.0)))
            y_upper = max(y_upper_cfg, y_lower_start_cfg)
            y_lower_start = min(y_upper_cfg, y_lower_start_cfg)
            y_lower_bound = min(y_lower_bound_cfg, y_upper)
            self._spawn_y_controller = self._make_controller(cfg, y_lower_start, y_lower_bound, y_lower_start)
            self._spawn_y_context = {"y_upper": y_upper, "y_lower_start": y_lower_start, "y_lower_bound": y_lower_bound}

        y_upper = float(self._spawn_y_context["y_upper"])
        y_lower = float(self._spawn_y_controller.value)
        progress = self._spawn_y_controller.progress() or 0.0
        extra = {
            "spawn_y_window_sr": self._spawn_y_controller.last_window_sr,
            "spawn_y_last_increment": float(self._spawn_y_controller.last_delta),
            "spawn_y_last_adjustment": str(self._spawn_y_controller.last_adjustment),
        }
        return y_lower, y_upper, progress, extra

    # ── X axis ───────────────────────────────────────────────────────────────

    def _sample_spawn_x(self, env: Any, episode: int, base: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cfg = self.spawn_x_curriculum
        if self._spawn_x_stage_mode == "off" or not cfg:
            return base
        poses = self._get_base_poses(env, base)
        if poses is None:
            return base
        idx = self._agent_index(env, self.agent_id)
        if idx < 0 or idx >= poses.shape[0]:
            return base

        mode = str(cfg.get("mode", "linear")).strip().lower()
        if mode == "success_adaptive":
            low, high, progress, extra = self._resolve_x_bounds(cfg, poses, idx)
        else:
            x_start = float(cfg.get("x_start", poses[idx, 0]))
            x_final = float(cfg.get("x_final", -1.0))
            ramp = int(cfg.get("ramp_episodes", 1000))
            progress = min(max(float(episode) / max(ramp, 1), 0.0), 1.0)
            x_min = x_start + (x_final - x_start) * progress
            low, high = min(x_start, x_min), max(x_start, x_min)
            extra = {}

        sampled_x = float(self._rng(env).uniform(low, high))
        poses[idx, 0] = sampled_x

        result = dict(base) if isinstance(base, dict) else {}
        result["options"] = {"poses": poses}
        result["spawn_x_curriculum"] = {"spawn_x_mode": mode, "spawn_x": sampled_x,
                                         "spawn_x_min": float(low), "spawn_x_max": float(high),
                                         "spawn_x_progress": float(progress), **extra}
        return result

    def _resolve_x_bounds(self, cfg: Dict[str, Any], poses: np.ndarray, idx: int) -> Tuple[float, float, float, Dict]:
        if self._spawn_x_controller is None:
            x_ref = float(poses[idx, 0])
            x_upper_cfg = float(cfg.get("x_upper", cfg.get("x_start", x_ref)))
            x_lower_start_cfg = float(cfg.get("x_lower_start", cfg.get("x_start", x_ref)))
            x_lower_bound_cfg = float(cfg.get("x_lower_bound", cfg.get("x_final", -1.0)))
            x_upper = max(x_upper_cfg, x_lower_start_cfg)
            x_lower_start = min(x_upper_cfg, x_lower_start_cfg)
            x_lower_bound = min(x_lower_bound_cfg, x_upper)
            self._spawn_x_controller = self._make_controller(cfg, x_lower_start, x_lower_bound, x_lower_start)
            self._spawn_x_context = {"x_upper": x_upper, "x_lower_start": x_lower_start, "x_lower_bound": x_lower_bound}

        x_upper = float(self._spawn_x_context["x_upper"])
        x_lower = float(self._spawn_x_controller.value)
        progress = self._spawn_x_controller.progress() or 0.0
        extra = {
            "spawn_x_window_sr": self._spawn_x_controller.last_window_sr,
            "spawn_x_last_increment": float(self._spawn_x_controller.last_delta),
            "spawn_x_last_adjustment": str(self._spawn_x_controller.last_adjustment),
        }
        return x_lower, x_upper, progress, extra

    # ── Target Y axis ────────────────────────────────────────────────────────

    def _sample_target_spawn_y(self, env: Any, episode: int, base: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cfg = self.target_spawn_y_curriculum
        if self._target_spawn_y_stage_mode == "off" or not cfg or not self.target_id:
            return base
        poses = self._get_base_poses(env, base)
        if poses is None:
            return base
        t_idx = self._agent_index(env, self.target_id)
        if t_idx < 0 or t_idx >= poses.shape[0]:
            return base

        mode = str(cfg.get("mode", "linear")).strip().lower()
        if mode == "success_adaptive":
            low, high, progress, extra = self._resolve_target_y_bounds(cfg, poses, t_idx)
        else:
            y_start = float(cfg.get("y_start", poses[t_idx, 1]))
            y_final = float(cfg.get("y_final", y_start))
            ramp = int(cfg.get("ramp_episodes", 1000))
            progress = min(max(float(episode) / max(ramp, 1), 0.0), 1.0)
            y_min = y_start + (y_final - y_start) * progress
            low, high = min(y_start, y_min), max(y_start, y_min)
            extra = {}

        sampled_y = float(self._rng(env).uniform(low, high))
        poses[t_idx, 1] = sampled_y

        result = dict(base) if isinstance(base, dict) else {}
        result["options"] = {"poses": poses}
        result["target_spawn_y_curriculum"] = {"target_spawn_y_mode": mode, "target_spawn_y": sampled_y,
                                                "target_spawn_y_min": float(low), "target_spawn_y_max": float(high),
                                                "target_spawn_y_progress": float(progress), **extra}
        return result

    def _resolve_target_y_bounds(self, cfg: Dict[str, Any], poses: np.ndarray, t_idx: int) -> Tuple[float, float, float, Dict]:
        if self._target_spawn_y_controller is None:
            y_ref = float(poses[t_idx, 1])
            y_upper_cfg = float(cfg.get("y_upper", cfg.get("y_start", y_ref)))
            y_lower_start_cfg = float(cfg.get("y_lower_start", cfg.get("y_start", y_ref)))
            y_lower_bound_cfg = float(cfg.get("y_lower_bound", cfg.get("y_final", y_ref)))
            y_upper = max(y_upper_cfg, y_lower_start_cfg)
            y_lower_start = min(y_upper_cfg, y_lower_start_cfg)
            y_lower_bound = min(y_lower_bound_cfg, y_upper)
            self._target_spawn_y_controller = self._make_controller(cfg, y_lower_start, y_lower_bound, y_lower_start)
            self._target_spawn_y_context = {"y_upper": y_upper, "y_lower_start": y_lower_start, "y_lower_bound": y_lower_bound}

        y_upper = float(self._target_spawn_y_context["y_upper"])
        y_lower = float(self._target_spawn_y_controller.value)
        progress = self._target_spawn_y_controller.progress() or 0.0
        extra = {
            "target_spawn_y_window_sr": self._target_spawn_y_controller.last_window_sr,
            "target_spawn_y_last_increment": float(self._target_spawn_y_controller.last_delta),
            "target_spawn_y_last_adjustment": str(self._target_spawn_y_controller.last_adjustment),
        }
        return y_lower, y_upper, progress, extra

    # ── Target X axis ────────────────────────────────────────────────────────

    def _sample_target_spawn_x(self, env: Any, episode: int, base: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cfg = self.target_spawn_x_curriculum
        if self._target_spawn_x_stage_mode == "off" or not cfg or not self.target_id:
            return base
        poses = self._get_base_poses(env, base)
        if poses is None:
            return base
        t_idx = self._agent_index(env, self.target_id)
        if t_idx < 0 or t_idx >= poses.shape[0]:
            return base

        mode = str(cfg.get("mode", "linear")).strip().lower()
        if mode == "success_adaptive":
            low, high, progress, extra = self._resolve_target_x_bounds(cfg, poses, t_idx)
        else:
            x_start = float(cfg.get("x_start", poses[t_idx, 0]))
            x_final = float(cfg.get("x_final", x_start))
            ramp = int(cfg.get("ramp_episodes", 1000))
            progress = min(max(float(episode) / max(ramp, 1), 0.0), 1.0)
            x_min = x_start + (x_final - x_start) * progress
            low, high = min(x_start, x_min), max(x_start, x_min)
            extra = {}

        sampled_x = float(self._rng(env).uniform(low, high))
        poses[t_idx, 0] = sampled_x

        result = dict(base) if isinstance(base, dict) else {}
        result["options"] = {"poses": poses}
        result["target_spawn_x_curriculum"] = {"target_spawn_x_mode": mode, "target_spawn_x": sampled_x,
                                                "target_spawn_x_min": float(low), "target_spawn_x_max": float(high),
                                                "target_spawn_x_progress": float(progress), **extra}
        return result

    def _resolve_target_x_bounds(self, cfg: Dict[str, Any], poses: np.ndarray, t_idx: int) -> Tuple[float, float, float, Dict]:
        if self._target_spawn_x_controller is None:
            x_ref = float(poses[t_idx, 0])
            x_upper_cfg = float(cfg.get("x_upper", cfg.get("x_start", x_ref)))
            x_lower_start_cfg = float(cfg.get("x_lower_start", cfg.get("x_start", x_ref)))
            x_lower_bound_cfg = float(cfg.get("x_lower_bound", cfg.get("x_final", x_ref)))
            x_upper = max(x_upper_cfg, x_lower_start_cfg)
            x_lower_start = min(x_upper_cfg, x_lower_start_cfg)
            x_lower_bound = min(x_lower_bound_cfg, x_upper)
            self._target_spawn_x_controller = self._make_controller(cfg, x_lower_start, x_lower_bound, x_lower_start)
            self._target_spawn_x_context = {"x_upper": x_upper, "x_lower_start": x_lower_start, "x_lower_bound": x_lower_bound}

        x_upper = float(self._target_spawn_x_context["x_upper"])
        x_lower = float(self._target_spawn_x_controller.value)
        progress = self._target_spawn_x_controller.progress() or 0.0
        extra = {
            "target_spawn_x_window_sr": self._target_spawn_x_controller.last_window_sr,
            "target_spawn_x_last_increment": float(self._target_spawn_x_controller.last_delta),
            "target_spawn_x_last_adjustment": str(self._target_spawn_x_controller.last_adjustment),
        }
        return x_lower, x_upper, progress, extra

    # ── Mirrored-Y stage ─────────────────────────────────────────────────────

    def _apply_mirrored_y(self, env: Any, sampled: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect attacker y about target y (deterministic, no probability).

        This is a dedicated curriculum stage sequenced after the y phase.
        It takes whatever y was already set (by spawn_y_curriculum or the
        default pose) and flips it to the opposite side of the target.
        """
        if not self.target_id:
            return sampled
        options = sampled.get("options")
        if not isinstance(options, dict):
            return sampled
        poses = options.get("poses")
        if poses is None:
            return sampled
        poses = np.asarray(poses, dtype=np.float32)
        a_idx = self._agent_index(env, self.agent_id)
        t_idx = self._agent_index(env, self.target_id)
        if a_idx < 0 or t_idx < 0 or poses.ndim != 2 or poses.shape[0] <= max(a_idx, t_idx):
            return sampled
        target_y = float(poses[t_idx, 1])
        poses[a_idx, 1] = 2.0 * target_y - poses[a_idx, 1]
        options["poses"] = poses
        y_meta = sampled.get("spawn_y_curriculum")
        if isinstance(y_meta, dict):
            y_meta["spawn_y"] = float(poses[a_idx, 1])
            y_meta["spawn_y_mirrored"] = True
        sampled["mirrored_y_curriculum"] = {"mirrored_y_active": True, "target_y": target_y}
        return sampled


__all__ = ["CoordinateSpawnCurriculum"]
