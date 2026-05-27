"""Per-env spawn state manager.

:class:`SpawnManager` owns all spawn-related configuration and mutable state
that was previously scattered across ``F110ParallelEnv.__init__``,
``_sample_random_spawn``, ``_sample_centerline_spawn``, and the spawn block in
``reset()``.  The env coordinator delegates spawn resolution to this class on
every reset.

Spawn precedence (highest → lowest)
-------------------------------------
0. Deterministic :class:`~src.env.types.SpawnPlan` (eval / offline replay)
1. Explicit ``options["poses"]``
2. Centerline-relative policy (``spawn_policy: centerline_relative``)
3. Random named-point sampling (``random_spawn.enabled: true``)
4. Current ``start_poses`` (configured or learned during training)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.env.spawn import (
    CenterlineSpawnFn,
    SpawnPlan,
    SpawnRequest,
    SpawnResult,
    extract_spawn_points,
    resolve_reset_spawn,
    sample_centerline_relative_spawn,
)

logger = logging.getLogger(__name__)


class SpawnManager:
    """Owns spawn config parsing, selection, and episode state for one env.

    Parameters
    ----------
    cfg:
        Merged environment config (post :func:`~src.core.spawn_config.normalize_spawn_config`).
    map_data:
        Preloaded map data object, or ``None``.  Used to extract initial spawn
        points; refreshed via :meth:`update_map_data` when maps cycle.
    possible_agents:
        Ordered list of all agent IDs.
    agent_index:
        Mapping ``agent_id → int index``.
    rng:
        The env's random-number generator (shared for determinism).
    seed:
        Current integer seed (updated by :meth:`reseed`).
    map_split_mode:
        ``"train"`` or ``"eval"``; affects centerline-relative spawn mode.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        map_data: Optional[Any],
        *,
        possible_agents: Sequence[str],
        agent_index: Mapping[str, int],
        rng: np.random.Generator,
        seed: int,
        map_split_mode: str = "train",
        init_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._possible_agents: list = list(possible_agents)
        self._n_agents: int = len(self._possible_agents)
        self._agent_index: Dict[str, int] = dict(agent_index)
        self._rng = rng
        self._seed = seed
        self._map_split_mode = str(map_split_mode).strip().lower()

        # Spawn policy config
        self._spawn_policy: Optional[str] = cfg.get("spawn_policy")
        self._spawn_centerline_cfg: Mapping[str, Any] = cfg.get("spawn_centerline", {}) or {}
        self._spawn_offsets_cfg: Mapping[str, Any] = cfg.get("spawn_offsets", {}) or {}
        self._spawn_target_cfg: Mapping[str, Any] = cfg.get("spawn_target", {}) or {}
        self._spawn_ego_cfg: Mapping[str, Any] = cfg.get("spawn_ego", {}) or {}

        # Random-spawn config (legacy flat or normalized nested)
        random_spawn_raw = cfg.get("random_spawn") or cfg.get("spawn_random")
        allow_reuse = False
        if isinstance(random_spawn_raw, Mapping):
            self._random_spawn_requested: bool = bool(random_spawn_raw.get("enabled", True))
            allow_reuse = bool(random_spawn_raw.get("allow_reuse", False))
        else:
            self._random_spawn_requested = bool(random_spawn_raw)
            allow_reuse = bool(cfg.get("random_spawn_allow_reuse", False))
        self._random_spawn_allow_reuse: bool = allow_reuse

        # Spawn points from initial map data
        # Prefer caller-supplied metadata (env passes its already-parsed YAML meta);
        # fall back to map_data.metadata if present, else empty dict.
        if init_metadata is not None:
            effective_meta: Mapping[str, Any] = init_metadata
        elif map_data is not None:
            effective_meta = getattr(map_data, "metadata", None) or {}
        else:
            effective_meta = {}
        self._spawn_points: Dict[str, np.ndarray] = extract_spawn_points(map_data, effective_meta)
        self._spawn_point_names: list = sorted(self._spawn_points.keys())
        self._random_spawn_enabled: bool = (
            self._random_spawn_requested and bool(self._spawn_points)
        )

        # Episode-scoped state
        self._centerline_index: int = 0
        self._last_spawn_metadata: Dict[str, Any] = {}
        self._last_spawn_mapping: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reseed(self, seed: int, rng: np.random.Generator) -> None:
        """Update seed and RNG after a reset with a new seed."""
        self._seed = seed
        self._rng = rng

    def reset_episode(self) -> None:
        """Clear episode-scoped state at the start of each reset."""
        self._last_spawn_metadata = {}
        self._last_spawn_mapping = {}

    def update_map_data(
        self,
        map_data: Optional[Any],
        metadata: Mapping[str, Any],
    ) -> None:
        """Refresh spawn points after a map cycle.

        Parameters
        ----------
        map_data:
            New map data object (may be ``None``).
        metadata:
            Parsed map YAML metadata dict from the new map.
        """
        self._spawn_points = extract_spawn_points(map_data, metadata)
        self._spawn_point_names = sorted(self._spawn_points.keys())
        if not self._spawn_points:
            self._random_spawn_enabled = False
        else:
            self._random_spawn_enabled = self._random_spawn_requested

    # ------------------------------------------------------------------
    # Spawn resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        options: Optional[Mapping[str, Any]],
        *,
        centerline: Optional[np.ndarray] = None,
        walls: Optional[Any] = None,
        start_poses: Optional[np.ndarray] = None,
        spawn_plan: Optional[SpawnPlan] = None,
    ) -> SpawnResult:
        """Resolve spawn poses for one episode reset.

        Parameters
        ----------
        options:
            Options dict forwarded from ``env.reset(options=…)``.
        centerline:
            Current centerline point array (for ``centerline_relative``
            policy).
        walls:
            Wall arrays for lateral-offset clamping.
        start_poses:
            Current ``env.start_poses`` used as the lowest-priority
            fallback.
        spawn_plan:
            Optional deterministic plan for eval/replay (highest priority).

        Returns
        -------
        SpawnResult
            Resolved poses, velocities, and metadata.
        """
        request = SpawnRequest(
            agent_ids=self._possible_agents,
            n_agents=self._n_agents,
            agent_index=self._agent_index,
            options=options,
            current_start_poses=start_poses,
            random_spawn_enabled=self._random_spawn_enabled,
            random_spawn_allow_reuse=self._random_spawn_allow_reuse,
            spawn_points=self._spawn_points,
            spawn_point_names=self._spawn_point_names,
            rng=self._rng,
            seed=self._seed,
        )

        centerline_fn: Optional[CenterlineSpawnFn] = None
        if self._spawn_policy is not None or centerline is not None:
            _cl = centerline
            _walls = walls
            def _cl_fn() -> Optional[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
                return self._sample_centerline(_cl, _walls)
            centerline_fn = _cl_fn

        result = resolve_reset_spawn(
            request,
            centerline_spawn_fn=centerline_fn,
            spawn_plan=spawn_plan,
        )
        self._last_spawn_metadata = dict(result.metadata)
        self._last_spawn_mapping = dict(result.spawn_mapping)
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_spawn_metadata(self) -> Dict[str, Any]:
        """Metadata dict from the most recent :meth:`resolve` call."""
        return self._last_spawn_metadata

    @property
    def last_spawn_mapping(self) -> Dict[str, str]:
        """``agent_id → spawn_point_name`` from the most recent :meth:`resolve`."""
        return self._last_spawn_mapping

    @property
    def spawn_points(self) -> Dict[str, np.ndarray]:
        """Named spawn points available on the current map."""
        return self._spawn_points

    @property
    def random_spawn_enabled(self) -> bool:
        """True when random named-point spawning is active."""
        return self._random_spawn_enabled

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_centerline(
        self,
        centerline: Optional[np.ndarray],
        walls: Optional[Any],
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
        result = sample_centerline_relative_spawn(
            spawn_policy=self._spawn_policy,
            centerline=centerline,
            map_split_mode=self._map_split_mode,
            spawn_centerline_cfg=self._spawn_centerline_cfg,
            spawn_offsets_cfg=self._spawn_offsets_cfg,
            spawn_target_cfg=self._spawn_target_cfg,
            spawn_ego_cfg=self._spawn_ego_cfg,
            agent_ids=self._possible_agents,
            rng=self._rng,
            current_index=self._centerline_index,
            walls=walls,
        )
        if result is None:
            return None
        self._centerline_index = result.next_index
        return result.poses, result.velocities, result.metadata
