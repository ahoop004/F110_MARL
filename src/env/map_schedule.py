"""Per-episode map bundle selection and loading.

``MapScheduler`` owns all map-cycling state that was previously scattered
across ``F110ParallelEnv.__init__``, ``_select_map_bundle``, and
``_maybe_cycle_map``.  It also centralises the map-load config assembly,
which is the *centerline autoload policy*: the decision about which
centerline/walls config keys to forward to ``MapLoader.load()``.

The env coordinator calls :meth:`select_next_bundle` at the start of each
episode reset and :meth:`load_bundle` when a new bundle has been selected.
``active_bundle`` is updated by the env after ``_apply_map_data`` succeeds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.utils.map_loader import MapData, MapLoader


class MapScheduler:
    """Owns per-episode map bundle selection and data loading.

    Parameters
    ----------
    cfg:
        The merged environment config dict (same one passed to
        ``F110ParallelEnv.__init__``).
    rng:
        The environment's random-number generator.  Passed in so the
        scheduler shares the same RNG stream as the env (deterministic
        replay depends on this).
    """

    def __init__(self, cfg: Mapping[str, Any], *, rng: np.random.Generator) -> None:
        self._rng = rng

        # Resolved map root directory
        self._map_root = Path(
            cfg.get("map_dir") or cfg.get("map_root") or Path.cwd()
        ).resolve()

        # Cycling policy
        self._cycle_mode = str(cfg.get("map_cycle", "")).strip().lower()
        self._pick_mode = str(cfg.get("map_pick", "first")).strip().lower()
        self._epoch_shuffle = bool(cfg.get("epoch_shuffle", False))

        # Bundle lists
        bundles_all = list(cfg.get("map_bundles") or [])
        bundles_train = list(cfg.get("map_bundles_train") or [])
        bundles_eval = list(cfg.get("map_bundles_eval") or [])
        if not bundles_train and bundles_all:
            bundles_train = list(bundles_all)
        # eval stays empty unless explicitly configured

        self._cycle_order: Dict[str, List[str]] = {
            "train": bundles_train,
            "eval": bundles_eval,
        }
        self._cycle_indices: Dict[str, int] = {"train": 0, "eval": 0}

        if self._epoch_shuffle:
            for key, order in self._cycle_order.items():
                if order:
                    self._rng.shuffle(order)
                    self._cycle_indices[key] = 0

        # Active bundle tracking
        self._active_bundle: Optional[str] = (
            cfg.get("map_bundle_active") or cfg.get("map_bundle") or None
        )

        # Static map-load config (centerline / walls autoload policy)
        self._map_ext: str = str(cfg.get("map_ext", ".png"))
        self._centerline_csv: Optional[str] = cfg.get("centerline_csv") or None
        self._walls_autoload: bool = bool(cfg.get("walls_autoload", True))
        self._walls_csv: Optional[str] = cfg.get("walls_csv") or None
        self._track_threshold: Optional[float] = cfg.get("track_threshold")
        self._track_inverted: bool = bool(cfg.get("track_inverted", False))

        self._loader = MapLoader(base_dir=Path.cwd())

        # In-memory cache: (bundle, map_ext, centerline_render, centerline_features) → MapData.
        # Avoids re-parsing YAML/image/centerline when the same bundle is cycled back.
        self._cache: Dict[Tuple, MapData] = {}

    # ------------------------------------------------------------------
    # Active bundle
    # ------------------------------------------------------------------

    @property
    def active_bundle(self) -> Optional[str]:
        """The map bundle currently in use."""
        return self._active_bundle

    @active_bundle.setter
    def active_bundle(self, value: Optional[str]) -> None:
        self._active_bundle = value

    # ------------------------------------------------------------------
    # Bundle selection
    # ------------------------------------------------------------------

    def select_next_bundle(self, split_mode: str) -> Optional[str]:
        """Return the bundle to use for the next episode, or ``None``.

        Returns ``None`` when cycling is disabled (``map_cycle`` is not
        ``"per_episode"``), or when no bundles are configured for the
        current split.

        Parameters
        ----------
        split_mode:
            ``"train"`` or ``"eval"``; selects which bundle list to use.
        """
        if self._cycle_mode != "per_episode":
            return None
        mode = "eval" if split_mode == "eval" else "train"
        bundles = self._cycle_order.get(mode) or []
        if not bundles:
            return None

        if self._pick_mode == "random":
            return bundles[int(self._rng.integers(0, len(bundles)))]
        if self._pick_mode == "first":
            return bundles[0]

        # Round-robin
        idx = int(self._cycle_indices.get(mode, 0))
        bundle = bundles[idx % len(bundles)]
        idx += 1
        if idx >= len(bundles):
            if self._epoch_shuffle:
                self._rng.shuffle(bundles)
            idx = 0
        self._cycle_indices[mode] = idx
        return bundle

    # ------------------------------------------------------------------
    # Map-load config assembly (centerline autoload policy)
    # ------------------------------------------------------------------

    def build_load_config(
        self,
        bundle: str,
        *,
        map_ext: str,
        centerline_render: bool,
        centerline_features: bool,
    ) -> Dict[str, Any]:
        """Build the config dict for ``MapLoader.load()`` for *bundle*.

        This method centralises the *centerline autoload policy*: the
        decision to forward ``centerline_autoload``, ``centerline_csv``,
        render/feature flags, and walls config to the loader.
        """
        return {
            "map_dir": str(self._map_root),
            "map_bundle": bundle,
            "map_ext": map_ext,
            # Centerline autoload policy
            "centerline_autoload": True,
            "centerline_csv": self._centerline_csv,
            "centerline_render": centerline_render,
            "centerline_features": centerline_features,
            # Walls
            "walls_autoload": self._walls_autoload,
            "walls_csv": self._walls_csv,
            # Track mask
            "track_threshold": self._track_threshold,
            "track_inverted": self._track_inverted,
        }

    def load_bundle(
        self,
        bundle: str,
        *,
        map_ext: str,
        centerline_render: bool,
        centerline_features: bool,
    ) -> MapData:
        """Load *bundle* and return a populated :class:`~src.utils.map_loader.MapData`.

        Results are cached by ``(bundle, map_ext, centerline_render,
        centerline_features)`` so revisiting the same bundle within a run
        does not re-parse YAML / image / centerline from disk.
        """
        cache_key = (bundle, map_ext, centerline_render, centerline_features)
        if cache_key in self._cache:
            return self._cache[cache_key]
        cfg = self.build_load_config(
            bundle,
            map_ext=map_ext,
            centerline_render=centerline_render,
            centerline_features=centerline_features,
        )
        result = self._loader.load(cfg)
        self._cache[cache_key] = result
        return result

    def invalidate_cache(self, bundle: Optional[str] = None) -> None:
        """Clear cached map data.

        Parameters
        ----------
        bundle:
            When given, removes only entries for that bundle.  When ``None``,
            clears the entire cache.
        """
        if bundle is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache if k[0] == bundle]
            for key in keys_to_remove:
                del self._cache[key]

    def load_from_path(self, map_path: str, map_ext: str) -> MapData:
        """Load map data from a raw YAML path without bundle-config context.

        Used by ``F110ParallelEnv.update_map`` for runtime map hot-swaps.
        Does not load centerline or walls (those remain in-memory on the env).
        Result is **not** cached since hot-swap paths are one-off.
        """
        path = Path(map_path).resolve()
        cfg: Dict[str, Any] = {
            "map_dir": str(path.parent),
            "map_yaml": path.name,
            "map": path.name,
            "map_ext": map_ext,
        }
        return self._loader.load(cfg)
