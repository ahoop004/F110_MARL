"""Offline RL dataset writer — streams TransitionRecord objects to chunked .npz files.

File layout
-----------
``{output_dir}/``
├── ``metadata.json``                    — schema version, dimensions, totals
├── ``transitions_000000.npz``           — first chunk (up to chunk_size rows)
├── ``transitions_000001.npz``           — second chunk …
└── …

Each ``.npz`` contains parallel arrays (same first dimension N):

    obs          float32 (N, obs_dim)
    action_norm  float32 (N, action_dim)
    action_phys  float32 (N, action_dim)
    reward       float32 (N,)
    next_obs     float32 (N, obs_dim)
    terminated   bool    (N,)
    truncated    bool    (N,)
    global_state float32 (N, global_state_dim)   — zero-dim if not available
    map_id       object  (N,)  — str or ""
    spawn_id     object  (N,)  — str or ""
    episode_id   object  (N,)  — str
    step_idx     int32   (N,)
    agent_id     object  (N,)  — str

``metadata.json`` is written (or updated) on :meth:`close`.  It records the
``DATASET_SCHEMA_VERSION`` so consumers can detect forward-incompatible changes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from env.types import TransitionRecord

_log = logging.getLogger(__name__)

DATASET_SCHEMA_VERSION = "2.0"
SUPPORTED_DATASET_SCHEMA_VERSIONS = frozenset({"1.0", DATASET_SCHEMA_VERSION})


def detect_dataset_schema(path: str | Path) -> str:
    """Return a supported dataset schema without guessing from chunk fields."""
    metadata_path = Path(path)
    if metadata_path.is_dir():
        metadata_path = metadata_path / "metadata.json"
    with open(metadata_path) as handle:
        version = str((json.load(handle) or {}).get("schema_version", ""))
    if version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported dataset schema version: {version or '<missing>'}")
    return version


class DatasetWriter:
    """Streams :class:`~env.types.TransitionRecord` objects to chunked ``.npz`` files.

    Parameters
    ----------
    output_dir:
        Directory where chunk files and ``metadata.json`` are written.
        Created on first flush if it does not exist.
    chunk_size:
        Number of transitions per ``.npz`` file.  Default 10 000.
    metadata:
        Extra key/value pairs written into ``metadata.json`` (e.g. scenario
        name, run_id, obs_dim).

    Usage::

        writer = DatasetWriter("datasets/ppo_run1", chunk_size=5000,
                               metadata={"run_id": run_id, "algo": "ppo"})
        # ... inside training loop ...
        writer.add(record)
        # ... at training end ...
        writer.close()

    Or use as a context manager::

        with DatasetWriter("datasets/run1") as writer:
            writer.add(record)
    """

    def __init__(
        self,
        output_dir: str | Path,
        chunk_size: int = 10_000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._dir = Path(output_dir)
        self._chunk_size = max(1, int(chunk_size))
        self._extra_meta: Dict[str, Any] = dict(metadata or {})

        self._buffer: List["TransitionRecord"] = []
        self._chunk_idx = 0
        self._total = 0
        self._closed = False

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add(self, record: "TransitionRecord") -> None:
        """Buffer one transition.  Flushes automatically when buffer is full."""
        if self._closed:
            raise RuntimeError("DatasetWriter is closed — cannot add more records.")
        self._buffer.append(record)
        if len(self._buffer) >= self._chunk_size:
            self._flush()

    def _flush(self) -> None:
        """Write buffered transitions to a new chunk file."""
        if not self._buffer:
            return
        self._dir.mkdir(parents=True, exist_ok=True)

        n = len(self._buffer)
        obs_dim = len(self._buffer[0].obs)
        act_dim = len(self._buffer[0].action_norm)
        gs_dim = len(self._buffer[0].global_state)

        obs_arr         = np.zeros((n, obs_dim),  dtype=np.float32)
        act_norm_arr    = np.zeros((n, act_dim),  dtype=np.float32)
        act_phys_arr    = np.zeros((n, act_dim),  dtype=np.float32)
        reward_arr      = np.zeros(n,             dtype=np.float32)
        next_obs_arr    = np.zeros((n, obs_dim),  dtype=np.float32)
        terminated_arr  = np.zeros(n,             dtype=bool)
        truncated_arr   = np.zeros(n,             dtype=bool)
        gs_arr          = np.zeros((n, gs_dim),   dtype=np.float32) if gs_dim > 0 else np.empty((n, 0), dtype=np.float32)
        map_id_arr      = np.empty(n, dtype=object)
        spawn_id_arr    = np.empty(n, dtype=object)
        episode_id_arr  = np.empty(n, dtype=object)
        step_idx_arr    = np.zeros(n,             dtype=np.int32)
        agent_id_arr    = np.empty(n, dtype=object)
        lap_crossed_arr = np.zeros(n, dtype=bool)
        lap_count_arr = np.zeros(n, dtype=np.int32)
        target_laps_arr = np.ones(n, dtype=np.int32)
        race_completed_arr = np.zeros(n, dtype=bool)
        terminal_reason_arr = np.empty(n, dtype=object)
        lifecycle_status_arr = np.empty(n, dtype=object)
        finish_position_arr = np.full(n, -1, dtype=np.int32)
        mask_keys = ("active_mask", "finished_mask", "crashed_mask", "truncated_mask")
        mask_dim = max(
            (
                len(np.asarray(rec.lifecycle_masks.get("active_mask", [])))
                for rec in self._buffer
            ),
            default=0,
        )
        lifecycle_mask_arr = np.zeros((n, len(mask_keys), mask_dim), dtype=bool)

        for i, rec in enumerate(self._buffer):
            obs_arr[i]        = rec.obs
            act_norm_arr[i]   = rec.action_norm
            act_phys_arr[i]   = rec.action_phys
            reward_arr[i]     = rec.reward
            next_obs_arr[i]   = rec.next_obs
            terminated_arr[i] = rec.terminated
            truncated_arr[i]  = rec.truncated
            if gs_dim > 0:
                gs_arr[i]     = rec.global_state
            map_id_arr[i]     = rec.map_id or ""
            spawn_id_arr[i]   = rec.spawn_id or ""
            episode_id_arr[i] = rec.episode_id
            step_idx_arr[i]   = rec.step_idx
            agent_id_arr[i]   = rec.agent_id
            lap_crossed_arr[i] = rec.lap_crossed
            lap_count_arr[i] = rec.lap_count
            target_laps_arr[i] = rec.target_laps
            race_completed_arr[i] = rec.race_completed
            terminal_reason_arr[i] = rec.terminal_reason or ""
            lifecycle_status_arr[i] = rec.lifecycle_status
            finish_position_arr[i] = rec.finish_position if rec.finish_position is not None else -1
            for mask_idx, key in enumerate(mask_keys):
                mask = np.asarray(rec.lifecycle_masks.get(key, []), dtype=bool).reshape(-1)
                lifecycle_mask_arr[i, mask_idx, : min(mask_dim, mask.size)] = mask[:mask_dim]

        chunk_path = self._dir / f"transitions_{self._chunk_idx:06d}.npz"
        np.savez_compressed(
            chunk_path,
            obs=obs_arr,
            action_norm=act_norm_arr,
            action_phys=act_phys_arr,
            reward=reward_arr,
            next_obs=next_obs_arr,
            terminated=terminated_arr,
            truncated=truncated_arr,
            global_state=gs_arr,
            map_id=map_id_arr,
            spawn_id=spawn_id_arr,
            episode_id=episode_id_arr,
            step_idx=step_idx_arr,
            agent_id=agent_id_arr,
            lap_crossed=lap_crossed_arr,
            lap_count=lap_count_arr,
            target_laps=target_laps_arr,
            race_completed=race_completed_arr,
            terminal_reason=terminal_reason_arr,
            lifecycle_status=lifecycle_status_arr,
            finish_position=finish_position_arr,
            lifecycle_masks=lifecycle_mask_arr,
            lifecycle_mask_keys=np.asarray(mask_keys, dtype=object),
        )
        _log.info("DatasetWriter: wrote %d transitions → %s", n, chunk_path)

        self._total += n
        self._chunk_idx += 1
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _write_metadata(self) -> None:
        meta: Dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "total_transitions": self._total,
            "num_chunks": self._chunk_idx,
            "chunk_size": self._chunk_size,
        }
        meta.update(self._extra_meta)
        meta_path = self._dir / "metadata.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        _log.info("DatasetWriter: metadata → %s  (%d total transitions)", meta_path, self._total)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush remaining buffer, write metadata, and mark writer as closed."""
        if self._closed:
            return
        self._flush()
        self._write_metadata()
        self._closed = True

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total_written(self) -> int:
        """Total transitions flushed to disk (excludes current buffer)."""
        return self._total

    @property
    def buffer_size(self) -> int:
        """Number of transitions currently buffered but not yet flushed."""
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------

class DatasetHook:
    """TrainingHook adapter that writes every agent step to a DatasetWriter.

    Parameters
    ----------
    writer:
        An open :class:`DatasetWriter` instance.  ``close()`` is **not**
        called automatically — do it in ``on_training_end`` or via context
        manager at the call site.

    Example::

        writer = DatasetWriter("datasets/run1", metadata={"algo": "ppo"})
        hook = DatasetHook(writer)
        trainer = OnPolicyTrainer(..., hooks=[..., hook])
        trainer.train(n_episodes=1000)
        writer.close()
    """

    requires_transition_record = True

    def __init__(self, writer: DatasetWriter) -> None:
        self._writer = writer

    def on_step(self, record: "TransitionRecord") -> None:
        self._writer.add(record)

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        pass

    def on_update(self, metrics: Dict) -> None:
        pass

    def on_training_end(self) -> None:
        self._writer.close()
