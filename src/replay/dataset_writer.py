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
RACE_DATASET_SCHEMA_VERSION = "3.0"
SUPPORTED_DATASET_SCHEMA_VERSIONS = frozenset({"1.0", DATASET_SCHEMA_VERSION, RACE_DATASET_SCHEMA_VERSION})


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
        Must be absent or empty. Reserved immediately with incomplete metadata;
        reopening or resuming an existing dataset is not supported.
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
        self._dir.mkdir(parents=True, exist_ok=True)
        if any(self._dir.iterdir()):
            raise FileExistsError(f"Dataset directory must be empty: {self._dir}")
        # Exclusive creation reserves an empty directory against a second writer.
        self._write_metadata(exclusive=True)
        from core.provenance import PhysicsEpisodeLog
        self._physics_log = PhysicsEpisodeLog(self._dir)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add(self, record: "TransitionRecord") -> None:
        """Buffer one transition.  Flushes automatically when buffer is full."""
        if self._closed:
            raise RuntimeError("DatasetWriter is closed — cannot add more records.")
        self._physics_log.write(record.episode_id, record.map_id, record.info.get('physics'))
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
        with chunk_path.open("xb") as chunk_file:
            np.savez_compressed(
                chunk_file,
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

    def _write_metadata(self, *, exclusive: bool = False, complete: bool = False) -> None:
        meta: Dict[str, Any] = {
            **self._extra_meta,
            "schema_version": DATASET_SCHEMA_VERSION,
            "total_transitions": self._total,
            "num_chunks": self._chunk_idx,
            "chunk_size": self._chunk_size,
            "complete": complete,
        }
        meta_path = self._dir / "metadata.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "x" if exclusive else "w") as f:
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
        self._write_metadata(complete=True)
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


class RaceDatasetWriter(DatasetWriter):
    """Versioned shared frames, incremental compressed chunks and clip index.

    max_bytes bounds serialized frame payload before compression. The small
    metadata/index files are additional; max_frames also bounds index growth.
    """

    def __init__(self, output_dir, *, config, metadata=None):
        from replay.race_recorder import recording_config
        self.config = recording_config(config)
        self.payload_bytes = 0
        self.storage_full = False
        self.exhausted_windows = set()
        self.window_usage = [dict(index=i, **w, frames=0, serialized_frame_bytes=0, exhausted=False)
                             for i, w in enumerate(self.config['windows'])]
        self._accepted_ends = {}
        self._accepted_contexts = {}
        super().__init__(output_dir, chunk_size=self.config['chunk_frames'], metadata=metadata)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(complete=exc_type is None)

    def can_record(self, progress):
        """Whether this progress point has any unexhausted allocation."""
        if self.storage_full:
            return False
        if not self.window_usage:
            return True
        return any(progress is not None and w['start_step'] <= progress < w['end_step']
                   and not w['exhausted'] for w in self.window_usage)

    def add_event(self, event):
        from replay.race_recorder import plain
        if self._closed:
            raise RuntimeError("RaceDatasetWriter is closed")
        kind, row = event
        key = (row['episode_id'], row.get('recording_window_index'))
        if kind == 'frame':
            if self.storage_full:
                return
            row = plain(row)
            encoded = json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n'
            size = len(encoded.encode('utf-8'))
            window = None
            if self.window_usage:
                index = row.get('recording_window_index')
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.window_usage):
                    raise ValueError('Windowed recording frame requires a valid recording_window_index')
                window = self.window_usage[index]
                progress = row.get('recording_progress')
                if progress is None or not window['start_step'] <= progress < window['end_step']:
                    raise ValueError('Recording frame progress is outside its assigned window')
                if index in self.exhausted_windows:
                    return
                if window['frames'] >= window['max_frames'] or window['serialized_frame_bytes'] + size > window['max_bytes']:
                    self.exhausted_windows.add(index)
                    window['exhausted'] = True
                    self._write_metadata()
                    return
            if (self._total + len(self._buffer) >= self.config['max_frames'] or
                    self.payload_bytes + size > self.config['max_bytes']):
                self.storage_full = True
                self._write_metadata()
                return
            self.payload_bytes += size
            if window is not None:
                window['frames'] += 1
                window['serialized_frame_bytes'] += size
                if window['frames'] >= window['max_frames'] or window['serialized_frame_bytes'] >= window['max_bytes']:
                    self.exhausted_windows.add(window['index'])
                    window['exhausted'] = True
            self._buffer.append((row['episode_id'], row['physics_index'], encoded))
            self._accepted_ends[key] = row['physics_index']
            self._accepted_contexts[key] = dict(
                all_cars_terminal=all(not s['active'] for s in row['post_state'].values()),
                policy_version_end=row['policy_version'])
            if 'team_policy_versions' in row:
                self._accepted_contexts[key]['team_policy_versions_end'] = row['team_policy_versions']
            if len(self._buffer) >= self._chunk_size:
                self._flush()
        elif kind == 'clip':
            row = plain(row)
            if row['status'] == 'closed':
                accepted = self._accepted_ends.get(key, -1)
                end = row['end_physics_index']
                if end is not None and end > accepted:
                    reason = 'window_storage_limit' if row.get('recording_window_index') in self.exhausted_windows else 'storage_limit'
                    row.update(end_physics_index=accepted, complete=False, end_reason=reason,
                               post_context_complete=False, **self._accepted_contexts.get(key,
                                   dict(all_cars_terminal=False, policy_version_end=None)))
                    if accepted < 0 and 'team_policy_versions_end' in row:
                        row['team_policy_versions_end'] = None
                self._flush()
            with (self._dir / 'clips.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        else:
            raise ValueError(f"Unknown race recording event: {kind}")

    def _flush(self):
        if not self._buffer:
            return
        import gzip
        name = f'frames_{self._chunk_idx:06d}.jsonl.gz'
        spans = {}
        with (self._dir / name).open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                for episode_id, step, encoded in self._buffer:
                    stream.write(encoded.encode('utf-8'))
                    span = spans.setdefault(episode_id, [step, step])
                    span[0], span[1] = min(span[0], step), max(span[1], step)
        with (self._dir / 'frame_chunks.jsonl').open('a', encoding='utf-8') as manifest:
            manifest.write(json.dumps(dict(path=name, episodes=spans, frames=len(self._buffer))) + '\n')
        self._total += len(self._buffer)
        self._chunk_idx += 1
        self._buffer.clear()
        self._write_metadata()

    def _write_metadata(self, *, exclusive=False, complete=False):
        meta = {**self._extra_meta, 'schema_version': RACE_DATASET_SCHEMA_VERSION,
                'format': 'shared_race_frames', 'complete': complete,
                'total_frames': self._total, 'num_chunks': self._chunk_idx,
                'recording': self.config, 'serialized_frame_bytes': self.payload_bytes,
                'storage_full': self.storage_full,
                'recording_windows': self.window_usage,
                'frame_contract': {'version': '1.0', 'interval': 'one physics step',
                    'state': 'explicit pre_state and post_state',
                    'learner_observation': 'pre_decision; repeated on held-action substeps',
                    'reward': 'this physics interval', 'commands': 'physical simulator input',
                    'missing_commands': None, 'velocity': 'body frame',
                    'collision_pairing': 'unavailable', 'privileged': ['pre_state', 'post_state']}}
        path = self._dir / 'metadata.json'
        if exclusive:
            with path.open('x') as stream:
                json.dump(meta, stream, indent=2)
        else:
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(meta, indent=2))
            temporary.replace(path)

    def close(self, *, complete=True):
        if self._closed:
            return
        self._flush()
        self._write_metadata(complete=complete)
        self._closed = True


class RaceDatasetHook(DatasetHook):
    requires_transition_record = False

    @property
    def recording_config(self):
        return self._writer.config

    def on_step(self, record):
        pass

    def on_race_record(self, event):
        self._writer.add_event(event)

    @property
    def storage_full(self):
        return self._writer.storage_full

    @property
    def exhausted_windows(self):
        return self._writer.exhausted_windows
