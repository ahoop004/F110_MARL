#!/usr/bin/env python3
"""Replay recorded dataset episodes with the F110 renderer.

Usage
-----
    python replay.py <dataset_dir> [options]

Examples
--------
    python replay.py datasets/run1                         # all episodes in order
    python replay.py datasets/run1 --episode ep_0042      # filter by episode ID substring
    python replay.py datasets/run1 --map Budapest_map     # filter by map substring
    python replay.py datasets/run1 --speed 2.0            # double speed
    python replay.py datasets/run1 --list                 # list episodes and exit

Recording
---------
    python run.py --scenario scenarios/legacy/ppo_time_trial.yaml --no-wandb \\
        --dataset-dir datasets/run1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Must match F110ParallelEnv._central_state_keys (order matters)
_STATE_KEYS: Tuple[str, ...] = (
    "poses_x",
    "poses_y",
    "poses_theta",
    "linear_vels_x",
    "linear_vels_y",
    "ang_vels_z",
    "collisions",
)
_N_KEYS = len(_STATE_KEYS)
_V2_KEYS_PER_AGENT = _N_KEYS + 5

_DEFAULT_TIMESTEP = 0.01
_DEFAULT_ACTION_REPEAT = 2


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir: Path) -> Dict[str, np.ndarray]:
    """Load and concatenate all transition chunks from *dataset_dir*."""
    metadata = dataset_dir / 'metadata.json'
    if metadata.exists() and json.loads(metadata.read_text()).get('schema_version') == '3.0':
        from src.replay.race_reader import iter_race_frames
        frames = list(iter_race_frames(dataset_dir))
        return {'race_frame': np.asarray(frames, dtype=object),
                'episode_id': np.asarray([f['episode_id'] for f in frames], dtype=object),
                'map_id': np.asarray([f['map_id'] for f in frames], dtype=object),
                'step_idx': np.asarray([f['physics_index'] for f in frames], dtype=np.int64)}
    chunks = sorted(dataset_dir.glob("transitions_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No transition chunks found in {dataset_dir}")
    parts: Dict[str, List[np.ndarray]] = {}
    for path in chunks:
        d = np.load(path, allow_pickle=True)
        for key in d.files:
            parts.setdefault(key, []).append(d[key])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def group_episodes(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map episode_id → row-index array sorted by step_idx."""
    episode_ids = data["episode_id"]
    step_idxs = data["step_idx"]
    groups: Dict[str, List[int]] = {}
    for i, ep in enumerate(episode_ids):
        groups.setdefault(str(ep), []).append(i)
    result = {}
    for ep, indices in groups.items():
        arr = np.array(indices, dtype=np.int64)
        result[ep] = arr[np.argsort(step_idxs[arr])]
    return result


# ---------------------------------------------------------------------------
# Global-state decoding
# ---------------------------------------------------------------------------

def unpack_global_state(
    gs: np.ndarray,
    n_agents: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Decode the packed global-state vector into per-agent render-obs dicts.

    Layout (n_agents agents, 7 keys each):
        [poses_x[0..n-1] | poses_y[0..n-1] | poses_theta[0..n-1] | ...]
    """
    if n_agents is not None:
        n = max(1, int(n_agents))
    elif len(gs) % _V2_KEYS_PER_AGENT == 0:
        n = max(1, len(gs) // _V2_KEYS_PER_AGENT)
    else:
        n = max(1, len(gs) // _N_KEYS)
    result: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        entry: Dict[str, float] = {}
        for k_idx, key in enumerate(_STATE_KEYS):
            vec_idx = k_idx * n + i
            entry[key] = float(gs[vec_idx]) if vec_idx < len(gs) else 0.0
        entry["collision"] = bool(entry.get("collisions", 0.0) > 0.5)
        result[f"car_{i}"] = entry
    return result


def deduplicate_by_step(
    idxs: np.ndarray,
    step_idxs: np.ndarray,
) -> List[int]:
    """Return one row index per unique step (first seen), sorted by step."""
    seen: Set[int] = set()
    rows: List[int] = []
    for i in idxs:
        s = int(step_idxs[i])
        if s not in seen:
            seen.add(s)
            rows.append(int(i))
    rows.sort(key=lambda i: int(step_idxs[i]))
    return rows


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def load_map(map_id: str, maps_dir: Path) -> Tuple[str, str, dict]:
    """Return (map_path_no_ext, map_ext, meta) for renderer.update_map()."""
    import yaml  # type: ignore[import]

    map_dir = maps_dir / map_id
    if not map_dir.is_dir():
        raise FileNotFoundError(f"Map directory not found: {map_dir}")

    map_ext = ".png"
    for ext in (".png", ".pgm"):
        if (map_dir / f"{map_id}{ext}").exists():
            map_ext = ext
            break

    yaml_path = map_dir / f"{map_id}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Map YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    return str(map_dir / map_id), map_ext, meta


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(
    dataset_dir: Path,
    *,
    maps_dir: Path,
    filter_episode: Optional[str] = None,
    filter_map: Optional[str] = None,
    speed: float = 1.0,
    list_only: bool = False,
    window_width: int = 1280,
    window_height: int = 800,
    action_repeat: int = _DEFAULT_ACTION_REPEAT,
    timestep: float = _DEFAULT_TIMESTEP,
    filter_clip: Optional[str] = None,
) -> None:
    metadata = dataset_dir / 'metadata.json'
    if metadata.exists() and json.loads(metadata.read_text()).get('schema_version') == '3.0':
        return replay_races(dataset_dir, maps_dir=maps_dir, filter_episode=filter_episode,
                            filter_map=filter_map, filter_clip=filter_clip, speed=speed,
                            list_only=list_only, window_width=window_width, window_height=window_height)
    print(f"Loading dataset: {dataset_dir}")
    data = load_dataset(dataset_dir)
    lifecycle_masks = data.get("lifecycle_masks")
    dataset_n_agents = (
        int(lifecycle_masks.shape[-1])
        if lifecycle_masks is not None and lifecycle_masks.ndim == 3
        else None
    )
    episodes = group_episodes(data)
    total_loaded = len(data["episode_id"])
    print(f"  {total_loaded:,} transitions across {len(episodes)} episode(s)")

    # Apply filters
    if filter_episode:
        episodes = {k: v for k, v in episodes.items() if filter_episode in k}
    if filter_map:
        episodes = {
            k: v for k, v in episodes.items()
            if filter_map in str(data["map_id"][v[0]])
        }

    if list_only or not episodes:
        print(f"\n{'Episode ID':<52}  {'Map':<20}  Frames")
        print("-" * 84)
        for ep, idxs in sorted(episodes.items()):
            ep_map = str(data["map_id"][idxs[0]])
            frames = len(deduplicate_by_step(idxs, data["step_idx"]))
            print(f"  {ep:<50}  {ep_map:<20}  {frames}")
        if not episodes:
            print("  (no episodes match the given filters)")
        return

    from render.renderer import EnvRenderer

    frame_dt = (timestep * action_repeat) / max(speed, 1e-9)
    renderer: Optional[EnvRenderer] = None
    current_map: Optional[str] = None

    try:
        ep_list = sorted(episodes.items())
        for ep_num, (ep_id, idxs) in enumerate(ep_list, 1):
            ep_map = str(data["map_id"][idxs[0]])
            frames = deduplicate_by_step(idxs, data["step_idx"])
            print(f"\n[{ep_num}/{len(ep_list)}] {ep_id}  map={ep_map}  frames={len(frames)}")

            if renderer is None:
                renderer = EnvRenderer(window_width, window_height)

            if ep_map != current_map:
                try:
                    map_path, map_ext, map_meta = load_map(ep_map, maps_dir)
                    renderer.update_map(map_path, map_ext, map_meta=map_meta)
                    current_map = ep_map
                except FileNotFoundError as exc:
                    print(f"  WARN: {exc} — skipping episode")
                    continue

            renderer.reset_state()

            for frame_idx, row in enumerate(frames):
                gs = data["global_state"][row]
                render_obs = unpack_global_state(gs, n_agents=dataset_n_agents)

                try:
                    renderer.update_obs(render_obs)
                    renderer.dispatch_events()
                    renderer.flip()
                except Exception:
                    print("  Window closed.")
                    return

                if getattr(renderer, "has_exit", False):
                    print("  Window closed.")
                    return

                if frame_dt > 0:
                    time.sleep(frame_dt)

            # Pause between episodes so the last frame stays visible
            if ep_num < len(ep_list):
                print("  Press Enter for next episode, Ctrl-C to quit.")
                try:
                    input()
                except EOFError:
                    pass

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass


def replay_races(dataset_dir, *, maps_dir, filter_episode=None, filter_map=None,
                 filter_clip=None, speed=1., list_only=False, window_width=1280, window_height=800):
    from src.replay.race_reader import load_clips, clip_frames, render_state
    if speed <= 0:
        raise ValueError('Playback speed must be positive')
    clips = [c for c in load_clips(dataset_dir)
             if (not filter_episode or filter_episode in c['episode_id'])
             and (not filter_map or filter_map in str(c['map_id']))
             and (not filter_clip or filter_clip in c['clip_id'])]
    if list_only:
        for c in clips:
            print(f"{c['clip_id']}  {c['map_id']}  {c['kind']}  "
                  f"physics={c['start_physics_index']}..{c['end_physics_index']}  "
                  f"complete={c['complete']}  reasons={c['retention_reasons']} "
                  f"end={c.get('end_reason', 'open')} all_cars_terminal={c['all_cars_terminal']}")
        return
    from render.renderer import EnvRenderer
    renderer = None
    try:
        for clip in clips:
            frames = clip_frames(dataset_dir, clip)
            first = next(frames, None)
            if first is None:
                continue
            if renderer is None:
                renderer = EnvRenderer(window_width, window_height)
            path, extension, meta = load_map(clip['map_id'], maps_dir)
            renderer.update_map(path, extension, map_meta=meta)
            renderer.reset_state()
            print(f"Replaying {clip['clip_id']} ({clip['kind']}); timing from recorded physics intervals")
            renderer.update_obs(render_state(first['pre_state']))
            renderer.dispatch_events()
            renderer.flip()
            from itertools import chain
            for frame in chain([first], frames):
                time.sleep(max(0., frame['simulation_time_end_s'] - frame['simulation_time_s']) / speed)
                renderer.update_obs(render_state(frame['post_state']))
                renderer.dispatch_events()
                renderer.flip()
                if getattr(renderer, 'has_exit', False):
                    return
    except KeyboardInterrupt:
        pass
    finally:
        if renderer is not None:
            renderer.close()


# CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay recorded F110 dataset episodes with the renderer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dataset_dir", type=Path,
                   help="Directory containing transitions_*.npz files")
    p.add_argument("--episode", type=str, default=None,
                   help="Filter: episode ID substring (e.g. 'ep_0042')")
    p.add_argument('--clip', type=str, default=None, help='Shared-frame dataset: clip ID substring')
    p.add_argument("--map", type=str, default=None, dest="map_filter",
                   help="Filter: map bundle substring (e.g. 'Budapest')")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0)")
    p.add_argument("--list", action="store_true",
                   help="List available episodes and exit without rendering")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--maps-dir", type=Path,
                   default=Path(__file__).parent / "maps",
                   help="Root maps directory (default: ./maps)")
    p.add_argument("--action-repeat", type=int, default=_DEFAULT_ACTION_REPEAT,
                   help=f"Action repeat used during recording (default {_DEFAULT_ACTION_REPEAT})")
    p.add_argument("--timestep", type=float, default=_DEFAULT_TIMESTEP,
                   help=f"Physics timestep used during recording (default {_DEFAULT_TIMESTEP})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    replay(
        args.dataset_dir,
        maps_dir=args.maps_dir,
        filter_episode=args.episode,
        filter_map=args.map_filter,
        speed=args.speed,
        list_only=args.list,
        window_width=args.width,
        window_height=args.height,
        action_repeat=args.action_repeat,
        timestep=args.timestep,
        filter_clip=args.clip,
    )


if __name__ == "__main__":
    main()
