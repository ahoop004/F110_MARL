#!/usr/bin/env python3
"""Watch for training best checkpoints and trigger evaluation runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINTS_ROOT = ROOT_DIR / "outputs" / "checkpoints"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch for training best checkpoints and run eval.py"
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        type=str,
        help="Path to run checkpoint directory (outputs/checkpoints/<algo>/<run_id>)",
    )
    target_group.add_argument(
        "--run-id",
        type=str,
        help="Run ID to watch (searches outputs/checkpoints)",
    )
    target_group.add_argument(
        "--latest",
        action="store_true",
        help="Watch latest run for specified algorithm",
    )

    parser.add_argument(
        "--algo",
        type=str,
        help="Algorithm name (required with --latest)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=str(ROOT_DIR / "output" / "evals"),
        help="Directory to write eval JSON outputs",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to state file tracking evaluated checkpoints",
    )
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Evaluate existing best checkpoints at startup",
    )
    parser.add_argument(
        "--stable-checks",
        type=int,
        default=2,
        help="Consecutive size checks required before eval",
    )
    parser.add_argument(
        "--stable-interval",
        type=float,
        default=1.0,
        help="Seconds between size checks before eval",
    )
    parser.add_argument(
        "--stable-timeout",
        type=float,
        default=120.0,
        help="Timeout waiting for checkpoint to stabilize",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit",
    )
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Args passed to eval.py after --",
    )

    args = parser.parse_args()
    if args.latest and not args.algo:
        parser.error("--algo is required with --latest")
    return args


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except Exception:
        return None


def _parse_created_at(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def _find_run_dir_by_run_id(run_id: str) -> Path:
    if not CHECKPOINTS_ROOT.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {CHECKPOINTS_ROOT}")

    for algo_dir in CHECKPOINTS_ROOT.iterdir():
        if not algo_dir.is_dir():
            continue
        for run_dir in algo_dir.iterdir():
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "run_metadata.json"
            metadata = _load_json(meta_path)
            if metadata and metadata.get("run_id") == run_id:
                return run_dir

    raise FileNotFoundError(f"Run ID '{run_id}' not found in {CHECKPOINTS_ROOT}")


def _find_latest_run_dir(algo: str) -> Path:
    algo_dir = CHECKPOINTS_ROOT / algo
    if not algo_dir.exists():
        raise FileNotFoundError(f"No checkpoints found for algorithm '{algo}'")

    candidates: List[Tuple[datetime, Path]] = []
    for run_dir in algo_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "run_metadata.json"
        metadata = _load_json(meta_path)
        if not metadata:
            continue
        created_at = _parse_created_at(metadata.get("created_at"))
        candidates.append((created_at, run_dir))

    if not candidates:
        raise FileNotFoundError(f"No run metadata found for algorithm '{algo}'")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.run_id:
        return _find_run_dir_by_run_id(args.run_id)
    return _find_latest_run_dir(args.algo)


def _extract_episode(filename: str) -> Optional[int]:
    import re
    match = re.search(r"ep(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def _is_training_best_filename(name: str) -> bool:
    return "_best_ep" in name and "_eval_" not in name


def _list_training_best(run_dir: Path) -> List[Path]:
    candidates: List[Tuple[int, Path]] = []
    for path in run_dir.glob("*.pt"):
        if not _is_training_best_filename(path.name):
            continue
        episode = _extract_episode(path.name)
        episode_key = episode if episode is not None else -1
        candidates.append((episode_key, path))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"processed": {}}
    data = _load_json(state_path)
    if not isinstance(data, dict):
        return {"processed": {}}
    processed = data.get("processed")
    if not isinstance(processed, dict):
        return {"processed": {}}
    return {"processed": processed}


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w") as handle:
        json.dump(state, handle, indent=2)


def _needs_eval(path: Path, processed: Dict[str, Any]) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    key = str(path)
    entry = processed.get(key)
    current = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    if not entry:
        return True
    return (
        entry.get("mtime_ns") != current["mtime_ns"]
        or entry.get("size") != current["size"]
    )


def _update_processed(path: Path, processed: Dict[str, Any]) -> None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    processed[str(path)] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _prune_processed(processed: Dict[str, Any], existing: List[Path]) -> None:
    existing_keys = {str(path) for path in existing}
    for key in list(processed.keys()):
        if key not in existing_keys:
            processed.pop(key, None)


def _wait_for_stable(
    path: Path,
    checks: int,
    interval: float,
    timeout: float,
) -> bool:
    start = time.time()
    last_size = None
    stable = 0
    while time.time() - start < timeout:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            stable = 0
            time.sleep(interval)
            continue
        if last_size is not None and size == last_size:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
            last_size = size
        time.sleep(interval)
    return False


def _run_eval(checkpoint_path: Path, output_dir: Path, eval_args: List[str]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_path.stem}.json"

    cmd = [
        sys.executable,
        str(ROOT_DIR / "eval.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--save-results",
        str(output_path),
    ]
    if eval_args:
        cmd.extend(eval_args)

    print(f"[watch] Running eval: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"[watch] Eval finished with code {result.returncode}")
    return result.returncode


def main() -> int:
    args = _parse_args()
    run_dir = _resolve_run_dir(args)
    run_dir = run_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    metadata = _load_json(run_dir / "run_metadata.json") or {}
    run_id = metadata.get("run_id") or run_dir.name

    output_dir = Path(args.eval_output_dir)
    output_dir = output_dir / run_id

    state_path = Path(args.state_file) if args.state_file else output_dir / "watch_state.json"
    state = _load_state(state_path)
    processed = state["processed"]

    best_checkpoints = _list_training_best(run_dir)
    if not args.evaluate_existing and not processed:
        for path in best_checkpoints:
            _update_processed(path, processed)
        _save_state(state_path, state)

    print(f"[watch] Watching: {run_dir}")
    print(f"[watch] Eval outputs: {output_dir}")
    print(f"[watch] State file: {state_path}")

    while True:
        best_checkpoints = _list_training_best(run_dir)
        _prune_processed(processed, best_checkpoints)

        for checkpoint in best_checkpoints:
            if not _needs_eval(checkpoint, processed):
                continue

            print(f"[watch] New best checkpoint: {checkpoint.name}")
            if not _wait_for_stable(
                checkpoint,
                checks=args.stable_checks,
                interval=args.stable_interval,
                timeout=args.stable_timeout,
            ):
                print(f"[watch] Checkpoint did not stabilize: {checkpoint}")
                continue

            _run_eval(checkpoint, output_dir, args.eval_args)
            _update_processed(checkpoint, processed)
            _save_state(state_path, state)

        if args.once:
            break
        time.sleep(args.poll_interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
