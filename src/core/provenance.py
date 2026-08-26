"""Local run provenance shared by checkpoints, metrics, and datasets."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from core.map_selection import resolve_bundle_yaml


PROVENANCE_VERSION = "1.0"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _configured_bundles(environment: Mapping[str, Any]) -> Iterable[str]:
    bundles: set[str] = set()
    for key in ("maps", "map_bundles", "map_bundles_train", "map_bundles_eval"):
        value = environment.get(key)
        if isinstance(value, str):
            bundles.add(value)
        elif isinstance(value, (list, tuple)):
            bundles.update(str(item) for item in value)
    for key in ("map", "map_bundle"):
        value = environment.get(key)
        if isinstance(value, str):
            bundles.add(value)
    return sorted(bundle for bundle in bundles if bundle)


def collect_map_protocols(environment: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Hash every configured map YAML and capture its finish-line contract."""
    map_root = Path(str(environment.get("map_dir") or environment.get("map_root") or "maps"))
    if not map_root.is_absolute():
        map_root = (Path.cwd() / map_root).resolve()

    protocols: Dict[str, Dict[str, Any]] = {}
    for bundle in _configured_bundles(environment):
        try:
            yaml_path = resolve_bundle_yaml(map_root, bundle)
        except FileNotFoundError:
            continue
        metadata = yaml.safe_load(yaml_path.read_text()) or {}
        finish = (metadata.get("annotations") or {}).get("finish_line", {})
        protocols[bundle] = {
            "yaml_path": str(yaml_path),
            "yaml_sha256": _sha256_bytes(yaml_path.read_bytes()),
            "finish_line_version": int(finish.get("version", 1)),
        }
    return protocols


def _git_state(repo_root: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_run_provenance(
    scenario: Mapping[str, Any],
    *,
    scenario_path: str | Path,
    run_id: str,
    algorithm: str,
    trainable_agents: Iterable[str],
) -> Dict[str, Any]:
    """Build one JSON-safe provenance block after CLI overrides are applied."""
    source = Path(scenario_path).expanduser().resolve()
    environment = scenario.get("environment", {}) or {}
    experiment = scenario.get("experiment", {}) or {}
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "version": PROVENANCE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "algorithm": algorithm,
        "scenario_name": experiment.get("name"),
        "scenario_path": str(source),
        "scenario_source_sha256": _sha256_bytes(source.read_bytes()),
        "resolved_config_sha256": _canonical_config_hash(scenario),
        "seed": experiment.get("seed"),
        "trainable_agents": list(trainable_agents),
        "target_laps": environment.get("target_laps", 1),
        "max_steps": environment.get("max_steps"),
        "map_split": {
            "train": list(environment.get("map_bundles_train") or []),
            "eval": list(environment.get("map_bundles_eval") or []),
        },
        "map_protocols": collect_map_protocols(environment),
        "git": _git_state(repo_root),
    }


def provenance_mismatches(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[str]:
    """Describe contract mismatches between a checkpoint and evaluation run."""
    mismatches: list[str] = []
    for key in ("algorithm", "scenario_source_sha256", "resolved_config_sha256"):
        if stored.get(key) != current.get(key):
            mismatches.append(
                f"{key}: checkpoint={stored.get(key)!r}, current={current.get(key)!r}"
            )

    def map_hashes(value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {
            str(name): details.get("yaml_sha256")
            for name, details in value.items()
            if isinstance(details, Mapping)
        }

    if map_hashes(stored.get("map_protocols")) != map_hashes(current.get("map_protocols")):
        mismatches.append("map_protocols: configured map names or YAML hashes differ")
    return mismatches
