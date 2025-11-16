from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from f110x.runner.context import RunnerContext

PathLogPoint = Tuple[int, int, str, float, float, float, float, Optional[float], Optional[float]]


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    cleaned = cleaned.strip("-")
    return cleaned or "scenario"


def slugify_suffix(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value))
    cleaned = cleaned.strip("-")
    return cleaned or None


def resolve_run_suffix(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    candidates = [
        os.environ.get("F110_RUN_SUFFIX"),
    ]
    if isinstance(metadata, Mapping):
        candidates.append(metadata.get("run_suffix"))
    candidates.extend(
        [
            os.environ.get("WANDB_NAME"),
            os.environ.get("WANDB_RUN_NAME"),
        ]
    )
    if isinstance(metadata, Mapping):
        candidates.append(metadata.get("wandb_run_name"))
    candidates.extend(
        [
            os.environ.get("RUN_CONFIG_HASH"),
            os.environ.get("WANDB_RUN_PATH"),
            os.environ.get("RUN_ITER"),
            os.environ.get("RUN_SEED"),
            os.environ.get("WANDB_RUN_ID"),
        ]
    )
    if isinstance(metadata, Mapping):
        candidates.append(metadata.get("wandb_run_id"))
    for candidate in candidates:
        slug = slugify_suffix(candidate)
        if slug:
            return slug
    return None


def resolve_episode_cause_code(
    *,
    success: bool,
    attacker_crashed: bool,
    defender_crashed: bool,
    truncated: bool,
) -> int:
    if success:
        return 0
    if attacker_crashed and defender_crashed:
        return 2
    if attacker_crashed:
        return 1
    if truncated:
        return 3
    return 4


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        return value.__dict__
    except Exception:
        return str(value)


class PlotArtifactLogger:
    """Persist per-episode path traces and summary metrics under plots/."""

    def __init__(self, context: RunnerContext, *, run_suffix: Optional[str]) -> None:
        raw_cfg = getattr(context.cfg, "raw", {}) or {}
        if not isinstance(raw_cfg, Mapping):
            raw_cfg = {"value": raw_cfg}

        scenario_name: Optional[str] = None
        meta_block = raw_cfg.get("meta") if isinstance(raw_cfg, Mapping) else None
        if isinstance(meta_block, Mapping):
            scenario_name = meta_block.get("name")
        if not scenario_name:
            try:
                scenario_name = str(context.cfg.main.get("experiment_name"))
            except Exception:
                scenario_name = None
        slug = slugify(scenario_name or "scenario")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_label = run_suffix or timestamp
        base_dir = Path("plots") / slug / run_label / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)

        self._base_dir = base_dir
        self.run_label = run_label
        self.timestamp = timestamp
        self.raw_config_snapshot: Mapping[str, Any] = raw_cfg
        self.path_log_file = base_dir / f"paths_{run_label}.csv"
        self.metrics_file = base_dir / f"metrics_{run_label}.csv"
        self._path_header_written = self.path_log_file.exists() and self.path_log_file.stat().st_size > 0
        self._metrics_header_written = self.metrics_file.exists() and self.metrics_file.stat().st_size > 0

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def log_path_points(self, points: Sequence[PathLogPoint], cause_code: int) -> None:
        if not points:
            return
        self.path_log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.path_log_file.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if not self._path_header_written:
                writer.writerow(
                    [
                        "episode",
                        "step",
                        "agent_id",
                        "x",
                        "y",
                        "theta",
                        "step_reward",
                        "cause_code",
                        "target_distance",
                        "pressure_active",
                    ]
                )
                self._path_header_written = True
            for (
                ep_num,
                step_idx,
                agent_id,
                x_val,
                y_val,
                theta_val,
                step_reward,
                target_distance,
                pressure_active,
            ) in points:
                writer.writerow(
                    [
                        ep_num,
                        step_idx,
                        agent_id,
                        x_val,
                        y_val,
                        theta_val,
                        step_reward,
                        cause_code,
                        "" if target_distance is None else target_distance,
                        "" if pressure_active is None else pressure_active,
                    ]
                )

    def log_episode_metrics(self, row: Mapping[str, Any]) -> None:
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_file.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if not self._metrics_header_written:
                writer.writerow(
                    [
                        "episode",
                        "steps",
                        "success",
                        "success_rate_window",
                        "success_rate_total",
                        "defender_survival_steps",
                        "avg_relative_distance",
                        "pressure_coverage",
                        "collisions",
                        "cause_code",
                    ]
                )
                self._metrics_header_written = True
            writer.writerow(
                [
                    row.get("episode", ""),
                    row.get("steps", ""),
                    row.get("success", ""),
                    row.get("success_rate_window", ""),
                    row.get("success_rate_total", ""),
                    row.get("defender_survival_steps", ""),
                    row.get("avg_relative_distance", ""),
                    row.get("pressure_coverage", ""),
                    row.get("collisions", ""),
                    row.get("cause_code", ""),
                ]
            )

    def write_run_config_snapshot(self, run_suffix: Optional[str]) -> None:
        payload = {
            "run_suffix": run_suffix,
            "timestamp": self.timestamp,
            "scenario": self.raw_config_snapshot,
        }
        config_path = self.base_dir / f"run_config_{self.run_label}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)


__all__ = [
    "PlotArtifactLogger",
    "resolve_episode_cause_code",
    "resolve_run_suffix",
    "slugify",
    "slugify_suffix",
]
