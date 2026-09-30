"""CSV and JSON export logging for training runs.

Provides file-based logging for post-training analysis and reproducibility.
Compatible with v1 PlotArtifactLogger format.
"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CSVLogger:
    """Logger for exporting metrics to CSV and JSON files.

    Creates files in a structured output directory:
        outputs/{scenario}/{run_id}/
            - race_metrics.jsonl       # MAPPO episode and agent facts
            - update_metrics.csv       # Optimizer diagnostics
            - episode_metrics.csv      # PPO episodes; optional MAPPO export
            - agent_metrics.csv        # Optional detailed CSV export
            - config_snapshot.json     # Full scenario configuration
            - run_summary.json         # Final training summary

    Example:
        >>> logger = CSVLogger(
        ...     output_dir="outputs/gaplock_ppo/run_001",
        ...     scenario_config=scenario,
        ... )
        >>> logger.log_training_episode(episode=0, reward=1.0, info={}, metrics={})
        >>> logger.save_summary(summary_stats)
    """

    def __init__(
        self,
        output_dir: str,
        scenario_config: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """Initialize CSV logger.

        Args:
            output_dir: Directory to save output files
            scenario_config: Full scenario configuration dict (saved as config_snapshot.json)
            enabled: Enable/disable logging (default: True)
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.scenario_config = scenario_config
        self.provenance = dict(provenance or {})
        self._tables = {}
        self._jsonl = {}
        self._pending_rows = 0
        self._last_flush = time.monotonic()
        settings = (scenario_config or {}).get("logging", {})
        self.csv_exports = bool(settings.get("csv_exports", False))
        self.collector_progress = bool(settings.get("collector_progress", False))
        self.flush_every = settings.get("flush_every", 64)
        self.flush_interval = settings.get("flush_interval_seconds", 10.)
        if isinstance(self.flush_every, bool) or not isinstance(self.flush_every, int) or self.flush_every < 1:
            raise ValueError("logging.flush_every must be a positive integer")
        if (isinstance(self.flush_interval, bool) or not isinstance(self.flush_interval, (int, float))
                or not math.isfinite(self.flush_interval) or self.flush_interval <= 0):
            raise ValueError("logging.flush_interval_seconds must be positive and finite")

        if not self.enabled:
            return

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV files
        self.episode_metrics_file = self.output_dir / "episode_metrics.csv"
        self.agent_metrics_file = self.output_dir / "agent_metrics.csv"

        # Save config snapshot
        if scenario_config:
            self.save_config_snapshot(scenario_config)

    def log_training_episode(
        self,
        episode: int,
        reward: float,
        info: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """Write the current hook contract without requiring legacy metrics classes."""
        if not self.enabled:
            return

        race = metrics.get("race_record")
        if race is not None:
            self.log_jsonl("race_metrics.jsonl", race)
            if not self.csv_exports:
                return

        row: Dict[str, Any] = {
            "episode": int(episode),
            "reward": float(reward),
            "outcome": info.get("outcome"),
            "map_bundle": info.get("map_bundle"),
            "spawn_id": info.get("spawn_id") or info.get("spawn_point"),
            "episode_steps": metrics.get("episode_steps"),
            "lap_count": info.get("lap_count"),
            "lap_time_s": metrics.get("lap_time_s"),
            "finish_position": info.get("finish_position"),
            "terminal_reason": info.get("terminal_reason"),
        }
        # Optimizer diagnostics belong to update_metrics.csv, on the update clock.
        if "train/environment_steps" in metrics:
            row["environment_steps"] = metrics["train/environment_steps"]
        for key, value in metrics.items():
            if not key.startswith(("train/", "perf/")) and (
                    isinstance(value, (str, int, float, bool)) or value is None):
                row.setdefault(key.replace("/", "_"), value)
        if race is not None:
            row.update({key: value for key, value in race.items()
                        if not isinstance(value, (dict, list))})
            row["team_reward_components"] = json.dumps(race.get("team_reward_components", {}), sort_keys=True)
        self._write_episode_row(row)

        if race is not None:
            identity = {key: race.get(key) for key in (
                "run_id", "environment_id", "episode_id", "environment_episode", "map_id",
                "environment_seed", "policy_version_start", "policy_version_end",
                "reported_at_environment_steps", "race_mode", "phase")}
            for aid, facts in race["agents"].items():
                agent_row = {"episode": episode, **identity, "agent_id": aid, **facts,
                             "spawn_id": race["spawn_ids"].get(aid)}
                # Opponent reward was never evaluated; leave it unavailable.
                agent_row["reward_components"] = (json.dumps(facts["reward_components"], sort_keys=True)
                                                   if "reward_components" in facts else None)
                self._write_agent_row(agent_row)
            return

        # Without a race record, this is the only per-agent source (heuristic runs).
        agent_fields = {
            "reward": metrics.get("agent_rewards"),
            "individual_reward": metrics.get("agent_individual_rewards"),
            "outcome": metrics.get("agent_outcomes"),
            "terminal_reason": metrics.get("agent_terminal_reasons"),
            "finish_position": metrics.get("agent_finish_positions"),
            "lap_count": metrics.get("agent_lap_counts"),
        }
        agent_ids = {
            str(agent_id)
            for values in agent_fields.values()
            if isinstance(values, dict)
            for agent_id in values
        }
        for agent_id in sorted(agent_ids):
            agent_row = {"episode": int(episode), "agent_id": agent_id}
            for field, values in agent_fields.items():
                if isinstance(values, dict):
                    agent_row[field] = values.get(agent_id)
            self._write_agent_row(agent_row)

    def _write_episode_row(self, row_data: Dict[str, Any]):
        self._write_row(self.episode_metrics_file, row_data)

    def _write_agent_row(self, row_data: Dict[str, Any]):
        self._write_row(self.agent_metrics_file, row_data)

    def log_update(self, metrics: Dict[str, Any]):
        if self.enabled:
            self._write_row(self.output_dir / "update_metrics.csv", {
                key: value for key, value in metrics.items()
                if isinstance(value, (str, int, float, bool)) or value is None})

    def log_collector_progress(self, metrics: Dict[str, Any]):
        if self.enabled and self.collector_progress:
            self._write_row(self.output_dir / "collector_progress.csv", metrics)

    def log_jsonl(self, filename: str, row: Dict[str, Any]):
        """Keep one buffered source record; optional CSVs are debugging exports."""
        if not self.enabled:
            return
        path = self.output_dir / filename
        stream = self._jsonl.get(path)
        if stream is None:
            stream = self._jsonl[path] = path.open("w", encoding="utf-8")
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        self._maybe_flush()

    def _write_row(self, path: Path, row: Dict[str, Any]):
        table = self._tables.setdefault(path, {"fields": [], "pending": [], "stream": None})
        table["pending"].append(dict(row))
        self._maybe_flush()

    def _maybe_flush(self):
        self._pending_rows += 1
        if (self._pending_rows >= self.flush_every
                or time.monotonic() - self._last_flush >= self.flush_interval):
            self.flush()

    def flush(self):
        """Flush batches; expand a CSV schema at most once per batch."""
        for path, table in self._tables.items():
            rows = table["pending"]
            if not rows:
                continue
            fields = list(dict.fromkeys([*table["fields"], *(key for row in rows for key in row)]))
            stream = table["stream"]
            if stream is None:
                stream = path.open("w", newline="")
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
            else:
                if fields != table["fields"]:
                    stream.flush()
                    temporary = path.with_suffix(".csv.tmp")
                    with path.open(newline="") as old, temporary.open("w", newline="") as new:
                        expanded = csv.DictWriter(new, fieldnames=fields)
                        expanded.writeheader()
                        expanded.writerows(csv.DictReader(old))
                    stream.close()
                    temporary.replace(path)
                    stream = path.open("a", newline="")
                writer = csv.DictWriter(stream, fieldnames=fields)
            table.update(stream=stream, fields=fields)
            writer.writerows(rows)
            rows.clear()
            stream.flush()
        for stream in self._jsonl.values():
            stream.flush()
        self._pending_rows = 0
        self._last_flush = time.monotonic()

    def save_config_snapshot(self, config: Dict[str, Any]):
        """Save scenario configuration snapshot to JSON.

        Args:
            config: Full scenario configuration dict
        """
        if not self.enabled:
            return

        config_file = self.output_dir / "config_snapshot.json"

        # Add metadata
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'provenance': self.provenance,
            'config': config,
        }

        with open(config_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

    def save_summary(self, summary: Dict[str, Any]):
        """Save final training summary to JSON.

        Args:
            summary: Summary statistics dict
        """
        if not self.enabled:
            return

        summary_file = self.output_dir / "run_summary.json"

        # Add timestamp
        summary['timestamp'] = datetime.now().isoformat()

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def close(self):
        """Close CSV files and flush buffers."""
        self.flush()
        for table in self._tables.values():
            if table["stream"] is not None:
                table["stream"].close()
        for stream in self._jsonl.values():
            stream.close()
        self._tables.clear()
        self._jsonl.clear()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_jsonl"):
            self.close()


__all__ = ['CSVLogger']
