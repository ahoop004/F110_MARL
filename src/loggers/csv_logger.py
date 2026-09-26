"""CSV and JSON export logging for training runs.

Provides file-based logging for post-training analysis and reproducibility.
Compatible with v1 PlotArtifactLogger format.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CSVLogger:
    """Logger for exporting metrics to CSV and JSON files.

    Creates files in a structured output directory:
        outputs/{scenario}/{run_id}/
            - episode_metrics.csv      # Per-episode aggregate metrics
            - agent_metrics.csv        # Per-agent per-episode metrics
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
        self._race_started = False

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
        for key, value in metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row.setdefault(key.replace("/", "_"), value)
        race = metrics.get("race_record")
        if race is not None:
            row.update({key: value for key, value in race.items()
                        if not isinstance(value, (dict, list))})
            row["team_reward_components"] = json.dumps(race.get("team_reward_components", {}), sort_keys=True)
            with (self.output_dir / "race_metrics.jsonl").open("a" if self._race_started else "w") as stream:
                stream.write(json.dumps(race, sort_keys=True) + "\n")
            self._race_started = True
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
        if self.enabled:
            self._write_row(self.output_dir / "collector_progress.csv", metrics)

    def _write_row(self, path: Path, row: Dict[str, Any]):
        """Retain late fields; expand existing headers atomically with bounded memory."""
        table = self._tables.get(path)
        if table is None:
            fields = list(row)
            stream = path.open("w", newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
        else:
            stream, writer, fields = table
            extra = [key for key in row if key not in fields]
            if extra:
                stream.flush()
                fields = [*fields, *extra]
                temporary = path.with_suffix(".csv.tmp")
                with path.open(newline="") as old, temporary.open("w", newline="") as new:
                    expanded = csv.DictWriter(new, fieldnames=fields)
                    expanded.writeheader()
                    expanded.writerows(csv.DictReader(old))
                stream.close()
                temporary.replace(path)
                stream = path.open("a", newline="")
                writer = csv.DictWriter(stream, fieldnames=fields)
        self._tables[path] = stream, writer, fields
        writer.writerow(row)
        stream.flush()

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
        for stream, _, _ in self._tables.values():
            stream.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


__all__ = ['CSVLogger']
