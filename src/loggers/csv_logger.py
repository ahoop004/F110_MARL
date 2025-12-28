"""CSV and JSON export logging for training runs.

Provides file-based logging for post-training analysis and reproducibility.
Compatible with v1 PlotArtifactLogger format.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
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
        ...     output_dir="outputs/gaplock_sac/run_001",
        ...     scenario_config=scenario,
        ... )
        >>> logger.log_episode(episode=0, metrics=episode_metrics, agent_metrics=agent_metrics)
        >>> logger.save_summary(summary_stats)
    """

    def __init__(
        self,
        output_dir: str,
        scenario_config: Optional[Dict[str, Any]] = None,
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

        if not self.enabled:
            return

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV files
        self.episode_metrics_file = self.output_dir / "episode_metrics.csv"
        self.agent_metrics_file = self.output_dir / "agent_metrics.csv"

        # CSV writers (initialized on first write)
        self._episode_writer = None
        self._agent_writer = None
        self._episode_csv = None
        self._agent_csv = None

        # Episode metrics fieldnames (determined from first episode)
        self._episode_fieldnames = None
        self._agent_fieldnames = None

        # Save config snapshot
        if scenario_config:
            self.save_config_snapshot(scenario_config)

    def log_episode(
        self,
        episode: int,
        metrics: Any,  # EpisodeMetrics
        agent_metrics: Optional[Dict[str, Any]] = None,
        rolling_stats: Optional[Dict[str, float]] = None,
    ):
        """Log episode metrics to CSV.

        Args:
            episode: Episode number
            metrics: EpisodeMetrics instance (primary agent)
            agent_metrics: Dict mapping agent_id -> per-agent metrics (optional)
            rolling_stats: Rolling statistics dict (optional)
        """
        if not self.enabled:
            return

        # Convert metrics to dict
        episode_data = metrics.to_dict()
        episode_data['episode'] = episode

        # Add rolling stats if provided
        if rolling_stats:
            for key, value in rolling_stats.items():
                if key not in ['outcome_counts', 'outcome_rates', 'total_episodes']:
                    episode_data[f'rolling_{key}'] = value

        # Write episode metrics
        self._write_episode_row(episode_data)

        # Write per-agent metrics if provided
        if agent_metrics:
            for agent_id, agent_data in agent_metrics.items():
                agent_row = {
                    'episode': episode,
                    'agent_id': agent_id,
                    **agent_data,
                }
                self._write_agent_row(agent_row)

    def _write_episode_row(self, row_data: Dict[str, Any]):
        """Write row to episode metrics CSV.

        Args:
            row_data: Row data dict
        """
        # Initialize CSV writer on first write
        if self._episode_writer is None:
            # Determine fieldnames from first row
            self._episode_fieldnames = list(row_data.keys())

            # Open CSV file
            self._episode_csv = open(self.episode_metrics_file, 'w', newline='')
            self._episode_writer = csv.DictWriter(
                self._episode_csv,
                fieldnames=self._episode_fieldnames,
                extrasaction='ignore'
            )
            self._episode_writer.writeheader()

        # Write row
        self._episode_writer.writerow(row_data)
        self._episode_csv.flush()

    def _write_agent_row(self, row_data: Dict[str, Any]):
        """Write row to agent metrics CSV.

        Args:
            row_data: Row data dict
        """
        # Initialize CSV writer on first write
        if self._agent_writer is None:
            # Determine fieldnames from first row
            self._agent_fieldnames = list(row_data.keys())

            # Open CSV file
            self._agent_csv = open(self.agent_metrics_file, 'w', newline='')
            self._agent_writer = csv.DictWriter(
                self._agent_csv,
                fieldnames=self._agent_fieldnames,
                extrasaction='ignore'
            )
            self._agent_writer.writeheader()

        # Write row
        self._agent_writer.writerow(row_data)
        self._agent_csv.flush()

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
        if self._episode_csv:
            self._episode_csv.close()
        if self._agent_csv:
            self._agent_csv.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


__all__ = ['CSVLogger']
