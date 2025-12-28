"""Run metadata tracking and persistence.

Stores run configuration, metrics, and checkpoint information.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class RunMetadata:
    """Metadata for a training run.

    Tracks run configuration, W&B info, checkpoints, and metrics.
    """

    # Run identification
    run_id: str
    scenario_name: str
    algorithm: str
    seed: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # W&B integration
    wandb_run_id: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_url: Optional[str] = None

    # Checkpoint tracking
    checkpoint_dir: str = ""
    latest_checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    checkpoints: list = field(default_factory=list)

    # Training progress
    episodes_completed: int = 0
    total_episodes: int = 0
    training_time_seconds: float = 0.0

    # Performance metrics
    best_metric_value: Optional[float] = None
    best_metric_episode: Optional[int] = None
    latest_metric_value: Optional[float] = None

    # Configuration snapshot
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "initialized"  # initialized, running, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary.

        Returns:
            Dictionary representation of metadata
        """
        return asdict(self)

    def to_json(self) -> str:
        """Convert metadata to JSON string.

        Returns:
            JSON string representation of metadata
        """
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str) -> None:
        """Save metadata to JSON file.

        Args:
            path: Path to save metadata file (typically run_metadata.json)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'RunMetadata':
        """Load metadata from JSON file.

        Args:
            path: Path to metadata file

        Returns:
            RunMetadata instance

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_scenario(
        cls,
        run_id: str,
        scenario: Dict[str, Any],
        checkpoint_dir: str,
        wandb_config: Optional[Dict[str, Any]] = None
    ) -> 'RunMetadata':
        """Create metadata from scenario configuration.

        Args:
            run_id: Unique run identifier
            scenario: Scenario configuration dict
            checkpoint_dir: Path to checkpoint directory
            wandb_config: W&B configuration (optional)

        Returns:
            RunMetadata instance
        """
        # Extract scenario info
        experiment = scenario.get('experiment', {})
        scenario_name = experiment.get('name', 'unknown')
        total_episodes = experiment.get('episodes', 0)
        seed = experiment.get('seed', 0)

        # Find learnable agent algorithm
        agents = scenario.get('agents', {})
        algorithm = 'unknown'
        for agent_id, agent_config in agents.items():
            algo = agent_config.get('algorithm', '')
            if algo.lower() not in ['ftg', 'pp', 'pure_pursuit']:
                algorithm = algo
                break

        # W&B info
        wandb_info = {}
        if wandb_config:
            wandb_info['wandb_project'] = wandb_config.get('project')
            wandb_info['wandb_run_name'] = wandb_config.get('name')

        return cls(
            run_id=run_id,
            scenario_name=scenario_name,
            algorithm=algorithm,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            total_episodes=total_episodes,
            config_snapshot=scenario,
            **wandb_info
        )

    def update_progress(
        self,
        episodes_completed: int,
        training_time_seconds: float,
        latest_metric_value: Optional[float] = None
    ) -> None:
        """Update training progress.

        Args:
            episodes_completed: Number of episodes completed
            training_time_seconds: Total training time in seconds
            latest_metric_value: Latest metric value (optional)
        """
        self.episodes_completed = episodes_completed
        self.training_time_seconds = training_time_seconds
        if latest_metric_value is not None:
            self.latest_metric_value = latest_metric_value

    def update_best(
        self,
        metric_value: float,
        episode: int,
        checkpoint_path: str
    ) -> None:
        """Update best model information.

        Args:
            metric_value: Best metric value
            episode: Episode number where best occurred
            checkpoint_path: Path to best checkpoint file
        """
        self.best_metric_value = metric_value
        self.best_metric_episode = episode
        self.best_checkpoint = checkpoint_path

    def add_checkpoint(
        self,
        checkpoint_path: str,
        episode: int,
        metric_value: Optional[float] = None,
        checkpoint_type: str = "periodic"
    ) -> None:
        """Add checkpoint to tracking list.

        Args:
            checkpoint_path: Path to checkpoint file
            episode: Episode number
            metric_value: Metric value at checkpoint (optional)
            checkpoint_type: Type of checkpoint (periodic, best, final)
        """
        checkpoint_info = {
            'path': checkpoint_path,
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'type': checkpoint_type,
        }
        if metric_value is not None:
            checkpoint_info['metric_value'] = metric_value

        self.checkpoints.append(checkpoint_info)
        self.latest_checkpoint = checkpoint_path

    def update_wandb_info(
        self,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        url: Optional[str] = None
    ) -> None:
        """Update W&B run information.

        Args:
            run_id: W&B run ID
            run_name: W&B run name
            url: W&B run URL
        """
        if run_id:
            self.wandb_run_id = run_id
        if run_name:
            self.wandb_run_name = run_name
        if url:
            self.wandb_url = url

    def mark_completed(self) -> None:
        """Mark run as completed."""
        self.status = "completed"

    def mark_failed(self) -> None:
        """Mark run as failed."""
        self.status = "failed"

    def mark_running(self) -> None:
        """Mark run as running."""
        self.status = "running"

    def get_progress_pct(self) -> float:
        """Get training progress percentage.

        Returns:
            Progress percentage (0.0-100.0)
        """
        if self.total_episodes == 0:
            return 0.0
        return (self.episodes_completed / self.total_episodes) * 100.0

    def __str__(self) -> str:
        """String representation of metadata."""
        lines = [
            f"Run: {self.run_id}",
            f"Scenario: {self.scenario_name}",
            f"Algorithm: {self.algorithm}",
            f"Seed: {self.seed}",
            f"Status: {self.status}",
            f"Progress: {self.episodes_completed}/{self.total_episodes} ({self.get_progress_pct():.1f}%)",
        ]

        if self.best_metric_value is not None:
            lines.append(f"Best Metric: {self.best_metric_value:.4f} @ episode {self.best_metric_episode}")

        if self.wandb_url:
            lines.append(f"W&B: {self.wandb_url}")

        return "\n".join(lines)
