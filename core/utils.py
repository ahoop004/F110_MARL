"""Utility functions for training: checkpointing, logging, etc."""
from typing import Dict, Any, Optional
from pathlib import Path
import json
import torch
import numpy as np
from datetime import datetime


def save_checkpoint(
    agents: Dict[str, Any],
    episode: int,
    checkpoint_dir: str,
    metrics: Optional[Dict[str, float]] = None,
    prefix: str = "checkpoint"
) -> str:
    """Save agent checkpoints and training state.

    Args:
        agents: Dictionary of agent_id -> Agent
        episode: Current episode number
        checkpoint_dir: Directory to save checkpoints
        metrics: Optional metrics to save with checkpoint
        prefix: Checkpoint file prefix

    Returns:
        checkpoint_path: Path to saved checkpoint directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint subdirectory with episode number
    checkpoint_name = f"{prefix}_episode_{episode}"
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_path.mkdir(exist_ok=True)

    # Save each agent's checkpoint
    for agent_id, agent in agents.items():
        agent_checkpoint_path = checkpoint_path / f"{agent_id}.pt"
        agent.save(str(agent_checkpoint_path))

    # Save metadata
    metadata = {
        'episode': episode,
        'timestamp': datetime.now().isoformat(),
        'agent_ids': list(agents.keys()),
        'metrics': metrics or {},
    }

    metadata_path = checkpoint_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(checkpoint_path)


def load_checkpoint(
    agents: Dict[str, Any],
    checkpoint_path: str
) -> Dict[str, Any]:
    """Load agent checkpoints and training state.

    Args:
        agents: Dictionary of agent_id -> Agent (must be pre-initialized)
        checkpoint_path: Path to checkpoint directory

    Returns:
        metadata: Dictionary containing episode number and metrics
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load metadata
    metadata_path = checkpoint_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load each agent's checkpoint
    for agent_id in metadata['agent_ids']:
        if agent_id in agents:
            agent_checkpoint_path = checkpoint_path / f"{agent_id}.pt"
            agents[agent_id].load(str(agent_checkpoint_path))
        else:
            print(f"Warning: Agent '{agent_id}' found in checkpoint but not in current agents dict")

    return metadata


class SimpleLogger:
    """Simple console and CSV logger for training metrics."""

    def __init__(self, log_dir: Optional[str] = None, verbose: bool = True):
        """Initialize logger.

        Args:
            log_dir: Directory to save CSV logs (None = no file logging)
            verbose: If True, print to console
        """
        self.verbose = verbose
        self.log_dir = Path(log_dir) if log_dir else None
        self.metrics_history = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.log_dir / "training_metrics.csv"
            self._csv_initialized = False

    def log(self, episode: int, metrics: Dict[str, float]):
        """Log metrics for an episode.

        Args:
            episode: Episode number
            metrics: Dictionary of metric_name -> value
        """
        # Add episode number
        log_entry = {'episode': episode, **metrics}
        self.metrics_history.append(log_entry)

        # Console logging
        if self.verbose:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"Episode {episode}: {metrics_str}")

        # CSV logging
        if self.log_dir:
            self._log_to_csv(log_entry)

    def _log_to_csv(self, log_entry: Dict[str, Any]):
        """Write log entry to CSV file."""
        import csv

        # Initialize CSV with headers on first write
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        # Append row
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)

    def get_history(self) -> list:
        """Get complete metrics history."""
        return self.metrics_history

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics over all episodes."""
        if not self.metrics_history:
            return {}

        # Extract all metric names (excluding 'episode')
        metric_names = set()
        for entry in self.metrics_history:
            metric_names.update(k for k in entry.keys() if k != 'episode')

        summary = {}
        for metric in metric_names:
            values = [entry[metric] for entry in self.metrics_history if metric in entry]
            if values:
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)
                summary[f"{metric}_min"] = np.min(values)
                summary[f"{metric}_max"] = np.max(values)

        return summary


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_episode_metrics(episode_rewards: Dict[str, float], episode_lengths: Dict[str, int]) -> Dict[str, float]:
    """Compute summary metrics from episode data.

    Args:
        episode_rewards: Dictionary of agent_id -> total episode reward
        episode_lengths: Dictionary of agent_id -> episode length

    Returns:
        metrics: Dictionary of aggregated metrics
    """
    metrics = {}

    # Per-agent metrics
    for agent_id, reward in episode_rewards.items():
        metrics[f"{agent_id}_reward"] = reward
        metrics[f"{agent_id}_length"] = episode_lengths.get(agent_id, 0)

    # Aggregate metrics (mean across agents)
    if episode_rewards:
        metrics['mean_reward'] = np.mean(list(episode_rewards.values()))
        metrics['total_reward'] = np.sum(list(episode_rewards.values()))

    if episode_lengths:
        metrics['mean_length'] = np.mean(list(episode_lengths.values()))

    return metrics
