"""Checkpoint management for training runs.

Handles saving, loading, and cleanup of model checkpoints.
"""

import os
import glob
import shutil
import torch
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .run_metadata import RunMetadata


class CheckpointManager:
    """Manages model checkpoints for a training run.

    Features:
    - Save/load model state, optimizer state, and training state
    - Automatic checkpoint cleanup (keep best + latest + periodic)
    - Checkpoint naming with episode numbers
    - Integration with RunMetadata tracking
    """

    def __init__(
        self,
        checkpoint_dir: str,
        run_metadata: RunMetadata,
        keep_best_n: int = 3,
        keep_latest_n: int = 2,
        keep_every_n_episodes: Optional[int] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            run_metadata: RunMetadata instance for tracking
            keep_best_n: Number of best checkpoints to keep (default: 3)
            keep_latest_n: Number of latest checkpoints to keep (default: 2)
            keep_every_n_episodes: Save checkpoint every N episodes (None = disabled)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.run_metadata = run_metadata
        self.keep_best_n = keep_best_n
        self.keep_latest_n = keep_latest_n
        self.keep_every_n_episodes = keep_every_n_episodes

        # Metadata file path
        self.metadata_path = self.checkpoint_dir / "run_metadata.json"

        # Update metadata with checkpoint dir
        self.run_metadata.checkpoint_dir = str(self.checkpoint_dir)

    def save_checkpoint(
        self,
        episode: int,
        agent_states: Dict[str, Any],
        optimizer_states: Optional[Dict[str, Any]] = None,
        training_state: Optional[Dict[str, Any]] = None,
        checkpoint_type: str = "periodic",
        metric_value: Optional[float] = None,
    ) -> str:
        """Save checkpoint to disk.

        Args:
            episode: Current episode number
            agent_states: Dict mapping agent_id -> agent state dict
            optimizer_states: Dict mapping agent_id -> optimizer state dict (optional)
            training_state: Additional training state (e.g., curriculum stage) (optional)
            checkpoint_type: Type of checkpoint (periodic, best, final)
            metric_value: Metric value at this checkpoint (optional)

        Returns:
            Path to saved checkpoint file

        Checkpoint format:
            checkpoint_ep{episode:06d}.pt or checkpoint_best_ep{episode:06d}.pt
        """
        # Generate checkpoint filename
        if checkpoint_type == "best":
            filename = f"checkpoint_best_ep{episode:06d}.pt"
        elif checkpoint_type == "final":
            filename = f"checkpoint_final_ep{episode:06d}.pt"
        else:
            filename = f"checkpoint_ep{episode:06d}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # Build checkpoint dict
        checkpoint = {
            'episode': episode,
            'agent_states': agent_states,
            'checkpoint_type': checkpoint_type,
        }

        if optimizer_states is not None:
            checkpoint['optimizer_states'] = optimizer_states

        if training_state is not None:
            checkpoint['training_state'] = training_state

        if metric_value is not None:
            checkpoint['metric_value'] = metric_value

        # Save to disk
        torch.save(checkpoint, checkpoint_path)

        # Update metadata
        self.run_metadata.add_checkpoint(
            checkpoint_path=str(checkpoint_path),
            episode=episode,
            metric_value=metric_value,
            checkpoint_type=checkpoint_type
        )

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """Load checkpoint from disk.

        Args:
            checkpoint_path: Path to specific checkpoint (optional)
            load_best: Load best checkpoint instead of latest (default: False)

        Returns:
            Checkpoint dictionary containing:
                - episode: int
                - agent_states: Dict[str, Any]
                - optimizer_states: Dict[str, Any] (if present)
                - training_state: Dict[str, Any] (if present)
                - metric_value: float (if present)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        # Determine which checkpoint to load
        if checkpoint_path is None:
            if load_best and self.run_metadata.best_checkpoint:
                checkpoint_path = self.run_metadata.best_checkpoint
            elif self.run_metadata.latest_checkpoint:
                checkpoint_path = self.run_metadata.latest_checkpoint
            else:
                # Find latest checkpoint in directory
                checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load from disk
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        return checkpoint

    def cleanup_checkpoints(
        self,
        best_checkpoints: Optional[List[Tuple[str, float]]] = None
    ) -> None:
        """Clean up old checkpoints based on retention policy.

        Keeps:
        - Best N checkpoints (by metric value)
        - Latest N checkpoints (by episode)
        - Periodic checkpoints (every N episodes)

        Args:
            best_checkpoints: List of (checkpoint_path, metric_value) tuples for best models
        """
        # Get all checkpoint files
        all_checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))

        if not all_checkpoints:
            return

        # Parse checkpoint files
        checkpoint_info = []
        for ckpt_path in all_checkpoints:
            # Extract episode number from filename
            filename = ckpt_path.name
            episode = self._extract_episode_from_filename(filename)

            if episode is None:
                continue

            is_best = "best" in filename
            is_final = "final" in filename

            checkpoint_info.append({
                'path': ckpt_path,
                'episode': episode,
                'is_best': is_best,
                'is_final': is_final,
            })

        # Sort by episode
        checkpoint_info.sort(key=lambda x: x['episode'])

        # Determine which checkpoints to keep
        keep_paths = set()

        # 1. Keep best checkpoints
        if best_checkpoints:
            # Sort by metric value (descending)
            sorted_best = sorted(best_checkpoints, key=lambda x: x[1], reverse=True)
            for ckpt_path, _ in sorted_best[:self.keep_best_n]:
                keep_paths.add(Path(ckpt_path))

        # Also keep checkpoints marked as "best" in filename
        for info in checkpoint_info:
            if info['is_best']:
                keep_paths.add(info['path'])

        # 2. Keep latest N checkpoints
        for info in checkpoint_info[-self.keep_latest_n:]:
            keep_paths.add(info['path'])

        # 3. Keep final checkpoint
        for info in checkpoint_info:
            if info['is_final']:
                keep_paths.add(info['path'])

        # 4. Keep periodic checkpoints
        if self.keep_every_n_episodes is not None:
            for info in checkpoint_info:
                if info['episode'] % self.keep_every_n_episodes == 0:
                    keep_paths.add(info['path'])

        # Delete checkpoints not in keep list
        for info in checkpoint_info:
            if info['path'] not in keep_paths:
                try:
                    info['path'].unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete checkpoint {info['path']}: {e}")

    def save_metadata(self) -> None:
        """Save run metadata to disk."""
        self.run_metadata.save(str(self.metadata_path))

    def load_metadata(self) -> RunMetadata:
        """Load run metadata from disk.

        Returns:
            RunMetadata instance

        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        return RunMetadata.load(str(self.metadata_path))

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints in directory.

        Returns:
            List of checkpoint info dicts with keys:
                - path: str
                - episode: int
                - is_best: bool
                - is_final: bool
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))

        checkpoints = []
        for ckpt_path in checkpoint_files:
            filename = ckpt_path.name
            episode = self._extract_episode_from_filename(filename)

            if episode is None:
                continue

            checkpoints.append({
                'path': str(ckpt_path),
                'episode': episode,
                'is_best': "best" in filename,
                'is_final': "final" in filename,
            })

        # Sort by episode
        checkpoints.sort(key=lambda x: x['episode'])

        return checkpoints

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in directory.

        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Return latest checkpoint
        return checkpoints[-1]['path']

    def _extract_episode_from_filename(self, filename: str) -> Optional[int]:
        """Extract episode number from checkpoint filename.

        Args:
            filename: Checkpoint filename (e.g., checkpoint_ep000042.pt)

        Returns:
            Episode number, or None if parsing fails
        """
        import re
        match = re.search(r'ep(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """Get information needed to resume training.

        Returns:
            Dict with resume information:
                - checkpoint_path: str
                - episode: int
                - metadata: RunMetadata
            Returns None if no checkpoints available
        """
        # Check if metadata exists
        if not self.metadata_path.exists():
            return None

        # Load metadata
        try:
            metadata = self.load_metadata()
        except Exception:
            return None

        # Get latest checkpoint
        checkpoint_path = metadata.latest_checkpoint
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            checkpoint_path = self._find_latest_checkpoint()

        if not checkpoint_path:
            return None

        # Load checkpoint to get episode number
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            episode = checkpoint['episode']
        except Exception:
            return None

        return {
            'checkpoint_path': checkpoint_path,
            'episode': episode,
            'metadata': metadata,
        }
