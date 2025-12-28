# V2 Checkpoint & Run Management Plan

## Overview
Implement a unified run identification, checkpointing, and logging system for v2 that:
- Aligns with legacy patterns for W&B compatibility
- Simplifies and modernizes the implementation
- Integrates cleanly with existing v2 architecture
- Provides consistent run tracking across all outputs

---

## 1. Run Identification System

### 1.1 Run Suffix Resolution (Priority Order)

Create `core/run_id.py` with the same priority-based resolution as legacy:

```python
def resolve_run_id() -> str:
    """Resolve unique run identifier from environment and config.

    Priority order (first non-empty wins):
    1. F110_RUN_SUFFIX env var
    2. WANDB_RUN_NAME env var
    3. WANDB_RUN_ID env var
    4. RUN_CONFIG_HASH env var
    5. RUN_SEED env var
    6. Auto-generate: {timestamp}_{random_suffix}

    Returns:
        Slugified run identifier (alphanumeric + hyphens/underscores only)
    """
    candidates = [
        os.environ.get("F110_RUN_SUFFIX"),
        os.environ.get("WANDB_RUN_NAME"),
        os.environ.get("WANDB_RUN_ID"),
        os.environ.get("RUN_CONFIG_HASH"),
        os.environ.get("RUN_SEED"),
    ]

    for candidate in candidates:
        if candidate and (slug := slugify(candidate)):
            return slug

    # Auto-generate if none found
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(4)
    return f"{timestamp}-{random_suffix}"


def slugify(value: str) -> str:
    """Convert string to URL-safe slug."""
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "-"
        for ch in str(value)
    )
    return cleaned.strip("-") or ""
```

**Key differences from legacy:**
- Always returns a valid ID (auto-generates if needed)
- Simplified candidate list (removed redundant sources)
- Clearer documentation

### 1.2 Run Metadata Tracking

Create `core/run_metadata.py`:

```python
@dataclass
class RunMetadata:
    """Metadata for a training run."""

    run_id: str
    scenario_name: str
    algorithm: str
    start_time: datetime
    config_hash: str
    seed: int

    # W&B info (if enabled)
    wandb_run_id: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Paths
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "algorithm": self.algorithm,
            "start_time": self.start_time.isoformat(),
            "config_hash": self.config_hash,
            "seed": self.seed,
            "wandb_run_id": self.wandb_run_id,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "checkpoint_dir": str(self.checkpoint_dir),
            "log_dir": str(self.log_dir),
        }

    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
```

---

## 2. Checkpoint Management System

### 2.1 Directory Structure

```
outputs/
├── checkpoints/
│   └── {scenario_slug}/
│       └── {run_id}/
│           ├── best_model.pt           # Best model by success rate
│           ├── latest_model.pt         # Most recent checkpoint
│           ├── checkpoint_ep{N}.pt     # Periodic checkpoints
│           └── metadata.json           # Run metadata
│
└── logs/
    └── {scenario_slug}/
        └── {run_id}/
            ├── training_metrics.csv    # Episode metrics
            ├── eval_metrics.csv        # Evaluation results
            ├── config.yaml             # Full scenario config
            └── console.log             # Console output
```

### 2.2 Checkpoint Manager Implementation

Create `core/checkpoint_manager.py`:

```python
class CheckpointManager:
    """Manages model checkpointing during training."""

    def __init__(
        self,
        checkpoint_dir: Path,
        scenario_name: str,
        run_id: str,
        save_interval: int = 100,  # Save every N episodes
        keep_last_n: int = 3,      # Keep N most recent checkpoints
    ):
        self.checkpoint_dir = checkpoint_dir / slugify(scenario_name) / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self.keep_last_n = keep_last_n

        # Track best model
        self.best_metric = -float('inf')
        self.best_episode = 0

        # Track periodic checkpoints
        self.periodic_checkpoints: list[Path] = []

    def save_checkpoint(
        self,
        agent: Any,
        episode: int,
        metrics: dict,
        is_best: bool = False,
    ) -> Optional[Path]:
        """Save a checkpoint.

        Args:
            agent: Agent with save() method
            episode: Current episode number
            metrics: Training metrics dict
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        saved_path = None

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            agent.save(str(best_path))
            self.best_episode = episode
            self.best_metric = metrics.get('success_rate', 0.0)
            saved_path = best_path

            # Save best metadata
            metadata_path = self.checkpoint_dir / "best_metadata.json"
            self._save_checkpoint_metadata(
                metadata_path, episode, metrics, is_best=True
            )

        # Save latest model (always)
        latest_path = self.checkpoint_dir / "latest_model.pt"
        agent.save(str(latest_path))

        # Save periodic checkpoint
        if episode % self.save_interval == 0:
            periodic_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"
            agent.save(str(periodic_path))
            self.periodic_checkpoints.append(periodic_path)
            saved_path = periodic_path

            # Cleanup old periodic checkpoints
            self._cleanup_old_checkpoints()

        return saved_path

    def load_best(self, agent: Any) -> bool:
        """Load best model checkpoint.

        Args:
            agent: Agent with load() method

        Returns:
            True if loaded successfully, False otherwise
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            return False

        agent.load(str(best_path))
        return True

    def load_latest(self, agent: Any) -> bool:
        """Load latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest_model.pt"
        if not latest_path.exists():
            return False

        agent.load(str(latest_path))
        return True

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old periodic checkpoints, keeping only last N."""
        if len(self.periodic_checkpoints) > self.keep_last_n:
            # Sort by episode number
            sorted_checkpoints = sorted(
                self.periodic_checkpoints,
                key=lambda p: int(p.stem.split('ep')[1])
            )

            # Remove oldest
            to_remove = sorted_checkpoints[:-self.keep_last_n]
            for path in to_remove:
                if path.exists():
                    path.unlink()
                self.periodic_checkpoints.remove(path)

    def _save_checkpoint_metadata(
        self,
        path: Path,
        episode: int,
        metrics: dict,
        is_best: bool = False,
    ) -> None:
        """Save checkpoint metadata to JSON."""
        metadata = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "is_best": is_best,
            "metrics": metrics,
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
```

### 2.3 Best Model Tracking

Create `core/best_model_tracker.py`:

```python
class BestModelTracker:
    """Tracks best model based on success rate."""

    def __init__(
        self,
        metric_key: str = 'success_rate',
        window_size: int = 100,
        higher_is_better: bool = True,
    ):
        self.metric_key = metric_key
        self.window_size = window_size
        self.higher_is_better = higher_is_better

        self.best_value = -float('inf') if higher_is_better else float('inf')
        self.best_episode = 0

        # Rolling window for smoothing
        self.recent_values: deque = deque(maxlen=window_size)

    def update(self, episode: int, metrics: dict) -> bool:
        """Update tracker with new metrics.

        Args:
            episode: Current episode number
            metrics: Metrics dict

        Returns:
            True if this is a new best, False otherwise
        """
        value = metrics.get(self.metric_key)
        if value is None:
            return False

        # Add to rolling window
        self.recent_values.append(value)

        # Compute smoothed value
        smoothed = sum(self.recent_values) / len(self.recent_values)

        # Check if new best
        is_better = (
            smoothed > self.best_value if self.higher_is_better
            else smoothed < self.best_value
        )

        if is_better:
            self.best_value = smoothed
            self.best_episode = episode
            return True

        return False
```

---

## 3. W&B Integration

### 3.1 W&B Logger Enhancement

Update `loggers/wandb_logger.py` to capture run ID:

```python
class WandbLogger:
    """W&B logging with run ID capture."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        resume: Optional[str] = None,
    ):
        import wandb

        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            group=group,
            tags=tags,
            config=config,
            resume=resume,
        )

        # Capture run ID in environment
        if self.run is not None:
            os.environ["WANDB_RUN_ID"] = str(self.run.id)
            if self.run.name:
                os.environ["WANDB_RUN_NAME"] = str(self.run.name)

        # Define step metrics
        self.run.define_metric("episode")
        self.run.define_metric("train/*", step_metric="episode")
        self.run.define_metric("eval/*", step_metric="episode")
        self.run.define_metric("curriculum/*", step_metric="episode")

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_artifact(self, path: Path, name: str, type: str = "model") -> None:
        """Log artifact (model checkpoint, config, etc.)."""
        if self.run is not None:
            artifact = wandb.Artifact(name, type=type)
            artifact.add_file(str(path))
            self.run.log_artifact(artifact)
```

### 3.2 Metric Filtering

Add selective metric logging (similar to legacy):

```python
def filter_metrics_for_wandb(metrics: dict, phase: str = "train") -> dict:
    """Filter metrics for W&B logging to avoid clutter.

    Args:
        metrics: Full metrics dict
        phase: 'train' or 'eval'

    Returns:
        Filtered metrics dict
    """
    # Always include these
    base_keys = {
        "episode",
        "episode_reward",
        "success_rate",
        "avg_reward",
        "best_reward",
    }

    # Include curriculum metrics
    curriculum_keys = {
        k for k in metrics.keys()
        if k.startswith("curriculum/")
    }

    # Include outcome metrics
    outcome_keys = {
        k for k in metrics.keys()
        if any(outcome in k for outcome in [
            "target_crash", "self_crash", "collision",
            "timeout", "idle_stop", "target_finish"
        ])
    }

    return {
        k: v for k, v in metrics.items()
        if k in base_keys or k in curriculum_keys or k in outcome_keys
    }
```

---

## 4. Training Loop Integration

### 4.1 Enhanced Training Loop

Update `core/enhanced_training.py`:

```python
class EnhancedTrainingLoop:
    """Training loop with integrated checkpointing and logging."""

    def __init__(
        self,
        env,
        agents: dict,
        agent_rewards: dict,
        max_episodes: int,
        scenario_name: str,
        wandb_logger: Optional[WandbLogger] = None,
        console_logger: Optional[ConsoleLogger] = None,
        spawn_curriculum: Optional[SpawnCurriculumManager] = None,
        checkpoint_config: Optional[dict] = None,
    ):
        self.env = env
        self.agents = agents
        self.agent_rewards = agent_rewards
        self.max_episodes = max_episodes
        self.scenario_name = scenario_name

        # Loggers
        self.wandb_logger = wandb_logger
        self.console_logger = console_logger
        self.spawn_curriculum = spawn_curriculum

        # Run identification
        self.run_id = resolve_run_id()

        # Checkpoint management
        ckpt_cfg = checkpoint_config or {}
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(ckpt_cfg.get("dir", "outputs/checkpoints")),
            scenario_name=scenario_name,
            run_id=self.run_id,
            save_interval=ckpt_cfg.get("save_interval", 100),
            keep_last_n=ckpt_cfg.get("keep_last_n", 3),
        )

        # Best model tracking
        self.best_tracker = BestModelTracker(
            metric_key='success_rate',
            window_size=100,
        )

        # Run metadata
        self.run_metadata = RunMetadata(
            run_id=self.run_id,
            scenario_name=scenario_name,
            algorithm=list(agents.values())[0].__class__.__name__,
            start_time=datetime.now(),
            config_hash=self._compute_config_hash(),
            seed=env.seed if hasattr(env, 'seed') else 0,
            wandb_run_id=os.environ.get("WANDB_RUN_ID"),
            wandb_project=wandb_logger.run.project if wandb_logger else None,
            checkpoint_dir=self.checkpoint_manager.checkpoint_dir,
        )

        # Save run metadata
        self.run_metadata.save(
            self.checkpoint_manager.checkpoint_dir / "run_metadata.json"
        )

    def train(self) -> dict:
        """Run training loop with checkpointing."""
        for episode in range(1, self.max_episodes + 1):
            # Run episode
            episode_metrics = self._run_episode(episode)

            # Check for new best model
            is_best = self.best_tracker.update(episode, episode_metrics)

            # Save checkpoint
            primary_agent = list(self.agents.values())[0]
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                agent=primary_agent,
                episode=episode,
                metrics=episode_metrics,
                is_best=is_best,
            )

            # Log checkpoint event
            if checkpoint_path:
                self.console_logger.print_info(
                    f"Saved checkpoint: {checkpoint_path.name}"
                )

            # Log to W&B
            if self.wandb_logger:
                filtered_metrics = filter_metrics_for_wandb(episode_metrics)
                self.wandb_logger.log(filtered_metrics, step=episode)

                # Log best model as artifact
                if is_best:
                    best_path = self.checkpoint_manager.checkpoint_dir / "best_model.pt"
                    self.wandb_logger.log_artifact(
                        best_path,
                        name=f"{self.scenario_name}_best",
                        type="model",
                    )

        return self._get_training_stats()

    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for reproducibility."""
        import hashlib
        config_str = json.dumps(self.env.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]
```

---

## 5. CLI Integration

### 5.1 Update run_v2.py

Add checkpoint-related arguments:

```python
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default="outputs/checkpoints",
    help="Directory for saving checkpoints"
)
parser.add_argument(
    "--checkpoint-interval",
    type=int,
    default=100,
    help="Save checkpoint every N episodes"
)
parser.add_argument(
    "--resume-from",
    type=str,
    help="Resume training from checkpoint path"
)
parser.add_argument(
    "--load-best",
    action="store_true",
    help="Load best model before evaluation"
)
```

### 5.2 Resume Training Support

```python
def resume_training(
    checkpoint_path: str,
    agent: Any,
    training_loop: EnhancedTrainingLoop,
) -> int:
    """Resume training from checkpoint.

    Returns:
        Starting episode number
    """
    # Load checkpoint
    agent.load(checkpoint_path)

    # Load metadata to get episode number
    metadata_path = Path(checkpoint_path).parent / "best_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("episode", 0)

    return 0
```

---

## 6. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `core/run_id.py` with run ID resolution
- [ ] Create `core/run_metadata.py` with metadata tracking
- [ ] Create `core/checkpoint_manager.py` with checkpoint management
- [ ] Create `core/best_model_tracker.py` with best model tracking

### Phase 2: Logging Enhancement
- [ ] Update `loggers/wandb_logger.py` with run ID capture
- [ ] Add metric filtering for W&B
- [ ] Create `loggers/csv_logger.py` for local CSV logging

### Phase 3: Training Loop Integration
- [ ] Update `core/enhanced_training.py` with checkpoint integration
- [ ] Add resume training support
- [ ] Add best model evaluation

### Phase 4: CLI Enhancement
- [ ] Update `run_v2.py` with checkpoint arguments
- [ ] Add resume training command
- [ ] Add checkpoint visualization command

### Phase 5: Testing
- [ ] Test checkpoint saving/loading
- [ ] Test W&B artifact logging
- [ ] Test resume training
- [ ] Test best model tracking

---

## 7. Usage Examples

### Training with Checkpoints

```bash
# Start new training run
python run_v2.py \
    --scenario scenarios/v2/gaplock_sac.yaml \
    --wandb \
    --checkpoint-interval 50

# Resume from checkpoint
python run_v2.py \
    --scenario scenarios/v2/gaplock_sac.yaml \
    --resume-from outputs/checkpoints/gaplock-sac/run123/best_model.pt \
    --wandb

# Evaluate best model
python run_v2.py \
    --scenario scenarios/v2/gaplock_sac.yaml \
    --load-best \
    --episodes 100 \
    --render
```

### Programmatic Usage

```python
from core.checkpoint_manager import CheckpointManager
from core.best_model_tracker import BestModelTracker

# Create checkpoint manager
ckpt_mgr = CheckpointManager(
    checkpoint_dir=Path("outputs/checkpoints"),
    scenario_name="gaplock_sac",
    run_id="run123",
)

# Create best tracker
tracker = BestModelTracker(metric_key='success_rate')

# During training
for episode in range(1000):
    metrics = train_episode()

    # Check if best
    is_best = tracker.update(episode, metrics)

    # Save checkpoint
    ckpt_mgr.save_checkpoint(
        agent=agent,
        episode=episode,
        metrics=metrics,
        is_best=is_best,
    )

# Load best model
ckpt_mgr.load_best(agent)
```

---

## 8. Migration from Legacy

### Backward Compatibility

To maintain compatibility with legacy runs:

1. **Environment Variables**: Support same env vars (F110_RUN_SUFFIX, WANDB_RUN_ID, etc.)
2. **W&B Project Names**: Keep same project naming
3. **Checkpoint Naming**: Use similar `{algo}_best_{run_id}.pt` pattern
4. **Metric Keys**: Keep consistent metric naming for comparison

### Migration Script

Create `scripts/migrate_legacy_checkpoints.py`:

```python
#!/usr/bin/env python3
"""Migrate legacy checkpoints to v2 format."""

def migrate_checkpoint(legacy_path: Path, v2_checkpoint_dir: Path):
    """Migrate a single checkpoint."""
    # Extract run ID from filename
    run_id = extract_run_id_from_legacy(legacy_path)

    # Create v2 directory structure
    scenario_name = extract_scenario_from_legacy(legacy_path)
    v2_path = v2_checkpoint_dir / slugify(scenario_name) / run_id
    v2_path.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint
    shutil.copy(legacy_path, v2_path / "best_model.pt")

    # Generate metadata
    metadata = RunMetadata(
        run_id=run_id,
        scenario_name=scenario_name,
        algorithm=extract_algo_from_legacy(legacy_path),
        # ... other fields
    )
    metadata.save(v2_path / "run_metadata.json")
```

---

## 9. Monitoring & Debugging

### Checkpoint Health Check

Create `scripts/check_checkpoints.py`:

```python
#!/usr/bin/env python3
"""Check checkpoint health and integrity."""

def check_checkpoint_health(checkpoint_dir: Path):
    """Verify checkpoint integrity."""
    for scenario_dir in checkpoint_dir.iterdir():
        for run_dir in scenario_dir.iterdir():
            # Check required files
            required = ["best_model.pt", "run_metadata.json"]
            missing = [f for f in required if not (run_dir / f).exists()]

            if missing:
                print(f"⚠️  {run_dir}: Missing {missing}")
            else:
                print(f"✓  {run_dir}: OK")

            # Check file sizes
            best_model = run_dir / "best_model.pt"
            if best_model.exists() and best_model.stat().st_size == 0:
                print(f"⚠️  {run_dir}: best_model.pt is empty")
```

### W&B Sync Status

```python
def check_wandb_sync_status(run_metadata_path: Path):
    """Check if run is synced to W&B."""
    with open(run_metadata_path) as f:
        metadata = json.load(f)

    wandb_run_id = metadata.get("wandb_run_id")
    if not wandb_run_id:
        return "Not logged to W&B"

    # Check if run exists in W&B
    # ... (use W&B API)
```
