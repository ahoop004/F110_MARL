"""Console, CSV, and optional W&B logging for training hooks."""
from .console import ConsoleLogger
from .csv_logger import CSVLogger
try:
    from .wandb_logger import WandbLogger
except ImportError as exc:
    class WandbLogger:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "WandbLogger requires the `wandb` package. Install with: pip install wandb"
            ) from exc


__all__ = ["WandbLogger", "ConsoleLogger", "CSVLogger"]
