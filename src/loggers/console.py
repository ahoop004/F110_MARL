"""Rich console logging for F110 training.

Provides beautiful terminal output with progress bars, tables,
and color-coded metrics display.
"""

from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel


class ConsoleLogger:
    """Logger for rich terminal output during training.

    Provides progress bars, metrics tables, and formatted output
    for training runs. Uses the rich library for beautiful terminal UI.

    Example:
        >>> from loggers import ConsoleLogger
        >>>
        >>> logger = ConsoleLogger()
        >>> logger.print_header("Training PPO on Gaplock")
        >>>
        >>> # During training
        >>> logger.log_episode(
        ...     episode=0,
        ...     outcome="target_crash",
        ...     reward=125.5,
        ...     steps=450,
        ...     success_rate=0.75,
        ... )
    """

    def __init__(self, verbose: bool = True):
        """Initialize console logger.

        Args:
            verbose: Whether to enable verbose output
        """
        self.console = Console()
        self.verbose = verbose

    def print_header(self, title: str, subtitle: Optional[str] = None):
        """Print formatted header.

        Args:
            title: Main title
            subtitle: Optional subtitle

        Example:
            >>> logger.print_header(
            ...     "Training PPO Agent",
            ...     "Gaplock task - 1500 episodes"
            ... )
        """
        self.console.rule(f"[bold blue]{title}[/bold blue]")
        if subtitle:
            self.console.print(f"[dim]{subtitle}[/dim]")
        self.console.print()

    def print_config(self, config: Dict[str, Any]):
        """Print configuration as formatted table.

        Args:
            config: Configuration dict to display

        Example:
            >>> logger.print_config({
            ...     'algorithm': 'ppo',
            ...     'lr': 0.0005,
            ...     'gamma': 0.995,
            ... })
        """
        table = Table(title="Configuration", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")

        for key, value in config.items():
            table.add_row(str(key), str(value))

        self.console.print(table)
        self.console.print()

    def log_episode(
        self,
        episode: int,
        outcome: str,
        reward: float,
        steps: int,
        success_rate: Optional[float] = None,
        avg_reward: Optional[float] = None,
    ):
        """Log episode results.

        Args:
            episode: Episode number
            outcome: Episode outcome type
            reward: Total episode reward
            steps: Number of steps
            success_rate: Rolling success rate (optional)
            avg_reward: Rolling average reward (optional)

        Example:
            >>> logger.log_episode(
            ...     episode=42,
            ...     outcome="target_crash",
            ...     reward=125.5,
            ...     steps=450,
            ...     success_rate=0.75,
            ...     avg_reward=85.2,
            ... )
        """
        if not self.verbose:
            return

        # Color code outcome
        outcome_color = "green" if "target_crash" in outcome or "success" in outcome else "red"

        msg = f"[dim]Episode {episode:4d}[/dim] | "
        msg += f"[{outcome_color}]{outcome:15s}[/{outcome_color}] | "
        msg += f"Reward: {reward:7.1f} | "
        msg += f"Steps: {steps:4d}"

        if success_rate is not None:
            msg += f" | Success: {success_rate:5.1%}"

        if avg_reward is not None:
            msg += f" | Avg: {avg_reward:6.1f}"

        self.console.print(msg)

    def log_eval_inline(
        self,
        eval_episode: int,
        outcome: str,
        success: bool,
        reward: float,
        steps: int,
        rolling_success_rate: float,
        spawn_point: str,
    ):
        """Log eval episode inline (for alternating train/eval mode).

        Args:
            eval_episode: Eval episode number
            outcome: Episode outcome type
            success: Whether episode was successful
            reward: Total episode reward
            steps: Number of steps
            rolling_success_rate: Rolling success rate from eval metrics tracker
            spawn_point: Spawn point used for this episode
        """
        if not self.verbose:
            return

        # Color code outcome
        outcome_color = "green" if success else "red"

        msg = f"[dim cyan]Eval {eval_episode:4d}[/dim cyan] | "
        msg += f"[{outcome_color}]{outcome:15s}[/{outcome_color}] | "
        msg += f"Reward: {reward:7.1f} | "
        msg += f"Steps: {steps:4d} | "
        msg += f"Success: {rolling_success_rate:5.1%} | "
        msg += f"[dim]{spawn_point}[/dim]"

        self.console.print(msg)

    def print_summary(self, stats: Dict[str, Any]):
        """Print training summary as formatted table.

        Args:
            stats: Statistics dict to display

        Example:
            >>> logger.print_summary({
            ...     'total_episodes': 1500,
            ...     'success_rate': 0.68,
            ...     'avg_reward': 82.4,
            ... })
        """
        self.console.print()
        table = Table(title="Training Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        for key, value in stats.items():
            # Format value based on type
            if isinstance(value, float):
                if 'rate' in key.lower():
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            table.add_row(key.replace('_', ' ').title(), formatted_value)

        self.console.print(table)

    def print_outcome_distribution(self, outcome_counts: Dict[str, int]):
        """Print outcome distribution table.

        Args:
            outcome_counts: Dict mapping outcome names to counts

        Example:
            >>> logger.print_outcome_distribution({
            ...     'target_crash': 1020,
            ...     'self_crash': 350,
            ...     'timeout': 130,
            ... })
        """
        total = sum(outcome_counts.values())
        if total == 0:
            return

        self.console.print()
        table = Table(title="Outcome Distribution", show_header=True)
        table.add_column("Outcome", style="cyan")
        table.add_column("Count", style="yellow", justify="right")
        table.add_column("Rate", style="green", justify="right")

        # Sort by count descending
        sorted_outcomes = sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True)

        for outcome, count in sorted_outcomes:
            rate = count / total
            table.add_row(
                outcome.replace('_', ' ').title(),
                str(count),
                f"{rate:.1%}",
            )

        self.console.print(table)

    def create_progress(self, total: int, description: str = "Training") -> Progress:
        """Create a progress bar for training.

        Args:
            total: Total number of episodes
            description: Description for progress bar

        Returns:
            Progress instance

        Example:
            >>> progress = logger.create_progress(1500, "Training PPO")
            >>> with progress:
            ...     task = progress.add_task("[cyan]Episodes", total=1500)
            ...     for episode in range(1500):
            ...         # Train episode
            ...         progress.update(task, advance=1)
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )

    def print_success(self, message: str):
        """Print success message.

        Args:
            message: Success message to display
        """
        self.console.print(f"[green]✓[/green] {message}")

    def print_warning(self, message: str):
        """Print warning message.

        Args:
            message: Warning message to display
        """
        self.console.print(f"[yellow]⚠[/yellow] {message}")

    def print_error(self, message: str):
        """Print error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"[red]✗[/red] {message}")

    def print_info(self, message: str):
        """Print info message.

        Args:
            message: Info message to display
        """
        self.console.print(f"[blue]ℹ[/blue] {message}")


__all__ = ['ConsoleLogger']
