"""Rich console logging for F110 training.

Provides formatted messages and evaluation summary tables.
"""

from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table


class ConsoleLogger:
    """Formatted console messages and evaluation summaries."""

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
