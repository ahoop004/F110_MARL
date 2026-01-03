"""Rich-based live console dashboard for training visualization.

Provides real-time training metrics display with episode statistics.
"""

from collections import deque
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class RichConsole:
    """Live-updating console dashboard for training metrics.

    Displays real-time statistics during training with auto-updating dashboard.
    Focused on gaplock task metrics: attack/target success rates, failure modes.

    Example:
        >>> console = RichConsole(refresh_rate=4.0)
        >>> console.start()
        >>> for episode in range(1500):
        ...     # ... training ...
        ...     console.update_episode(episode, outcome, reward, steps, outcome_stats)
        >>> console.stop()
    """

    def __init__(
        self,
        refresh_rate: float = 0.1,
        enabled: bool = True,
    ):
        """Initialize Rich console dashboard.

        Args:
            refresh_rate: Display refresh rate in Hz (default: 0.1, only updates on explicit calls)
            enabled: Enable/disable Rich console (default: True, auto-disabled if Rich unavailable)
        """
        self.enabled = enabled and RICH_AVAILABLE
        self.refresh_rate = refresh_rate

        if not self.enabled:
            return

        self.console = Console()
        self.live = None

        # Current episode stats
        self.current_episode = 0
        self.current_steps = 0
        self.current_return = 0.0
        self.current_outcome = "UNKNOWN"

        # Outcome statistics (rolling window)
        self.attack_success_rate = 0.0
        self.target_success_rate = 0.0
        self.idle_stop_rate = 0.0
        self.truncation_rate = 0.0
        self.attacker_crash_rate = 0.0
        self.collision_rate = 0.0

        # Performance tracking
        self.start_time = datetime.now()
        self.episodes_per_sec = 0.0
        self.last_update_time = None
        self.episode_times = deque(maxlen=100)

    def start(self):
        """Start live display."""
        if not self.enabled:
            return

        self.start_time = datetime.now()
        self.live = Live(
            self._generate_layout(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=False,
        )
        self.live.start()

    def stop(self):
        """Stop live display."""
        if not self.enabled or self.live is None:
            return

        self.live.stop()

    def update_episode(
        self,
        episode: int,
        outcome: str,
        reward: float,
        steps: int,
        outcome_stats: Optional[Dict[str, float]] = None,
    ):
        """Update dashboard with latest episode results.

        Args:
            episode: Episode number
            outcome: Episode outcome (e.g., "TARGET_CRASH", "SELF_CRASH")
            reward: Total episode reward
            steps: Number of steps in episode
            outcome_stats: Dict of outcome statistics (rates from MetricsTracker)
        """
        if not self.enabled or self.live is None:
            return

        # Update current episode
        self.current_episode = episode
        self.current_steps = steps
        self.current_return = reward
        self.current_outcome = outcome

        # Update outcome statistics
        if outcome_stats:
            # Extract outcome rates from outcome_rates dict
            outcome_rates = outcome_stats.get('outcome_rates', {})

            # Attack success = target_crash rate (clean attack success)
            self.attack_success_rate = outcome_rates.get('target_crash', 0.0)

            # Target success = defender successfully survived/escaped
            # Includes: finish line, attacker crashed, attacker idle, time ran out
            # Excludes: collision (mutual failure)
            self.target_success_rate = (
                outcome_rates.get('target_finish', 0.0) +
                outcome_rates.get('self_crash', 0.0) +
                outcome_rates.get('idle_stop', 0.0) +
                outcome_rates.get('timeout', 0.0)
            )

            # Failure modes (individual breakdown)
            self.idle_stop_rate = outcome_rates.get('idle_stop', 0.0)
            self.truncation_rate = outcome_rates.get('timeout', 0.0)
            self.attacker_crash_rate = outcome_rates.get('self_crash', 0.0)
            self.collision_rate = outcome_rates.get('collision', 0.0)

        # Update performance tracking
        now = datetime.now()
        if self.last_update_time:
            episode_time = (now - self.last_update_time).total_seconds()
            self.episode_times.append(episode_time)
            if len(self.episode_times) > 0:
                avg_episode_time = sum(self.episode_times) / len(self.episode_times)
                self.episodes_per_sec = 1.0 / avg_episode_time if avg_episode_time > 0 else 0.0
        self.last_update_time = now

        # Update display
        self.live.update(self._generate_layout())

    def _generate_layout(self) -> Layout:
        """Generate dashboard layout.

        Returns:
            Rich Layout with metrics panels
        """
        layout = Layout()

        # Create main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )

        # Header
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds

        header_text = Text()
        header_text.append("F110 MARL Training", style="bold cyan")
        header_text.append(f" | Episode {self.current_episode}", style="bold white")
        header_text.append(f" | Elapsed: {elapsed_str}", style="dim")
        header_text.append(f" | {self.episodes_per_sec:.2f} ep/s", style="dim")

        layout["header"].update(Panel(header_text, style="cyan"))

        # Body - metrics table
        layout["body"].update(self._create_metrics_table())

        return layout

    def _create_metrics_table(self) -> Table:
        """Create metrics table with current statistics.

        Returns:
            Rich Table with training metrics
        """
        table = Table(title="Training Metrics", show_header=True, header_style="bold magenta")

        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", justify="right", style="white", width=20)

        # Current episode info
        table.add_row("", "")  # Spacer
        table.add_row("[bold]Current Episode", "")
        table.add_row("  Outcome", self._format_outcome(self.current_outcome))
        table.add_row("  Steps", f"{self.current_steps}")
        table.add_row("  Return", f"{self.current_return:.2f}")

        # Success rates
        table.add_row("", "")  # Spacer
        table.add_row("[bold]Success Rates", "")
        table.add_row(
            "  Attack Success",
            self._format_percentage(self.attack_success_rate, good_threshold=0.5)
        )
        table.add_row(
            "  Target Success",
            self._format_percentage(self.target_success_rate, good_threshold=0.0, invert=True)
        )

        # Failure modes
        table.add_row("", "")  # Spacer
        table.add_row("[bold]Failure Modes", "")
        table.add_row(
            "  Attacker Crash",
            self._format_percentage(self.attacker_crash_rate, good_threshold=0.2, invert=True)
        )
        table.add_row(
            "  Collision",
            self._format_percentage(self.collision_rate, good_threshold=0.1, invert=True)
        )
        table.add_row(
            "  Idle Stop",
            self._format_percentage(self.idle_stop_rate, good_threshold=0.1, invert=True)
        )
        table.add_row(
            "  Truncation",
            self._format_percentage(self.truncation_rate, good_threshold=0.2, invert=True)
        )

        return table

    def _format_outcome(self, outcome: str) -> Text:
        """Format outcome with color coding.

        Args:
            outcome: Outcome string

        Returns:
            Colored Text object
        """
        outcome_colors = {
            'target_crash': 'green',
            'self_crash': 'red',
            'collision': 'yellow',
            'timeout': 'blue',
            'idle_stop': 'magenta',
            'target_finish': 'cyan',
        }

        color = outcome_colors.get(outcome, 'white')
        return Text(outcome.upper(), style=f"bold {color}")

    def _format_percentage(
        self,
        value: float,
        good_threshold: float = 0.5,
        invert: bool = False
    ) -> Text:
        """Format percentage with color coding.

        Args:
            value: Value between 0 and 1
            good_threshold: Threshold for "good" performance
            invert: If True, lower is better (for failure modes)

        Returns:
            Colored Text object
        """
        pct_str = f"{value * 100:.1f}%"

        # Determine color based on performance
        if invert:
            # Lower is better (failure modes)
            if value < good_threshold * 0.5:
                color = "green"
            elif value < good_threshold:
                color = "yellow"
            else:
                color = "red"
        else:
            # Higher is better (success rates)
            if value >= good_threshold:
                color = "green"
            elif value >= good_threshold * 0.5:
                color = "yellow"
            else:
                color = "red"

        return Text(pct_str, style=f"bold {color}")


__all__ = ['RichConsole']
