"""Unified logging utilities for console summaries and W&B emission."""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_RICH = False

Number = Optional[float]


class LogSink(ABC):
    """Abstract base class for logger sinks."""

    @abstractmethod
    def start(self, context: Mapping[str, Any]) -> None:
        """Signal that logging is about to begin with shared context."""

    @abstractmethod
    def stop(self) -> None:
        """Flush and release any resources held by the sink."""

    @abstractmethod
    def log_metrics(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        *,
        step: Optional[float] = None,
    ) -> None:
        """Record structured metrics for the supplied phase."""

    @abstractmethod
    def log_event(
        self,
        level: str,
        message: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Emit an informational or warning event."""


class Logger:
    """Primary façade that fans log requests out to registered sinks."""

    def __init__(self, sinks: Optional[Iterable[LogSink]] = None) -> None:
        self._sinks: Sequence[LogSink] = tuple(sinks or ())
        self._context: Dict[str, Any] = {}
        self._started = False

    def start(self, context: Optional[Mapping[str, Any]] = None) -> None:
        """Begin logging and share immutable context with sinks."""

        if context:
            self._context.update({key: context[key] for key in context})
        if self._started:
            # Allow context updates without re-notifying sinks.
            return
        self._started = True
        snapshot = dict(self._context)
        for sink in self._sinks:
            sink.start(snapshot)

    def update_context(self, **context: Any) -> None:
        """Update shared context and re-broadcast to sinks."""

        if not context:
            return
        self._context.update(context)
        snapshot = dict(self._context)
        for sink in self._sinks:
            sink.start(snapshot)

    def log_metrics(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        *,
        step: Optional[float] = None,
    ) -> None:
        if not metrics:
            return
        for sink in self._sinks:
            sink.log_metrics(phase, metrics, step=step)

    def log_event(
        self,
        level: str,
        message: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        for sink in self._sinks:
            sink.log_event(level, message, extra)

    def info(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None:
        self.log_event("info", message, extra=extra)

    def warning(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None:
        self.log_event("warn", message, extra=extra)

    def stop(self) -> None:
        for sink in reversed(self._sinks):
            sink.stop()


def _format_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return bool(value)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, (np.bool_, np.bool8)):  # pragma: no cover - numpy optional
            return bool(value)
        if isinstance(value, (np.integer, np.floating)):  # pragma: no cover - numpy optional
            if np.isnan(value):
                return None
            return bool(value)
    except Exception:  # pragma: no cover - numpy optional
        pass
    return None


def _format_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


if _HAS_RICH:

    class ConsoleSink(LogSink):
        """Lean console dashboard rendered with ``rich``."""

        def __init__(
            self,
            *,
            refresh_per_second: float = 4.0,
            event_history: int = 5,
        ) -> None:
            self._refresh_rate = refresh_per_second
            self._event_history = int(event_history)
            self._console = Console()
            self._live: Optional[Live] = None
            self._context: Dict[str, Any] = {}
            self._phase_state: Dict[str, Dict[str, Any]] = {"train": {}, "eval": {}}
            self._events: Deque[Tuple[str, str]] = deque(maxlen=self._event_history)

        # ------------------------------------------------------------------
        # LogSink interface
        # ------------------------------------------------------------------
        def start(self, context: Mapping[str, Any]) -> None:
            self._context.update(context)
            if self._live is None:
                self._live = Live(
                    self._render(),
                    console=self._console,
                    refresh_per_second=self._refresh_rate,
                    auto_refresh=False,
                )
                self._live.start()
            else:
                self._refresh()

        def stop(self) -> None:
            if self._live is not None:
                self._live.stop()
                self._live = None
            self._context.clear()
            self._phase_state = {"train": {}, "eval": {}}
            self._events.clear()

        def log_metrics(
            self,
            phase: str,
            metrics: Mapping[str, Any],
            *,
            step: Optional[float] = None,
        ) -> None:
            tracked = self._phase_state.get(phase)
            if tracked is None:
                return

            episode = _format_int(metrics.get(f"{phase}/episode", step))
            total = _format_int(metrics.get(f"{phase}/episodes_total"))
            steps = _format_int(metrics.get(f"{phase}/steps"))
            collisions = _format_int(metrics.get(f"{phase}/collisions"))
            collision_rate = _format_number(metrics.get(f"{phase}/collision_rate"))
            mode = metrics.get(f"{phase}/reward_task") or metrics.get(f"{phase}/reward_mode")
            cause = metrics.get(f"{phase}/cause")
            success = _format_bool(metrics.get(f"{phase}/success"))
            success_rate = _format_number(metrics.get(f"{phase}/success_rate"))
            success_total = _format_number(metrics.get(f"{phase}/success_total"))
            idle = _format_bool(metrics.get(f"{phase}/idle"))
            epsilon = (
                _format_number(metrics.get("train/epsilon"))
                if phase == "train"
                else _format_number(metrics.get(f"{phase}/epsilon"))
            )
            primary_agent = metrics.get(f"{phase}/primary_agent")
            primary_return = _format_number(metrics.get(f"{phase}/return"))
            return_mean = _format_number(metrics.get(f"{phase}/return_mean"))
            return_window = _format_int(metrics.get(f"{phase}/return_window"))
            return_best = _format_number(metrics.get(f"{phase}/return_best"))
            buffer_fraction = _format_number(metrics.get(f"{phase}/buffer_fraction"))
            defender_crash = _format_bool(metrics.get(f"{phase}/defender_crashed"))
            attacker_crash = _format_bool(metrics.get(f"{phase}/attacker_crashed"))
            spawn_enabled_raw = metrics.get(f"{phase}/random_spawn_enabled")
            spawn_stage = metrics.get(f"{phase}/spawn_stage")
            spawn_success_rate = _format_number(metrics.get(f"{phase}/spawn_success_rate"))
            spawn_stage_success_rate = _format_number(metrics.get(f"{phase}/spawn_stage_success_rate"))
            spawn_structured_success_rate = _format_number(metrics.get(f"{phase}/spawn_structured_success_rate"))
            spawn_random_success_rate = _format_number(metrics.get(f"{phase}/spawn_random_success_rate"))
            defender_stage = metrics.get(f"{phase}/defender_stage")
            defender_success_rate = _format_number(metrics.get(f"{phase}/defender_success_rate"))
            defender_stage_success_rate = _format_number(metrics.get(f"{phase}/defender_stage_success_rate"))
            defender_stage_index = _format_number(metrics.get(f"{phase}/defender_stage_index"))

            if episode is not None:
                tracked["episode"] = episode
            if total is not None:
                tracked["total"] = total
            if steps is not None:
                tracked["steps"] = steps
            if collisions is not None:
                tracked["collisions"] = collisions
            if collision_rate is not None:
                tracked["collision_rate"] = collision_rate
            if mode is not None:
                tracked["mode"] = mode
            if cause is not None:
                tracked["cause"] = cause
            if success is not None:
                tracked["success"] = success
            if success_rate is not None:
                tracked["success_rate"] = success_rate
            if idle is not None:
                tracked["idle"] = idle
            if epsilon is not None:
                tracked["epsilon"] = epsilon
            if primary_agent is not None:
                tracked["primary_agent"] = primary_agent
            if primary_return is not None:
                tracked["primary_return"] = primary_return
            if return_mean is not None:
                tracked["return_mean"] = return_mean
            if return_window is not None:
                tracked["return_window"] = return_window
            if return_best is not None:
                tracked["return_best"] = return_best
            if buffer_fraction is not None:
                tracked["buffer_fraction"] = buffer_fraction
            if success_total is not None:
                tracked["success_total"] = success_total
            if defender_crash is not None:
                tracked["defender_crash"] = defender_crash
            if attacker_crash is not None:
                tracked["attacker_crash"] = attacker_crash
            if spawn_enabled_raw is not None:
                tracked["spawn_enabled"] = bool(spawn_enabled_raw)
            if spawn_stage is not None:
                tracked["spawn_stage"] = str(spawn_stage)
            if spawn_success_rate is not None:
                tracked["spawn_success_rate"] = spawn_success_rate
            if spawn_stage_success_rate is not None:
                tracked["spawn_stage_success_rate"] = spawn_stage_success_rate
            if spawn_structured_success_rate is not None:
                tracked["spawn_structured_success_rate"] = spawn_structured_success_rate
            if spawn_random_success_rate is not None:
                tracked["spawn_random_success_rate"] = spawn_random_success_rate
            if defender_stage is not None:
                tracked["defender_stage"] = str(defender_stage)
            if defender_success_rate is not None:
                tracked["defender_success_rate"] = defender_success_rate
            if defender_stage_success_rate is not None:
                tracked["defender_stage_success_rate"] = defender_stage_success_rate
            if defender_stage_index is not None:
                tracked["defender_stage_index"] = defender_stage_index

            agents = tracked.setdefault("agents", {})
            for key, value in metrics.items():
                agent_prefix = f"{phase}/agent/"
                if key.startswith(agent_prefix):
                    remainder = key[len(agent_prefix) :]
                    agent_id, _, metric_name = remainder.partition("/")
                    if not metric_name:
                        continue
                    entry = agents.setdefault(agent_id, {})
                    formatted = _format_number(value)
                    if metric_name == "return":
                        entry["return"] = formatted
                    elif metric_name == "collisions":
                        entry["collisions"] = formatted
                    elif metric_name == "avg_speed":
                        entry["speed"] = formatted
                    elif metric_name == "collision_step":
                        entry["collision_step"] = formatted
                    elif metric_name == "lap_count":
                        entry["lap_count"] = formatted

            finish_hits: Dict[str, bool] = {}
            finish_prefix = f"{phase}/finish_line_hit/"
            for key, value in metrics.items():
                if key.startswith(finish_prefix):
                    agent_id = key[len(finish_prefix) :]
                    hit = _format_bool(value)
                    if hit is not None:
                        finish_hits[agent_id] = hit
            if finish_hits:
                tracked["finish_line_hits"] = finish_hits

            self._refresh()

        def log_event(
            self,
            level: str,
            message: str,
            extra: Optional[Mapping[str, Any]] = None,
        ) -> None:
            fragments = [message]
            if extra:
                extras = ", ".join(f"{key}={extra[key]}" for key in sorted(extra))
                if extras:
                    fragments.append(f"({extras})")
            timestamp = datetime.now().strftime("%H:%M:%S")
            entry = f"[{timestamp}] [{level.upper()}] {' '.join(fragments)}"
            self._events.append(entry)
            self._refresh()

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _refresh(self) -> None:
            if self._live is not None:
                self._live.update(self._render(), refresh=True)

        def _render(self) -> Group:
            sections = []
            for phase, state in (("train", self._phase_state["train"]), ("eval", self._phase_state["eval"])):
                if state:
                    panel = self._render_phase(phase.upper(), state)
                    sections.append(panel)

            if self._events:
                sections.append(self._render_events())

            if not sections:
                sections.append(Panel("Waiting for telemetry…", title="F110"))

            return Group(*sections)

        def _render_phase(self, title: str, state: Mapping[str, Any]) -> Panel:
            summary_lines = []
            episode = state.get("episode")
            total = state.get("total")
            if episode is not None:
                if total:
                    summary_lines.append(f"Episode {episode}/{total}")
                else:
                    summary_lines.append(f"Episode {episode}")
            elif total:
                summary_lines.append(f"Total episodes: {total}")

            steps = state.get("steps")
            collisions = state.get("collisions")
            collision_rate = state.get("collision_rate")
            buffer_fraction = state.get("buffer_fraction")
            metrics_line_parts = []
            if steps is not None:
                metrics_line_parts.append(f"Steps {steps}")
            if collisions is not None:
                metrics_line_parts.append(f"Collisions {collisions}")
            if collision_rate is not None:
                metrics_line_parts.append(f"CollRate {collision_rate:.2f}")
            success_total = state.get("success_total")
            if success_total is not None:
                metrics_line_parts.append(f"SuccessTot {int(success_total)}")
            if metrics_line_parts:
                summary_lines.append("  ".join(metrics_line_parts))
            if buffer_fraction is not None:
                summary_lines.append(f"Buffer: {buffer_fraction * 100:.0f}%")

            mode = state.get("mode")
            cause = state.get("cause")
            if mode:
                summary_lines.append(f"Mode: {mode}")
            if cause:
                summary_lines.append(f"Cause: {cause}")

            success = state.get("success")
            if success is not None:
                summary_lines.append(f"Success: {'yes' if success else 'no'}")
            success_rate = state.get("success_rate")
            if success_rate is not None:
                summary_lines.append(f"Success rate: {success_rate * 100:.1f}%")
            if success_total is not None and success_total > 0:
                summary_lines.append(f"Success total: {int(success_total)}")
            finish_hits = state.get("finish_line_hits")
            if finish_hits:
                labels = ", ".join(f"{aid}={'yes' if hit else 'no'}" for aid, hit in finish_hits.items())
                summary_lines.append(f"Finish line: {labels}")

            spawn_enabled = state.get("spawn_enabled")
            spawn_stage = state.get("spawn_stage")
            spawn_success_rate = state.get("spawn_success_rate")
            spawn_stage_success_rate = state.get("spawn_stage_success_rate")
            spawn_line_parts = []
            if spawn_enabled is not None:
                spawn_line_parts.append(f"Spawn random: {'on' if spawn_enabled else 'off'}")
            if spawn_stage:
                spawn_line_parts.append(f"Stage {spawn_stage}")
            stage_rate_display = spawn_stage_success_rate if spawn_stage_success_rate is not None else spawn_success_rate
            if stage_rate_display is not None:
                spawn_line_parts.append(f"Stage success {stage_rate_display * 100:.1f}%")
            if (
                spawn_success_rate is not None
                and spawn_stage_success_rate is not None
                and not math.isclose(spawn_success_rate, spawn_stage_success_rate)
            ):
                spawn_line_parts.append(f"(overall {spawn_success_rate * 100:.1f}%)")
            if spawn_line_parts:
                summary_lines.append(" ".join(spawn_line_parts))

            defender_stage = state.get("defender_stage")
            defender_stage_index = state.get("defender_stage_index")
            defender_success_rate = state.get("defender_success_rate")
            defender_stage_success_rate = state.get("defender_stage_success_rate")
            defender_line_parts = []
            if defender_stage:
                defender_line_parts.append(f"Defender stage {defender_stage}")
            if defender_stage_index is not None:
                defender_line_parts.append(f"(idx {int(defender_stage_index)})")
            stage_success_display = (
                defender_stage_success_rate if defender_stage_success_rate is not None else defender_success_rate
            )
            if stage_success_display is not None:
                defender_line_parts.append(f"Defender success {stage_success_display * 100:.1f}%")
            if (
                defender_success_rate is not None
                and defender_stage_success_rate is not None
                and not math.isclose(defender_success_rate, defender_stage_success_rate)
            ):
                defender_line_parts.append(f"(overall {defender_success_rate * 100:.1f}%)")
            if defender_line_parts:
                summary_lines.append(" ".join(defender_line_parts))

            idle = state.get("idle")
            if idle is not None:
                summary_lines.append(f"Idle stop: {'yes' if idle else 'no'}")

            epsilon = state.get("epsilon")
            if epsilon is not None:
                summary_lines.append(f"Epsilon: {epsilon:.3f}")

            defender_crash = state.get("defender_crash")
            attacker_crash = state.get("attacker_crash")
            crash_bits = []
            if defender_crash is not None:
                crash_bits.append(f"Defender crash: {'yes' if defender_crash else 'no'}")
            if attacker_crash is not None:
                crash_bits.append(f"Attacker crash: {'yes' if attacker_crash else 'no'}")
            if crash_bits:
                summary_lines.append("  ".join(crash_bits))

            primary_agent = state.get("primary_agent")
            primary_return = state.get("primary_return")
            return_mean = state.get("return_mean")
            return_best = state.get("return_best")
            return_window = state.get("return_window")
            avg_label = None
            if return_mean is not None:
                if return_window:
                    avg_label = f"avg{int(return_window)}ep {return_mean:.2f}"
                else:
                    avg_label = f"avg {return_mean:.2f}"
            if primary_agent and primary_return is not None:
                line = f"{primary_agent} return: {primary_return:.2f}"
                if avg_label is not None:
                    line += f" ({avg_label})"
                if return_best is not None:
                    line += f", best {return_best:.2f}"
                summary_lines.append(line)
            elif avg_label is not None:
                line = f"Return {avg_label}"
                if return_best is not None:
                    line += f", best {return_best:.2f}"
                summary_lines.append(line)

            summary_text = Text("\n".join(summary_lines) if summary_lines else "No telemetry yet.")

            agents_table = None
            agents = state.get("agents") or {}
            if agents:
                agents_table = Table(expand=True)
                agents_table.add_column("Agent", justify="left")
                agents_table.add_column("Return", justify="right")
                agents_table.add_column("Coll", justify="right")
                agents_table.add_column("AvgSpd", justify="right")
                for agent_id in sorted(agents):
                    entry = agents[agent_id]
                    ret = entry.get("return")
                    collisions_val = entry.get("collisions")
                    speed = entry.get("speed")
                    agents_table.add_row(
                        agent_id,
                        f"{ret:.2f}" if ret is not None else "—",
                        f"{collisions_val:.0f}" if collisions_val is not None else "—",
                        f"{speed:.2f}" if speed is not None else "—",
                    )

            components = [Align.left(summary_text)]
            if agents_table is not None:
                components.append(Align.left(agents_table))

            content = Group(*components) if len(components) > 1 else components[0]

            return Panel(content, title=title, border_style="cyan")

        def _render_events(self) -> Panel:
            events_table = Table.grid(padding=0)
            for entry in reversed(self._events):
                events_table.add_row(Text(entry))
            return Panel(events_table, title="Recent Events", border_style="magenta")


else:

    class ConsoleSink(LogSink):
        """Fallback console sink using standard :mod:`logging` output."""

        _LEVEL_MAP = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warn": logging.WARNING,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        def __init__(
            self,
            *,
            logger_name: str = "f110.console",
            level: int = logging.INFO,
            handler: Optional[logging.Handler] = None,
        ) -> None:
            self._logger = logging.getLogger(logger_name)
            self._logger.propagate = False
            self._level = level
            self._formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
            self._handler_owned = handler is None
            self._handler: Optional[logging.Handler] = None
            if handler is not None:
                handler.setLevel(level)
                if handler.formatter is None:
                    handler.setFormatter(self._formatter)
                self._logger.addHandler(handler)
                self._handler = handler
            self._logger.setLevel(level)

        def start(self, context: Mapping[str, Any]) -> None:
            if self._handler is None and self._handler_owned:
                handler = logging.StreamHandler()
                handler.setFormatter(self._formatter)
                handler.setLevel(self._level)
                self._logger.addHandler(handler)
                self._handler = handler
            if context:
                formatted = ", ".join(f"{key}={context[key]}" for key in sorted(context))
                self._logger.log(self._level, "Logger context: %s", formatted)

        def stop(self) -> None:
            if self._handler and self._handler_owned:
                self._logger.removeHandler(self._handler)
                self._handler.close()
                self._handler = None

        def log_metrics(
            self,
            phase: str,
            metrics: Mapping[str, Any],
            *,
            step: Optional[float] = None,
        ) -> None:
            if not metrics:
                return
            prefix = f"{phase.upper()} step={step}" if step is not None else phase.upper()
            filtered = {key: metrics[key] for key in sorted(metrics)}
            self._logger.log(self._level, "%s %s", prefix, filtered)

        def log_event(
            self,
            level: str,
            message: str,
            extra: Optional[Mapping[str, Any]] = None,
        ) -> None:
            level_no = self._LEVEL_MAP.get(level.lower(), self._level)
            if extra:
                formatted = ", ".join(f"{key}={value}" for key, value in sorted(extra.items()))
                message = f"{message} ({formatted})"
            self._logger.log(level_no, message)


class WandbSink(LogSink):
    """Sink that relays metrics to an instantiated wandb run."""

    def __init__(self, run: Any) -> None:  # type: ignore[valid-type]
        self._run = run
        self._train_step_max: float = 0.0
        self._phase_counters: Dict[str, int] = {}
        self._phase_stride: float = 1e-3

    def start(self, context: Mapping[str, Any]) -> None:  # noqa: D401 - context unused
        # wandb run already initialised upstream; nothing extra required.
        _ = context
        self._train_step_max = 0.0
        self._phase_counters.clear()

    def stop(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:  # pragma: no cover - wandb shutdown best effort
                pass

    def log_metrics(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        *,
        step: Optional[float] = None,
    ) -> None:
        if self._run is None:
            return
        payload: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                payload[key] = float(value)
            else:
                payload[key] = value
        if phase in {"train", "eval"}:
            keep_keys = {
                "train": {"train/return", "train/return_mean"},
                "eval": {"eval/return", "eval/return_mean"},
            }[phase]
            payload = {key: value for key, value in payload.items() if key in keep_keys}
        if not payload:
            return

        step_value: Optional[float]
        if step is None:
            step_value = None
        else:
            try:
                step_value = float(step)
            except (TypeError, ValueError):
                step_value = None

        wandb_step: Optional[float]
        if phase == "train":
            if step_value is None:
                step_value = self._train_step_max + 1.0
            self._train_step_max = max(self._train_step_max, step_value)
            wandb_step = step_value
        else:
            base = self._train_step_max if self._train_step_max > 0.0 else 0.0
            counter = self._phase_counters.get(phase, 0) + 1
            self._phase_counters[phase] = counter
            candidate = base + counter * self._phase_stride
            if candidate <= self._train_step_max:
                candidate = self._train_step_max + counter * self._phase_stride
            wandb_step = candidate

        try:
            if wandb_step is not None:
                self._run.log(payload, step=float(wandb_step))
            else:
                self._run.log(payload)
        except Exception:
            try:
                self._run.log(payload)
            except Exception:
                pass

    def log_event(
        self,
        level: str,
        message: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._run is None:
            return
        payload: Dict[str, Any] = {
            "event/level": level.upper(),
            "event/message": message,
        }
        if extra:
            for key, value in extra.items():
                payload[f"event/{key}"] = value
        self._run.log(payload)


class NullSink(LogSink):
    """Sink that intentionally discards all log requests."""

    def start(self, context: Mapping[str, Any]) -> None:
        _ = context

    def stop(self) -> None:
        return

    def log_metrics(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        *,
        step: Optional[float] = None,
    ) -> None:
        _ = (phase, metrics, step)

    def log_event(
        self,
        level: str,
        message: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        _ = (level, message, extra)


NULL_LOGGER = Logger(sinks=(NullSink(),))


__all__ = [
    "ConsoleSink",
    "LogSink",
    "Logger",
    "NULL_LOGGER",
    "NullSink",
    "WandbSink",
]
