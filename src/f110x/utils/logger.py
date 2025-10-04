"""Unified logging utilities for console summaries and W&B emission."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


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


class ConsoleSink(LogSink):
    """Structured console summaries for training and evaluation metrics."""

    def __init__(self) -> None:
        self._context: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # LogSink interface
    # ------------------------------------------------------------------
    def start(self, context: Mapping[str, Any]) -> None:
        self._context = dict(context)

    def stop(self) -> None:
        # Console output has no persistent resources.
        self._context.clear()

    def log_metrics(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        *,
        step: Optional[float] = None,
    ) -> None:
        if phase == "train":
            self._log_train(metrics, step)
        elif phase == "eval":
            self._log_eval(metrics, step)

    def log_event(
        self,
        level: str,
        message: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        prefix = level.upper()
        extras = ""
        if extra:
            extras = " " + self._format_keyvals(extra)
        print(f"[{prefix}] {message}{extras}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_train(self, metrics: Mapping[str, Any], step: Optional[float]) -> None:
        episode = self._coerce_int(metrics.get("train/episode", step))
        total = self._coerce_int(metrics.get("train/total_episodes"))
        header = f"[TRAIN {episode:03d}]" if episode is not None else "[TRAIN]"
        if episode is not None and total is not None and total > 0:
            header = f"[TRAIN {episode:03d}/{total}]"

        mode = metrics.get("train/reward_mode")
        cause = metrics.get("train/cause")
        steps = self._coerce_int(metrics.get("train/steps"))
        collisions = self._coerce_int(metrics.get("train/collisions_total"))
        success = metrics.get("train/success")
        epsilon = metrics.get("train/epsilon")
        primary_agent = metrics.get("train/primary_agent")
        primary_return = metrics.get("train/primary_return")

        fragments = []
        if mode:
            fragments.append(f"mode={mode}")
        if cause:
            fragments.append(f"cause={cause}")
        if steps is not None:
            fragments.append(f"steps={steps}")
        if collisions is not None:
            fragments.append(f"collisions={collisions}")
        if isinstance(success, bool):
            fragments.append(f"success={'yes' if success else 'no'}")
        if primary_agent and self._is_number(primary_return):
            fragments.append(
                f"return[{primary_agent}]={float(primary_return):.2f}"
            )
        elif self._is_number(primary_return):
            fragments.append(f"return={float(primary_return):.2f}")
        if self._is_number(epsilon):
            fragments.append(f"epsilon={float(epsilon):.3f}")

        print(f"{header} {' '.join(fragments)}".rstrip())

    def _log_eval(self, metrics: Mapping[str, Any], step: Optional[float]) -> None:
        episode = self._coerce_int(metrics.get("eval/episode", step))
        total = self._coerce_int(metrics.get("eval/total_episodes"))
        header = f"[EVAL {episode:03d}]" if episode is not None else "[EVAL]"
        if episode is not None and total is not None and total > 0:
            header = f"[EVAL {episode:03d}/{total}]"

        cause = metrics.get("eval/cause")
        steps = self._coerce_int(metrics.get("eval/steps"))
        collisions = self._coerce_int(metrics.get("eval/collisions_total"))
        defender = metrics.get("eval/defender_crashed")
        attacker = metrics.get("eval/attacker_crashed")
        primary_agent = metrics.get("eval/primary_agent")
        primary_return = metrics.get("eval/primary_return")

        fragments = []
        if cause:
            fragments.append(f"cause={cause}")
        if steps is not None:
            fragments.append(f"steps={steps}")
        if collisions is not None:
            fragments.append(f"collisions={collisions}")
        if isinstance(defender, bool):
            fragments.append(f"defender_crash={'yes' if defender else 'no'}")
        if isinstance(attacker, bool):
            fragments.append(f"attacker_crash={'yes' if attacker else 'no'}")
        if primary_agent and self._is_number(primary_return):
            fragments.append(
                f"return[{primary_agent}]={float(primary_return):.2f}"
            )
        elif self._is_number(primary_return):
            fragments.append(f"return={float(primary_return):.2f}")

        print(f"{header} {' '.join(fragments)}".rstrip())

    @staticmethod
    def _format_keyvals(pairs: Mapping[str, Any]) -> str:
        fragments = []
        for key, value in pairs.items():
            if isinstance(value, float):
                fragments.append(f"{key}={value:.3f}")
            else:
                fragments.append(f"{key}={value}")
        return " ".join(fragments)

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_number(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
        return False


class WandbSink(LogSink):
    """Sink that relays metrics to an instantiated wandb run."""

    def __init__(self, run: Any) -> None:  # type: ignore[valid-type]
        self._run = run

    def start(self, context: Mapping[str, Any]) -> None:  # noqa: D401 - context unused
        # wandb run already initialised upstream; nothing extra required.
        _ = context

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
        if not payload:
            return
        if step is not None:
            try:
                self._run.log(payload, step=float(step))
                return
            except Exception:
                # Fall back to default step if wandb rejects manual step.
                pass
        self._run.log(payload)

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
