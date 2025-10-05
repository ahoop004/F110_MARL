"""Unified logging utilities for console summaries and W&B emission."""
from __future__ import annotations

import math
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence


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
        self._use_live_panel = bool(os.environ.get("F110_LIVE_CONSOLE", "1") != "0" and sys.stdout.isatty())
        self._train_state: Dict[str, Any] = {}
        self._eval_state: Dict[str, Any] = {}
        self._event_buffer: Deque[str] = deque(maxlen=8)
        self._clear_sequence = "\x1b[2J\x1b[H"
        self._started = False

    # ------------------------------------------------------------------
    # LogSink interface
    # ------------------------------------------------------------------
    def start(self, context: Mapping[str, Any]) -> None:
        self._context = dict(context)
        first_start = not self._started
        self._started = True
        if self._use_live_panel:
            if first_start:
                self._train_state.clear()
                self._eval_state.clear()
                self._event_buffer.clear()
            self._render_panel()

    def stop(self) -> None:
        # Console output has no persistent resources.
        self._context.clear()
        self._started = False

    def log_metrics(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        *,
        step: Optional[float] = None,
    ) -> None:
        if phase == "train":
            if self._use_live_panel:
                self._update_phase_state("train", metrics, step)
                self._render_panel()
            else:
                self._log_train(metrics, step)
        elif phase == "eval":
            if self._use_live_panel:
                self._update_phase_state("eval", metrics, step)
                self._render_panel()
            else:
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
        line = f"[{prefix}] {message}{extras}"
        if self._use_live_panel:
            self._event_buffer.append(line)
            self._render_panel()
        else:
            print(line)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_panel(self) -> None:
        if not self._use_live_panel:
            return

        lines = self._compose_panel_lines()
        panel_lines = self._wrap_in_box(lines)

        sys.stdout.write(self._clear_sequence)
        for text in panel_lines:
            sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def _compose_panel_lines(self) -> List[str]:
        lines: List[str] = []

        if self._train_state:
            lines.extend(self._format_phase_section("TRAINING", self._train_state))
        if self._eval_state:
            if lines:
                lines.append("")
            lines.extend(self._format_phase_section("EVALUATION", self._eval_state))

        if self._event_buffer:
            if lines:
                lines.append("")
            lines.append("Recent Events:")
            for event in list(self._event_buffer)[-6:]:
                lines.append(f"  {event}")

        if not lines:
            lines.append("Waiting for metrics...")

        return lines

    def _format_phase_section(self, title: str, state: Mapping[str, Any]) -> List[str]:
        lines = [title]

        episode = state.get("episode")
        total = state.get("total")
        steps = state.get("steps")
        collisions = state.get("collisions_total")

        summary_parts = []
        if episode is not None and total is not None:
            summary_parts.append(f"Episode {episode:03d}/{total:03d}")
        elif episode is not None:
            summary_parts.append(f"Episode {episode:03d}")
        if steps is not None:
            summary_parts.append(f"Steps {steps}")
        if collisions is not None:
            summary_parts.append(f"Collisions {collisions}")
        if summary_parts:
            lines.append("  " + "  ".join(summary_parts))

        detail_parts = []
        cause = state.get("cause")
        if cause:
            detail_parts.append(f"Cause: {cause}")
        mode = state.get("mode")
        if mode:
            detail_parts.append(f"Mode: {mode}")
        success = state.get("success")
        if isinstance(success, bool):
            detail_parts.append(f"Success: {'yes' if success else 'no'}")
        epsilon = state.get("epsilon")
        if epsilon is not None and self._is_number(epsilon):
            detail_parts.append(f"Epsilon: {float(epsilon):.3f}")
        defender_flag = state.get("defender_crashed")
        if isinstance(defender_flag, bool):
            detail_parts.append(f"Defender crash: {'yes' if defender_flag else 'no'}")
        attacker_flag = state.get("attacker_crashed")
        if isinstance(attacker_flag, bool):
            detail_parts.append(f"Attacker crash: {'yes' if attacker_flag else 'no'}")
        if detail_parts:
            lines.append("  " + "  ".join(detail_parts))

        primary_agent = state.get("primary_agent")
        primary_return = state.get("primary_return")
        if primary_agent and self._is_number(primary_return):
            lines.append(
                f"  Primary {primary_agent}: return {float(primary_return):.2f}"
            )

        idle_truncated = state.get("idle_truncated")
        if isinstance(idle_truncated, bool):
            lines.append(f"  Idle truncated: {'yes' if idle_truncated else 'no'}")

        agents = state.get("agents") or {}
        if agents:
            lines.append("")
            lines.extend(self._format_agent_table(agents))

        return lines

    def _format_agent_table(self, agents: Mapping[str, Any]) -> List[str]:
        header = f"{'Agent':<10} {'Return':>10} {'AvgRet':>10} {'Coll':>6} {'AvgSpd':>8}"
        lines = [header, "-" * len(header)]

        for agent_id in sorted(agents):
            entry = agents[agent_id]
            last_return = entry.get("last_return")
            avg_return = entry.get("avg_return")
            collisions = entry.get("last_collisions")
            avg_speed = entry.get("last_speed")

            return_str = f"{float(last_return):.2f}" if self._is_number(last_return) else "-"
            avg_return_str = (
                f"{float(avg_return):.2f}" if self._is_number(avg_return) else "-"
            )
            collisions_str = (
                f"{int(collisions)}" if self._is_number(collisions) else "-"
            )
            avg_speed_str = (
                f"{float(avg_speed):.2f}" if self._is_number(avg_speed) else "-"
            )

            row = (
                f"{agent_id:<10} {return_str:>10} {avg_return_str:>10} "
                f"{collisions_str:>6} {avg_speed_str:>8}"
            )
            lines.append(row)

        return lines

    def _wrap_in_box(self, lines: List[str]) -> List[str]:
        term_size = shutil.get_terminal_size(fallback=(100, 24))
        columns = term_size.columns if term_size.columns > 0 else 100
        width = max(60, min(columns, 120))
        inner_width = max(10, width - 2)

        wrapped: List[str] = []
        top = "+" + "-" * inner_width + "+"
        bottom = "+" + "-" * inner_width + "+"
        wrapped.append(top)

        for line in lines:
            truncated = line[:inner_width]
            padding = " " * max(inner_width - len(truncated), 0)
            wrapped.append(f"|{truncated}{padding}|")

        wrapped.append(bottom)
        return wrapped

    def _update_phase_state(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        step: Optional[float],
    ) -> None:
        state = self._train_state if phase == "train" else self._eval_state

        episode_key = f"{phase}/episode"
        total_key = f"{phase}/total_episodes"
        steps_key = f"{phase}/steps"
        collisions_key = f"{phase}/collisions_total"

        episode_val = metrics.get(episode_key, step)
        episode = self._coerce_int(episode_val)
        total = self._coerce_int(metrics.get(total_key))
        steps_val = self._coerce_int(metrics.get(steps_key))
        collisions_val = self._coerce_int(metrics.get(collisions_key))

        state["episode"] = episode
        state["total"] = total
        state["steps"] = steps_val
        state["collisions_total"] = collisions_val
        state["cause"] = metrics.get(f"{phase}/cause")
        state["mode"] = metrics.get(f"{phase}/reward_task") or metrics.get(
            f"{phase}/reward_mode"
        )

        success_key = f"{phase}/success"
        if success_key in metrics:
            state["success"] = bool(metrics.get(success_key))

        if phase == "train":
            epsilon_val = metrics.get("train/epsilon")
            if self._is_number(epsilon_val):
                state["epsilon"] = float(epsilon_val)
            idle_flag = metrics.get("train/idle_truncated")
            if isinstance(idle_flag, bool):
                state["idle_truncated"] = idle_flag

        defender_key = f"{phase}/defender_crashed"
        if defender_key in metrics:
            state["defender_crashed"] = bool(metrics.get(defender_key))

        attacker_key = f"{phase}/attacker_crashed"
        if attacker_key in metrics:
            state["attacker_crashed"] = bool(metrics.get(attacker_key))

        primary_agent = metrics.get(f"{phase}/primary_agent")
        if primary_agent:
            state["primary_agent"] = primary_agent
        primary_return = metrics.get(f"{phase}/primary_return")
        if self._is_number(primary_return):
            state["primary_return"] = float(primary_return)

        agents: Dict[str, Any] = state.setdefault("agents", {})
        self._update_agent_metrics(agents, metrics, phase, episode)

    def _update_agent_metrics(
        self,
        agents: Dict[str, Any],
        metrics: Mapping[str, Any],
        phase: str,
        episode: Optional[int],
    ) -> None:
        return_prefix = f"{phase}/return_"
        collisions_prefix = f"{phase}/collision_count_"
        speed_prefix = f"{phase}/avg_speed_"

        for key, value in metrics.items():
            if key.startswith(return_prefix):
                agent_id = key[len(return_prefix) :]
                self._record_agent_return(agents, agent_id, value, episode)
            elif key.startswith(collisions_prefix):
                agent_id = key[len(collisions_prefix) :]
                self._record_agent_collision(agents, agent_id, value)
            elif key.startswith(speed_prefix):
                agent_id = key[len(speed_prefix) :]
                self._record_agent_speed(agents, agent_id, value)

    def _agent_entry(self, agents: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        if agent_id not in agents:
            agents[agent_id] = {
                "episodes": 0,
                "return_sum": 0.0,
                "last_episode": None,
                "last_return": None,
                "avg_return": None,
                "last_collisions": None,
                "last_speed": None,
            }
        return agents[agent_id]

    def _record_agent_return(
        self,
        agents: Dict[str, Any],
        agent_id: str,
        value: Any,
        episode: Optional[int],
    ) -> None:
        if not self._is_number(value):
            return
        entry = self._agent_entry(agents, agent_id)
        numeric = float(value)
        entry["last_return"] = numeric

        if episode is None:
            return
        last_episode = entry.get("last_episode")
        if last_episode == episode:
            return

        entry["episodes"] = int(entry.get("episodes", 0)) + 1
        entry["return_sum"] = float(entry.get("return_sum", 0.0)) + numeric
        entry["last_episode"] = episode
        count = entry["episodes"]
        if count > 0:
            entry["avg_return"] = entry["return_sum"] / count

    def _record_agent_collision(
        self, agents: Dict[str, Any], agent_id: str, value: Any
    ) -> None:
        if not self._is_number(value):
            return
        entry = self._agent_entry(agents, agent_id)
        entry["last_collisions"] = float(value)

    def _record_agent_speed(
        self, agents: Dict[str, Any], agent_id: str, value: Any
    ) -> None:
        if not self._is_number(value):
            return
        entry = self._agent_entry(agents, agent_id)
        entry["last_speed"] = float(value)

    def _log_train(self, metrics: Mapping[str, Any], step: Optional[float]) -> None:
        episode = self._coerce_int(metrics.get("train/episode", step))
        total = self._coerce_int(metrics.get("train/total_episodes"))
        header = f"[TRAIN {episode:03d}]" if episode is not None else "[TRAIN]"
        if episode is not None and total is not None and total > 0:
            header = f"[TRAIN {episode:03d}/{total}]"

        mode = metrics.get("train/reward_task") or metrics.get("train/reward_mode")
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
