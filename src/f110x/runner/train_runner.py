"""Training runner orchestrating engine rollouts and trainer updates."""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Mapping, Tuple

from f110x.federated import FederatedAverager, FederatedConfig
from f110x.engine.rollout import (
    BestReturnTracker,
    IdleTerminationTracker,
    build_trajectory_buffers,
    run_episode,
)
from f110x.runner.context import RunnerContext
from f110x.runner.eval_runner import EvalRunner
from f110x.runner.plot_logger import PlotArtifactLogger, resolve_episode_cause_code, resolve_run_suffix
from f110x.runner.rollout_helpers import build_rollout_hooks
from f110x.trainer.base import Trainer
from f110x.utils.builders import AgentBundle, AgentTeam
from f110x.utils.logger import Logger
from f110x.utils.output import resolve_output_dir, resolve_output_file
from f110x.utils.start_pose import StartPoseOption


TrainerUpdateHook = Callable[[str, Trainer, Optional[Dict[str, Any]]], None]


class SpawnCurriculumManager:
    """Gate random spawn usage based on recent success rate."""

    def __init__(self, context: RunnerContext, config: Dict[str, Any], logger: Logger) -> None:
        self.context = context
        self.logger = logger
        self.config = dict(config)

        self.success_window = max(1, int(self.config.get("success_window", self.config.get("window", 200))))
        self.activation_samples = max(1, int(self.config.get("activation_samples", self.success_window)))
        self.min_episode = max(0, int(self.config.get("min_episode", 0)))
        self.enable_rate = float(self.config.get("enable_rate", self.config.get("threshold", 0.7)))
        self.enable_patience = max(1, int(self.config.get("enable_patience", self.config.get("confirm_patience", 3))))
        self.disable_rate = float(self.config.get("disable_rate", self.config.get("revert_rate", 0.5)))
        self.disable_patience = max(1, int(self.config.get("disable_patience", self.config.get("revert_patience", 2))))
        self.cooldown = max(0, int(self.config.get("cooldown", 0)))
        self.persist = bool(self.config.get("persist", False))
        self.start_enabled = bool(self.config.get("start_enabled", False))

        self._success_history: Deque[float] = deque(maxlen=self.success_window)
        self._structured_history: Deque[float] = deque(maxlen=self.success_window)
        self._random_history: Deque[float] = deque(maxlen=self.success_window)
        self._enable_streak = 0
        self._disable_streak = 0
        self._last_toggle_episode: Optional[int] = None

        all_options = list(context.start_pose_options or [])
        self._all_options: List[StartPoseOption] = list(all_options)
        self._random_options: List[StartPoseOption] = [opt for opt in all_options if self._is_random_option(opt)]
        random_ids = {id(opt) for opt in self._random_options}
        self._baseline_options: List[StartPoseOption] = [opt for opt in all_options if id(opt) not in random_ids]
        if not self._baseline_options and all_options:
            self._baseline_options = list(all_options)

        self.random_capable = bool(self._random_options)
        self.stage_configs = self._compile_stage_configs(self.config.get("stages"))
        if self.stage_configs and not self.random_capable:
            self.stage_configs = []
        self.stage_mode = bool(self.stage_configs)
        self.current_stage_index = 0
        self._stage_histories: Dict[int, Deque[float]] = {}
        self._promote_streak = 0
        self._regress_streak = 0
        self.random_enabled = False
        if self.stage_mode:
            self._apply_stage(0, force=True)
        else:
            initial_state = self.start_enabled and self.random_capable
            self._apply_state(initial_state, force=True)

    @staticmethod
    def _is_random_option(option: StartPoseOption) -> bool:
        metadata = getattr(option, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return False
        if metadata.get("spawn_random_pool"):
            return True
        option_id = metadata.get("spawn_option_id")
        if isinstance(option_id, str) and option_id.strip().lower() == "spawn_random":
            return True
        return False

    @property
    def window(self) -> int:
        return self.success_window

    @property
    def success_rate(self) -> Optional[float]:
        if not self._success_history:
            return None
        return float(sum(self._success_history) / len(self._success_history))

    @staticmethod
    def _rate(history: Deque[float]) -> Optional[float]:
        if not history:
            return None
        return float(sum(history) / len(history))

    def _apply_state(self, enable: bool, *, force: bool = False) -> bool:
        target = bool(enable and self.random_capable)
        previous = self.random_enabled
        self.random_enabled = target
        state_changed = previous != target
        if state_changed:
            if self.random_enabled:
                self._random_history.clear()
            else:
                self._structured_history.clear()
        if target:
            selected = list(self._random_options or self._baseline_options or self._all_options)
        else:
            selected = list(self._baseline_options or self._random_options or self._all_options)

        self.context.start_pose_options = selected or None
        return state_changed and not force

    def _compile_stage_configs(self, raw_stages: Any) -> List[Dict[str, Any]]:
        configs: List[Dict[str, Any]] = []
        if raw_stages is None:
            return configs
        if isinstance(raw_stages, Mapping):
            raw_iter = [raw_stages]
        elif isinstance(raw_stages, Iterable):
            raw_iter = raw_stages
        else:
            return configs
        for idx, entry in enumerate(raw_iter):
            if not isinstance(entry, Mapping):
                continue
            option_ids_raw = entry.get("option_ids") or entry.get("options") or entry.get("ids")
            if option_ids_raw is None:
                option_ids = ()
            elif isinstance(option_ids_raw, (list, tuple, set)):
                option_ids = tuple(
                    str(value).strip()
                    for value in option_ids_raw
                    if isinstance(value, (str, int, float)) and str(value).strip()
                )
            else:
                option_ids = (str(option_ids_raw).strip(),)
            if option_ids == ("",):
                option_ids = ()
            max_stage_raw = entry.get("max_stage", entry.get("stage"))
            try:
                max_stage = int(max_stage_raw) if max_stage_raw is not None else None
            except (TypeError, ValueError):
                max_stage = None
            include_baseline = bool(entry.get("include_baseline", True))
            configs.append(
                {
                    "name": entry.get("name") or f"stage_{idx + 1}",
                    "option_ids": tuple(value for value in option_ids if value),
                    "max_stage": max_stage,
                    "include_baseline": include_baseline,
                    "enable_rate": entry.get("enable_rate"),
                    "enable_patience": entry.get("enable_patience"),
                    "disable_rate": entry.get("disable_rate"),
                    "disable_patience": entry.get("disable_patience"),
                }
            )
        return configs

    @staticmethod
    def _option_stage_id(option: StartPoseOption) -> Optional[str]:
        metadata = getattr(option, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            return None
        for key in ("spawn_curriculum_id", "curriculum_id", "option_id", "id"):
            value = metadata.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
        option_id = metadata.get("spawn_option_id")
        if (
            isinstance(option_id, str)
            and option_id.strip()
            and option_id.strip().lower() not in {"spawn_random"}
        ):
            return option_id.strip()
        return None

    @staticmethod
    def _option_stage_level(option: StartPoseOption) -> Optional[int]:
        metadata = getattr(option, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            return None
        value = metadata.get("spawn_curriculum_stage", metadata.get("curriculum_stage"))
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _match_stage_option(self, option: StartPoseOption, stage_cfg: Mapping[str, Any]) -> bool:
        option_ids: Tuple[str, ...] = tuple(stage_cfg.get("option_ids") or ())
        max_stage = stage_cfg.get("max_stage")
        if option_ids:
            stage_id = self._option_stage_id(option)
            if stage_id and stage_id in option_ids:
                return True
        if max_stage is not None:
            try:
                threshold = int(max_stage)
            except (TypeError, ValueError):
                threshold = None
            if threshold is not None:
                stage_level = self._option_stage_level(option)
                if stage_level is not None and stage_level <= threshold:
                    return True
        if not option_ids and max_stage is None:
            return True
        return False

    def _options_for_stage(self, index: int) -> List[StartPoseOption]:
        if not self.stage_configs or index <= 0:
            return list(self._baseline_options or self._all_options)
        stage_cfg = self.stage_configs[index - 1]
        include_baseline = bool(stage_cfg.get("include_baseline", True))
        selected: List[StartPoseOption] = []
        if include_baseline and self._baseline_options:
            selected.extend(self._baseline_options)
        matched = [
            opt
            for opt in self._random_options
            if self._match_stage_option(opt, stage_cfg)
        ]
        if not matched and not selected:
            matched = list(self._random_options or self._baseline_options or self._all_options)
        selected.extend(matched)
        if not selected:
            return list(self._all_options)
        return selected

    def _stage_history(self, index: int) -> Deque[float]:
        history = self._stage_histories.get(index)
        if history is None:
            history = deque(maxlen=self.success_window)
            self._stage_histories[index] = history
        return history

    def _stage_name(self, index: int) -> str:
        if index <= 0:
            return "structured"
        if 0 < index <= len(self.stage_configs):
            return str(self.stage_configs[index - 1].get("name") or f"stage_{index}")
        return f"stage_{index}"

    def _stage_state(
        self,
        changed: bool,
        *,
        success_rate: Optional[float],
        stage_success_rate: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "changed": changed,
            "enabled": self.current_stage_index > 0,
            "stage": self._stage_name(self.current_stage_index),
            "stage_index": self.current_stage_index,
            "success_rate": success_rate,
            "stage_success_rate": stage_success_rate,
            "structured_success_rate": None,
            "random_success_rate": None,
        }

    def _apply_stage(self, index: int, *, episode: Optional[int] = None, force: bool = False) -> None:
        index = max(0, min(index, len(self.stage_configs)))
        previous = getattr(self, "current_stage_index", 0)
        self.current_stage_index = index
        selected = self._options_for_stage(index)
        self.context.start_pose_options = selected or None
        self.random_enabled = index > 0
        self._stage_history(index).clear()
        if not force and index != previous:
            self.logger.info(
                "Spawn curriculum stage transition",
                extra={
                    "episode": episode,
                    "stage": self._stage_name(index),
                    "stage_index": index,
                },
            )
            self._last_toggle_episode = episode
        self._promote_streak = 0
        self._regress_streak = 0

    def _cooldown_ready(self, episode: int) -> bool:
        if self.cooldown <= 0:
            return True
        if self._last_toggle_episode is None:
            return True
        return (episode - self._last_toggle_episode) >= self.cooldown

    def observe(self, episode: int, success: Optional[bool]) -> Dict[str, Any]:
        if self.stage_mode:
            return self._observe_stage(episode, success)
        if success is not None:
            value = 1.0 if success else 0.0
            self._success_history.append(value)
            if self.random_enabled:
                self._random_history.append(value)
            else:
                self._structured_history.append(value)

        success_rate = self.success_rate
        structured_rate = self._rate(self._structured_history)
        random_rate = self._rate(self._random_history)
        stage_success_rate = random_rate if self.random_enabled else structured_rate
        history_ready = len(self._success_history) >= self.activation_samples
        changed = False

        if not self.random_capable:
            return {
                "changed": False,
                "enabled": self.random_enabled,
                "success_rate": success_rate,
                "stage_success_rate": stage_success_rate,
                "structured_success_rate": structured_rate,
                "random_success_rate": random_rate,
                "stage": "random" if self.random_enabled else "structured",
            }

        if not self.random_enabled:
            if episode < self.min_episode or not history_ready or success_rate is None:
                self._enable_streak = 0
            else:
                if success_rate >= self.enable_rate:
                    self._enable_streak += 1
                else:
                    self._enable_streak = 0

                if self._enable_streak >= self.enable_patience:
                    changed = self._apply_state(True)
                    self._last_toggle_episode = episode
                    self._enable_streak = 0
                    self._disable_streak = 0
        else:
            if not self.persist and history_ready and success_rate is not None:
                if success_rate < self.disable_rate:
                    self._disable_streak += 1
                else:
                    self._disable_streak = 0

                cooldown_ready = self._cooldown_ready(episode)
                if cooldown_ready and self._disable_streak >= self.disable_patience:
                    changed = self._apply_state(False)
                    self._last_toggle_episode = episode
                    self._disable_streak = 0
                    self._enable_streak = 0

        if changed:
            self.logger.info(
                "Spawn curriculum transition",
                extra={
                    "episode": episode,
                    "stage": "random" if self.random_enabled else "structured",
                    "success_rate": float(success_rate) if success_rate is not None else None,
                },
            )

        return {
            "changed": changed,
            "enabled": self.random_enabled,
            "success_rate": success_rate,
            "stage_success_rate": stage_success_rate,
            "structured_success_rate": structured_rate,
            "random_success_rate": random_rate,
            "stage": "random" if self.random_enabled else "structured",
        }

    def _observe_stage(self, episode: int, success: Optional[bool]) -> Dict[str, Any]:
        if success is not None:
            value = 1.0 if success else 0.0
            self._success_history.append(value)
            self._stage_history(self.current_stage_index).append(value)

        success_rate = self.success_rate
        stage_history = self._stage_history(self.current_stage_index)
        stage_success_rate = self._rate(stage_history)
        history_ready = len(self._success_history) >= self.activation_samples
        changed = False

        if not self.random_capable:
            return self._stage_state(
                changed,
                success_rate=success_rate,
                stage_success_rate=stage_success_rate,
            )

        if episode < self.min_episode or not history_ready or success_rate is None:
            self._promote_streak = 0
            self._regress_streak = 0
            return self._stage_state(
                changed,
                success_rate=success_rate,
                stage_success_rate=stage_success_rate,
            )

        if self.current_stage_index < len(self.stage_configs):
            next_index = self.current_stage_index + 1
            next_cfg = self.stage_configs[next_index - 1]
            enable_rate = float(next_cfg.get("enable_rate", self.enable_rate))
            enable_patience = int(next_cfg.get("enable_patience", self.enable_patience) or self.enable_patience)
            enable_patience = max(1, enable_patience)
            if success_rate >= enable_rate:
                self._promote_streak += 1
            else:
                self._promote_streak = 0
            if self._promote_streak >= enable_patience and self._cooldown_ready(episode):
                prev_index = self.current_stage_index
                self._apply_stage(next_index, episode=episode)
                changed = changed or (self.current_stage_index != prev_index)

        if (
            not self.persist
            and self.current_stage_index > 0
            and self.current_stage_index <= len(self.stage_configs)
        ):
            current_cfg = self.stage_configs[self.current_stage_index - 1]
            disable_rate = float(current_cfg.get("disable_rate", self.disable_rate))
            disable_patience = int(current_cfg.get("disable_patience", self.disable_patience) or self.disable_patience)
            disable_patience = max(1, disable_patience)
            if success_rate < disable_rate:
                self._regress_streak += 1
            else:
                self._regress_streak = 0
            if self._regress_streak >= disable_patience and self._cooldown_ready(episode):
                prev_index = self.current_stage_index
                self._apply_stage(self.current_stage_index - 1, episode=episode)
                changed = changed or (self.current_stage_index != prev_index)

        return self._stage_state(
            changed,
            success_rate=success_rate,
            stage_success_rate=stage_success_rate,
        )


class DefenderCurriculumManager:
    """Schedules defender heuristic parameters using attacker outcomes."""

    def __init__(
        self,
        *,
        controller: Any,
        config: Mapping[str, Any],
        logger: Logger,
    ) -> None:
        self.controller = controller
        self.logger = logger
        self.config = dict(config or {})

        self.success_window = max(1, int(self.config.get("success_window", self.config.get("window", 200))))
        self.activation_samples = max(1, int(self.config.get("activation_samples", self.success_window)))
        self.min_episode = max(0, int(self.config.get("min_episode", 0)))
        self.cooldown = max(0, int(self.config.get("cooldown", 0)))
        self.persist = bool(self.config.get("persist", False))
        self.enable_rate_default = float(self.config.get("enable_rate", 0.7))
        self.disable_rate_default = float(self.config.get("disable_rate", 0.5))
        self.enable_patience_default = max(1, int(self.config.get("enable_patience", 3)))
        self.disable_patience_default = max(1, int(self.config.get("disable_patience", 2)))

        raw_stages = self.config.get("stages") or []
        stages: List[Dict[str, Any]] = []
        for idx, stage in enumerate(raw_stages):
            if isinstance(stage, Mapping):
                entry = dict(stage)
            else:
                continue
            entry.setdefault("name", f"stage_{idx}")
            params = entry.get("params")
            if params is not None and not isinstance(params, Mapping):
                raise TypeError("Defender curriculum stage 'params' must be a mapping")
            stages.append(entry)
        if not stages:
            stages = [{"name": "baseline"}]
        baseline_params = stages[0].get("params")
        if not isinstance(baseline_params, Mapping) or not baseline_params:
            stages[0]["params"] = self._capture_params()
        else:
            stages[0]["params"] = dict(baseline_params)

        self.stages = stages
        self.history: Deque[float] = deque(maxlen=self.success_window)
        self.stage_histories: Dict[int, Deque[float]] = {}
        self.current_stage_index = 0
        self._last_change_episode: Optional[int] = None
        self._promote_streak = 0
        self._regress_streak = 0
        self._apply_stage(0, episode=0, initial=True)

    @property
    def window(self) -> int:
        return self.success_window

    @property
    def success_rate(self) -> Optional[float]:
        if not self.history:
            return None
        return float(sum(self.history) / len(self.history))

    def _stage_history(self, index: int) -> Deque[float]:
        history = self.stage_histories.get(index)
        if history is None:
            history = deque(maxlen=self.success_window)
            self.stage_histories[index] = history
        return history

    @staticmethod
    def _rate(history: Deque[float]) -> Optional[float]:
        if not history:
            return None
        return float(sum(history) / len(history))

    def _capture_params(self) -> Dict[str, Any]:
        exporter = getattr(self.controller, "export_config", None)
        if callable(exporter):
            snapshot = exporter()
        else:
            keys = ("max_speed", "min_speed", "steering_gain", "bubble_radius", "steer_smooth")
            snapshot = {key: getattr(self.controller, key) for key in keys if hasattr(self.controller, key)}
        return self._sanitize_params(snapshot)

    @staticmethod
    def _sanitize_params(params: Mapping[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        return cleaned

    def _apply_stage(self, index: int, *, episode: int, initial: bool = False) -> None:
        index = max(0, min(index, len(self.stages) - 1))
        stage = self.stages[index]
        params = stage.get("params")
        if isinstance(params, Mapping):
            self.controller.apply_config(params)
        applied = self._capture_params()
        stage["_applied_params"] = applied
        stage_name = stage.get("name", f"stage_{index}")
        previous_index = getattr(self, "current_stage_index", 0)
        self.current_stage_index = index
        self._promote_streak = 0
        self._regress_streak = 0
        if initial:
            self._last_change_episode = None
            return
        self._stage_history(index).clear()
        self._last_change_episode = episode
        if index != previous_index:
            self.logger.info(
                "Defender curriculum transition",
                extra={
                    "episode": episode,
                    "stage": stage_name,
                    "stage_index": index,
                    "defender_params": applied,
                },
            )

    def _cooldown_ready(self, episode: int) -> bool:
        if self.cooldown <= 0:
            return True
        if self._last_change_episode is None:
            return True
        return (episode - self._last_change_episode) >= self.cooldown

    def observe(self, episode: int, success: Optional[bool]) -> Dict[str, Any]:
        if success is not None:
            value = 1.0 if success else 0.0
            self.history.append(value)
            self._stage_history(self.current_stage_index).append(value)

        success_rate = self.success_rate
        history_ready = len(self.history) >= self.activation_samples
        changed = False

        if len(self.stages) <= 1:
            return self._state(changed, success_rate)

        if not history_ready or success_rate is None or episode < self.min_episode:
            self._promote_streak = 0
            self._regress_streak = 0
            return self._state(changed, success_rate)

        if self.current_stage_index < len(self.stages) - 1:
            next_stage_index = self.current_stage_index + 1
            next_stage = self.stages[next_stage_index]
            enable_rate = float(next_stage.get("enable_rate", self.enable_rate_default))
            enable_patience = max(1, int(next_stage.get("enable_patience", self.enable_patience_default)))
            if success_rate >= enable_rate:
                self._promote_streak += 1
            else:
                self._promote_streak = 0
            if self._promote_streak >= enable_patience and self._cooldown_ready(episode):
                self._apply_stage(next_stage_index, episode=episode)
                changed = True

        if not self.persist and self.current_stage_index > 0:
            current_stage = self.stages[self.current_stage_index]
            disable_rate = float(current_stage.get("disable_rate", self.disable_rate_default))
            disable_patience = max(1, int(current_stage.get("disable_patience", self.disable_patience_default)))
            if success_rate < disable_rate:
                self._regress_streak += 1
            else:
                self._regress_streak = 0
            if self._regress_streak >= disable_patience and self._cooldown_ready(episode):
                self._apply_stage(self.current_stage_index - 1, episode=episode)
                changed = True

        return self._state(changed, success_rate)

    def _state(self, changed: bool, success_rate: Optional[float]) -> Dict[str, Any]:
        stage = self.stages[self.current_stage_index]
        stage_history = self.stage_histories.get(self.current_stage_index)
        return {
            "changed": changed,
            "stage_index": self.current_stage_index,
            "stage": stage.get("name", f"stage_{self.current_stage_index}"),
            "success_rate": success_rate,
            "stage_success_rate": self._rate(stage_history) if stage_history else None,
            "params": dict(stage.get("_applied_params", {})),
        }


@dataclass
class TrainRunner:
    """Compose trainers, environments, and reward helpers for training loops."""

    context: RunnerContext
    best_model_path: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    trainer_map: Dict[str, Trainer] = field(init=False)
    _primary_bundle: AgentBundle = field(init=False)
    _primary_trainer: Optional[Trainer] = field(init=False, default=None)
    _logger: Logger = field(init=False)
    _eval_interval: int = field(init=False, default=0)
    _eval_episodes: int = field(init=False, default=1)
    _eval_runner: Optional[EvalRunner] = field(init=False, default=None)
    _federated_cfg: Optional[FederatedConfig] = field(init=False, default=None)
    _federated: Optional[FederatedAverager] = field(init=False, default=None)
    _federated_interval: int = field(init=False, default=0)
    _federated_round: int = field(init=False, default=0)
    _run_suffix: Optional[str] = field(init=False, default=None)
    _plot_logger: PlotArtifactLogger = field(init=False)
    _pressure_metric_distance: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:  # noqa: D401 - behaviour described in class docstring
        self._ensure_primary_agent()
        self._primary_bundle = self.context.primary_bundle
        try:
            self._primary_trainer = self.context.primary_trainer
        except RuntimeError:
            self._primary_trainer = None

        self.trainer_map = dict(self.context.trainer_map)
        self._run_suffix = resolve_run_suffix(getattr(self.context, "metadata", {}))
        self._plot_logger = PlotArtifactLogger(self.context, run_suffix=self._run_suffix)
        self._configure_output_paths()
        self._logger = self.context.logger
        self._apply_warm_start()
        self._trainer_stats: Dict[str, Dict[str, Any]] = {
            trainer_id: {} for trainer_id in self.trainer_map
        }
        interval_value = getattr(self.context, "eval_interval", 0)
        try:
            interval_int = int(interval_value)
        except (TypeError, ValueError):
            interval_int = 0
        self._eval_interval = max(interval_int, 0)
        episodes_value = getattr(self.context, "eval_episodes", 1)
        try:
            episodes_int = int(episodes_value)
        except (TypeError, ValueError):
            episodes_int = 1
        self._eval_episodes = max(episodes_int, 1)
        self._eval_runner = None
        self._plot_logger.write_run_config_snapshot(self._run_suffix)
        reward_params = self.context.reward_cfg.get("params")
        if not isinstance(reward_params, Mapping):
            reward_params = {}
        pressure_distance = reward_params.get("pressure_distance", 1.0)
        try:
            self._pressure_metric_distance = float(pressure_distance)
        except (TypeError, ValueError):
            self._pressure_metric_distance = 1.0
        if self._pressure_metric_distance <= 0:
            self._pressure_metric_distance = 1.0
        self._init_federated()

    def _apply_warm_start(self) -> None:
        try:
            warm_start_path = self.context.cfg.main.get("warm_start_checkpoint")
        except Exception:
            warm_start_path = None
        if not warm_start_path:
            return
        candidate = Path(warm_start_path).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists():
            self._logger.warning(
                "Warm-start checkpoint not found",
                extra={"path": str(candidate)},
            )
            return
        target_ids = list(self.context.trainable_agent_ids or self.trainer_map.keys())
        loaded_any = False
        for agent_id in target_ids:
            trainer = self.trainer_map.get(agent_id)
            if trainer is None:
                continue
            try:
                trainer.load(str(candidate))
            except Exception as exc:
                self._logger.warning(
                    "Failed to warm start trainer",
                    extra={"agent_id": agent_id, "path": str(candidate), "error": str(exc)},
                )
                continue
            loaded_any = True
            self._logger.info(
                "Warm-started trainer from checkpoint",
                extra={"agent_id": agent_id, "path": str(candidate)},
            )
        if not loaded_any:
            self._logger.warning(
                "Warm-start checkpoint ignored; no trainable trainers available",
                extra={"path": str(candidate)},
            )

    # ------------------------------------------------------------------
    def _init_federated(self) -> None:
        self._federated_cfg = None
        self._federated = None
        self._federated_interval = 0
        self._federated_round = 0

        try:
            base_cfg = dict(self.context.cfg.main.federated or {})
        except Exception:
            base_cfg = {}

        if not base_cfg or not base_cfg.get("enabled"):
            return

        env_enabled = os.environ.get("FED_ENABLED")
        if env_enabled and env_enabled.strip().lower() in {"0", "false", "off", "no"}:
            return

        total_raw = os.environ.get("FED_TOTAL_CLIENTS")
        client_raw = os.environ.get("FED_CLIENT_ID")
        if total_raw is None or client_raw is None:
            self._logger.warning(
                "Federated averaging enabled but FED_TOTAL_CLIENTS or FED_CLIENT_ID not set; disabling federated sync.",
                extra={"federated_enabled": True},
            )
            return

        try:
            total_clients = max(int(total_raw), 1)
            client_id = int(client_raw)
        except ValueError:
            self._logger.warning(
                "Invalid federated client identifiers; disabling federated sync.",
                extra={"FED_TOTAL_CLIENTS": total_raw, "FED_CLIENT_ID": client_raw},
            )
            return

        overrides = dict(base_cfg)
        env_interval = os.environ.get("FED_ROUND_INTERVAL") or os.environ.get("FED_INTERVAL")
        if env_interval:
            try:
                overrides["interval"] = int(env_interval)
            except ValueError:
                pass
        env_root = os.environ.get("FED_ROOT")
        if env_root:
            overrides["root"] = env_root
        env_agents = os.environ.get("FED_AGENTS")
        if env_agents:
            overrides["agents"] = [token.strip() for token in env_agents.split(",") if token.strip()]
        env_mode = os.environ.get("FED_AVG_MODE")
        if env_mode:
            overrides["mode"] = env_mode
        env_timeout = os.environ.get("FED_TIMEOUT")
        if env_timeout:
            try:
                overrides["timeout"] = float(env_timeout)
            except ValueError:
                pass
        env_weights = os.environ.get("FED_WEIGHTS")
        if env_weights:
            mapping: Dict[Any, float] = {}
            ordered: List[float] = []
            saw_mapping = False
            for token in env_weights.split(","):
                token = token.strip()
                if not token:
                    continue
                if ":" in token:
                    key, raw_val = token.split(":", 1)
                    try:
                        mapping[key.strip()] = float(raw_val)
                        saw_mapping = True
                    except ValueError:
                        continue
                else:
                    try:
                        ordered.append(float(token))
                    except ValueError:
                        continue
            if saw_mapping and mapping:
                overrides["weights"] = mapping
            elif ordered:
                overrides["weights"] = ordered
        env_checkpoint = os.environ.get("FED_CHECKPOINT_AFTER_SYNC")
        if env_checkpoint:
            if env_checkpoint.strip().lower() in {"0", "false", "off", "no"}:
                overrides["checkpoint_after_sync"] = False
            else:
                overrides["checkpoint_after_sync"] = True
        env_opt_strategy = os.environ.get("FED_OPTIMIZER_STRATEGY")
        if env_opt_strategy:
            overrides["optimizer_strategy"] = env_opt_strategy

        base_dir = getattr(self.context, "output_root", None)
        if base_dir is None:
            base_dir_path = self.checkpoint_dir.parent
        else:
            base_dir_path = Path(base_dir)

        try:
            fed_cfg = FederatedConfig.from_mapping(overrides, base_dir=base_dir_path)
        except Exception as exc:
            self._logger.warning(
                "Failed to initialise federated configuration",
                extra={"error": str(exc)},
            )
            return

        if fed_cfg.mode and fed_cfg.mode not in {"mean"}:
            self._logger.warning(
                f"Unsupported federated averaging mode '{fed_cfg.mode}'; defaulting to 'mean'."
            )
            fed_cfg = FederatedConfig(
                enabled=fed_cfg.enabled,
                interval=fed_cfg.interval,
                agents=fed_cfg.agents,
                root=fed_cfg.root,
                mode="mean",
                timeout=fed_cfg.timeout,
                weights=fed_cfg.weights,
            )

        if not fed_cfg.agents:
            self._logger.warning(
                "Federated averaging enabled but no agents listed; disabling federated sync."
            )
            return

        self._federated_cfg = fed_cfg
        self._federated_interval = max(fed_cfg.interval, 1)
        try:
            self._federated = FederatedAverager(
                fed_cfg,
                client_id=client_id,
                total_clients=total_clients,
                logger=self._logger,
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to construct federated averager; disabling federated sync.",
                extra={"error": str(exc)},
            )
            self._federated = None
            self._federated_cfg = None
            self._federated_interval = 0
            return

        self._logger.info(
            "Federated averaging enabled",
            extra={
                "clients": total_clients,
                "client_id": client_id,
                "interval": self._federated_interval,
                "agents": list(fed_cfg.agents),
                "root": str(fed_cfg.root),
            },
        )

    def _run_federated_sync(self, episode: int, trainers: Mapping[str, Trainer]) -> None:
        if not self._federated or not self._federated_interval:
            return

        next_round = self._federated_round + 1
        try:
            metrics = self._federated.sync(trainers, next_round)
        except TimeoutError as exc:
            self._logger.warning(
                "Federated averaging timed out",
                extra={"episode": episode, "round": next_round, "error": str(exc)},
            )
            self._federated = None
            return
        except Exception as exc:
            self._logger.warning(
                "Federated averaging failed",
                extra={"episode": episode, "round": next_round, "error": str(exc)},
            )
            return

        self._federated_round = next_round
        if not metrics:
            return
        metrics = dict(metrics)
        metrics.setdefault("federated/round", float(next_round))
        metrics.setdefault("federated/episode", float(episode))
        if self._federated_cfg:
            metrics.setdefault("federated/optimizer_strategy", self._federated_cfg.optimizer_strategy)
            metrics.setdefault(
                "federated/checkpoint_enabled",
                1.0 if self._federated_cfg.checkpoint_after_sync else 0.0,
            )
            metrics.setdefault("federated/interval", float(self._federated_interval))
        try:
            self.team.reset_actions()
        except Exception:
            pass
        for trainer in trainers.values():
            reset_fn = getattr(trainer, "reset_noise_schedule", None)
            if callable(reset_fn):
                try:
                    reset_fn(restart=True)
                except TypeError:
                    reset_fn()
        checkpoint_saved = False
        if self._federated_cfg and self._federated_cfg.checkpoint_after_sync:
            checkpoint_saved = self._save_primary_model()
            if checkpoint_saved:
                metrics.setdefault("federated/checkpoint_saved", 1.0)

        if self._federated_cfg and self._federated_cfg.optimizer_strategy == "reset":
            for trainer in trainers.values():
                reset_opt = getattr(trainer, "reset_optimizers", None)
                if callable(reset_opt):
                    reset_opt()
        self._logger.log_metrics("federated", metrics, step=episode)

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    @property
    def team(self) -> AgentTeam:
        return self.context.team

    @property
    def env(self):
        return self.context.env

    @property
    def primary_agent_id(self) -> str:
        return self.context.primary_agent_id or self._primary_bundle.agent_id

    @property
    def primary_bundle(self) -> AgentBundle:
        return self._primary_bundle

    @property
    def primary_trainer(self) -> Optional[Trainer]:
        return self._primary_trainer

    @property
    def trainable_agent_ids(self) -> List[str]:
        return list(self.context.trainable_agent_ids)

    @property
    def opponent_agent_ids(self) -> List[str]:
        primary_id = self.primary_agent_id
        return [bundle.agent_id for bundle in self.team.agents if bundle.agent_id != primary_id]

    @property
    def roster_metadata(self) -> Dict[str, Any]:
        return {
            "agent_ids": [bundle.agent_id for bundle in self.team.agents],
            "roles": dict(self.team.roles),
            "trainable": list(self.context.trainable_agent_ids),
        }

    def run(
        self,
        *,
        episodes: int,
        update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        trainer_update_hook: Optional[TrainerUpdateHook] = None,
        update_start: int = 0,
    ) -> List[Dict[str, Any]]:
        env = self.env
        team = self.team
        trainer_map = self.trainer_map
        results: List[Dict[str, Any]] = []

        logger = self._logger

        spawn_manager: Optional[SpawnCurriculumManager] = None
        spawn_cfg = self.context.cfg.env.get("spawn_curriculum", {})
        if isinstance(spawn_cfg, dict) and spawn_cfg:
            spawn_manager = SpawnCurriculumManager(
                context=self.context,
                config=spawn_cfg,
                logger=logger,
            )

        primary_id = self.primary_agent_id
        primary_bundle = self.primary_bundle
        roster = team.roster

        attacker_id = team.primary_role("attacker")
        if attacker_id is None:
            attacker_id = primary_id or (roster.agent_ids[0] if roster.agent_ids else None)

        defender_id = team.primary_role("defender")
        if defender_id is None and attacker_id is not None:
            defender_id = roster.first_other(attacker_id)
        if defender_id is None:
            for bundle in team.agents:
                if attacker_id is None or bundle.agent_id != attacker_id:
                    defender_id = bundle.agent_id
                    break

        defender_manager: Optional[DefenderCurriculumManager] = None
        if defender_id:
            defender_bundle = getattr(team, "by_id", {}).get(defender_id)
            if defender_bundle is not None:
                curriculum_cfg = defender_bundle.metadata.get("policy_curriculum", {})
                controller = getattr(defender_bundle, "controller", None)
                if curriculum_cfg and hasattr(controller, "apply_config"):
                    defender_manager = DefenderCurriculumManager(
                        controller=controller,
                        config=curriculum_cfg,
                        logger=logger,
                    )

        primary_cfg = primary_bundle.metadata.get("config", {})
        recent_window = max(1, int(primary_cfg.get("rolling_avg_window", 10)))
        if spawn_manager is not None:
            recent_window = max(recent_window, spawn_manager.window)
        if defender_manager is not None:
            recent_window = max(recent_window, defender_manager.window)
        best_tracker = BestReturnTracker(recent_window)
        recent_returns: Deque[float] = deque(maxlen=recent_window)
        recent_success: Deque[float] = deque(maxlen=recent_window)
        total_successes: int = 0

        reward_cfg = self.context.reward_cfg
        truncation_penalty = self._resolve_reward_value(reward_cfg, "truncation_penalty")
        params_block = reward_cfg.get("params") if isinstance(reward_cfg.get("params"), dict) else {}

        def _extract_reward_param(name: str, default: float) -> float:
            value = reward_cfg.get(name)
            if value is None and isinstance(params_block, dict):
                value = params_block.get(name)
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        idle_speed_threshold = _extract_reward_param("idle_speed_threshold", 0.4)
        idle_patience_steps = int(round(_extract_reward_param("idle_patience_steps", 200)))
        idle_tracker = IdleTerminationTracker(
            idle_speed_threshold,
            idle_patience_steps,
            agent_ids=self.trainable_agent_ids,
        )
        reward_sharing_cfg = reward_cfg.get("shared_reward")

        trajectory_buffers, off_policy_ids = build_trajectory_buffers(team, trainer_map)

        hooks = build_rollout_hooks(
            self.context,
            team,
            env,
            deterministic=False,
        )
        reward_factory = hooks.reward_factory
        compute_actions = hooks.compute_actions
        prepare_next = hooks.prepare_next_observation
        reset_env = hooks.reset_fn

        def on_offpolicy_flush(agent_id: str, trainer: Trainer, buffer) -> None:
            for _ in range(buffer.updates_per_step):
                stats = trainer.update()
                self._record_trainer_stats(agent_id, stats)
                if trainer_update_hook:
                    trainer_update_hook(agent_id, trainer, stats)

        if self.context.render_interval:
            def should_render(ep_index: int, _step: int) -> bool:
                return (ep_index + 1) % self.context.render_interval == 0
        else:
            should_render = None

        agent_ids = list(env.possible_agents)
        update_after = max(1, int(self.context.update_after or 1))
        _ = update_start  # kept for compatibility with legacy callers

        total_episodes = int(episodes)
        logger.start({
            "mode": "train",
            "primary_agent": primary_id,
            "train/episodes_total": total_episodes,
        })
        logger.update_context(
            mode="train",
            primary_agent=primary_id,
            **{"train/episodes_total": total_episodes},
        )

        for episode_idx in range(int(episodes)):
            path_points: List[Tuple[int, int, str, float, float, float, float, Optional[float], Optional[float]]] = []
            rollout = run_episode(
                env=env,
                team=team,
                trainer_map=trainer_map,
                trajectory_buffers=trajectory_buffers,
                reward_wrapper_factory=reward_factory,
                compute_actions=compute_actions,
                prepare_next_observation=prepare_next,
                idle_tracker=idle_tracker,
                episode_index=episode_idx,
                reset_fn=reset_env,
                agent_ids=agent_ids,
                render_condition=should_render,
                on_offpolicy_flush=on_offpolicy_flush,
                path_logger=path_points,
                reward_sharing=reward_sharing_cfg,
                pressure_metric_distance=self._pressure_metric_distance,
            )

            returns = dict(rollout.returns)
            reward_breakdown = dict(rollout.reward_breakdown)
            finish_line_hits = dict(rollout.finish_line_hits or {})
            lap_counts: Dict[str, float] = {}
            lap_array = getattr(env, "lap_counts", None)
            id_to_index = getattr(env, "_agent_id_to_index", {})
            if lap_array is not None and isinstance(id_to_index, dict):
                for aid in agent_ids:
                    idx = id_to_index.get(aid)
                    if idx is None:
                        continue
                    if 0 <= idx < len(lap_array):
                        lap_counts[aid] = float(lap_array[idx])

            if truncation_penalty:
                for agent_id, truncated in rollout.truncations.items():
                    if not truncated:
                        continue
                    agent_breakdown = reward_breakdown.setdefault(agent_id, {})
                    if "truncation_penalty" in agent_breakdown:
                        continue
                    returns[agent_id] = returns.get(agent_id, 0.0) + truncation_penalty
                    agent_breakdown["truncation_penalty"] = (
                        agent_breakdown.get("truncation_penalty", 0.0) + truncation_penalty
                    )

            if rollout.idle_triggered:
                logger.info(
                    "Idle stop triggered",
                    extra={
                        "episode": episode_idx + 1,
                        "steps": rollout.steps,
                        "idle_patience_steps": idle_patience_steps,
                        "idle_speed_threshold": idle_speed_threshold,
                    },
                )

            epsilon_val = self._resolve_primary_epsilon()
            exploration_noise = self._resolve_primary_exploration_noise()

            defender_crashed: Optional[bool] = None
            defender_survival_steps: Optional[int] = None
            if defender_id is not None:
                defender_step = rollout.collision_steps.get(defender_id, -1)
                defender_crashed = defender_step >= 0
                defender_survival_steps = (
                    int(defender_step) if defender_crashed else rollout.steps
                )

            attacker_crashed: Optional[bool] = None
            if attacker_id is not None:
                attacker_step = rollout.collision_steps.get(attacker_id, -1)
                attacker_crashed = attacker_step >= 0

            target_finished: Optional[bool] = None
            if defender_id is not None:
                try:
                    target_laps = int(getattr(env, "target_laps", 1) or 1)
                except (TypeError, ValueError):
                    target_laps = 1
                target_finished = bool(finish_line_hits.get(defender_id, False)) or (
                    target_laps > 0 and float(lap_counts.get(defender_id, 0.0)) >= float(target_laps)
                )

            attacker_win: Optional[bool] = None
            if defender_crashed is not None and attacker_crashed is not None:
                attacker_win = bool(defender_crashed and not attacker_crashed)
            elif defender_crashed is not None:
                attacker_win = bool(defender_crashed)

            target_win: Optional[bool] = None
            if defender_crashed is not None and attacker_crashed is not None:
                if defender_crashed and attacker_crashed:
                    target_win = False
                else:
                    target_win = bool((not defender_crashed) and (bool(target_finished) or attacker_crashed))
            elif defender_crashed is not None:
                target_win = bool((not defender_crashed) and bool(target_finished))

            success: Optional[bool] = attacker_win

            attacker_components = reward_breakdown.get(attacker_id, {}) if attacker_id is not None else {}
            assisted_success: Optional[bool] = None
            episode_record_finish_agent: Optional[str] = None
            if success and attacker_id is not None:
                success_reward_val = float(attacker_components.get("success_reward", 0.0) or 0.0)
                kamikaze_reward_val = float(attacker_components.get("kamikaze_success", 0.0) or 0.0)
                assisted_success = (success_reward_val > 0.0) or (kamikaze_reward_val > 0.0)
                if not assisted_success:
                    logger.log_event(
                        "debug",
                        "Defender crash without explicit success reward",
                        extra={
                            "episode": episode_idx + 1,
                            "steps": rollout.steps,
                            "spawn_option": rollout.spawn_option,
                            "spawn_points": rollout.spawn_points,
                        },
                    )

            if not success and attacker_id is not None:
                success_reward_val = float(attacker_components.get("success_reward", 0.0) or 0.0)
                kamikaze_reward_val = float(attacker_components.get("kamikaze_success", 0.0) or 0.0)
                if success_reward_val > 0.0 or kamikaze_reward_val > 0.0:
                    success = True
                    assisted_success = True

            if episode_record_finish_agent is None and defender_id and finish_line_hits.get(defender_id):
                episode_record_finish_agent = defender_id

            if success is None:
                finish_agent = primary_id or (agent_ids[0] if agent_ids else None)
                if finish_agent is not None:
                    try:
                        target_laps = int(getattr(env, "target_laps", 1) or 1)
                    except (TypeError, ValueError):
                        target_laps = 1
                    if target_laps <= 0:
                        target_laps = 1
                    finish_hit = bool(finish_line_hits.get(finish_agent, False))
                    lap_count = float(lap_counts.get(finish_agent, 0.0))
                    success = finish_hit or lap_count >= float(target_laps)
                    if success and finish_hit and episode_record_finish_agent is None:
                        episode_record_finish_agent = finish_agent

            cause_code = resolve_episode_cause_code(
                success=bool(success),
                attacker_crashed=bool(attacker_crashed),
                defender_crashed=bool(defender_crashed),
                truncated=any(rollout.truncations.values()) or rollout.idle_triggered,
            )
            self._plot_logger.log_path_points(path_points, cause_code)

            if success:
                total_successes += 1
                logger.info(
                    "Gaplock success recorded",
                    extra={
                        "episode": episode_idx + 1,
                        "steps": rollout.steps,
                        "spawn_option": rollout.spawn_option,
                        "spawn_points": rollout.spawn_points,
                    },
                )

            collisions_total = int(sum(rollout.collisions.values()))
            episode_record: Dict[str, Any] = {
                "episode": episode_idx + 1,
                "steps": rollout.steps,
                "cause": rollout.cause,
                "reward_task": rollout.reward_task,
                "reward_mode": rollout.reward_mode,
                "returns": returns,
                "reward_breakdown": reward_breakdown,
                "success": success,
                "attacker_win": attacker_win,
                "target_win": target_win,
                "target_finished": target_finished,
                "assisted_success": assisted_success,
                "collisions_total": collisions_total,
                "idle_truncated": rollout.idle_triggered,
                "finish_line_hits": finish_line_hits,
                "lap_counts": dict(lap_counts),
            }
            if episode_record_finish_agent:
                episode_record["finish_line_agent"] = episode_record_finish_agent

            if defender_crashed is not None:
                episode_record["defender_crashed"] = defender_crashed
            if attacker_crashed is not None:
                episode_record["attacker_crashed"] = attacker_crashed
            if defender_survival_steps is not None:
                episode_record["defender_survival_steps"] = defender_survival_steps

            if rollout.spawn_points:
                episode_record["spawn_points"] = dict(rollout.spawn_points)
            if rollout.spawn_option is not None:
                episode_record["spawn_option"] = rollout.spawn_option
            if epsilon_val is not None:
                episode_record["epsilon"] = epsilon_val
            if exploration_noise is not None:
                episode_record["exploration_noise"] = exploration_noise

            for aid in agent_ids:
                episode_record[f"collision_count_{aid}"] = int(rollout.collisions.get(aid, 0))
                step_val = rollout.collision_steps.get(aid, -1)
                if step_val >= 0:
                    episode_record[f"collision_step_{aid}"] = int(step_val)
                episode_record[f"avg_speed_{aid}"] = float(rollout.average_speeds.get(aid, 0.0))

            primary_return = float(returns.get(primary_id, 0.0))
            recent_returns.append(primary_return)
            rolling_return = float(sum(recent_returns) / len(recent_returns))

            if success is not None:
                recent_success.append(1.0 if success else 0.0)
            success_rate = (
                float(sum(recent_success) / len(recent_success)) if recent_success else None
            )

            spawn_state: Optional[Dict[str, Any]] = None
            if spawn_manager is not None:
                spawn_state = spawn_manager.observe(episode_idx + 1, success)
            defender_state: Optional[Dict[str, Any]] = None
            if defender_manager is not None:
                defender_state = defender_manager.observe(episode_idx + 1, success)

            collision_rate = float(collisions_total) / max(float(rollout.steps), 1.0)
            best_average = best_tracker.best if best_tracker.best != float("-inf") else None
            buffer_fraction = None
            if primary_id:
                buffer_fraction = self._trainer_stats.get(primary_id, {}).get("buffer_fraction")
            if buffer_fraction is None:
                for stats_snapshot in self._trainer_stats.values():
                    candidate = stats_snapshot.get("buffer_fraction")
                    if candidate is not None:
                        buffer_fraction = candidate
                        break

            success_rate_total = float(total_successes) / float(episode_idx + 1)
            metrics: Dict[str, Any] = {
                "train/episode": float(episode_idx + 1),
                "train/episodes_total": float(total_episodes),
                "train/steps": float(rollout.steps),
                "train/return": primary_return,
                "train/return_mean": rolling_return,
                "train/collisions": float(collisions_total),
                "train/collision_rate": collision_rate,
                "train/idle": bool(rollout.idle_triggered),
                "train/reward_task": rollout.reward_task,
                "train/episode_cause": rollout.cause,
                "train/success_total": float(total_successes),
                "train/success_rate_total": success_rate_total,
            }
            if primary_id:
                metrics["train/primary_agent"] = primary_id
            if success is not None:
                metrics["train/success"] = bool(success)
                episode_record["success"] = bool(success)
            if assisted_success is not None:
                metrics["train/assisted_success"] = bool(assisted_success)
            if success_rate is not None:
                metrics["train/success_rate"] = success_rate
            if epsilon_val is not None:
                metrics["train/epsilon"] = float(epsilon_val)
            if exploration_noise is not None:
                metrics["train/exploration_noise"] = float(exploration_noise)
            if attacker_crashed is not None:
                metrics["train/attacker_crashed"] = bool(attacker_crashed)
            if defender_crashed is not None:
                metrics["train/defender_crashed"] = bool(defender_crashed)
            if attacker_win is not None:
                metrics["train/attacker_win"] = bool(attacker_win)
            if target_win is not None:
                metrics["train/target_win"] = bool(target_win)
            if target_finished is not None:
                metrics["train/target_finished"] = bool(target_finished)
            if defender_survival_steps is not None:
                metrics["train/defender_survival_steps"] = float(defender_survival_steps)
            attacker_metrics_id = attacker_id or primary_id
            avg_relative_distance = None
            pressure_ratio = None
            if attacker_metrics_id:
                avg_relative_distance = rollout.average_relative_distance.get(attacker_metrics_id)
                pressure_ratio = rollout.pressure_coverage.get(attacker_metrics_id)
            if avg_relative_distance is not None:
                metrics["train/avg_relative_distance"] = float(avg_relative_distance)
                episode_record["avg_relative_distance"] = float(avg_relative_distance)
            if pressure_ratio is not None:
                metrics["train/pressure_coverage"] = float(pressure_ratio)
                episode_record["pressure_coverage"] = float(pressure_ratio)
            if defender_survival_steps is not None:
                episode_record["defender_survival_steps"] = defender_survival_steps
            episode_record["success_rate_total"] = success_rate_total
            if best_average is not None:
                metrics["train/return_best"] = float(best_average)
            metrics["train/return_window"] = float(recent_returns.maxlen)
            if buffer_fraction is not None:
                metrics["train/buffer_fraction"] = float(buffer_fraction)
                episode_record["buffer_fraction"] = float(buffer_fraction)

            def _coerce_update_metric(value: Any) -> Optional[float]:
                if value is None:
                    return None
                if isinstance(value, bool):
                    return float(value)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(number):
                    return None
                return number

            for trainer_id, stats in self._trainer_stats.items():
                if not stats:
                    continue
                for key, value in stats.items():
                    numeric = _coerce_update_metric(value)
                    if numeric is None:
                        continue
                    metrics[f"train/update/{trainer_id}/{key}"] = numeric

            for agent_id, trainer in self.trainer_map.items():
                accessor = getattr(trainer, "exploration_noise", None)
                if not callable(accessor):
                    continue
                try:
                    value = accessor()
                except Exception:
                    continue
                if value is None:
                    continue
                try:
                    noise_value = float(value)
                except (TypeError, ValueError):
                    continue
                metrics[f"train/agent/{agent_id}/exploration_noise"] = noise_value
                episode_record[f"exploration_noise_{agent_id}"] = noise_value
            if spawn_state is not None:
                metrics["train/random_spawn_enabled"] = bool(spawn_state.get("enabled", False))
                stage_value = spawn_state.get("stage")
                if stage_value is not None:
                    metrics["train/spawn_stage"] = stage_value
                    episode_record["spawn_stage"] = stage_value
                spawn_rate = spawn_state.get("success_rate")
                if spawn_rate is not None:
                    metrics["train/spawn_success_rate"] = float(spawn_rate)
                    episode_record["spawn_success_rate"] = float(spawn_rate)
                stage_rate = spawn_state.get("stage_success_rate")
                if stage_rate is not None:
                    metrics["train/spawn_stage_success_rate"] = float(stage_rate)
                    episode_record["spawn_stage_success_rate"] = float(stage_rate)
                structured_rate = spawn_state.get("structured_success_rate")
                if structured_rate is not None:
                    metrics["train/spawn_structured_success_rate"] = float(structured_rate)
                random_rate = spawn_state.get("random_success_rate")
                if random_rate is not None:
                    metrics["train/spawn_random_success_rate"] = float(random_rate)
                episode_record["random_spawn_enabled"] = bool(spawn_state.get("enabled", False))
            if defender_state is not None:
                stage_name = defender_state.get("stage")
                if stage_name is not None:
                    metrics["train/defender_stage"] = stage_name
                    episode_record["defender_stage"] = stage_name
                stage_index = defender_state.get("stage_index")
                if stage_index is not None:
                    metrics["train/defender_stage_index"] = float(stage_index)
                    episode_record["defender_stage_index"] = int(stage_index)
                defender_success = defender_state.get("success_rate")
                if defender_success is not None:
                    metrics["train/defender_success_rate"] = float(defender_success)
                    episode_record["defender_success_rate"] = float(defender_success)
                defender_stage_success = defender_state.get("stage_success_rate")
                if defender_stage_success is not None:
                    metrics["train/defender_stage_success_rate"] = float(defender_stage_success)
                    episode_record["defender_stage_success_rate"] = float(defender_stage_success)
                params_snapshot = defender_state.get("params") or {}
                if params_snapshot:
                    episode_record["defender_params"] = dict(params_snapshot)
                    for key, value in params_snapshot.items():
                        if isinstance(value, (int, float)):
                            metrics[f"train/defender_param/{key}"] = float(value)
            episode_record["success_total"] = total_successes

            for aid, value in returns.items():
                metrics[f"train/agent/{aid}/return"] = float(value)
            for aid, count in rollout.collisions.items():
                metrics[f"train/agent/{aid}/collisions"] = float(count)
            for aid, speed in rollout.average_speeds.items():
                metrics[f"train/agent/{aid}/avg_speed"] = float(speed)
            for aid, step_val in rollout.collision_steps.items():
                if step_val >= 0:
                    metrics[f"train/agent/{aid}/collision_step"] = float(step_val)
            for aid, breakdown in reward_breakdown.items():
                for name, value in breakdown.items():
                    metrics[f"train/reward/{aid}/{name}"] = float(value)
            if finish_line_hits:
                metrics["train/finish_line_any"] = float(any(finish_line_hits.values()))
                for aid, hit in finish_line_hits.items():
                    metrics[f"train/finish_line_hit/{aid}"] = 1.0 if hit else 0.0

            logger.log_metrics("train", metrics, step=episode_idx + 1)
            publish_metrics = getattr(env, "update_render_metrics", None)
            if callable(publish_metrics):
                try:
                    publish_metrics("train", metrics, step=episode_idx + 1)
                except Exception:
                    pass
            self._plot_logger.log_episode_metrics(
                {
                    "episode": episode_idx + 1,
                    "steps": rollout.steps,
                    "success": int(success) if success is not None else "",
                    "attacker_win": int(attacker_win) if attacker_win is not None else "",
                    "target_win": int(target_win) if target_win is not None else "",
                    "target_finished": int(target_finished) if target_finished is not None else "",
                    "success_rate_window": success_rate if success_rate is not None else "",
                    "success_rate_total": success_rate_total,
                    "defender_survival_steps": defender_survival_steps if defender_survival_steps is not None else "",
                    "avg_relative_distance": avg_relative_distance if avg_relative_distance is not None else "",
                    "pressure_coverage": pressure_ratio if pressure_ratio is not None else "",
                    "collisions": collisions_total,
                    "cause_code": cause_code,
                }
            )

            logger.info(
                "Episode finished",
                extra={
                    "episode": episode_idx + 1,
                    "steps": rollout.steps,
                    "cause": rollout.cause or "unknown",
                },
            )

            results.append(episode_record)

            if update_callback:
                payload: Dict[str, Any] = {
                    "train/episode": float(episode_idx + 1),
                    "train/return": primary_return,
                }
                for aid, value in returns.items():
                    payload[f"train/agent/{aid}/return"] = float(value)
                update_callback(payload)

            if (episode_idx + 1) % update_after == 0:
                for trainer_id, trainer in trainer_map.items():
                    if trainer_id in off_policy_ids:
                        continue
                    stats = trainer.update()
                    self._record_trainer_stats(trainer_id, stats)
                    if trainer_update_hook:
                        trainer_update_hook(trainer_id, trainer, stats)

            ppo_return = returns.get(primary_id, 0.0)
            new_best_avg = best_tracker.observe(ppo_return)
            if new_best_avg is not None:
                saved = self._save_primary_model()
                if saved:
                    logger.info(
                        "New best model checkpoint",
                        extra={
                            "episode": episode_idx + 1,
                            "avg_return": float(new_best_avg),
                            "path": str(self.best_model_path),
                        },
                    )
                    metrics.setdefault("federated/checkpoint_saved", 1.0)

            if (
                self._federated
                and self._federated_interval > 0
                and (episode_idx + 1) % self._federated_interval == 0
            ):
                self._run_federated_sync(episode_idx + 1, trainer_map)

            for trainer in trainer_map.values():
                reset_noise = getattr(trainer, "reset_noise_schedule", None)
                if callable(reset_noise):
                    reset_noise()

            if self._eval_interval and (episode_idx + 1) % self._eval_interval == 0:
                self._run_periodic_evaluation(
                    episode_idx + 1,
                    trainer_map,
                )

        for trainer_id, trainer in trainer_map.items():
            buffer = trajectory_buffers.get(trainer_id)
            flushed = False
            if buffer is not None:
                flushed = buffer.flush()
            if buffer is not None and trainer_id in off_policy_ids:
                if flushed:
                    for _ in range(buffer.updates_per_step):
                        stats = trainer.update()
                        self._record_trainer_stats(trainer_id, stats)
                        if trainer_update_hook:
                            trainer_update_hook(trainer_id, trainer, stats)
                continue
            stats = trainer.update()
            self._record_trainer_stats(trainer_id, stats)
            if trainer_update_hook:
                trainer_update_hook(trainer_id, trainer, stats)

        self.context.trainer_map = dict(trainer_map)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_trainer_stats(
        self,
        trainer_id: str,
        stats: Optional[Dict[str, Any]],
    ) -> None:
        if not stats:
            return
        snapshot = self._trainer_stats.setdefault(trainer_id, {})
        snapshot.update(stats)

    def _ensure_primary_agent(self) -> None:
        if self.context.primary_agent_id:
            return
        candidates = list(self.context.trainable_agent_ids)
        if not candidates and self.context.team.agents:
            candidates = [self.context.team.agents[0].agent_id]
        if not candidates:
            raise RuntimeError("RunnerContext does not expose any agents to select as primary")
        self.context.set_primary_agent(candidates[0])

    def _configure_output_paths(self) -> None:
        output_root = self.context.output_root
        output_root.mkdir(parents=True, exist_ok=True)
        self.context.cfg.main.schema.output_root = str(output_root)

        bundle_cfg = dict(self._primary_bundle.metadata.get("config", {}))
        if not bundle_cfg and self._primary_bundle.algo.lower() == "ppo":
            bundle_cfg = self.context.cfg.ppo.to_dict()

        save_dir_value = bundle_cfg.get("save_dir", "checkpoints")
        self.checkpoint_dir = resolve_output_dir(save_dir_value, output_root)
        bundle_cfg["save_dir"] = str(self.checkpoint_dir)

        checkpoint_name = bundle_cfg.get(
            "checkpoint_name",
            f"{self._primary_bundle.algo.lower()}_best.pt",
        )
        checkpoint_name = self._apply_run_suffix(checkpoint_name)
        bundle_cfg["checkpoint_name"] = checkpoint_name
        self.best_model_path = self.checkpoint_dir / checkpoint_name
        self._primary_bundle.metadata["config"] = bundle_cfg

        main_checkpoint = self.context.cfg.main.checkpoint
        if main_checkpoint:
            resolved = resolve_output_file(main_checkpoint, output_root)
            self.context.cfg.main.schema.checkpoint = str(resolved)

    def _resolve_primary_epsilon(self) -> Optional[float]:
        trainer = self.primary_trainer
        if trainer is None:
            return None
        accessor = getattr(trainer, "epsilon", None)
        if callable(accessor):
            try:
                value = accessor()
            except Exception:  # pragma: no cover - defensive guard around custom trainers
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _resolve_primary_exploration_noise(self) -> Optional[float]:
        trainer = self.primary_trainer
        if trainer is None:
            return None
        accessor = getattr(trainer, "exploration_noise", None)
        if callable(accessor):
            try:
                value = accessor()
            except Exception:
                return None
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _save_primary_model(self) -> bool:
        controller = self._primary_bundle.controller
        save_fn = getattr(controller, "save", None)
        if not callable(save_fn):
            return False
        current_name = self.best_model_path.name
        suffixed = self._apply_run_suffix(current_name)
        if suffixed != current_name:
            self.best_model_path = self.checkpoint_dir / suffixed
            bundle_cfg = dict(self._primary_bundle.metadata.get("config", {}))
            bundle_cfg["checkpoint_name"] = suffixed
            self._primary_bundle.metadata["config"] = bundle_cfg
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_fn(str(self.best_model_path))
        return True

    def _run_periodic_evaluation(
        self,
        train_episode: int,
        trainer_map: Dict[str, Trainer],
    ) -> None:
        if self._eval_episodes <= 0:
            return
        self.context.trainer_map = dict(trainer_map)
        if self._eval_runner is None:
            try:
                self._eval_runner = EvalRunner(self.context)
            except Exception as exc:
                self._logger.warning(
                    "Failed to initialise periodic evaluator",
                    extra={
                        "episode": train_episode,
                        "error": str(exc),
                    },
                )
                self._eval_runner = None
                return
        else:
            self._eval_runner.trainer_map = dict(trainer_map)
            self._eval_runner.context.trainer_map = dict(trainer_map)

        self._logger.info(
            "Starting periodic evaluation",
            extra={
                "train_episode": train_episode,
                "eval_interval": self._eval_interval,
                "eval_episodes": self._eval_episodes,
            },
        )
        try:
            self.team.reset_actions()
            self._eval_runner.run(
                episodes=self._eval_episodes,
                auto_load=False,
                force_render=False,
            )
        except Exception as exc:
            self._logger.warning(
                "Periodic evaluation failed",
                extra={
                    "train_episode": train_episode,
                    "error": str(exc),
                },
            )
        finally:
            self.team.reset_actions()
            self._logger.update_context(mode="train")

    def _apply_run_suffix(self, checkpoint_name: str) -> str:
        suffix = self._run_suffix
        if not suffix:
            suffix = resolve_run_suffix(getattr(self.context, "metadata", {}))
            self._run_suffix = suffix
        if not suffix:
            return checkpoint_name
        base = Path(checkpoint_name)
        stem = base.stem
        if stem.endswith(f"_{suffix}"):
            return base.name
        return f"{stem}_{suffix}{base.suffix}"

    @staticmethod
    def _resolve_reward_value(cfg: Dict[str, Any], key: str) -> float:
        """Resolve reward configuration values, honouring nested params."""

        raw = cfg.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

        params = cfg.get("params")
        if isinstance(params, dict):
            raw = params.get(key)
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass

        return 0.0


__all__ = ["TrainRunner", "TrainerUpdateHook"]
