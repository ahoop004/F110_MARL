"""Evaluation-gated PPO map curriculum with retention and whole-bundle stopping."""
from __future__ import annotations

import json
import math

from training.hooks import EvaluationCheckpointHook


def validate_map_curriculum(scenario):
    cfg = scenario["map_curriculum"]
    env = scenario["environment"]
    evaluation = scenario.get("evaluation", {})
    if not isinstance(cfg, dict) or set(cfg) - {"success_threshold", "required_evaluations", "episodes_per_map"}:
        raise ValueError("map_curriculum accepts success_threshold, required_evaluations, episodes_per_map")
    threshold = cfg.get("success_threshold", 0.9)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ValueError("map_curriculum.success_threshold must be in (0, 1]")
    for key, default in (("required_evaluations", 2), ("episodes_per_map", 10)):
        value = cfg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"map_curriculum.{key} must be a positive integer")
    maps = env.get("map_bundles_eval", [])
    initial = env.get("map_bundles_train", [])
    if not maps or len(set(maps)) != len(maps) or len(initial) != 1 or initial[0] not in maps:
        raise ValueError("Map curriculum requires unique evaluation maps and exactly one initial training map")
    learners = [a for a in scenario["agents"].values() if a.get("trainable", False)]
    if len(scenario["agents"]) != 1 or len(learners) != 1 or learners[0]["algorithm"] != "ppo":
        raise ValueError("Map curriculum requires a single PPO vehicle")
    if scenario.get("curriculum"):
        raise ValueError("Map and spawn curricula cannot be combined")
    if (env.get("map_cycle") != "per_episode" or env.get("map_pick") != "round_robin"
            or env.get("epoch_shuffle", False)):
        raise ValueError("Map curriculum requires per_episode round_robin scheduling without epoch_shuffle")
    if env.get("max_steps", 0) <= 0 or not env.get("episode_termination", {}).get("lap_completion"):
        raise ValueError("Map curriculum requires bounded training episodes with lap completion")
    if (not evaluation.get("enabled") or not evaluation.get("every_steps")
            or evaluation.get("selection_strategy") != "map_curriculum"
            or evaluation.get("episodes") != len(maps) * cfg.get("episodes_per_map", 10)):
        raise ValueError("Map curriculum requires step-based evaluation, map_curriculum selection and episodes_per_map * maps episodes")
    if (evaluation.get("target_laps", 0) <= 0 or evaluation.get("max_steps", 0) <= 0
            or not evaluation.get("terminate_on_track_limit")
            or not evaluation.get("terminate_on_collision")
            or not env.get("track_limits", {}).get("enabled")):
        raise ValueError("Map curriculum requires finite clean-lap evaluation with track-limit and collision termination")
    final = evaluation.get("final_test")
    if not final or final.get("episodes", 0) < len(maps) or final["episodes"] % len(maps):
        raise ValueError("Map curriculum requires a final_test with equal coverage of every map")


class MapCurriculum:
    def __init__(self, maps, initial, *, success_threshold=0.9, required_evaluations=2, episodes_per_map=10):
        self.maps = list(maps)
        self.active = [initial]
        self.threshold = success_threshold
        self.required = required_evaluations
        self.episodes_per_map = episodes_per_map
        self.stage_streak = 0
        self.bundle_streak = 0
        self.complete = False

    def observe(self, summary):
        per_map = summary.get("per_map", {})
        rates = {}
        for name in self.maps:
            row = per_map.get(name, {})
            count = row.get("episodes", 0)
            if count != self.episodes_per_map:
                raise ValueError(f"Expected {self.episodes_per_map} evaluation episodes for {name}, got {count}")
            if "strict_clean_finish_count" not in row:
                raise ValueError(f"Missing strict clean-finish facts for {name}")
            rate = row["strict_clean_finish_count"] / count
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError(f"Invalid clean completion rate for {name}")
            rates[name] = rate
        for name, rate in rates.items():
            summary[f"map/{name}/clean_completion_rate"] = rate
        passed = {name for name, rate in rates.items() if rate >= self.threshold}
        self.bundle_streak = self.bundle_streak + 1 if len(passed) == len(self.maps) else 0
        self.stage_streak = self.stage_streak + 1 if all(name in passed for name in self.active) else 0
        added = None
        if self.bundle_streak >= self.required:
            self.complete = True
        elif self.stage_streak >= self.required:
            candidates = [name for name in self.maps if name not in self.active and name not in passed]
            if candidates:
                # Closest to passing first; configured order breaks ties.
                added = max(candidates, key=lambda name: rates[name])
                self.active.append(added)
                self.stage_streak = 0
        finish = summary.get("mean_clean_finish_time_s")
        summary["curriculum_selection_score"] = [len(passed), min(rates.values()),
            sum(rates.values()) / len(rates), -float(finish) if finish is not None else -1e30]
        summary["curriculum"] = dict(active_maps=list(self.active), added_map=added,
            clean_completion_rates=rates, stage_streak=self.stage_streak,
            bundle_streak=self.bundle_streak, complete=self.complete)
        return added

    def training_schedule(self):
        # Half the episodes on the newest map, half shared by earlier maps.
        previous = self.active[:-1]
        return [item for name in previous for item in (self.active[-1], name)] or list(self.active)


class CurriculumEvaluator:
    """Keep the normal evaluation report and attach curriculum decisions."""
    def __init__(self, evaluator, curriculum, scheduler, console):
        self.evaluator = evaluator
        self.curriculum = curriculum
        self.scheduler = scheduler
        self.console = console

    def __getattr__(self, name):
        return getattr(self.evaluator, name)

    def evaluate(self):
        summary = self.evaluator.evaluate()
        added = self.curriculum.observe(summary)
        if added:
            self.scheduler.set_training_bundles(self.curriculum.training_schedule())
            self.console.print_info(f"Map curriculum added {added}; retaining {self.curriculum.active[:-1]}")
        self.console.print_info("Map curriculum clean completion: " + ", ".join(
            f"{name}={rate:.0%}" for name, rate in summary["curriculum"]["clean_completion_rates"].items()))
        return summary


class MapCurriculumCheckpointHook(EvaluationCheckpointHook):
    @property
    def should_stop(self):
        return self._evaluator.curriculum.complete

    def _evaluate_checkpoint(self, completed_episodes):
        if self.should_stop:
            return
        super()._evaluate_checkpoint(completed_episodes)
        state = dict(active_maps=self._evaluator.curriculum.active,
                     complete=self.should_stop, environment_steps=self._environment_steps,
                     stage_streak=self._evaluator.curriculum.stage_streak,
                     bundle_streak=self._evaluator.curriculum.bundle_streak)
        (self._dir / "curriculum_state.json").write_text(json.dumps(state, indent=2) + "\n")
        if self._wandb is not None:
            self._wandb.log_metrics({"curriculum/active_map_count": len(state["active_maps"]),
                "curriculum/stage_streak": state["stage_streak"],
                "curriculum/bundle_streak": state["bundle_streak"],
                "curriculum/complete": int(self.should_stop)})
        if self.should_stop:
            self._save(self._dir / "curriculum_passed.pt", metadata={"map_curriculum": state})
            self._console.print_info("All maps passed the curriculum gate; stopping PPO updates for final validation.")
