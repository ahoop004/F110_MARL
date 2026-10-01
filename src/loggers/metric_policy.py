"""Dashboard selection only; training and checkpoint selection retain all facts."""
from fnmatch import fnmatchcase


AXES = {
    "episode": "episode/number",
    "train": "train/environment_steps",
    "perf": "train/environment_steps",
    "eval": "eval/environment_steps",
    "collector": "collector/elapsed_seconds",
    "curriculum": "train/environment_steps",
    "selfplay": "selfplay/environment_steps",
    "selfplay_eval": "selfplay_eval/environment_steps",
}

CORE = (
    "episode/skill/*", "eval/skill_*", "eval/retention_*", "eval/base_skill_*", "eval/curriculum_*",
    "episode/reward", "episode/steps", "episode/lap_count", "episode/lap_time_s",
    "episode/completed", "episode/failed", "episode/timeout", "episode/net_progress_laps",
    "episode/reward/*", "episode/individual_reward/*",
    "episode/team/completion_rate", "episode/team/all_finished",
    "episode/team/failure_rate", "episode/team/timeout_rate",
    "episode/team/first_place", "episode/team/sweep", "episode/team/rank_score",
    "train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl",
    "train/*/policy_loss", "train/*/value_loss", "train/*/entropy", "train/*/approx_kl",
    "train/learning_rate", "train/*/learning_rate", "train/kl_early_stop",
    "perf/end_to_end_env_steps_per_second", "perf/update_seconds",
    "eval/completion_rate", "eval/learner_failure_rate", "eval/timeout_rate",
    "eval/mean_net_progress", "eval/mean_clean_finish_time_s", "eval/clean_finish_count",
    "eval/race_count", "eval/evaluation_seconds", "eval/map/*/clean_completion_rate",
    "eval/focal_completion_rate", "eval/focal_opponent_win_rate",
    "curriculum/*",
    "selfplay/rolling100/*/reward", "selfplay/rolling100/*/win",
    "selfplay/rolling100/*/finish_rate", "selfplay/rolling100/*/any_crash",
    "selfplay_eval/*/finish_rate", "selfplay_eval/*/win", "selfplay_eval/*/both_finished",
    "selfplay_eval/*/progress_laps", "selfplay_eval/*/crash_count", "selfplay_eval/is_best",
)
ATTACK = (
    "episode/attack_successes", "episode/attack_target_crashes", "episode/attack_eligible_crashes",
    "episode/attack_ego_failed", "eval/attack_score", "eval/attack_successes",
    "eval/attack_successes_per_minute", "eval/attack_ego_crash_rate",
    "eval/attack_target_crashes", "eval/attack_eligible_crashes",
)
RACING = (
    "eval/win_rate", "eval/team_both_finished_rate", "eval/team_first_place",
    "eval/team_sweep", "eval/team_rank_score", "eval/team_rank_penalty_score",
    "eval/mean_valid_lap_time_s", "eval/valid_laps", "eval/fastest_valid_lap_s",
)
LAP_COMPLETION = (
    "eval/mean_valid_lap_time_s", "eval/valid_laps", "eval/fastest_valid_lap_s",
    "eval/team_both_finished_rate",
)
COMPLETION_STRATEGIES = {"lap_time", "completion_progress", "completion_safety",
                         "team_completion", "map_curriculum"}


class MetricPolicy:
    def __init__(self, config=None, scenario=None):
        self.config = config or {}
        scenario = scenario or {}
        profile = self.config.get("profile", "auto")
        if profile not in {"auto", "attack", "lap_completion", "racing", "debug"}:
            raise ValueError("wandb.logging.profile must be auto, attack, lap_completion, racing, or debug")
        self.debug = profile == "debug"
        self.attack = profile == "attack" or (profile == "auto" and bool(
            scenario.get("environment", {}).get("attack_task")))
        learners = sum(bool(a.get("trainable")) for a in scenario.get("agents", {}).values())
        self.multi_agent = learners > 1
        self.shared_reward = scenario.get("mappo", {}).get("reward_mode") == "team_shared"
        self.finite_training = scenario.get("environment", {}).get("episode_termination", {}).get("lap_completion", True)
        self.selection_strategy = scenario.get("evaluation", {}).get("selection_strategy")
        self.lap_completion = profile == "lap_completion" or (
            profile == "auto" and not self.attack and
            self.selection_strategy in COMPLETION_STRATEGIES)
        patterns = (*CORE, *(ATTACK if self.attack else LAP_COMPLETION if self.lap_completion else RACING))
        self._names = set(patterns)
        self._patterns = tuple(pattern for pattern in patterns if "*" in pattern)

    def group_enabled(self, group):
        groups = self.config.get("groups", {})
        default = self.debug or group not in {"collector", "reward_components"}
        return bool(groups.get(group, default))

    def accepts(self, key):
        namespace = key.split("/", 1)[0]
        group = {"episode": "train", "selfplay": "train", "selfplay_eval": "eval",
                 "perf": "performance"}.get(namespace, namespace)
        if not self.group_enabled(group):
            return False
        components = "reward_component" in key
        if components and not self.group_enabled("reward_components"):
            return False
        allowlist = self.config.get("metrics")
        if allowlist is not None:
            rules = allowlist if isinstance(allowlist, dict) else dict.fromkeys(allowlist, True)
            matches = [bool(value) for pattern, value in rules.items() if fnmatchcase(key, pattern)]
            return bool(matches) and all(matches)
        if self.debug or components or namespace == "collector":
            return True
        if self.lap_completion and (key.startswith(("episode/reward/", "episode/individual_reward/", "eval/focal_"))
                or key in {"episode/team/first_place", "episode/team/sweep", "episode/team/rank_score"}):
            return False
        if self.shared_reward and key.startswith("episode/reward/"):
            return False
        if not self.finite_training and key in {
                "episode/completed", "episode/team/completion_rate", "episode/team/all_finished",
                "episode/team/first_place", "episode/team/sweep", "episode/team/rank_score"}:
            return False
        # Supporting racer outcomes matter in asymmetric attack teams, but are
        # aliases of completion in the single-learner attack task.
        if not self.multi_agent and key.startswith("eval/focal_"):
            return False
        if self.attack and not self.multi_agent and key in {"episode/failed", "eval/learner_failure_rate"}:
            return False
        if self.selection_strategy == "lap_time" and key in {
                "eval/offtrack_error_m_s_per_lap", "eval/boundary_violation_lap_rate"}:
            return True
        return key in self._names or any(fnmatchcase(key, pattern) for pattern in self._patterns)

    def filter(self, metrics):
        axes = set(AXES.values())
        selected = {key: value for key, value in metrics.items()
                    if key not in axes and self.accepts(key)}
        # Keep the x-axis with an allowed measurement, even under an allowlist.
        # Disabled groups never generate axis-only W&B events.
        for key in tuple(selected):
            axis = AXES.get(key.split("/", 1)[0])
            if axis in metrics:
                selected[axis] = metrics[axis]
        return selected
