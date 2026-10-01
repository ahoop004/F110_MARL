"""Logging controls must affect real output without altering source metrics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from loggers.metric_policy import MetricPolicy
from loggers.wandb_logger import WandbLogger


@pytest.fixture
def fake_wandb(monkeypatch):
    import wandb
    calls, definitions = [], []
    run = SimpleNamespace(id="test", name="test", url=None)
    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(wandb, "config", SimpleNamespace(update=lambda *args, **kwargs: None))
    monkeypatch.setattr(wandb, "define_metric", lambda *args, **kwargs: definitions.append((args, kwargs)))
    monkeypatch.setattr(wandb, "log", lambda row, **kwargs: calls.append(row))
    return calls, definitions


def test_real_logger_reads_scenario_controls_and_drops_disabled_groups(fake_wandb):
    calls, definitions = fake_wandb
    logger = WandbLogger("test", config={"wandb": {"logging": {
        "groups": {"train": False, "eval": False, "define_metrics": False}}}})
    logger.log_metrics({"train/policy_loss": .2, "train/environment_steps": 100})
    logger.log_metrics({"episode/reward": 5., "episode/number": 3})
    logger.log_metrics({"eval/completion_rate": 1., "eval/environment_steps": 100})
    assert calls == []
    assert definitions == []


def test_allowlist_supports_patterns_and_preserves_axes(fake_wandb):
    calls, _ = fake_wandb
    logger = WandbLogger("test", logging_config={"metrics": {"eval/attack_*": True,
                                                           "eval/attack_target_crashes": False}})
    source = {"eval/attack_score": .5, "eval/attack_target_crashes": 3,
              "eval/collision_rate": .2, "eval/environment_steps": 100}
    original = dict(source)
    logger.log_metrics(source)
    assert calls == [{"eval/attack_score": .5, "eval/environment_steps": 100}]
    assert source == original


def test_attack_defaults_remove_aliases_but_preserve_selection_inputs():
    scenario = yaml.safe_load(Path("scenarios/mappo_1v1_attack.yaml").read_text())
    policy = MetricPolicy(scenario["wandb"]["logging"], scenario)
    source = {key: 1. for key in [
        "eval/attack_score", "eval/attack_ego_crash_rate", "eval/attack_successes_per_minute",
        "eval/mean_net_progress", "eval/self_crash_rate", "eval/opponent_finish_rate",
        "eval/focal_completion_rate", "eval/valid_lap_time_sample_count", "eval/environment_steps"]}
    original = dict(source)
    result = policy.filter(source)
    assert set(result) == {"eval/attack_score", "eval/attack_ego_crash_rate",
                           "eval/attack_successes_per_minute", "eval/mean_net_progress",
                           "eval/environment_steps"}
    assert source == original
    assert not policy.accepts("episode/reward_component/progress/car_0")
    assert not policy.accepts("collector/worker_messages")


def test_debug_and_component_opt_in(fake_wandb):
    scenario = yaml.safe_load(Path("scenarios/mappo_1v1_attack.yaml").read_text())
    config = deepcopy(scenario["wandb"]["logging"])
    config["profile"] = "debug"
    policy = MetricPolicy(config, scenario)
    assert policy.accepts("episode/reward_component/progress/car_0")
    assert policy.accepts("collector/worker_messages")
    assert policy.accepts("eval/mean_speed")
    config["profile"] = "auto"
    config["groups"]["reward_components"] = True
    assert MetricPolicy(config, scenario).accepts("episode/reward_component/progress/car_0")


def test_defaults_cover_selfplay_and_namespace_axes(fake_wandb):
    calls, definitions = fake_wandb
    logger = WandbLogger("test")
    logger.log_metrics({"selfplay/rolling100/team_a/reward": 2.,
                        "selfplay/team_a/reward": 3., "selfplay/environment_steps": 100})
    assert calls == [{"selfplay/rolling100/team_a/reward": 2., "selfplay/environment_steps": 100}]
    assert (("eval/*",), {"step_metric": "eval/environment_steps"}) in definitions


def test_scenario_configs_use_current_profiles():
    paths = [Path("configs/wandb.yaml"), *Path("scenarios").rglob("*.yaml")]
    for path in paths:
        config = yaml.safe_load(path.read_text()) or {}
        logging = config.get("wandb", {}).get("logging")
        if logging is not None:
            policy = MetricPolicy(logging, config)
            assert policy.accepts("train/policy_loss"), str(path)
            assert policy.accepts("eval/completion_rate"), str(path)


def test_continuous_training_and_shared_rewards_omit_inapplicable_metrics():
    policy = MetricPolicy(scenario={
        'mappo': {'reward_mode': 'team_shared'},
        'environment': {'episode_termination': {'lap_completion': False}}})
    assert not policy.accepts('episode/completed')
    assert not policy.accepts('episode/team/all_finished')
    assert not policy.accepts('episode/reward/car_0')
    assert policy.accepts('episode/individual_reward/car_0')
    assert policy.accepts('episode/reward')
    assert policy.accepts('eval/completion_rate')


@pytest.mark.parametrize('num_envs', [1, 8])
@pytest.mark.parametrize('strategy', ['lap_time', 'completion_progress', 'completion_safety',
                                    'team_completion', 'map_curriculum'])
def test_completion_profiles_keep_laps_and_outcomes_without_racing_clutter(fake_wandb, num_envs, strategy):
    calls, _ = fake_wandb
    scenario = {'experiment': {'num_envs': num_envs},
                'agents': {'a': {'trainable': True}, 'b': {'trainable': True}},
                'evaluation': {'selection_strategy': strategy, 'enabled': False}}
    logger = WandbLogger('test', config=scenario)
    source = dict.fromkeys([
        'episode/reward', 'episode/lap_count', 'episode/lap_time_s',
        'episode/team/all_finished', 'episode/team/failure_rate', 'episode/team/timeout_rate',
        'episode/reward/a', 'episode/individual_reward/a', 'episode/team/first_place',
        'episode/team/sweep', 'episode/team/rank_score', 'episode/number'], 1.)
    logger.log_metrics(source)
    assert set(calls[-1]) == {'episode/reward', 'episode/lap_count', 'episode/lap_time_s',
                             'episode/team/all_finished', 'episode/team/failure_rate',
                             'episode/team/timeout_rate', 'episode/number'}
    logger.log_metrics(dict.fromkeys([
        'eval/completion_rate', 'eval/team_both_finished_rate', 'eval/mean_valid_lap_time_s',
        'eval/mean_clean_finish_time_s', 'eval/learner_failure_rate', 'eval/environment_steps',
        'eval/team_first_place', 'eval/team_rank_score', 'eval/win_rate',
        'eval/focal_completion_rate', 'eval/focal_opponent_win_rate'], 1.))
    assert set(calls[-1]) == {'eval/completion_rate', 'eval/team_both_finished_rate',
                             'eval/mean_valid_lap_time_s', 'eval/mean_clean_finish_time_s',
                             'eval/learner_failure_rate', 'eval/environment_steps'}


def test_completion_debug_racing_and_allowlist_restore_detailed_metrics():
    scenario = {'evaluation': {'selection_strategy': 'team_completion'}}
    for profile in ('debug', 'racing'):
        policy = MetricPolicy({'profile': profile}, scenario)
        assert not policy.lap_completion
        assert policy.accepts('eval/team_rank_score')
        assert policy.accepts('episode/individual_reward/car_0')
    policy = MetricPolicy({'metrics': ['eval/team_rank_score']}, scenario)
    assert policy.accepts('eval/team_rank_score')
    assert MetricPolicy({'profile': 'lap_completion'}).lap_completion
