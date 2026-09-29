"""Attack episodes finish on learner laps while the target remains active."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.scenario import load_and_expand_scenario, resolve_evaluation_protocol, validate_scenario, ScenarioError
from core.setup import create_training_setup
from metrics.racing_eval import aggregate_eval_episodes, create_episode_facts, episode_race_record, update_agent_step_facts


DIRECTORY = Path('scenarios').resolve()


def scenario(lora=False):
    name = 'mappo_1v1_attack' + ('_lora' if lora else '')
    return load_and_expand_scenario(str(DIRECTORY / (name + '.yaml')))


@pytest.mark.parametrize('lora', [False, True])
@pytest.mark.parametrize('mode', ['train', 'eval'])
def test_five_learner_laps_finish_with_timeout_and_without_target_finish(lora, mode):
    config = scenario(lora)
    assert config['environment']['max_steps'] == config['evaluation']['max_steps'] == 40000
    assert config['environment']['target_laps'] == config['evaluation']['target_laps'] == 5
    assert resolve_evaluation_protocol(config, 'final')['max_steps'] == 40000
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        config['environment'][key] = ['circle_map']
    env, _, _ = create_training_setup(config, mode=mode, scenario_dir=DIRECTORY)
    try:
        env.reset(seed=42)
        assert env.max_steps == 40000
        assert env.lifecycle.finish_on_laps
        assert env.lifecycle.lap_finish_agents == {'car_0'}
        # Move past both historical timeouts without simulating thousands of
        # stationary steps; accepted lap events exercise the real lifecycle.
        env._elapsed_steps = 12001
        for _ in range(25):
            env.lifecycle.record_lap_crossing('car_1', step=env._elapsed_steps)
        for _ in range(4):
            env.lifecycle.record_lap_crossing('car_0', step=env._elapsed_steps)
        _, _, terms, truncs, infos = env.step({})
        assert not any(terms.values()) and not any(truncs.values())
        assert env.agents == ['car_0', 'car_1']
        assert infos['car_1']['lap_count'] == 25
        assert not infos['car_1']['race_completed']

        env.lifecycle.record_lap_crossing('car_0', step=env._elapsed_steps)
        _, _, terms, truncs, infos = env.step({})
        assert terms['car_0'] and not terms['car_1']
        assert not any(truncs.values())
        assert env.episode_done and not env.agents
        assert infos['car_0']['terminal_reason'] == 'race_complete'
        assert infos['car_0']['lap_count'] == 5
        assert infos['car_0']['attack']['horizon_steps'] == 40000
        assert infos['car_0']['attack']['target_laps'] == 5
        assert not infos['car_0']['attack']['ego_failed']

        facts = create_episode_facts(episode=0, agent_ids=['car_0', 'car_1'],
                                    trainable_ids=['car_0'], opponent_ids=['car_1'])
        update_agent_step_facts(facts, step_idx=1, infos=infos, terminations=terms)
        record = episode_race_record(facts, timestep=env.timestep)
        assert record['agents']['car_0']['attack_target_laps'] == 5
        assert record['agents']['car_0']['finished']
        summary = aggregate_eval_episodes([facts], timestep=env.timestep)
        assert summary['attack_score_basis'] == 'scheduled_minutes'
        assert summary['attack_score_budget'] == pytest.approx(2000 / 60)
        assert summary['attack_score'] == 0.
        env.reset(seed=42)
        assert env.lifecycle.records['car_0'].lap_count == 0
        assert len(env.agents) == 2
        # A stalled learner remains active until the configured final step,
        # then truncates without being treated as an ego crash or lap finish.
        env._elapsed_steps = env.max_steps - 2
        _, _, terms, truncs, _ = env.step({})
        assert not any(terms.values()) and not any(truncs.values())
        _, _, terms, truncs, infos = env.step({})
        assert not any(terms.values()) and all(truncs.values())
        assert env.episode_done and not env.agents
        assert infos['car_0']['time_limit']
        assert not infos['car_0']['race_completed']
        assert not infos['car_0']['attack']['ego_failed']
    finally:
        env.close()


@pytest.mark.parametrize('phase,field,value', [
    ('environment', 'max_steps', -1),
    ('environment', 'max_steps', True),
    ('environment', 'max_steps', 1.5),
    ('environment', 'lap_completion', False),
    ('environment', 'lap_finish_agents', ['car_0', 'car_1']),
    ('environment', 'lap_finish_agents', ['car_1']),
    ('environment', 'lap_finish_agents', None),
    ('environment', 'mode', 'all_agents'),
    ('evaluation', 'lap_completion', False),
    ('evaluation', 'episode_termination_mode', 'all_agents'),
])
def test_attack_rejects_missing_learner_bound_or_stopping_target(phase, field, value):
    config = scenario()
    config['environment']['max_steps'] = config['evaluation']['max_steps'] = 0
    section = config[phase]
    if phase == 'environment' and field != 'max_steps':
        section = section['episode_termination']
    section[field] = value
    with pytest.raises(ScenarioError, match='attack_task'):
        validate_scenario(config)


def test_timed_attack_remains_supported_and_final_can_inherit_or_set_no_timeout():
    config = scenario()
    config['evaluation']['final_test']['max_steps'] = 0
    validate_scenario(config)
    assert resolve_evaluation_protocol(config, 'final')['max_steps'] == 0
    legacy = deepcopy(config)
    legacy['environment']['max_steps'] = legacy['evaluation']['max_steps'] = 1200
    legacy['environment']['episode_termination']['lap_completion'] = False
    legacy['environment']['episode_termination'].pop('lap_finish_agents')
    legacy['evaluation']['lap_completion'] = False
    legacy['evaluation']['final_test'].pop('max_steps')
    validate_scenario(legacy)
    legacy['evaluation']['final_test']['max_steps'] = 0
    with pytest.raises(ScenarioError, match='evaluation.final.*finite horizon'):
        validate_scenario(legacy)


def test_attack_score_uses_scheduled_laps_across_early_failures_and_keeps_zero_successes():
    episodes = []
    for episode, laps, success, failed in [(0, 20, 3, False), (1, 10, 0, True)]:
        facts = create_episode_facts(episode=episode, agent_ids=['car_0', 'car_1'],
                                    trainable_ids=['car_0'], opponent_ids=['car_1'])
        update_agent_step_facts(facts, step_idx=1, infos={'car_0': {'attack': dict(
            success=success, target_crash=False, eligible_crash=False, ego_failed=failed,
            horizon_steps=0, target_laps=laps)}})
        episodes.append(facts)
    summary = aggregate_eval_episodes(episodes, timestep=.05)
    assert summary['attack_score_budget'] == 30
    assert summary['attack_ego_crash_rate'] == .5
    assert summary['attack_score'] == pytest.approx((3 - 2) / 30)
    # A time-limited episode cannot silently change the units of a lap score.
    episodes[0].agents['car_0'].attack_horizon_steps = 1200
    with pytest.raises(ValueError, match='consistent time or lap budget'):
        aggregate_eval_episodes(episodes, timestep=.05)


@pytest.mark.parametrize('mode,finishers,lap_completion,accepted', [
    ('all_trainable', ['car_0'], True, True),
    ('all_trainable', ['car_1'], True, False),
    ('all_trainable', ['car_0'], False, False),
    ('all_agents', ['car_0'], True, False),
    ('all_agents', ['car_0', 'car_1'], True, True),
    ('any_agent', ['car_0'], True, True),
    ('any_agent', [], True, False),
])
def test_checkpoint_evaluation_requires_a_lap_bound_for_its_termination_group(
        mode, finishers, lap_completion, accepted):
    from env.collision_state import RaceLifecycle
    from training.mappo_evaluator import DeterministicMAPPOEvaluator

    env = SimpleNamespace(max_steps=0, possible_agents=['car_0', 'car_1'],
        episode_termination_mode=mode,
        lifecycle=RaceLifecycle(['car_0', 'car_1'], 20,
                                finish_on_laps=lap_completion, lap_finish_agents=finishers))
    kwargs = dict(env=env, trainable_ids=['car_0'], other_agents={},
                  obs_composers={}, action_composer=None, episodes=1, base_seed=42)
    if accepted:
        assert DeterministicMAPPOEvaluator(**kwargs).env is env
    else:
        with pytest.raises(ValueError, match='finite max_steps or lap completion'):
            DeterministicMAPPOEvaluator(**kwargs)
