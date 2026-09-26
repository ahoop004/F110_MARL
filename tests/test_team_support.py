"""The support objective must reward useful delay, not proximity or crashes."""
from copy import deepcopy

import pytest

from wrappers.rewards.interaction import TeamSupportComponent
from training.hooks import EvaluationCheckpointHook
from core.scenario import load_and_expand_scenario


def context():
    ego = {'centerline': {'progress_delta': .001},
           'frenet_neighbors': [{'agent_id': 'car_2', 'delta_s': -2., 'delta_d': .1}]}
    return {'info': ego, 'track_length': 100., 'opponent_agent_ids': ['car_2', 'car_3'],
            'all_infos': {'car_0': {'centerline': {'progress_delta': .002}},
                          'car_1': ego,
                          'car_2': {'centerline': {'progress_delta': .0005}},
                          'car_3': {'centerline': {'progress_delta': .001}}}}


def test_blocking_rewards_teammate_progress_advantage_and_penalizes_losing_ground():
    reward = TeamSupportComponent({})
    c = context()
    parts = reward.compute(c)
    assert parts['team_support/progress'] == pytest.approx(.2)
    assert parts['team_support/blocking'] == pytest.approx((.2-.05)/2)
    c['all_infos']['car_2']['centerline']['progress_delta'] = .004
    assert reward.compute(c)['team_support/blocking'] == pytest.approx(-.1)


@pytest.mark.parametrize('change', ['ahead', 'far', 'lateral', 'crashed_opponent',
                                    'finished_opponent', 'reverse', 'stopped_teammate'])
def test_unhelpful_proximity_does_not_earn_blocking_credit(change):
    c = context()
    n = c['info']['frenet_neighbors'][0]
    if change == 'ahead': n['delta_s'] = 2
    if change == 'far': n['delta_s'] = -20
    if change == 'lateral': n['delta_d'] = 2
    if change == 'crashed_opponent': c['all_infos']['car_2']['terminal_reason'] = 'collision'
    if change == 'finished_opponent': c['all_infos']['car_2']['terminal_reason'] = 'race_complete'
    if change == 'reverse': c['info']['centerline']['progress_delta'] = -.001
    if change == 'stopped_teammate': c['all_infos']['car_0']['centerline']['progress_delta'] = 0
    assert 'team_support/blocking' not in TeamSupportComponent({}).compute(c)


@pytest.mark.parametrize('aid', ['car_0', 'car_1'])
@pytest.mark.parametrize('cause', ['collision', 'race_complete', 'boundary'])
def test_terminated_or_outside_learner_generates_no_support_bonus(aid, cause):
    c = context()
    if cause == 'boundary':
        c['all_infos'][aid]['track_limits'] = {'exceeded': True}
    else:
        c['all_infos'][aid]['terminal_reason'] = cause
    assert TeamSupportComponent({}).compute(c) == {}


def test_support_checkpoint_selection_prioritizes_progress_car_not_both_finishing():
    score = lambda s: EvaluationCheckpointHook.selection_score(s, 'asymmetric_support')
    base = dict(focal_completion_rate=1., focal_opponent_win_rate=1., team_collision_rate=0.,
                focal_mean_clean_finish_time_s=20., focal_mean_net_progress=3., team_both_finished_rate=0.)
    assert score(base) > score({**base, 'focal_opponent_win_rate': .5, 'team_both_finished_rate': 1.})
    assert score(base) > score({**base, 'focal_completion_rate': .9, 'focal_mean_clean_finish_time_s': 1.})
    assert score(base) > score({**base, 'team_collision_rate': .5})
    assert score(base) > score({**base, 'focal_mean_clean_finish_time_s': 30.})


def test_scenarios_match_task_physics_and_training_protocol():
    full = load_and_expand_scenario('scenarios/mappo_2v2_asymmetric.yaml')
    lora = load_and_expand_scenario('scenarios/mappo_2v2_asymmetric_lora.yaml')
    for key in ('agents', 'environment', 'evaluation'):
        assert full[key] == lora[key]
    a, b = deepcopy(full['training_defaults']), deepcopy(lora['training_defaults'])
    assert b.pop('lora')['per_agent_log_std']
    assert a == b
    assert a['require_pretrained_actor']
    assert full['mappo']['actor_mode'] == 'independent'
    assert lora['mappo']['actor_mode'] == 'shared'
    play = load_and_expand_scenario('scenarios/render/mappo_2v2_asymmetric.yaml')
    assert play['agents'] == full['agents']
    assert play['mappo'] == full['mappo']
    assert play['experiment']['evaluation_only'] and play['experiment']['num_envs'] == 1


def test_role_metrics_use_explicit_progress_car_even_when_agent_order_changes():
    from metrics.racing_eval import AgentEpisodeFacts, EvalEpisodeFacts, aggregate_eval_episodes
    agents = {
        'car_1': AgentEpisodeFacts('car_1', 'trainable', terminal_reason='time_limit', net_progress=.5),
        'car_0': AgentEpisodeFacts('car_0', 'trainable', finish_step=10,
            finish_elapsed_steps=10, terminal_reason='race_complete', finish_position=1, net_progress=3.),
        'car_2': AgentEpisodeFacts('car_2', 'opponent', terminal_reason='time_limit'),
        'car_3': AgentEpisodeFacts('car_3', 'opponent', terminal_reason='collision', collision_step=4),
    }
    episode = EvalEpisodeFacts(episode=0, steps=10, agents=agents, trainable_team=['car_1', 'car_0'],
                              opponent_team=['car_2', 'car_3'])
    summary = aggregate_eval_episodes([episode], focal_agent_id='car_0', timestep=.05)
    assert summary['focal_completion_rate'] == 1
    assert summary['focal_opponent_win_rate'] == 1
    assert summary['team_both_finished_rate'] == 0
    assert summary['focal_mean_clean_finish_time_s'] == .5
