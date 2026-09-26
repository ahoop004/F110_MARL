"""Incident scoring must agree across training, evaluation, and initialization arms."""
import pytest

from metrics.race_penalties import POLICY, terminal_penalty_event
from metrics.racing_eval import create_episode_facts, update_agent_step_facts, aggregate_eval_episodes
from wrappers.rewards.composer import RewardComposer
from training.hooks import EvaluationCheckpointHook

TEAM = ['car_0', 'car_1']
OPPONENTS = ['car_2', 'car_3']


def composer():
    return RewardComposer.from_config({'team_race_penalties': {'enabled': True, 'policy': POLICY}})


def context(infos):
    return {'all_infos': infos, 'trainable_agent_ids': TEAM, 'opponent_agent_ids': OPPONENTS}


def test_terminal_incident_paid_once_despite_persistent_collision_flags():
    rewards = composer()
    infos = {'car_2': {'terminal_reason': 'collision', 'terminal_step': 12, 'collision': True}}
    assert rewards.compute(context(infos), team=True)[0] == pytest.approx(.125)
    for _ in range(5):
        assert rewards.compute(context(infos), team=True) == (0., {})
    rewards.reset()
    assert rewards.compute(context(infos), team=True)[0] == pytest.approx(.125)


@pytest.mark.parametrize('reason', ['race_complete', 'time_limit', None])
def test_no_penalty_for_finish_timeout_or_unattributed_collision_flag(reason):
    assert terminal_penalty_event('car_2', {'terminal_reason': reason, 'terminal_step': 2,
                                          'collision': True}) is None


def test_one_for_one_contact_has_negative_penalty_score_and_metrics_agree():
    rewards = composer()
    facts = create_episode_facts(episode=0, agent_ids=TEAM + OPPONENTS,
                                 trainable_ids=TEAM, opponent_ids=OPPONENTS)
    infos = {aid: {'terminal_reason': 'collision', 'terminal_step': 5}
             for aid in ['car_0', 'car_2']}
    value, parts = rewards.compute(context(infos), team=True)
    assert parts == {'race_penalties/own': -.5, 'race_penalties/opponents': .125}
    assert value == -.375
    for step in range(6, 10):
        update_agent_step_facts(facts, step_idx=step, infos=infos,
                               terminations={'car_0': True, 'car_2': True})
    summary = aggregate_eval_episodes([facts], timestep=.05)
    assert summary['mean_team_penalty_score'] == value
    assert summary['mean_own_penalty_points'] == summary['mean_opponent_penalty_points'] == 1
    assert len(summary['race_penalty_events']) == 2
    assert summary['team_rank_penalty_score'] == value
    score = EvaluationCheckpointHook.selection_score
    assert score({**summary, 'team_rank_penalty_score': 0.}, 'team_combined_penalties') > score(summary, 'team_combined_penalties')


def test_boundary_dnf_recorded_but_not_synthesized_from_other_signals():
    assert terminal_penalty_event('car_1', {'terminal_reason': 'track_boundary',
                                          'terminal_step': 8}).kind == 'boundary_dnf'
    assert terminal_penalty_event('car_1', {'track_limits': {'exceeded': True}}) is None


def test_local_collision_reward_disabled_to_avoid_double_charge():
    reward = RewardComposer.from_file('configs/reward/tasks/race_team_2v2_penalties.yaml')
    _, parts = reward.compute({'info': {'collision': True}, 'timestep': .05})
    assert not any('collision' in name for name in parts)


def test_scratch_pretrained_pair_differs_only_in_initialization_and_name():
    from core.scenario import load_and_expand_scenario
    a = load_and_expand_scenario('scenarios/mappo_2v2_race.yaml', overrides=['training_defaults.update_version="parallel-mappo-grouped256-batch2048-v1"',
         'training_defaults.batch_size=2048',
         'training_defaults.rollout_steps_per_env=256',
         'training_defaults.checkpoint_every_steps=1024000',
         'wandb.group="mappo-2v2-penalties-current-physics"',
         'wandb.tags=["mappo","2v2","terminal-incidents-v1","current-pretrain-physics","racing-mpc-opponents"]',
         'wandb.notes="Matched physics, observations, racing MPC opponents and rewards. Both-finished '
         'rate, then rank plus recorded penalties select checkpoints; clean finish time breaks successful '
         'ties."',
         'experiment.name="mappo_2v2_penalties_scratch"',
         'experiment.num_envs=400',
         'experiment.num_workers=100',
         'experiment.worker_startup_batch_size=8',
         'experiment.worker_startup_timeout_s=600',
         'experiment.worker_response_timeout_s=120',
         'experiment.terminal_recent_episodes=100',
         'experiment.terminal_every_updates=10',
         'experiment.terminal_diagnostic_every_updates=100',
         'experiment.terminal_episode_detail=false',
         'evaluation.selection_strategy="team_combined_penalties"',
         'evaluation.every_steps=1024000',
         'agents.car_0.reward.task.name="race_team_2v2_penalties"',
         'agents.car_0.reward.task.description="Shared completion/placement reward with recorded terminal '
         'race penalties."',
         'agents.car_0.reward.reward.collision.enabled=false',
         'agents.car_0.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}',
         'agents.car_1.reward.task.name="race_team_2v2_penalties"',
         'agents.car_1.reward.task.description="Shared completion/placement reward with recorded terminal '
         'race penalties."',
         'agents.car_1.reward.reward.collision.enabled=false',
         'agents.car_1.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}'])
    b = load_and_expand_scenario('scenarios/mappo_2v2_race.yaml', overrides=['training_defaults.update_version="parallel-mappo-grouped256-batch2048-v1"',
         'training_defaults.batch_size=2048',
         'training_defaults.pretrained_actor_checkpoint="../outputs/L_map_pretrain/L_map_best_model.pt"',
         'training_defaults.rollout_steps_per_env=256',
         'training_defaults.checkpoint_every_steps=1024000',
         'wandb.group="mappo-2v2-penalties-current-physics"',
         'wandb.tags=["mappo","2v2","terminal-incidents-v1","current-pretrain-physics","racing-mpc-opponents"]',
         'wandb.notes="Matched physics, observations, racing MPC opponents and rewards. Both-finished '
         'rate, then rank plus recorded penalties select checkpoints; clean finish time breaks successful '
         'ties."',
         'experiment.name="mappo_2v2_penalties_pretrained"',
         'experiment.num_envs=400',
         'experiment.num_workers=100',
         'experiment.worker_startup_batch_size=8',
         'experiment.worker_startup_timeout_s=600',
         'experiment.worker_response_timeout_s=120',
         'experiment.terminal_recent_episodes=100',
         'experiment.terminal_every_updates=10',
         'experiment.terminal_diagnostic_every_updates=100',
         'experiment.terminal_episode_detail=false',
         'evaluation.selection_strategy="team_combined_penalties"',
         'evaluation.every_steps=1024000',
         'agents.car_0.reward.task.name="race_team_2v2_penalties"',
         'agents.car_0.reward.task.description="Shared completion/placement reward with recorded terminal '
         'race penalties."',
         'agents.car_0.reward.reward.collision.enabled=false',
         'agents.car_0.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}',
         'agents.car_1.reward.task.name="race_team_2v2_penalties"',
         'agents.car_1.reward.task.description="Shared completion/placement reward with recorded terminal '
         'race penalties."',
         'agents.car_1.reward.reward.collision.enabled=false',
         'agents.car_1.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}'])
    assert a['training_defaults']['pretrained_actor_checkpoint'] is None
    assert b['training_defaults']['pretrained_actor_checkpoint'] == '../outputs/L_map_pretrain/L_map_best_model.pt'
    b['training_defaults']['pretrained_actor_checkpoint'] = None
    b['experiment']['name'] = a['experiment']['name']
    assert a == b
