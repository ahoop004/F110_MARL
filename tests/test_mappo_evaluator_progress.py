"""Evaluation progress uses live race facts without affecting selection results."""
from types import SimpleNamespace

import pytest
import torch

from env.collision_state import RaceLifecycle
from metrics.racing_eval import create_episode_facts, update_agent_step_facts
from training.mappo_evaluator import DeterministicMAPPOEvaluator


def evaluator():
    env = SimpleNamespace(max_steps=40000, target_laps=5, timestep=.05,
                          map_name='Spa_map', lifecycle=RaceLifecycle(['car_0', 'car_1'], 5,
                          finish_on_laps=True, lap_finish_agents=['car_0']))
    return DeterministicMAPPOEvaluator(env=env, trainable_ids=['car_0'], other_agents={},
        obs_composers={}, action_composer=None, episodes=8, base_seed=10042)


def test_progress_throttles_running_reports_but_always_reports_episode_boundaries(monkeypatch):
    import training.mappo_evaluator as module
    now = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    ev = evaluator()
    rows = []
    assert ev.set_progress_callback(rows.append) is None
    facts = create_episode_facts(episode=0, agent_ids=['car_0', 'car_1'],
                                trainable_ids=['car_0'], opponent_ids=['car_1'])
    ev._report_progress(facts, 0, 0, 'starting')
    assert rows[-1]['episode'] == 1 and rows[-1]['episodes'] == 8
    assert rows[-1]['laps'] == 'car_0:0/5'
    update_agent_step_facts(facts, step_idx=50, infos={'car_0': {'lap_count': 2,
        'attack': dict(success=1, target_crash=True, eligible_crash=True,
                       ego_failed=False, horizon_steps=40000, target_laps=5)}})
    now[0] = .5
    ev._report_progress(facts, 0, 50)
    assert len(rows) == 1
    now[0] = 1.
    ev._report_progress(facts, 0, 50)
    assert rows[-1]['sim_seconds'] == 2.5
    assert rows[-1]['laps'] == 'car_0:2/5'
    assert rows[-1]['attack_successes'] == rows[-1]['target_crashes'] == 1
    assert not any('reward' in key for key in rows[-1])
    update_agent_step_facts(facts, step_idx=51, infos={'car_0': {
        'terminal_reason': 'track_boundary'}}, terminations={'car_0': True})
    ev._report_progress(facts, 0, 51, 'complete')
    assert len(rows) == 3
    assert rows[-1]['outcome'] == 'car_0:track_boundary'
    assert ev.set_progress_callback(None) == rows.append


def test_failed_evaluation_clears_progress_and_restores_actor_mode():
    ev = evaluator()
    actor = torch.nn.Linear(1, 1)
    ev.bind_agent(SimpleNamespace(actor=actor, last_raw_actions={}))
    rows = []
    ev.set_progress_callback(rows.append)

    def fail(**kwargs):
        raise RuntimeError('reset failed')

    ev.env.reset = fail
    with pytest.raises(RuntimeError, match='reset failed'):
        ev.evaluate()
    assert rows == [None]
    assert actor.training
