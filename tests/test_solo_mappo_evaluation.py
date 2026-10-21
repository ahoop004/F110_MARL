"""Solo MAPPO inference keeps the source actor's observations and physical contract."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from agents.common.networks import Actor
from agents.common.solo_evaluation import SoloMAPPOActor
from core.scenario import load_and_expand_scenario
from core.setup import build_obs_composer, resolve_training_params
from env.spaces_builder import build_action_spaces


@pytest.fixture
def solo_setup():
    scenario = load_and_expand_scenario('scenarios/mappo_solo_lap_completion_transfer.yaml')
    cfg = scenario['agents']['car_0']
    cfg['params'].update(pi_hidden_dims=[8, 8])
    composer = build_obs_composer(cfg, scenario['environment'], Path('scenarios'), action_dim=2)
    space, _ = build_action_spaces(['car_0'], scenario['environment']['vehicle_params'])
    params = resolve_training_params(cfg, scenario)
    actors = {aid: Actor(192, 2, [8, 8], 'leaky_relu') for aid in ('car_0', 'car_1')}
    # Deliberately omit critic and optimizer: neither may be required for inference.
    checkpoint = dict(algorithm='mappo', actor_mode='independent', lora_contract=None,
        agent_ids=list(actors), actors={aid: actor.state_dict() for aid, actor in actors.items()},
        obs_dims={aid: 192 for aid in actors}, action_dim=2,
        action_low=space.low, action_high=space.high,
        action_contract=params['_action_contract'], physics_contract=params['_physics_contract'],
        observation_contracts={aid: composer.contract for aid in actors},
        actor_hidden_dims=[8, 8], activation='leaky_relu')
    return scenario, composer, space, params, checkpoint, actors


@pytest.mark.parametrize('source_agent', ['car_0', 'car_1'])
def test_solo_actor_actions_match_original_and_preserve_input(solo_setup, source_agent):
    _, composer, space, params, checkpoint, actors = solo_setup
    policy = SoloMAPPOActor(checkpoint, source_agent, composer, space.low, space.high, params)
    observation = np.random.default_rng(1).normal(size=192).astype(np.float32)
    observation[158:] = 0
    observation[188 + checkpoint['agent_ids'].index(source_agent)] = 1
    original = observation.copy()
    with torch.inference_mode():
        expected, _ = actors[source_agent].get_action(torch.from_numpy(observation).unsqueeze(0),
                                                      deterministic=True)
    np.testing.assert_array_equal(policy.predict(observation), expected[0].numpy())
    np.testing.assert_array_equal(observation, original)
    assert not hasattr(policy, 'critic')
    assert not hasattr(policy, 'optimizer')


@pytest.mark.parametrize('mismatch', ['physics', 'action_bounds', 'observation', 'lora', 'missing_actor'])
def test_solo_actor_rejects_incompatible_checkpoint(solo_setup, mismatch):
    _, composer, space, params, checkpoint, _ = solo_setup
    checkpoint = deepcopy(checkpoint)
    if mismatch == 'physics':
        checkpoint['physics_contract']['vehicle_params']['wheel_actuators']['wheel_speed_max'] = 100.
    elif mismatch == 'action_bounds':
        checkpoint['action_high'][1] = 100.
    elif mismatch == 'observation':
        checkpoint['observation_contracts']['car_0']['observation']['frenet_neighbors']['agent_ids'].reverse()
    elif mismatch == 'lora':
        checkpoint['lora_contract'] = {'rank': 4}
    else:
        del checkpoint['actors']['car_0']
    with pytest.raises(ValueError):
        SoloMAPPOActor(checkpoint, 'car_0', composer, space.low, space.high, params)


@pytest.mark.parametrize('source_agent', ['car_0', 'car_1'])
def test_cli_solo_evaluation_without_a_critic(tmp_path, monkeypatch, solo_setup, source_agent):
    import run
    from agents.mappo import MAPPOAgent
    from agents.ppo import PPOAgent
    scenario, _, _, _, checkpoint, _ = solo_setup
    scenario['environment']['map_bundles_eval'] = ['circle_map']
    scenario['evaluation'].update(max_steps=4)
    scenario['evaluation']['final_test'].update(episodes=2)
    checkpoint_path = tmp_path / 'best_model.pt'
    torch.save(checkpoint, checkpoint_path)
    original_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    captured = []
    predict = SoloMAPPOActor.predict

    def checked_predict(policy, observation):
        assert observation.shape == (192,)
        np.testing.assert_array_equal(observation[158:188], np.zeros(30))
        expected_identity = np.zeros(4)
        expected_identity[['car_0', 'car_1'].index(source_agent)] = 1
        np.testing.assert_array_equal(observation[188:], expected_identity)
        captured.append(observation.copy())
        return predict(policy, observation)

    def forbidden_init(*args, **kwargs):
        raise AssertionError('Solo inference must not construct a full PPO or MAPPO learner')

    monkeypatch.setattr(PPOAgent, '__init__', forbidden_init)
    monkeypatch.setattr(MAPPOAgent, '__init__', forbidden_init)
    monkeypatch.setattr(SoloMAPPOActor, 'predict', checked_predict)
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *a, **kw: scenario)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario',
        'scenarios/mappo_solo_lap_completion_transfer.yaml', '--eval', '--eval-actor', source_agent,
        '--checkpoint', str(checkpoint_path), '--eval-protocol', 'final',
        '--output-dir', str(tmp_path / 'eval'), '--no-render', '--no-wandb', '--quiet'])
    run.main()
    report = json.loads((tmp_path / 'eval' / 'evaluation_report.json').read_text())
    assert len(captured) == 8
    assert report['seeds'] == [20042, 20043]
    assert report['actor_evaluation']['source_actor'] == source_agent
    assert report['evaluation_provenance']['actor_evaluation'] == report['actor_evaluation']
    assert report['actor_evaluation']['preserve_source_identity']
    assert report['target_laps'] == 3
    assert report['max_steps'] == 4
    assert report['checkpoint_sha256'] == original_hash
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == original_hash


def test_actor_selection_requires_evaluation(monkeypatch):
    import run
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario',
        'scenarios/mappo_solo_lap_completion_transfer.yaml', '--eval-actor', 'car_0'])
    with pytest.raises(ValueError, match='--eval-actor requires --eval'):
        run.main()
