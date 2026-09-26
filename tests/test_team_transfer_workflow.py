"""Team identity, physical-time rewards, and evaluated MAPPO checkpoints."""
import numpy as np
import pytest

from env.centerline_state import build_relative_frenet_facts
from wrappers.observations.neighbors import FrenetNeighborsComponent
from wrappers.rewards.completion import StepTimePenaltyComponent
from training.hooks import EvaluationCheckpointHook


def test_team_flag_follows_identity_when_neighbors_reorder():
    component = FrenetNeighborsComponent(max_neighbors=3, include_team=True)
    teams = {"car_0": "a", "car_1": "a", "car_2": "b"}
    states = {"car_0": {"s": 0}, "car_1": {"s": 3}, "car_2": {"s": 1}}
    for teammate_s, expected in [(3, [0, 1, 0]), (.5, [1, 0, 0])]:
        states["car_1"]["s"] = teammate_s
        facts = build_relative_frenet_facts(states, track_length=100, closed=True,
                                            agent_teams=teams)
        out = component.compute({}, {"frenet_neighbors": facts["car_0"]})
        assert out.shape == (18,)
        np.testing.assert_array_equal(out[5::6], expected)
        np.testing.assert_array_equal(out[4::6], [1, 1, 0])
    with pytest.raises(ValueError, match="agent_teams"):
        component.compute({}, {"frenet_neighbors": [{"agent_id": "car_1"}]})


def test_time_cost_is_invariant_to_physics_step_partition():
    reward = StepTimePenaltyComponent({"per_second": -.00025})
    totals = [sum(reward.compute({"timestep": dt})["step_time/penalty"]
                  for _ in range(round(800 / dt))) for dt in (.01, .05, .1)]
    assert totals == pytest.approx([-.2, -.2, -.2])


@pytest.mark.parametrize("strategy,key", [("team_combined", "team_rank_score"),
                                         ("team_first_place", "team_first_place"),
                                         ("team_sweep", "team_sweep")])
def test_checkpoint_ranks_completion_then_objective_then_safety(strategy, key):
    baseline = {"team_both_finished_rate": 1., key: .5, "team_collision_rate": 0.,
                "mean_net_progress": 3.}
    score = EvaluationCheckpointHook.selection_score
    assert score(baseline, strategy) > score({**baseline, "team_both_finished_rate": .5, key: 1.}, strategy)
    assert score({**baseline, key: .6}, strategy) > score(baseline, strategy)
    assert score(baseline, strategy) > score({**baseline, "team_collision_rate": .1}, strategy)


@pytest.mark.parametrize("pretrained", [False, True])
def test_mappo_training_writes_evaluated_checkpoint(tmp_path, monkeypatch, pretrained):
    import json
    import sys
    import run
    from core.scenario import load_and_expand_scenario
    from utils.torch_io import safe_load

    scenario = load_and_expand_scenario('scenarios/mappo_2v2_race.yaml')
    scenario['experiment'].update(episodes=1, torch_threads=1)
    scenario['environment'].update(max_steps=4, map_bundles_eval=['circle_map'])
    scenario['evaluation'].update(every_episodes=1, episodes=2, max_steps=4)
    scenario['training_defaults'].update(n_steps=4, n_epochs=1, batch_size=4, device='cpu')
    extra_args = []
    if pretrained:
        from pathlib import Path
        from agents.ppo import PPOAgent
        from env.spaces_builder import build_action_spaces
        pretrain = load_and_expand_scenario('scenarios/ppo_lap_completion_pretrain.yaml')
        cfg = pretrain['agents']['car_0']
        composer = run.build_obs_composer(cfg, pretrain['environment'], Path('scenarios'))
        params = run.resolve_training_params(cfg, pretrain)
        params.update(device='cpu', n_steps=4, _observation_contract=composer.contract)
        space, _ = build_action_spaces(["car_0"], pretrain["environment"]["vehicle_params"])
        actor = PPOAgent(composer.obs_dim, space.low, space.high, params)
        checkpoint = tmp_path / 'source.pt'
        actor.save(str(checkpoint))
        extra_args = ['--pretrained-actor', str(checkpoint)]
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *_args, **_kw: scenario)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', 'scenarios/mappo_2v2_race.yaml',
                                    '--no-wandb', '--quiet', '--output-dir', str(tmp_path), *extra_args])
    # Evaluation must not alter training RNG streams, actor weights, or mode.
    import random
    import torch
    from training.mappo_evaluator import DeterministicMAPPOEvaluator
    original = DeterministicMAPPOEvaluator.evaluate
    def checked_evaluate(evaluator):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        torch_state = torch.random.get_rng_state().clone()
        weights = {key: value.clone() for key, value in evaluator.agent.actor.state_dict().items()}
        raw_actions = {aid: value.copy() for aid, value in evaluator.agent.last_raw_actions.items()}
        training = evaluator.agent.actor.training
        result = original(evaluator)
        after = np.random.get_state()
        assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
        np.testing.assert_array_equal(numpy_state[1], after[1])
        assert random.getstate() == python_state
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        assert evaluator.agent.actor.training == training
        assert evaluator.agent.last_raw_actions.keys() == raw_actions.keys()
        for aid, value in raw_actions.items():
            np.testing.assert_array_equal(evaluator.agent.last_raw_actions[aid], value)
        for key, value in evaluator.agent.actor.state_dict().items():
            assert torch.equal(value, weights[key])
        return result
    monkeypatch.setattr(DeterministicMAPPOEvaluator, 'evaluate', checked_evaluate)
    run.main()
    records = [json.loads(line) for line in (tmp_path / 'evaluation_history.jsonl').read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record['environment_steps'] == 4
    assert record['selection_strategy'] == 'team_combined'
    assert record['evaluation_protocol']['seeds'] == [10042, 10043]
    assert 'team_rank_score' in record
    assert 'circle_map' in record['per_map']
    checkpoint = safe_load(str(tmp_path / 'best_model.pt'), map_location='cpu')
    assert checkpoint['checkpoint_selection'] == record
    assert checkpoint['obs_dim'] == 176  # 158 driving inputs plus three 6-value neighbors.
    assert (tmp_path / 'final_model.pt').exists()


@pytest.mark.parametrize('name', ['ppo_lap_completion_pretrain', 'mappo_2v2_race'])
def test_validation_entry_points_cannot_train(name, monkeypatch):
    import sys
    import run
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', f'scenarios/{name}.yaml', '--set', 'experiment.evaluation_only=true', '--quiet'])
    with pytest.raises(ValueError, match='evaluation-only'):
        run.main()
