from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.ppo import PPOAgent
from agents.mappo import MAPPOAgent
from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup
from physics.dynamic_models import accl_constraints
from run import resolve_training_params
from training.marl_trainer import MARLTrainer
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer


LOW = np.array([-0.4189, -20.0], dtype=np.float32)
HIGH = np.array([0.4189, 20.0], dtype=np.float32)
CONSTRAINTS = dict(speed_control="acceleration", max_acceleration=5.0,
                   max_deceleration=5.0, prevent_reverse=True, speed_index=1)


def composer(dt=0.01, high=HIGH):
    return ActionComposer.from_config(LOW, high, CONSTRAINTS, decision_dt=dt)


def test_acceleration_integrates_brakes_holds_and_resets_without_windup():
    action_composer = composer(high=np.array([0.4189, 0.1]))
    action = np.array([0.5, 1.0], dtype=np.float32)
    for expected in (.05, .1, .1, .1):
        physical = action_composer.process(action)
        np.testing.assert_allclose(physical, [.20945, expected], atol=1e-7)
        np.testing.assert_array_equal(action, [.5, 1.0])
    assert action_composer.process([0, 0])[1] == pytest.approx(.1)
    assert action_composer.process([0, -1])[1] == pytest.approx(.05)
    for _ in range(4):
        assert action_composer.process([0, -1])[1] == pytest.approx(0, abs=1e-8)
    assert action_composer.process([0, 1])[1] == pytest.approx(.05)
    action_composer.reset()
    assert action_composer.process([0, 1])[1] == pytest.approx(.05)


@pytest.mark.parametrize("dt,decisions", [(.01, 100), (.05, 20)])
def test_integration_uses_decision_interval(dt, decisions):
    actions = composer(dt)
    for _ in range(decisions):
        speed = actions.process([0, 1])[1]
    assert speed == pytest.approx(5.0)


def test_direct_speed_mode_retains_existing_mapping():
    actions = ActionComposer.from_config(LOW, HIGH, {"prevent_reverse": True})
    for command, expected in [(-1, 0), (0, 0), (.5, 10), (1, 20), (.5, 10)]:
        assert actions.process([0, command])[1] == expected


@pytest.mark.parametrize("overrides,dt", [({"speed_control": "typo"}, .01),
    ({"max_acceleration": 0}, .01), ({"max_deceleration": -1}, .01),
    ({}, None), ({}, float("nan")), ({"speed_index": 2}, .01)])
def test_invalid_speed_control_is_rejected(overrides, dt):
    with pytest.raises(ValueError):
        ActionComposer.from_config(LOW, HIGH, {**CONSTRAINTS, **overrides}, decision_dt=dt)


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("repeat", [1, 3])
def test_frenet_environment_observes_applied_reference_rate(mode, repeat):
    path = Path("scenarios/ppo_lap_completion_pretrain_frenet.yaml").resolve()
    scenario = load_and_expand_scenario(str(path))
    scenario["environment"]["action_repeat"] = repeat
    cfg = scenario["agents"]["car_0"]
    dt = .01 * repeat
    observations = ObservationComposer.from_file(str(path.parent / cfg["observation"]), scenario["environment"])
    actions = ActionComposer.from_config(LOW, HIGH, cfg["action_constraints"], decision_dt=dt)
    env, _, _ = create_training_setup(scenario, mode=mode, scenario_dir=path.parent)
    try:
        env.reset(seed=42)
        for command, expected_speed, expected_rate in [(1, 5*dt, 5), (1, 10*dt, 5),
                (0, 10*dt, 0), (-1, 5*dt, -5), (-1, 0, -5), (-1, 0, 0)]:
            physical = actions.process([0, command])
            assert physical[1] == pytest.approx(expected_speed, abs=1e-7)
            for _ in range(repeat):
                raw, _, _, _, infos = env.step({"car_0": physical})
                assert raw["car_0"]["speed_reference_rate"] == pytest.approx(expected_rate, abs=2e-6)
            wrapped = observations.wrap(raw["car_0"], infos["car_0"])
            assert wrapped[115] == pytest.approx(expected_rate / 5, abs=1e-6)
            assert wrapped[116] == pytest.approx(expected_speed / 20, abs=1e-6)
        p = env.params
        for velocity in (0, .8, 5, 10, 19):
            assert accl_constraints(velocity, 100, p['v_switch'], p['a_max'], p['v_min'], p['v_max']) == 5
        assert accl_constraints(20, 100, p['v_switch'], p['a_max'], p['v_min'], p['v_max']) == 0
    finally:
        env.close()


def test_mappo_agents_have_independent_integrators():
    trainer = MARLTrainer(
        env=SimpleNamespace(), trainable_ids=["car_0", "car_1"],
        agent=SimpleNamespace(agent_ids=["car_0", "car_1"]), other_agents={},
        obs_composers={}, reward_composers={}, action_composer=composer(),
    )
    left, right = trainer.action_composers.values()
    assert left.process([0, 1])[1] == pytest.approx(.05)
    assert right.process([0, 0])[1] == 0
    assert left.process([0, 1])[1] == pytest.approx(.1)
    assert right.process([0, 1])[1] == pytest.approx(.05)


def test_checkpoint_control_contract_prevents_same_shape_misinterpretation(tmp_path):
    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain_frenet.yaml")
    params = resolve_training_params(scenario['agents']['car_0'], scenario)
    params.update(pi_hidden_dims=[4], vf_hidden_dims=[4], n_steps=2, device="cpu")
    accelerated = PPOAgent(3, LOW, HIGH, params)
    direct = PPOAgent(3, LOW, HIGH, {**params, '_action_contract': {'speed_control': 'direct'}})
    path = tmp_path / 'actor.pt'
    direct.save(str(path))
    legacy = torch.load(path, weights_only=False)
    del legacy['action_contract']
    torch.save(legacy, path)
    direct.load(str(path))
    with pytest.raises(ValueError, match='action contract'):
        accelerated.load(str(path))
    accelerated.save(str(path))
    accelerated.load(str(path))
    for field, value in [('decision_dt', .02), ('max_acceleration', 3.0)]:
        changed = PPOAgent(3, LOW, HIGH, {**params,
            '_action_contract': {**params['_action_contract'], field: value}})
        with pytest.raises(ValueError, match='action contract'):
            changed.load(str(path))
    changed_bounds = PPOAgent(3, LOW, np.array([.4189, 10.0]), params)
    with pytest.raises(ValueError, match='action bounds'):
        changed_bounds.load(str(path))
    with pytest.raises(ValueError, match='action contract'):
        direct.load(str(path))
    recipient = MAPPOAgent(3, 2, LOW, HIGH, ['car_0', 'car_1'], params)
    recipient.load_pretrained_actor(str(path))
    wrong = MAPPOAgent(3, 2, LOW, HIGH, ['car_0', 'car_1'],
        {**params, '_action_contract': {'speed_control': 'direct'}})
    with pytest.raises(ValueError, match='action contract'):
        wrong.load_pretrained_actor(str(path))
    recipient.save(str(path))
    recipient.load(str(path))
    with pytest.raises(ValueError, match='action contract'):
        wrong.load(str(path))
