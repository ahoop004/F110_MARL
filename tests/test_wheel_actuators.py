"""Analytical checks for the independent steering/wheel actuator component."""
import numpy as np
import pytest

from core.scenario import load_yaml_config
from physics.vehicle import WheelActuators


@pytest.fixture
def config():
    return load_yaml_config('configs/vehicle/wheel_actuator_development.yaml')['wheel_actuators']


def test_unsaturated_step_matches_exponential_for_both_actuators(config):
    actuator = WheelActuators(config)
    actuator.command(0.2, 10.0)
    initial = actuator.state
    for elapsed in (0, 0.002, 0.01, 0.1, 1, 10):
        expected = [0.2 * -np.expm1(-elapsed / 0.5),
                    10 * -np.expm1(-elapsed / 0.15)]
        np.testing.assert_allclose(actuator.sample(elapsed), expected, atol=1e-14)
        np.testing.assert_array_equal(actuator.state, initial)
    state = actuator.advance(0.1)
    assert state[1] != 0  # Wheel spins independently; no chassis speed input.
    assert state[1] < actuator.reference[1]
    state[:] = 0
    assert actuator.state[1] > 0  # Public snapshots do not expose internal storage.


def test_rate_limited_step_has_linear_region_then_exponential_tail(config):
    actuator = WheelActuators(config)
    actuator.command(0.0, 100.0)
    # e=100 rad/s, limit=100 rad/s², tau=.15 s => .85 s linear region.
    for elapsed in (.1, .5, .85):
        assert actuator.sample(elapsed)[1] == pytest.approx(100 * elapsed)
    assert actuator.sample(1.0)[1] == pytest.approx(100 - 15 * np.exp(-1))
    assert actuator.sample(100.0)[1] == pytest.approx(100.0)


def test_asymmetric_braking_limit_and_reverse(config):
    config['wheel_rate_min'] = -50.0
    actuator = WheelActuators(config)
    actuator.reset(forward_speed=5.0)  # 100 rad/s rolling initial state.
    actuator.command(0.0, -100.0)
    assert actuator.sample(1.0)[1] == pytest.approx(50.0)
    assert actuator.sample(3.0)[1] == pytest.approx(-50.0)
    assert actuator.sample(100.0)[1] == pytest.approx(-100.0)


@pytest.mark.parametrize('substeps', [1, 2, 10, 100])
def test_partition_invariance_across_saturation_transition(config, substeps):
    actuator = WheelActuators(config)
    actuator.command(.4, 100)
    expected = actuator.sample(1.5)
    for _ in range(substeps):
        actuator.advance(1.5 / substeps)
    np.testing.assert_allclose(actuator.state, expected, atol=1e-12, rtol=1e-12)


def test_reset_supports_rolling_or_explicit_slip_and_clears_old_command(config):
    actuator = WheelActuators(config)
    actuator.command(.4, 400)
    actuator.advance(2)
    actuator.reset(forward_speed=2.0)
    np.testing.assert_allclose(actuator.state, [0, 40])
    np.testing.assert_array_equal(actuator.state, actuator.reference)
    np.testing.assert_array_equal(actuator.advance(1), [0, 40])
    actuator.reset(forward_speed=2.0, wheel_speed=60.0)
    np.testing.assert_array_equal(actuator.state, [0, 60])
    actuator.reset()
    np.testing.assert_array_equal(actuator.state, [0, 0])


def test_commands_saturate_and_instances_keep_independent_state(config):
    first = WheelActuators(config)
    second = WheelActuators(config)
    config['wheel_radius'] = 10
    config['calibration']['status'] = 'measured'
    first.command(1, 500)
    np.testing.assert_array_equal(first.reference, [.4189, 400])
    for _ in range(100):
        first.advance(.1)
    assert 0 <= first.state[0] <= .4189
    assert 0 <= first.state[1] <= 400
    np.testing.assert_array_equal(second.state, [0, 0])
    assert first.params['wheel_radius'] == .05
    assert first.params['calibration']['status'] == 'uncalibrated'
    with pytest.raises(TypeError):
        first.params['wheel_radius'] = 1
    with pytest.raises(TypeError):
        first.params['calibration']['status'] = 'measured'


@pytest.mark.parametrize('key,value', [
    ('wheel_radius', 0), ('wheel_radius', np.nan),
    ('steering_time_constant', -1), ('wheel_speed_time_constant', np.inf),
    ('wheel_rate_min', 0), ('wheel_rate_max', 0),
    ('steering_min', -.5 * np.pi), ('steering_max', .5 * np.pi),
    ('wheel_speed_min', 1), ('version', True), ('wheel_radius', '.05'),
    ('wheel_radius', True), ('calibration', {}), ('delay', .1),
])
def test_invalid_configuration_is_rejected(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        WheelActuators(config)


def test_failed_operations_do_not_mutate_state_or_reference(config):
    actuator = WheelActuators(config)
    actuator.command(.1, 10)
    state, reference = actuator.state, actuator.reference
    for operation in (lambda: actuator.command(np.nan, 0),
                      lambda: actuator.reset(forward_speed=np.inf),
                      lambda: actuator.reset(wheel_speed=401),
                      lambda: actuator.advance(0),
                      lambda: actuator.advance(-1),
                      lambda: actuator.sample(np.nan)):
        with pytest.raises(ValueError):
            operation()
        np.testing.assert_array_equal(actuator.state, state)
        np.testing.assert_array_equal(actuator.reference, reference)


def test_development_profile_cannot_be_silently_used_as_training_config(config):
    from core.scenario import ScenarioError, load_and_expand_scenario, validate_scenario
    scenario = load_and_expand_scenario('scenarios/legacy/ppo.yaml')
    scenario['wheel_actuators'] = config
    with pytest.raises(ScenarioError, match='development profile'):
        validate_scenario(scenario)
