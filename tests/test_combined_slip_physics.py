"""Physical invariants for the reduced combined-slip vehicle (uncalibrated)."""
import numpy as np
import pytest

from core.scenario import load_yaml_config, load_and_expand_scenario, validate_scenario, ScenarioError
from physics.dynamic_models import combined_slip_dynamics, GRAVITY
from physics.tire_models import smooth_tire_force
from physics.vehicle import CombinedSlipVehicle


@pytest.fixture
def profile():
    return load_yaml_config('configs/vehicle/combined_slip_development.yaml')


def vehicle(profile, **overrides):
    return CombinedSlipVehicle({**profile['combined_slip_vehicle'], **overrides}, profile['wheel_actuators'])


@pytest.mark.parametrize('mu', [0.0, 0.3, 1.05])
def test_tire_force_limit_and_contact_slip_dissipation(mu):
    for u in (-5, -.05, 0, .05, 5):
        for v in (-2, 0, 2):
            for rolling_speed in (-10, 0, 10):
                fx, fy, kappa, alpha = smooth_tire_force(u, v, rolling_speed, mu, 10, 5, .5)
                assert np.all(np.isfinite([fx, fy, kappa, alpha]))
                assert np.hypot(fx, fy) <= mu + 1e-12
                assert fx * (u - rolling_speed) + fy * v <= 1e-12


def test_zero_slip_linear_stiffness_and_nonlinear_saturation():
    assert smooth_tire_force(4, 0, 4, 1, 10, 5, .5)[:2] == (0, 0)
    fx, _, _, _ = smooth_tire_force(4, 0, 4 + 4e-7, 1, 10, 5, .5)
    assert fx == pytest.approx(1e-6, rel=1e-6)
    _, fy, _, _ = smooth_tire_force(4, 4e-7, 4, 1, 10, 5, .5)
    assert fy == pytest.approx(-5e-7, rel=1e-6)
    fx, fy, _, _ = smooth_tire_force(0, 0, 20, 1, 10, 5, .5)
    assert fx == pytest.approx(1, abs=1e-8)
    assert fy == 0


def test_acceleration_and_braking_consume_cornering_capacity():
    _, pure_cornering, _, _ = smooth_tire_force(5, .5, 5, 1, 10, 5, .5)
    for wheel_speed in (0, 10):
        fx, combined_cornering, _, _ = smooth_tire_force(5, .5, wheel_speed, 1, 10, 5, .5)
        assert fx != 0
        assert abs(combined_cornering) < abs(pure_cornering)
        assert np.hypot(fx, combined_cornering) <= 1


def test_force_moment_balance_and_simultaneous_load_transfer(profile):
    car = vehicle(profile)
    car.reset(velocity=(3, .5), yaw_rate=.2, steering_angle=.15, wheel_speed=80)
    details = car.diagnostics()
    front, rear = details['tire_forces']
    fx_body = front[0] * np.cos(.15) - front[1] * np.sin(.15) + rear[0]
    fy_front = front[0] * np.sin(.15) + front[1] * np.cos(.15)
    p = car.params
    ax, ay = details['body_acceleration']
    assert p['m'] * ax == pytest.approx(fx_body)
    assert p['m'] * ay == pytest.approx(fy_front + rear[1])
    assert p['I'] * details['yaw_acceleration'] == pytest.approx(p['lf'] * fy_front - p['lr'] * rear[1])
    assert front[2] + rear[2] == pytest.approx(p['m'] * GRAVITY)
    assert front[2] == pytest.approx(p['m'] * (GRAVITY * p['lr'] - p['h'] * ax) / (p['lf'] + p['lr']))
    assert np.all(np.linalg.norm(details['tire_forces'][:, :2], axis=1)
                  <= p['mu'] * details['tire_forces'][:, 2] + 1e-12)
    # Verify wheel-frame contact velocities include both steering and yaw.
    vf_body = .5 + p['lf'] * .2
    np.testing.assert_allclose(details['contact_velocity'][0],
        [3 * np.cos(.15) + vf_body * np.sin(.15), -3 * np.sin(.15) + vf_body * np.cos(.15)])
    np.testing.assert_allclose(details['contact_velocity'][1], [3, .5 - p['lr'] * .2])
    rhs, _ = combined_slip_dynamics(car.state[:6], car.state[6:], car._dynamics_params)
    assert rhs[3] == pytest.approx(ax + .2 * .5)
    assert rhs[4] == pytest.approx(ay - .2 * 3)


def test_braking_transfers_load_forward_and_acceleration_rearward(profile):
    car = vehicle(profile)
    p = car.params
    static_front = p['m'] * GRAVITY * p['lr'] / (p['lf'] + p['lr'])
    car.reset(velocity=(3, 0), wheel_speed=0)
    braking = car.diagnostics()
    assert braking['body_acceleration'][0] < 0
    assert braking['tire_forces'][0, 2] > static_front
    car.reset(velocity=(3, 0), wheel_speed=100)
    accelerating = car.diagnostics()
    assert accelerating['body_acceleration'][0] > 0
    assert accelerating['tire_forces'][0, 2] < static_front


def test_wheel_spin_does_not_accelerate_chassis_without_grip(profile):
    car = vehicle(profile, mu=0)
    car.command(.3, 100)
    for _ in range(100):
        car.advance(.01)
    np.testing.assert_array_equal(car.state[:6], np.zeros(6))
    assert car.state[6] > 0
    assert car.state[7] > 0
    np.testing.assert_array_equal(car.diagnostics()['tire_forces'][:, :2], np.zeros((2, 2)))


def test_zero_force_free_motion_is_inertially_straight_despite_body_rotation(profile):
    car = vehicle(profile, mu=0)
    psi, vx, vy, r = .3, 2, .5, .4
    car.reset(pose=(1, -2, psi), velocity=(vx, vy), yaw_rate=r)
    car.command(.3, 100)
    result = car.advance(.5)
    world_velocity = [vx * np.cos(psi) - vy * np.sin(psi), vx * np.sin(psi) + vy * np.cos(psi)]
    np.testing.assert_allclose(result[:2], np.array([1, -2]) + .5 * np.array(world_velocity), atol=1e-11)
    assert result[2] == pytest.approx(psi + r * .5)
    assert np.linalg.norm(result[3:5]) == pytest.approx(np.hypot(vx, vy))


def test_zero_load_transfer_recovers_static_axle_loads(profile):
    car = vehicle(profile, h=0)
    car.reset(velocity=(2, 0), wheel_speed=100)
    loads = car.diagnostics()['tire_forces'][:, 2]
    p = car.params
    np.testing.assert_allclose(loads, p['m'] * GRAVITY * np.array([p['lr'], p['lf']]) / (p['lf'] + p['lr']))


def test_lower_grip_reduces_acceleration_at_equal_slip(profile):
    high = vehicle(profile, mu=1.0)
    low = vehicle(profile, mu=.3)
    for car in (high, low):
        car.reset(velocity=(2, 0), wheel_speed=100)
    assert 0 < low.diagnostics()['body_acceleration'][0] < high.diagnostics()['body_acceleration'][0]


def test_coupled_vehicle_straight_line_and_mirror_symmetry(profile):
    straight, left, right = (vehicle(profile) for _ in range(3))
    straight.command(0, 80)
    left.command(.2, 80)
    right.command(-.2, 80)
    for _ in range(100):
        for car in (straight, left, right):
            car.advance(.01)
    np.testing.assert_array_equal(straight.state[[1, 2, 4, 5, 6]], np.zeros(5))
    assert straight.state[0] > 0 and straight.state[3] > 0
    assert left.state[1] > 0 and left.state[5] > 0
    np.testing.assert_allclose(right.state, left.state * [1, -1, -1, 1, -1, -1, -1, 1], atol=1e-12)
    assert not np.isclose(straight.state[3], .05 * straight.state[7])  # traction needs slip


def test_reverse_motion_and_braking_through_zero_are_finite(profile):
    car = vehicle(profile)
    car.command(.15, -60)
    for _ in range(100):
        car.advance(.01)
    assert car.state[3] < 0 and car.state[5] < 0
    car.command(0, 60)
    for _ in range(200):
        assert np.all(np.isfinite(car.advance(.01)))
    assert car.state[3] > 0


def test_outer_step_partition_uses_correct_actuator_stage_times(profile):
    first, second = vehicle(profile), vehicle(profile)
    for car in (first, second):
        car.command(.2, 80)
    for _ in range(100):
        first.advance(.01)
    for _ in range(50):
        second.advance(.02)
    np.testing.assert_allclose(first.state, second.state, atol=1e-11, rtol=1e-11)


def test_long_maneuver_sequence_remains_finite_and_within_grip(profile):
    car = vehicle(profile)
    # One minute of deterministic turn/drive/brake/reverse cycles on an open plane.
    for steering, omega in [(0.2, 80), (-.15, 60), (0, 0), (.15, -40), (0, 0)] * 3:
        car.command(steering, omega)
        for step in range(400):
            assert np.all(np.isfinite(car.advance(.01)))
            if step % 20 == 0:
                forces = car.diagnostics()['tire_forces']
                assert np.all(forces[:, 2] >= 0)
                assert np.all(np.linalg.norm(forces[:, :2], axis=1)
                              <= car.params['mu'] * forces[:, 2] + 1e-12)


def test_invalid_step_or_initial_state_leaves_vehicle_unchanged(profile):
    car = vehicle(profile)
    car.command(.1, 80)
    car.advance(.05)
    state, reference = car.state, car.reference
    for operation in (lambda: car.advance(0), lambda: car.advance(np.nan),
                      lambda: car.reset(pose=(0, 0)),
                      lambda: car.reset(velocity=(np.nan, 0)),
                      lambda: car.reset(velocity=(25, 0))):
        with pytest.raises(ValueError):
            operation()
        np.testing.assert_array_equal(car.state, state)
        np.testing.assert_array_equal(car.reference, reference)


def test_chassis_timestep_convergence(profile):
    results = []
    for integration_step in (.008, .004, .002, .0005):
        car = vehicle(profile, max_integration_step=integration_step)
        car.reset(velocity=(2, 0))
        car.command(.2, 80)
        for _ in range(50):
            car.advance(.024)
        results.append(car.state)
    errors = [np.linalg.norm(value[:6] - results[-1][:6]) for value in results[:-1]]
    assert errors[0] > errors[1] > errors[2]
    np.testing.assert_allclose(results[-2], results[-1], atol=2e-5, rtol=2e-5)


def test_failed_step_and_reset_do_not_commit_partial_subsystem_state(profile):
    car = vehicle(profile, h=1.0)
    car.command(0, 400)
    state, reference = car.state, car.reference
    with pytest.raises(ValueError, match='normal load'):
        car.advance(.05)
    np.testing.assert_array_equal(car.state, state)
    np.testing.assert_array_equal(car.reference, reference)
    with pytest.raises(ValueError, match='normal load'):
        car.reset(wheel_speed=20)
    np.testing.assert_array_equal(car.state, state)
    np.testing.assert_array_equal(car.reference, reference)


def test_reset_and_snapshots_are_independent(profile):
    first, second = vehicle(profile), vehicle(profile)
    first.command(.2, 80)
    first.advance(.1)
    old_state = first.state
    old_state[:] = 100
    assert first.state[0] != 100
    np.testing.assert_array_equal(second.state, np.zeros(8))
    first.reset(pose=(1, 2, .3), velocity=(2, .1))
    assert first.state[7] == 40
    np.testing.assert_array_equal(first.reference, [0, 40])
    first.reset()
    np.testing.assert_array_equal(first.state, np.zeros(8))
    np.testing.assert_array_equal(first.reference, np.zeros(2))


@pytest.mark.parametrize('key,value', [
    ('model', 'mf61'), ('model_version', True), ('tire_model', 'pacejka'),
    ('drivetrain', 'rear_wheel_drive'), ('tire_id', ''), ('m', 0), ('I', -1),
    ('mu', np.nan), ('h', -1), ('slip_speed_floor', 0), ('max_integration_step', np.inf),
    ('front_longitudinal_stiffness', 0), ('rear_cornering_stiffness', '5'),
    ('wheel_radius', .05), ('calibration', {}),
])
def test_invalid_model_parameters_are_rejected(profile, key, value):
    with pytest.raises(ValueError):
        vehicle(profile, **{key: value})


def test_complete_development_profile_cannot_be_mistaken_for_training_config(profile):
    scenario = load_and_expand_scenario('scenarios/ppo.yaml')
    scenario['combined_slip_vehicle'] = profile['combined_slip_vehicle']
    with pytest.raises(ScenarioError, match='development profile'):
        validate_scenario(scenario)
