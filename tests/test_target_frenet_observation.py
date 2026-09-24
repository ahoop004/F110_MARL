from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup
from agents.ppo import PPOAgent
from wrappers.observations.neighbors import TargetFrenetComponent

SCALES = dict(delta_s=30., delta_d=5., delta_vs=20., delta_vd=20.)


def test_target_normalization_presence_and_no_clipping():
    component = TargetFrenetComponent(SCALES)
    info = {'target_frenet': dict(delta_s=90., delta_d=-2.5, delta_vs=10., delta_vd=-5.)}
    np.testing.assert_allclose(component.compute({}, info), [3., -.5, .5, -.25, 1.])
    np.testing.assert_array_equal(component.compute({}, {}), np.zeros(5))
    with pytest.raises(ValueError, match='positive'):
        TargetFrenetComponent({**SCALES, 'delta_s': 0.})


@pytest.mark.parametrize('scenario_name', ['ppo_1v1_racing_mpc_circle', 'ppo_1v1_mpc_traffic_circle'])
def test_target_slot_preserves_prefix_and_follows_configured_target(scenario_name):
    from run import build_obs_composers
    scenario = load_and_expand_scenario(f'scenarios/{scenario_name}.yaml')
    pretraining = load_and_expand_scenario('scenarios/ppo_lap_completion_pretrain.yaml')
    directory = Path('scenarios').resolve()
    env, _, _ = create_training_setup(scenario, scenario_dir=directory)
    try:
        old = build_obs_composers(pretraining['agents'], ['car_0'], pretraining['environment'], directory)['car_0']
        new = build_obs_composers(scenario['agents'], ['car_0'], scenario['environment'], directory)['car_0']
        raw, infos = env.reset(seed=42)
        assert new.obs_dim == 163 and old.obs_dim == 158
        vector = new.wrap(raw['car_0'], infos['car_0'])
        np.testing.assert_array_equal(vector[:158], old.wrap(raw['car_0'], infos['car_0']))
        assert infos['car_0']['target_frenet']['agent_id'] == 'car_1'
        assert vector[-1] == 1.
        if len(env.possible_agents) > 2:
            # Place traffic very close; the reserved target remains car_1.
            pose = env.sim.agent_poses[0].copy()
            pose[:2] += .7
            env.sim.agents[2].reset(pose)
            infos = env.step({})[4]
            assert infos['car_0']['target_frenet']['agent_id'] == 'car_1'
            env.configure_agent_targets({'car_0': 'car_3'})
            _, infos = env.reset(seed=42)
            assert infos['car_0']['target_frenet']['agent_id'] == 'car_3'
    finally:
        env.close()


def make_agent(contract, dimension):
    return PPOAgent(dimension, np.array([-1., -1.]), np.array([1., 1.]),
        dict(device='cpu', pi_hidden_dims=[16, 16], vf_hidden_dims=[16, 16], n_steps=4,
             _physics_contract={'version': 1}, _observation_contract=contract))


def contracts():
    from core.scenario import load_yaml_config
    obs = load_yaml_config(Path('configs/observations/rl_racer_simulated_wheel.yaml'))['observation']
    source = {'version': 1, 'observation': obs}
    destination = deepcopy(source)
    destination['observation']['target_frenet'] = {'enabled': True, 'maxima': SCALES}
    return source, destination


def test_expansion_preserves_policy_and_value_and_can_learn(tmp_path):
    source_contract, dest_contract = contracts()
    source, destination = make_agent(source_contract, 158), make_agent(dest_contract, 163)
    checkpoint = tmp_path / 'source.pt'
    source.save(checkpoint)
    destination.load(checkpoint, load_optimizer=False, observation_extension='target_frenet')
    x = torch.randn(12, 158)
    extended = torch.cat([x, torch.randn(12, 5) * 10.], dim=1)
    torch.testing.assert_close(source.actor(x)[0], destination.actor(extended)[0])
    torch.testing.assert_close(source.critic(x), destination.critic(extended))
    for name in ('actor', 'critic'):
        before, after = getattr(source, name), getattr(destination, name)
        torch.testing.assert_close(before.net[0].weight, after.net[0].weight[:, :158], rtol=0, atol=0)
        assert torch.count_nonzero(after.net[0].weight[:, 158:]) == 0
    destination.actor(extended)[0].sum().backward()
    assert torch.count_nonzero(destination.actor.net[0].weight.grad[:, 158:]) > 0
    expanded_checkpoint = tmp_path / 'expanded.pt'
    destination.save(expanded_checkpoint)
    restored = make_agent(dest_contract, 163)
    restored.load(expanded_checkpoint)
    torch.testing.assert_close(destination.actor(extended)[0], restored.actor(extended)[0])


def test_expansion_requires_opt_in_fresh_optimizer_and_identical_prefix(tmp_path):
    old, new = contracts()
    source, dest = make_agent(old, 158), make_agent(new, 163)
    checkpoint = tmp_path / 'source.pt'
    source.save(checkpoint)
    with pytest.raises(ValueError, match='observation_contract'):
        dest.load(checkpoint, load_optimizer=False)
    with pytest.raises(ValueError, match='fresh optimizer'):
        dest.load(checkpoint, observation_extension='target_frenet')
    changed = deepcopy(new)
    changed['observation']['frenet_vehicle_track']['maxima']['vx'] = 10.
    with pytest.raises(ValueError, match='prefix'):
        make_agent(changed, 163).load(checkpoint, load_optimizer=False, observation_extension='target_frenet')
    dest.physics_contract = {'version': 2}
    with pytest.raises(ValueError, match='physics_contract'):
        dest.load(checkpoint, load_optimizer=False, observation_extension='target_frenet')
