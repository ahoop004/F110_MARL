"""Specialists must retain actor identity through inference, updates and saves."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from agents.mappo import MAPPOAgent
from agents.ppo import PPOAgent
from training.parallel_mappo import CollectorAgent, infer_requests


PARAMS = dict(pi_hidden_dims=[8, 8], vf_hidden_dims=[8], n_steps=8,
              n_epochs=2, batch_size=3, device='cpu', learning_rate=.01,
              reward_mode='individual', critic_mode='agent_conditioned', team_return_mode='per_agent')


def make_agent(tmp_path, mode):
    source = PPOAgent(6, -np.ones(2), np.ones(2), PARAMS)
    source.actor.log_std.data.fill_(-.4)
    path = tmp_path / 'source.pt'
    source.save(str(path))
    params = {**PARAMS, 'actor_mode': 'independent' if mode == 'full' else 'shared'}
    if mode == 'lora':
        params['lora'] = dict(mode='per_agent', rank=2, alpha=2, train_log_std=True, per_agent_log_std=True)
    agent = MAPPOAgent(6, 4, -np.ones(2), np.ones(2), ['a', 'b'], params)
    agent.load_pretrained_actor(str(path))
    return agent, source, params


def collect(agent, ids_by_step, collector=None):
    for step, ids in enumerate(ids_by_step):
        obs = {aid: np.array([agent._agent_index[aid], step, .3, -.2, .5, 1.], np.float32) for aid in ids}
        state = np.full(4, step, np.float32)
        actions, lp = agent.act_batch(ids, np.stack([obs[aid] for aid in ids]))
        values = agent.evaluate_states(state, ids)
        for owner in [agent] + ([collector] if collector else []):
            owner.store_batch(ids, observations=obs, global_state=state, actions=actions,
                rewards={aid: (step + 1) * (1 if aid == 'a' else -2) for aid in ids},
                log_probs=lp, values=values,
                terminated={aid: step == len(ids_by_step)-1 or aid not in ids_by_step[step+1] for aid in ids},
                truncated={aid: False for aid in ids}, raw_actions=agent.last_raw_actions)


def specialist_state(agent, aid):
    if agent.actor_mode == 'independent':
        return deepcopy(agent.actor.actors[aid].state_dict())
    index = agent._agent_index[aid]
    return {k: v.clone() for k, v in agent.actor.state_dict().items()
            if k.startswith(f'adapters.{index}.') or k == f'log_stds.{index}'}


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_specialists_start_from_source_and_isolate_updates_with_adam_momentum(tmp_path, mode):
    agent, source, _ = make_agent(tmp_path, mode)
    obs = torch.randn(3, 6)
    mean, std = agent.actor(obs, adapter_indices=torch.tensor([1, 0, 1]))
    expected_mean, expected_std = source.actor(obs)
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, expected_std.expand_as(mean))
    collect(agent, [['b', 'a'], ['a', 'b'], ['b']])
    agent.update(np.zeros(4))
    agent.clear_buffers()
    before_a, before_b = specialist_state(agent, 'a'), specialist_state(agent, 'b')
    critic = deepcopy(agent.critic.state_dict())
    collect(agent, [['a'], ['a'], ['a']])
    agent.update(np.zeros(4))
    assert any(not torch.equal(v, specialist_state(agent, 'a')[k]) for k, v in before_a.items())
    assert all(torch.equal(v, specialist_state(agent, 'b')[k]) for k, v in before_b.items())
    assert any(not torch.equal(v, agent.critic.state_dict()[k]) for k, v in critic.items())
    if mode == 'lora':
        for k, v in source.actor.state_dict().items():
            assert torch.equal(v, agent.actor.base_state_dict()[k])
    else:
        assert not ({p.data_ptr() for p in agent.actor.actors['a'].parameters()}
                    & {p.data_ptr() for p in agent.actor.actors['b'].parameters()})


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_parallel_fragments_and_serial_updates_match(tmp_path, mode):
    agent, _, _ = make_agent(tmp_path, mode)
    parallel = deepcopy(agent)
    names = ('agent_ids', 'obs_dim', 'global_state_dim', 'action_dim', 'gamma',
             'gae_lambda', 'critic_mode', 'team_return_mode')
    collector = CollectorAgent({name: getattr(agent, name) for name in names}, 8)
    collect(agent, [['b', 'a'], ['a', 'b'], ['b'], ['b']], collector)
    collector.finish_fragment({aid: 0. for aid in agent.agent_ids})
    torch.manual_seed(7)
    agent.update(np.zeros(4))
    torch.manual_seed(7)
    parallel.update_rollouts(collector.take_fragments())
    for owner in ('actor', 'critic'):
        for k, v in getattr(agent, owner).state_dict().items():
            torch.testing.assert_close(v, getattr(parallel, owner).state_dict()[k])


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_checkpoint_and_parallel_inference_route_specialists(tmp_path, mode):
    from utils.torch_io import safe_load
    agent, _, params = make_agent(tmp_path, mode)
    collect(agent, [['b', 'a'], ['a', 'b'], ['b']])
    agent.update(np.zeros(4))
    path = tmp_path / 'best_model.pt'
    agent.save(str(path))
    saved = safe_load(str(path), map_location='cpu')
    assert saved['actor_routing'] == {'a': 0, 'b': 1}
    if mode == 'full':
        assert set(saved['actors']) == {'a', 'b'} and 'actor' not in saved
    restored = MAPPOAgent(6, 4, -np.ones(2), np.ones(2), ['a', 'b'], params)
    restored.load(str(path))
    obs = np.ones((2, 6), np.float32)
    requests = {0: ('act', (['b', 'a'], obs, np.zeros(4))),
                1: ('act', (['b'], obs[:1], np.zeros(4)))}
    torch.manual_seed(5)
    expected = infer_requests(agent, requests)
    torch.manual_seed(5)
    actual = infer_requests(restored, requests)
    for key in requests:
        for aid in expected[key][0]:
            np.testing.assert_array_equal(expected[key][0][aid], actual[key][0][aid])
    if mode == 'full':
        with pytest.raises(ValueError, match='actor_mode'):
            MAPPOAgent(6, 4, -np.ones(2), np.ones(2), ['a', 'b'], PARAMS).load(str(path))
        saved['actors'].pop('b')
        torch.save(saved, path)
        with pytest.raises(ValueError, match='every learner'):
            restored.load(str(path))


def test_full_and_lora_have_identical_initial_critic(tmp_path):
    torch.manual_seed(42)
    full, _, _ = make_agent(tmp_path, 'full')
    torch.manual_seed(42)
    lora, _, _ = make_agent(tmp_path, 'lora')
    for k, v in full.critic.state_dict().items():
        torch.testing.assert_close(v, lora.critic.state_dict()[k])


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_legacy_shared_mappo_can_initialize_both_arms_without_restoring_critic(tmp_path, mode):
    from utils.torch_io import safe_load
    source = MAPPOAgent(6, 4, -np.ones(2), np.ones(2), ['a', 'b'], PARAMS)
    path = tmp_path / 'shared.pt'
    source.save(str(path))
    checkpoint = safe_load(str(path), map_location='cpu')
    checkpoint.pop('actor_mode')  # Historical shared-actor format.
    torch.save(checkpoint, path)
    recipient, _, _ = make_agent(tmp_path, mode)
    critic = deepcopy(recipient.critic.state_dict())
    recipient.load_pretrained_actor(str(path))
    obs = torch.randn(3, 6)
    means, stds = recipient.actor(obs, adapter_indices=torch.tensor([1, 0, 1]))
    expected, std = source.actor(obs)
    torch.testing.assert_close(means, expected)
    torch.testing.assert_close(stds, std.expand_as(means))
    for k, v in critic.items():
        torch.testing.assert_close(v, recipient.critic.state_dict()[k])
    assert not recipient.optimizer.state
    assert recipient.pretrained_actor_source['algorithm'] == 'mappo'
