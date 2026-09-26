"""LoRA must preserve PPO likelihoods, route teammate credit, and freeze its base."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.common.lora import resolve_lora_config
from agents.mappo import MAPPOAgent
from agents.ppo import PPOAgent
from training.parallel_mappo import CollectorAgent, infer_requests


def make_agent(tmp_path, mode="shared", train_log_std=True):
    params = dict(pi_hidden_dims=[8, 8], vf_hidden_dims=[8], n_steps=8,
                  n_epochs=2, batch_size=3, device="cpu", learning_rate=.01,
                  team_return_mode="joint", reward_mode="team_shared", critic_mode="shared_team")
    source = PPOAgent(6, -np.ones(2), np.ones(2), params)
    with torch.no_grad():
        source.actor.log_std.copy_(torch.tensor([-.4, -.2]))
    path = tmp_path / "ppo.pt"
    source.save(str(path))
    agent = MAPPOAgent(6, 4, -np.ones(2), np.ones(2), ["a", "b"], {
        **params, "lora": dict(mode=mode, rank=2, alpha=2, train_log_std=train_log_std)})
    critic = deepcopy(agent.critic.state_dict())
    agent.load_pretrained_actor(str(path))
    for name, value in critic.items():
        torch.testing.assert_close(agent.critic.state_dict()[name], value)
    assert not agent.optimizer.state
    return agent, source


def collector_for(agent):
    names = ("agent_ids", "obs_dim", "global_state_dim", "global_state_contract_version",
             "action_dim", "action_low", "action_high", "observation_contract", "gamma",
             "gae_lambda", "critic_mode", "reward_mode", "team_return_mode", "team_reward_reduction")
    return CollectorAgent({name: getattr(agent, name) for name in names}, 8)


def collect(agent, collector=None):
    for step, ids in enumerate((["b", "a"], ["a", "b"], ["b"], ["b"])):
        # The first feature identifies the actor for the routing assertion below.
        obs = {aid: np.array([agent._agent_index[aid], step, .3, -.2, .5, 1.], np.float32)
               for aid in ids}
        state = np.full(4, step, dtype=np.float32)
        actions, log_probs = agent.act_batch(ids, np.stack([obs[aid] for aid in ids]))
        values = agent.evaluate_states(state, ids)
        terminated = {aid: (aid == "a" and step == 1) or step == 3 for aid in ids}
        for owner in (agent,) if collector is None else (agent, collector):
            owner.store_batch(ids, observations=obs, global_state=state, actions=actions,
                              rewards={aid: step + .5 for aid in ids}, log_probs=log_probs,
                              values=values, terminated=terminated,
                              truncated={aid: False for aid in ids}, raw_actions=agent.last_raw_actions)
            owner.store_team_step(ids, reward=step + .5, value=values[ids[0]], terminal=step == 3)


@pytest.mark.parametrize("mode", ["shared", "per_agent"])
@pytest.mark.parametrize("train_log_std", [True, False])
def test_initial_policy_and_ppo_likelihood_match_source(tmp_path, mode, train_log_std):
    agent, source = make_agent(tmp_path, mode, train_log_std)
    obs = torch.randn(3, 6)
    ids = ["b", "a", "b"]
    routes = torch.tensor([1, 0, 1]) if mode == "per_agent" else None
    expected_mean, expected_std = source.actor(obs)
    actual_mean, actual_std = agent.actor(obs, adapter_indices=routes)
    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_std, expected_std)
    with torch.no_grad():
        actions, old_lp, raw = agent.actor_actions(obs, ids, return_raw=True)
    lp, entropy = agent.actor.evaluate_actions(obs, actions, raw, adapter_indices=routes)
    source_lp, source_entropy = source.actor.evaluate_actions(obs, actions, raw)
    torch.testing.assert_close(lp, old_lp)
    torch.testing.assert_close(lp, source_lp)
    torch.testing.assert_close(entropy, source_entropy)
    trainable = {id(p) for p in agent.actor.parameters() if p.requires_grad}
    frozen = {id(p) for p in agent.actor.net.parameters()}
    optimizer = {id(p) for group in agent.optimizer.param_groups for p in group['params']}
    assert trainable <= optimizer and not frozen & optimizer
    assert agent.actor.log_std.requires_grad is train_log_std


@pytest.mark.parametrize("mode", ["shared", "per_agent"])
def test_update_freezes_base_and_keeps_identity_through_shuffle(tmp_path, monkeypatch, mode):
    agent, _ = make_agent(tmp_path, mode)
    before = deepcopy(agent.actor.state_dict())
    critic_before = deepcopy(agent.critic.state_dict())
    collect(agent)
    import agents.mappo as module
    original = module.ppo_minibatch_step
    seen = []

    def check_rows(agent, observations, *args, **kwargs):
        if mode == "per_agent":
            torch.testing.assert_close(kwargs['adapter_indices'], observations[:, 0].long())
            seen.extend(kwargs['adapter_indices'].tolist())
        return original(agent, observations, *args, **kwargs)

    monkeypatch.setattr(module, 'ppo_minibatch_step', check_rows)
    metrics = agent.update(np.zeros(4))
    assert all(np.isfinite(value) for value in metrics.values())
    for name, value in agent.actor.net.state_dict().items():
        assert torch.equal(value, before['net.' + name])
    assert any(not torch.equal(value, critic_before[name]) for name, value in agent.critic.state_dict().items())
    for index, bank in enumerate(agent.actor.adapters):
        assert any(not torch.equal(layer.B, before[f'adapters.{index}.{key}.B']) for key, layer in bank.items())
    if mode == "per_agent":
        assert seen.count(0) == 4 and seen.count(1) == 8


def test_distinct_adapters_route_parallel_inference_and_active_subset(tmp_path):
    agent, _ = make_agent(tmp_path, "per_agent")
    with torch.no_grad():
        for bank, value in zip(agent.actor.adapters, (.1, -.2)):
            for residual in bank.values():
                residual.B.fill_(value)
    requests = {
        (0, 0): ("act", (["b", "a"], np.ones((2, 6)), np.zeros(4))),
        (0, 1): ("act", (["b"], np.ones((1, 6)), np.zeros(4))),
    }
    results = infer_requests(agent, requests)
    for key, (_, (ids, observations, _)) in requests.items():
        for row, aid in enumerate(ids):
            lp, _ = agent.actor.evaluate_actions(
                torch.tensor(observations[row:row + 1], dtype=torch.float32),
                torch.tensor(results[key][0][aid][None]),
                torch.tensor(results[key][3][aid][None]),
                adapter_indices=torch.tensor([agent._agent_index[aid]]))
            assert lp.item() == pytest.approx(results[key][1][aid], abs=1e-6)
    batch, _ = agent.act_batch(["b", "a"], np.ones((2, 6)), deterministic=True)
    assert not np.allclose(batch['a'], batch['b'])
    for aid in ('a', 'b'):
        scalar, _ = agent.act(np.ones(6), deterministic=True, agent_id=aid)
        np.testing.assert_allclose(scalar, batch[aid], atol=1e-7)
    with pytest.raises(ValueError, match='agent_id'):
        agent.act(np.ones(6))
    with pytest.raises(ValueError, match='adapter index'):
        agent.actor(torch.ones(1, 6))


@pytest.mark.parametrize("mode", ["shared", "per_agent"])
def test_collector_update_matches_serial_with_unequal_teammate_lifetimes(tmp_path, mode):
    serial, _ = make_agent(tmp_path, mode)
    parallel = deepcopy(serial)
    collector = collector_for(serial)
    collect(serial, collector)
    state = np.zeros(4)
    collector.finish_fragment(serial.evaluate_states(state, serial.agent_ids))
    fragments = collector.take_fragments()
    assert np.concatenate([r[2] for r in fragments]).tolist() == [0, 0, 1, 1, 1, 1]
    rng = torch.get_rng_state()
    expected = serial.update(state)
    torch.set_rng_state(rng)
    actual = parallel.update_rollouts(fragments)
    assert actual == pytest.approx(expected, abs=1e-6)
    for left, right in zip(serial._optim_parameters, parallel._optim_parameters):
        torch.testing.assert_close(left, right)
    if mode == "per_agent":
        with pytest.raises(ValueError, match='identity'):
            parallel.update_rollouts([r[:2] for r in fragments])


@pytest.mark.parametrize("mode", ["shared", "per_agent"])
def test_checkpoint_roundtrip_restores_optimizer_and_rejects_wrong_contract(tmp_path, mode):
    agent, _ = make_agent(tmp_path, mode)
    collect(agent)
    agent.update(np.zeros(4))
    agent.clear_buffers()
    checkpoint = tmp_path / 'mappo.pt'
    agent.save(str(checkpoint))
    params = dict(pi_hidden_dims=[8, 8], vf_hidden_dims=[8], n_steps=8,
                  n_epochs=2, batch_size=3, device='cpu', learning_rate=.01,
                  team_return_mode='joint', reward_mode='team_shared', critic_mode='shared_team',
                  lora=agent.lora_config)
    def recipient(overrides=None, ids=None):
        return MAPPOAgent(6, 4, -np.ones(2), np.ones(2), ids or ['a', 'b'], {**params, **(overrides or {})})
    restored = recipient()
    (tmp_path / 'ppo.pt').unlink()  # Checkpoint contains the complete frozen base.
    restored.load(str(checkpoint))
    assert restored.pretrained_actor_source == agent.pretrained_actor_source
    assert restored.optimizer.state
    torch.testing.assert_close(restored.actor.state_dict(), agent.actor.state_dict())
    torch.testing.assert_close(restored.optimizer.state_dict(), agent.optimizer.state_dict())
    obs = np.ones((2, 6), dtype=np.float32)
    actual, _ = restored.act_batch(['b', 'a'], obs, deterministic=True)
    expected, _ = agent.act_batch(['b', 'a'], obs, deterministic=True)
    for aid in actual:
        np.testing.assert_array_equal(actual[aid], expected[aid])
    for override in (None, {**agent.lora_config, 'alpha': 3},
                     {**agent.lora_config, 'rank': 3},
                     {**agent.lora_config, 'mode': 'shared' if mode == 'per_agent' else 'per_agent'},
                     {**agent.lora_config, 'train_log_std': False}):
        wrong = recipient({'lora': override})
        before = deepcopy(wrong.actor.state_dict())
        with pytest.raises(ValueError, match='LoRA contract'):
            wrong.load(str(checkpoint))
        torch.testing.assert_close(before, wrong.actor.state_dict())
    with pytest.raises(ValueError, match='contract'):
        recipient(ids=['b', 'a']).load(str(checkpoint))
    # Optimizer restoration reproduces the next update as well as inference.
    torch.manual_seed(8)
    collect(agent)
    torch.manual_seed(8)
    collect(restored)
    rng = torch.get_rng_state()
    agent.update(np.zeros(4))
    torch.set_rng_state(rng)
    restored.update(np.zeros(4))
    torch.testing.assert_close(restored.actor.state_dict(), agent.actor.state_dict())
    torch.testing.assert_close(restored.critic.state_dict(), agent.critic.state_dict())


def test_absent_teammate_adapter_does_not_move_with_adam_momentum(tmp_path):
    agent, _ = make_agent(tmp_path, 'per_agent')
    collect(agent)
    agent.update(np.zeros(4))
    before = deepcopy(agent.actor.adapters[0].state_dict())
    obs = torch.ones(4, 6)
    actions, old_lp, raw = agent.actor_actions(obs, ['b'] * 4, return_raw=True)
    from agents.common import ppo_minibatch_step
    ppo_minibatch_step(agent, obs, torch.ones(4, 4), actions.detach(), old_lp.detach(),
                       torch.ones(4), torch.ones(4), raw.detach(), adapter_indices=torch.ones(4, dtype=torch.long))
    torch.testing.assert_close(before, agent.actor.adapters[0].state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize('config', [False, {'mode': 'attacker'}, {'rank': 0}, {'rank': True},
                                   {'alpha': float('nan')}, {'alpha': 0}, {'train_log_std': 'false'},
                                   {'dropout': .1}])
def test_invalid_lora_options_fail(config):
    with pytest.raises(ValueError, match='lora'):
        resolve_lora_config(config)


def test_random_frozen_actor_cannot_train_or_save(tmp_path):
    agent = MAPPOAgent(6, 4, -np.ones(2), np.ones(2), ['a', 'b'],
                       dict(hidden_dims=[8], device='cpu', lora={'rank': 2}))
    with pytest.raises(ValueError, match='pretrained PPO'):
        agent.act(np.ones(6))
    with pytest.raises(ValueError, match='pretrained PPO'):
        agent.save(str(tmp_path / 'bad.pt'))


@pytest.mark.parametrize('task,scenario_path,overrides', [('base',
  'scenarios/mappo_2v2_continuous.yaml',
  ['training_defaults.pretrained_actor_checkpoint="../outputs/L_map_pretrain/L_map_best_model.pt"',
   'experiment.name="mappo_2v2_base_pretrained"']),
 ('penalties',
  'scenarios/mappo_2v2_race.yaml',
  ['training_defaults.update_version="parallel-mappo-grouped256-batch2048-v1"',
   'training_defaults.batch_size=2048',
   'training_defaults.pretrained_actor_checkpoint="../outputs/L_map_pretrain/L_map_best_model.pt"',
   'training_defaults.rollout_steps_per_env=256',
   'training_defaults.checkpoint_every_steps=1024000',
   'wandb.group="mappo-2v2-penalties-current-physics"',
   'wandb.tags=["mappo","2v2","terminal-incidents-v1","current-pretrain-physics","racing-mpc-opponents"]',
   'wandb.notes="Matched physics, observations, racing MPC opponents and rewards. Both-finished '
   'rate, then rank plus recorded penalties select checkpoints; clean finish time breaks '
   'successful ties."',
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
   'agents.car_0.reward.task.description="Shared completion/placement reward with recorded '
   'terminal race penalties."',
   'agents.car_0.reward.reward.collision.enabled=false',
   'agents.car_0.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}',
   'agents.car_1.reward.task.name="race_team_2v2_penalties"',
   'agents.car_1.reward.task.description="Shared completion/placement reward with recorded '
   'terminal race penalties."',
   'agents.car_1.reward.reward.collision.enabled=false',
   'agents.car_1.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}'])])
@pytest.mark.parametrize('variant,mode,rank', [('shared', 'shared', 4), ('per_agent', 'per_agent', 4),
                                             ('shared_r8', 'shared', 8)])
def test_lora_scenarios_preserve_task_protocol(task, variant, mode, rank, scenario_path, overrides):
    from core.scenario import load_and_expand_scenario
    source = load_and_expand_scenario(scenario_path, overrides=overrides)
    import json
    adapted = load_and_expand_scenario(scenario_path, overrides=[*overrides,
        'training_defaults.lora=' + json.dumps(dict(mode=mode, rank=rank, alpha=float(rank), train_log_std=True))])
    config = adapted['training_defaults'].pop('lora')
    assert config == dict(mode=mode, rank=rank, alpha=float(rank), train_log_std=True)
    for key in ('experiment', 'wandb'):
        if key == 'experiment':
            adapted[key]['name'] = source[key]['name']
        else:
            adapted[key] = source[key]
    assert adapted == source


@pytest.mark.parametrize('mode', ['shared', 'per_agent'])
def test_lora_cli_parallel_training_and_checkpoint_evaluation(tmp_path, monkeypatch, mode):
    import sys
    import run
    from core.scenario import load_and_expand_scenario
    from env.spaces_builder import build_action_spaces
    from utils.torch_io import safe_load

    path = Path('scenarios/mappo_2v2_race.yaml').resolve()
    scenario = load_and_expand_scenario(str(path), overrides=['training_defaults.update_version="parallel-mappo-grouped256-batch2048-v1"',
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
    scenario['experiment'].update(num_envs=2, num_workers=1, episodes=2, torch_threads=1)
    scenario['environment'].update(max_steps=3, terminate_on_collision=False,
                                   map_bundles=['circle_map'], map_bundles_train=['circle_map'],
                                   map_bundles_eval=['circle_map'])
    scenario['evaluation'].update(every_steps=2, episodes=1, max_steps=3)
    scenario['training_defaults'].update(rollout_steps_per_env=2, device='cpu',
                                         lora=dict(mode=mode, rank=2, alpha=2, train_log_std=True))
    for aid in ('car_0', 'car_1'):
        scenario['agents'][aid]['params'].update(pi_hidden_dims=[8, 8], vf_hidden_dims=[8],
                                                 n_steps=4, n_epochs=1, batch_size=4, device='cpu')
    config = scenario['agents']['car_0']
    composer = run.build_obs_composer(config, scenario['environment'], path.parent)
    params = run.resolve_training_params(config, scenario)
    source_config = {**config, 'observation': '../configs/observations/rl_racer_simulated_wheel.yaml'}
    source_composer = run.build_obs_composer(source_config, scenario['environment'], path.parent)
    params['_observation_contract'] = source_composer.contract
    space, _ = build_action_spaces(['car_0'], scenario['environment']['vehicle_params'])
    source = PPOAgent(source_composer.obs_dim, space.low, space.high, params)
    checkpoint = tmp_path / 'source.pt'
    source.save(str(checkpoint))
    scenario['training_defaults']['pretrained_actor_checkpoint'] = str(checkpoint)
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *_args, **_kwargs: deepcopy(scenario))
    output = tmp_path / 'train'
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', str(path), '--no-wandb', '--quiet',
                                     '--output-dir', str(output)])
    run.main()
    saved = output / 'best_model.pt'
    payload = safe_load(str(saved), map_location='cpu')
    assert payload['lora_contract']['mode'] == mode
    assert payload['obs_dim'] == 176
    assert payload['checkpoint_selection']['environment_steps'] > 0
    for key, value in source.actor.net.state_dict().items():
        actual = payload['actor']['net.' + key]
        if key == '0.weight':
            assert torch.equal(actual[:, :158], value)
            assert actual[:, 158:].count_nonzero() == 0
        else:
            assert torch.equal(actual, value)
    assert any(value.count_nonzero() for key, value in payload['actor'].items()
               if key.startswith('adapters.') and key.endswith('.B'))
    checkpoint.unlink()
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', str(path), '--eval', '--checkpoint',
                                     str(saved), '--eval-episodes', '1', '--no-wandb', '--quiet',
                                     '--output-dir', str(tmp_path / 'eval')])
    run.main()  # Evaluation restores routing without needing the PPO source file.
