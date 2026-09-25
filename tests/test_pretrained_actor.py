import json

import numpy as np
import pytest
import torch

from agents.mappo import MAPPOAgent
from agents.ppo import PPOAgent
from training.hooks import EvaluationCheckpointHook
from training.ppo_evaluator import DeterministicPPOEvaluator


ACTION_LOW = np.array([-0.4, -5.0], dtype=np.float32)
ACTION_HIGH = np.array([0.4, 20.0], dtype=np.float32)


@pytest.mark.parametrize("scenario_name", [
    "mappo_2v2_completion",
    "mappo_2v2_asymmetric",
    "mappo_2v2_combined",
    "mappo_2v2_base_scratch",
    "mappo_2v2_base_pretrained",
    "mappo_2v2_penalties_scratch",
    "mappo_2v2_penalties_pretrained",
])
def test_mf61_2v2_scenario_shares_lidar_actor_and_keeps_roles(tmp_path, scenario_name):
    from pathlib import Path
    from core.agent_builder import get_trainable_agent_ids
    from core.scenario import load_and_expand_scenario, resolve_mappo_config
    from core.setup import create_training_setup
    from run import build_obs_composers, resolve_training_params
    from wrappers.actions.composer import ActionComposer

    scenario = load_and_expand_scenario(f"scenarios/{scenario_name}.yaml")
    from core.provenance import physics_contract
    pretraining = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    assert physics_contract(pretraining['environment']) == physics_contract(scenario['environment'])
    from core.scenario import load_yaml_config
    opponent_profile = load_yaml_config(Path('configs/controllers/racing_mpc.yaml'))
    ids = get_trainable_agent_ids(scenario["agents"])
    assert ids == ["car_0", "car_1"]
    parallel = scenario_name.startswith(("mappo_2v2_base_", "mappo_2v2_penalties_"))
    assert scenario["experiment"]["num_envs"] == (400 if parallel else 1)
    assert scenario["environment"]["vehicle_params"] == pretraining["environment"]["vehicle_params"]
    assert scenario["environment"]["lap_counting"]["count_initial_crossing_as_lap"] is False
    for aid in ["car_2", "car_3"]:
        assert scenario["agents"][aid] == {
            **opponent_profile, "role": "opponent",
            "target_id": "car_0" if aid == "car_2" else "car_1",
        }
    for aid in ids:
        assert scenario["agents"][aid]["action_constraints"]["prevent_reverse"] is True
    assert scenario["agents"]["car_0"]["params"] == scenario["agents"]["car_1"]["params"]

    env, opponents, _ = create_training_setup(scenario, mode="eval", scenario_dir=Path("scenarios").resolve())
    try:
        assert set(opponents) == {"car_2", "car_3"}
        raw, infos = env.reset(seed=42)
        assert len(env.agents) == 4
        composers = build_obs_composers(scenario["agents"], ids, scenario["environment"], Path("scenarios").resolve())
        obs_dim = 192 if scenario_name == "mappo_2v2_asymmetric" else 176
        assert [composers[aid].obs_dim for aid in ids] == [obs_dim, obs_dim]
        source_composer = build_obs_composers(pretraining['agents'], ['car_0'],
            pretraining['environment'], Path('scenarios').resolve())['car_0']
        assert source_composer.obs_dim == 158
        for aid in ids:
            from copy import deepcopy
            prefix = deepcopy(composers[aid].contract)
            prefix['observation'].pop('frenet_neighbors')
            assert prefix == source_composer.contract
            assert len(infos[aid]['frenet_neighbors']) == 3
            assert sum(n['is_teammate'] for n in infos[aid]['frenet_neighbors']) == 1
            wrapped = composers[aid].wrap(raw[aid], infos[aid])
            np.testing.assert_allclose(wrapped[:108], np.minimum(raw[aid]['lidar'] / 10., 1.), atol=1e-7)
            assert np.any(wrapped[:108] > 0.)
            slot_dim = 10 if obs_dim == 192 else 6
            slots = wrapped[158:158 + 3 * slot_dim].reshape(3, slot_dim)
            if obs_dim == 192:
                np.testing.assert_array_equal(wrapped[-4:], np.eye(4)[int(aid[-1])])
                for slot, neighbor in zip(slots, infos[aid]['frenet_neighbors']):
                    np.testing.assert_array_equal(slot[6:], np.eye(4)[int(neighbor['agent_id'][-1])])
            np.testing.assert_array_equal(slots[:, 4], 1.)
            assert slots[:, 5].sum() == 1.
        space = env.action_spaces["car_0"]
        source_params = resolve_training_params(pretraining["agents"]["car_0"], pretraining)
        params = resolve_training_params(scenario["agents"]["car_0"], scenario)
        source_params['_observation_contract'] = source_composer.contract
        params['_observation_contract'] = composers['car_0'].contract
        assert params["_action_contract"] == source_params["_action_contract"]
        assert params["learning_rate"] == 1e-4
        assert params["gamma"] == source_params["gamma"]
        source = PPOAgent(158, space.low, space.high, {**source_params, 'n_steps': 4, "device": "cpu"})
        checkpoint = tmp_path / "forward_frenet.pt"
        source.save(str(checkpoint))
        recipient = MAPPOAgent(
            obs_dim, len(env.get_global_state().vector), space.low, space.high, ids,
            {**params, **resolve_mappo_config(scenario), "device": "cpu"},
        )
        critic_before = {key: value.clone() for key, value in recipient.critic.state_dict().items()}
        recipient.load_pretrained_actor(str(checkpoint))
        for key, value in recipient.actor.state_dict().items():
            if key == "net.0.weight":
                torch.testing.assert_close(value[:, :158], source.actor.state_dict()[key])
                assert value[:, 158:].count_nonzero() == 0
            else:
                torch.testing.assert_close(value, source.actor.state_dict()[key])
        for key, value in recipient.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[key])
        assert not recipient.optimizer.state
        observations = np.random.default_rng(42).normal(size=(2, obs_dim)).astype(np.float32)
        actions, _ = recipient.act_batch(ids, observations, deterministic=True)
        for i, aid in enumerate(ids):
            np.testing.assert_allclose(actions[aid], source.predict(observations[i, :158]), atol=1e-7)
        recipient.actor.net(torch.from_numpy(observations)).sum().backward()
        assert recipient.actor.net[0].weight.grad[:, :108].count_nonzero() > 0
        assert recipient.actor.net[0].weight.grad[:, 158:].count_nonzero() > 0
        controls = [ActionComposer.from_config(
            space.low, space.high, scenario["agents"][aid]["action_constraints"], decision_dt=0.05,
        ) for aid in ids]
        assert controls[0].process([0, -1])[1] == pytest.approx(0.0)
        assert controls[1].process([0, 0])[1] == 0
        controls[0].reset()
        assert controls[0].process([0, 0])[1] == 0

        # A same-size reverse-enabled checkpoint must still be rejected.
        source.action_contract = {**source.action_contract, "prevent_reverse": False}
        source.save(str(checkpoint))
        with pytest.raises(ValueError, match="action contract") as error:
            recipient.load_pretrained_actor(str(checkpoint))
        assert f"checkpoint={source.action_contract!r}" in str(error.value)
        assert f"MAPPO={recipient.action_contract!r}" in str(error.value)
    finally:
        env.close()


@pytest.mark.parametrize('mismatch', ['opt_in', 'scales', 'order', 'dimension', 'layer'])
def test_neighbor_extension_rejects_incompatible_input_without_mutating_actor(tmp_path, mismatch):
    from copy import deepcopy
    base_contract = {'version': 1, 'observation': {'frenet_vehicle_track': {
        'enabled': True, 'points': 20, 'track_maxima': {'curvature': 1., 'width': 1.}}}}
    target_contract = deepcopy(base_contract)
    target_contract['observation']['frenet_neighbors'] = {'enabled': True, 'max_neighbors': 3}
    common = {'hidden_dims': [4], 'n_steps': 2, 'device': 'cpu'}
    source = PPOAgent(50, ACTION_LOW, ACTION_HIGH,
                      {**common, '_observation_contract': base_contract})
    checkpoint = tmp_path / 'source.pt'
    source.save(str(checkpoint))
    extension = 'frenet_neighbors'
    if mismatch == 'opt_in':
        extension = None
    elif mismatch == 'scales':
        target_contract['observation']['frenet_vehicle_track']['track_maxima']['width'] = 2.
    elif mismatch == 'order':
        target_contract['observation']['lidar'] = {'enabled': True}
    elif mismatch == 'layer':
        payload = torch.load(checkpoint, weights_only=False)
        payload['actor']['net.2.weight'] = torch.zeros(3, 4)
        torch.save(payload, checkpoint)
    recipient = MAPPOAgent(66 if mismatch == 'dimension' else 65, 12,
        ACTION_LOW, ACTION_HIGH, ['car_0', 'car_1'], {**common,
        '_observation_contract': target_contract,
        'pretrained_actor_observation_extension': extension})
    before = {k: v.clone() for k, v in recipient.actor.state_dict().items()}
    with pytest.raises(ValueError):
        recipient.load_pretrained_actor(str(checkpoint))
    for key, value in recipient.actor.state_dict().items():
        torch.testing.assert_close(value, before[key])


def _ppo(obs_dim=6, hidden_dims=None):
    return PPOAgent(
        obs_dim=obs_dim,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        params={"hidden_dims": hidden_dims or [16, 16], "device": "cpu"},
    )


def _mappo(obs_dim=6, hidden_dims=None):
    return MAPPOAgent(
        obs_dim=obs_dim,
        global_state_dim=12,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        agent_ids=["car_0", "car_1"],
        params={
            "hidden_dims": hidden_dims or [16, 16],
            "device": "cpu",
            "critic_mode": "shared_team",
            "reward_mode": "team_shared",
        },
    )


def test_mappo_loads_only_compatible_ppo_actor(tmp_path):
    ppo = _ppo()
    with torch.no_grad():
        for parameter in ppo.actor.parameters():
            parameter.fill_(0.125)
    checkpoint = tmp_path / "ppo.pt"
    ppo.save(str(checkpoint))

    mappo = _mappo()
    critic_before = {
        name: value.detach().clone() for name, value in mappo.critic.state_dict().items()
    }
    mappo.load_pretrained_actor(str(checkpoint))

    for value in mappo.actor.state_dict().values():
        assert torch.allclose(value, torch.full_like(value, 0.125))
    for name, value in mappo.critic.state_dict().items():
        assert torch.equal(value, critic_before[name])
    assert not mappo.optimizer.state


def test_mappo_rejects_incompatible_ppo_actor_contract(tmp_path):
    checkpoint = tmp_path / "ppo.pt"
    _ppo(obs_dim=5).save(str(checkpoint))

    with pytest.raises(ValueError, match="obs_dim"):
        _mappo(obs_dim=6).load_pretrained_actor(str(checkpoint))


class _SequenceEvaluator:
    def __init__(self, summaries):
        self._summaries = iter(summaries)

    def evaluate(self):
        return next(self._summaries)


class _SavingAgent:
    def __init__(self):
        self.version = 0

    def save(self, path):
        torch.save({"version": self.version}, path)


def test_evaluation_checkpoint_prefers_completion_over_reward(tmp_path):
    agent = _SavingAgent()
    evaluator = _SequenceEvaluator(
        [
            {
                "completion_rate": 0.25,
                "collision_rate": 0.0,
                "mean_progress": 0.8,
                "mean_finish_steps": 100.0,
                "mean_episode_reward": 1000.0,
            },
            {
                "completion_rate": 0.5,
                "collision_rate": 0.1,
                "mean_progress": 0.7,
                "mean_finish_steps": 150.0,
                "mean_episode_reward": -10.0,
            },
        ]
    )
    hook = EvaluationCheckpointHook(
        agent=agent,
        output_dir=str(tmp_path),
        evaluator=evaluator,
        evaluate_every=1,
    )

    hook.on_episode_end(0, 1000.0, {}, {})
    agent.version = 1
    hook.on_episode_end(1, -10.0, {}, {})

    checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
    assert checkpoint["version"] == 1
    assert checkpoint["checkpoint_selection"]["completion_rate"] == 0.5
    records = [json.loads(line) for line in (tmp_path / "evaluation_history.jsonl").read_text().splitlines()]
    assert [record["is_best"] for record in records] == [True, True]


class _ActorOwner:
    def __init__(self):
        self.actor = torch.nn.Linear(1, 1)

    def predict(self, observation):
        return np.zeros(2, dtype=np.float32)


class _Composer:
    def reset(self):
        pass

    def wrap(self, observation, info):
        return np.zeros(3, dtype=np.float32)

    def update_prev_action(self, action):
        pass


class _ActionComposer:
    def process(self, action):
        return action


class _OneStepFinishEnv:
    possible_agents = ["car_0"]
    timestep = 0.01

    def reset(self, seed=None, options=None):
        self.agents = ["car_0"]
        return {"car_0": {}}, {"car_0": {}}

    def step(self, actions):
        self.agents = []
        info = {
            "car_0": {
                "race_completed": True,
                "terminal_reason": "race_complete",
                "terminal_step": 1,
                "lap_count": 1,
            }
        }
        return {"car_0": {}}, {}, {"car_0": True}, {"car_0": False}, info

    def get_agent_state(self, agent_id):
        raise KeyError(agent_id)


def test_deterministic_ppo_evaluator_uses_environment_completion_facts():
    agent = _ActorOwner()
    agent.actor.train()
    evaluator = DeterministicPPOEvaluator(
        env=_OneStepFinishEnv(),
        rl_agent_id="car_0",
        other_agents={},
        obs_composer=_Composer(),
        action_composer=_ActionComposer(),
        episodes=2,
        base_seed=100,
    )

    summary = evaluator.evaluate(agent)

    assert summary["completion_rate"] == 1.0
    assert summary["collision_rate"] == 0.0
    assert summary["mean_finish_steps"] == 1.0
    assert summary["mean_clean_finish_time_s"] == pytest.approx(0.01)
    assert summary["finish_time_sample_count"] == 2
    assert summary["evaluation_protocol"]["seeds"] == [100, 101]
    assert agent.actor.training is True


def test_selection_evaluation_replans_when_opponent_terminates():
    class Env(_OneStepFinishEnv):
        possible_agents = ["car_0", "opponent"]

        def __init__(self):
            self.resets = []

        def reset(self, seed=None, options=None):
            self.steps = 0
            self.agents = self.possible_agents.copy()
            self.resets.append((seed, options))
            return {aid: {} for aid in self.agents}, {}

        def step(self, actions):
            assert set(actions) == set(self.agents)
            self.steps += 1
            self.agents = ["car_0"] if self.steps < 3 else []
            return ({aid: {} for aid in self.possible_agents}, {},
                    {"car_0": self.steps == 3, "opponent": self.steps == 1}, {}, {})

    env = Env()
    decisions = []
    agent = _ActorOwner()

    def predict(obs):
        decisions.append(env.steps)
        return np.zeros(2)

    agent.predict = predict
    evaluator = DeterministicPPOEvaluator(
        env=env, rl_agent_id="car_0", other_agents={"opponent": type(
            "Opponent", (), {"act": lambda self, obs: np.zeros(2)}
        )()}, obs_composer=_Composer(), action_composer=_ActionComposer(),
        episodes=2, base_seed=100, action_repeat=2,
    )
    evaluator.evaluate(agent)
    assert decisions == [0, 1, 0, 1]
    assert env.resets == [(100, {"map_episode_index": 0}), (101, {"map_episode_index": 1})]


def test_evaluation_resets_integrated_speed_each_episode():
    from wrappers.actions.composer import ActionComposer

    class Env(_OneStepFinishEnv):
        def __init__(self):
            self.speeds = []

        def step(self, actions):
            self.speeds.append(float(actions['car_0'][1]))
            return super().step(actions)

    actions = ActionComposer.from_config(ACTION_LOW, ACTION_HIGH,
        dict(speed_control='acceleration', max_acceleration=5, max_deceleration=5,
             prevent_reverse=True), decision_dt=.01)
    actions.process([0, 1])
    env = Env()
    agent = _ActorOwner()
    agent.predict = lambda obs: np.array([0., 1.], dtype=np.float32)
    evaluator = DeterministicPPOEvaluator(
        env=env, rl_agent_id='car_0', other_agents={}, obs_composer=_Composer(),
        action_composer=actions, episodes=2, base_seed=42,
    )
    evaluator.evaluate(agent)
    assert env.speeds == pytest.approx([.05, .05])


def test_fixed_evaluation_protocols_are_disjoint_and_inherit_training_horizon():
    import copy
    from core.scenario import ScenarioError, load_and_expand_scenario, resolve_evaluation_protocol

    for name in ("ppo_lap_completion_pretrain", "ppo_lap_completion_transfer"):
        scenario = load_and_expand_scenario(f"scenarios/{name}.yaml")
        original = copy.deepcopy(scenario)
        selection = resolve_evaluation_protocol(scenario, "selection")
        final = resolve_evaluation_protocol(scenario, "final")
        assert selection == dict(name="selection", seed=10042, episodes=8, max_steps=16000)
        assert final == dict(name="final", seed=20042, episodes=20, max_steps=16000)
        assert scenario == original
        scenario["evaluation"]["final_test"]["seed"] = 10049
        with pytest.raises(ScenarioError, match="disjoint"):
            resolve_evaluation_protocol(scenario, "selection")
        scenario["evaluation"]["final_test"]["seed"] = 10050
        scenario["evaluation"]["max_steps"] = 123
        assert resolve_evaluation_protocol(scenario, "final")["max_steps"] == 123
        del scenario["evaluation"]["final_test"]
        with pytest.raises(ScenarioError, match="requires evaluation.final_test"):
            resolve_evaluation_protocol(scenario, "final")


def test_finish_time_uses_elapsed_physics_steps_and_only_clean_finishes():
    from metrics.racing_eval import aggregate_eval_episodes, create_episode_facts, update_agent_step_facts

    episodes = []
    for index, reason in enumerate(("race_complete", "collision", "time_limit")):
        facts = create_episode_facts(episode=index, agent_ids=["car_0"], trainable_ids=["car_0"], opponent_ids=[])
        update_agent_step_facts(
            facts, step_idx=1, infos={"car_0": {
                "race_completed": reason != "time_limit", "terminal_reason": reason,
                "terminal_step": 0, "time_limit": reason == "time_limit",
            }},
        )
        episodes.append(facts)
    summary = aggregate_eval_episodes(episodes, timestep=.01)
    assert summary["mean_finish_steps"] == 0.0  # Existing ranking metric preserved.
    assert summary["mean_clean_finish_time_s"] == pytest.approx(.01)
    assert summary["clean_finish_count"] == summary["finish_time_sample_count"] == 1
    assert summary["collision_rate"] == pytest.approx(1 / 3)
    assert summary["timeout_rate"] == pytest.approx(1 / 3)
    failed = aggregate_eval_episodes(episodes[1:], timestep=.01)
    assert failed["mean_clean_finish_time_s"] is None
    assert failed["finish_time_sample_count"] == 0


def test_frenet_progress_selection_replaces_idle_but_prioritizes_completion(tmp_path):
    idle = dict(completion_rate=0.0, collision_rate=0.0, mean_progress=0.9,
                mean_net_progress=0.0, mean_finish_steps=None)
    progressing = {**idle, "collision_rate": 1.0, "mean_net_progress": 0.4}
    finished = {**idle, "completion_rate": 0.125, "mean_net_progress": 0.3, "mean_finish_steps": 1000.0}
    # Reward is irrelevant to the model-selection decision.
    evaluator = _SequenceEvaluator([idle, progressing, idle, finished])
    agent = _SavingAgent()
    hook = EvaluationCheckpointHook(agent, str(tmp_path), evaluator, 1,
                                    selection_strategy="completion_progress")
    for episode in range(4):
        agent.version = episode
        hook.on_episode_end(episode, 1000.0 if episode == 2 else -10.0, {}, {})
    records = [json.loads(line) for line in (tmp_path / "evaluation_history.jsonl").read_text().splitlines()]
    assert [record["is_best"] for record in records] == [True, True, False, True]
    assert all(record["selection_strategy"] == "completion_progress" for record in records)
    assert torch.load(tmp_path / "best_model.pt", weights_only=False)["version"] == 3
    score = EvaluationCheckpointHook.selection_score
    assert score(idle) > score(progressing)  # Preserve the baseline strategy.
    assert score(idle, "completion_progress") == score({**idle, "mean_net_progress": 1e-7}, "completion_progress")
    for progress in (None, float("nan")):
        with pytest.raises(ValueError, match="progress deltas"):
            score({**idle, "mean_net_progress": progress}, "completion_progress")


def test_complete_checkpoint_selection_ignores_overshoot_and_prefers_fast_safe_finish(tmp_path):
    fast = dict(completion_rate=1.0, collision_rate=0.0, mean_progress=0.0,
                mean_net_progress=3.50001, mean_finish_steps=3000.0)
    slow = {**fast, "mean_net_progress": 3.50010, "mean_finish_steps": 6000.0}
    unsafe = {**fast, "collision_rate": 0.125, "mean_finish_steps": 1000.0}
    agent = _SavingAgent()
    hook = EvaluationCheckpointHook(agent, str(tmp_path),
                                    _SequenceEvaluator([slow, fast, slow, unsafe]), 1,
                                    selection_strategy="completion_progress")
    for episode in range(4):
        agent.version = episode
        hook.on_episode_end(episode, 0.0, {}, {})
    records = [json.loads(line) for line in (tmp_path / "evaluation_history.jsonl").read_text().splitlines()]
    assert [record["is_best"] for record in records] == [True, True, False, False]
    assert torch.load(tmp_path / "best_model.pt", weights_only=False)["version"] == 1


def test_net_progress_ignores_spawn_position_counts_laps_and_cancels_reverse():
    from metrics.racing_eval import aggregate_eval_episodes, create_episode_facts, update_agent_step_facts

    results = []
    for spawn in (0.05, 0.95):
        facts = create_episode_facts(episode=0, agent_ids=["car_0"], trainable_ids=["car_0"], opponent_ids=[])
        position = spawn
        for index, delta in enumerate([0.0] + [0.1] * 12 + [-0.1] * 2, 1):
            position = (position + delta) % 1.0
            update_agent_step_facts(facts, step_idx=index, infos={"car_0": {
                "centerline": {"progress": position, "progress_delta": delta},
            }}, terminations={"car_0": index == 15})
            if index == 1:
                assert aggregate_eval_episodes([facts])["mean_net_progress"] == 0.0
        # Post-terminal physics must not grant the retired agent progress.
        update_agent_step_facts(facts, step_idx=16, infos={"car_0": {"centerline": {"progress_delta": 0.5}}})
        results.append(facts)
    assert aggregate_eval_episodes(results)["mean_net_progress"] == pytest.approx(1.0)
    missing = create_episode_facts(episode=1, agent_ids=["car_0"], trainable_ids=["car_0"], opponent_ids=[])
    assert aggregate_eval_episodes([missing])["mean_net_progress"] is None


def test_pretraining_reward_favors_forward_motion_and_penalizes_boundaries():
    import math
    from pathlib import Path
    from core.scenario import load_and_expand_scenario
    from run import build_reward_composer, resolve_training_params

    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    cfg = scenario["agents"]["car_0"]
    params = resolve_training_params(cfg, scenario)
    dt = scenario["environment"]["timestep"]
    gamma = params["gamma"]
    reward = build_reward_composer(cfg, Path("scenarios").resolve())
    def step(delta, **info):
        return reward.compute({'track_length': 350., "info": {
            "centerline": {"progress_delta": delta},
            'track_limits': {'exceeded': False}, **info}})[0]
    idle = step(0.0)
    forward = step(3.0 * dt / 350.0)  # Example physical trajectory, not a map dependency.
    backward = step(-3.0 * dt / 350.0)
    assert backward < idle == 0 < forward
    assert (forward + backward) / 2 == pytest.approx(idle)
    assert step(0.1, track_limits={'exceeded': True}) == -1.0
    assert forward == pytest.approx(3.0 * dt)
    trace_seconds = -dt / math.log(gamma * params["gae_lambda"])
    assert .8 < trace_seconds < .9
    # Model a ramp from rest to 3 m/s over two seconds, then sustained progress.
    moving_return = sum(gamma ** i * step(min(3.0, 1.5 * (i + 1) * dt) * dt / 350.0)
                        for i in range(2000))
    waiting_return = idle * (1 - gamma ** 2000) / (1 - gamma)
    assert moving_return > waiting_return


@pytest.mark.parametrize("violation", ["collision", "boundary", None])
def test_strict_clean_finish_rejects_same_step_finish_violation(violation):
    class Env(_OneStepFinishEnv):
        map_name = "L"

        def step(self, actions):
            obs, rewards, terms, truncs, infos = super().step(actions)
            infos["car_0"].update(collision=violation == "collision",
                track_limits={"exceeded": violation == "boundary", "offtrack_distance": 0.1})
            return obs, rewards, terms, truncs, infos

    evaluator = DeterministicPPOEvaluator(env=Env(), rl_agent_id="car_0", other_agents={},
        obs_composer=_Composer(), action_composer=_ActionComposer(), episodes=1, base_seed=100)
    summary = evaluator.evaluate(_ActorOwner())
    assert summary["completion_rate"] == 1.0
    assert summary["strict_clean_finish_count"] == int(violation is None)
    assert summary["per_map"]["L"]["strict_clean_finish_count"] == int(violation is None)
