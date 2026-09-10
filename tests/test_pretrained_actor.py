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


def test_reverse_frenet_2v2_scenario_transfers_actor_and_keeps_roles(tmp_path):
    from pathlib import Path
    from core.agent_builder import get_trainable_agent_ids
    from core.scenario import load_and_expand_scenario, resolve_mappo_config
    from core.setup import create_training_setup
    from run import build_obs_composers, resolve_training_params
    from wrappers.actions.composer import ActionComposer

    scenario = load_and_expand_scenario("scenarios/mappo_2v2_frenet_ppo_pretrained.yaml")
    pretraining = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain_frenet.yaml")
    baseline = load_and_expand_scenario("scenarios/mappo_2v2_team_shared.yaml")
    ids = get_trainable_agent_ids(scenario["agents"])
    assert ids == ["car_0", "car_1"]
    assert scenario["experiment"]["num_envs"] == 1
    assert scenario["environment"]["vehicle_params"] == pretraining["environment"]["vehicle_params"]
    assert scenario["environment"]["lap_counting"]["count_initial_crossing_as_lap"] is False
    for aid in ["car_2", "car_3"]:
        assert scenario["agents"][aid] == baseline["agents"][aid]
    for aid in ids:
        assert scenario["agents"][aid]["action_constraints"]["prevent_reverse"] is False
        assert scenario["agents"][aid]["observation"] == pretraining["agents"]["car_0"]["observation"]
    assert scenario["agents"]["car_0"]["params"] == scenario["agents"]["car_1"]["params"]

    env, opponents, _ = create_training_setup(scenario, mode="eval", scenario_dir=Path("scenarios").resolve())
    try:
        assert set(opponents) == {"car_2", "car_3"}
        env.reset(seed=42)
        assert len(env.agents) == 4
        composers = build_obs_composers(scenario["agents"], ids, scenario["environment"], Path("scenarios").resolve())
        assert [composers[aid].obs_dim for aid in ids] == [158, 158]
        space = env.action_spaces["car_0"]
        source_params = resolve_training_params(pretraining["agents"]["car_0"], pretraining)
        params = resolve_training_params(scenario["agents"]["car_0"], scenario)
        assert params["_action_contract"] == source_params["_action_contract"]
        assert params["learning_rate"] == 1e-4
        assert params["gamma"] == source_params["gamma"]
        source = PPOAgent(158, space.low, space.high, {**source_params, "device": "cpu"})
        checkpoint = tmp_path / "reverse_frenet.pt"
        source.save(str(checkpoint))
        recipient = MAPPOAgent(
            158, len(env.get_global_state().vector), space.low, space.high, ids,
            {**params, **resolve_mappo_config(scenario), "device": "cpu"},
        )
        critic_before = {key: value.clone() for key, value in recipient.critic.state_dict().items()}
        recipient.load_pretrained_actor(str(checkpoint))
        for key, value in recipient.actor.state_dict().items():
            torch.testing.assert_close(value, source.actor.state_dict()[key])
        for key, value in recipient.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[key])
        assert not recipient.optimizer.state
        observations = np.zeros((2, 158), dtype=np.float32)
        actions, _ = recipient.act_batch(ids, observations, deterministic=True)
        for aid in ids:
            np.testing.assert_allclose(actions[aid], source.predict(observations[0]), atol=1e-7)
        controls = [ActionComposer.from_config(
            space.low, space.high, scenario["agents"][aid]["action_constraints"], decision_dt=0.01,
        ) for aid in ids]
        assert controls[0].process([0, -1])[1] == pytest.approx(-0.05)
        assert controls[1].process([0, 0])[1] == 0
        controls[0].reset()
        assert controls[0].process([0, 0])[1] == 0

        # A same-size forward-only checkpoint must still be rejected.
        source.action_contract = {**source.action_contract, "prevent_reverse": True}
        source.save(str(checkpoint))
        with pytest.raises(ValueError, match="action contract"):
            recipient.load_pretrained_actor(str(checkpoint))
    finally:
        env.close()


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

    for name in ("ppo_lap_completion_pretrain", "ppo_lap_completion_pretrain_frenet"):
        scenario = load_and_expand_scenario(f"scenarios/{name}.yaml")
        original = copy.deepcopy(scenario)
        selection = resolve_evaluation_protocol(scenario, "selection")
        final = resolve_evaluation_protocol(scenario, "final")
        assert selection == dict(name="selection", seed=10042, episodes=8, max_steps=80000)
        assert final == dict(name="final", seed=20042, episodes=20, max_steps=80000)
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
