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

    def act(self, observation, deterministic=False):
        assert deterministic is True
        return np.zeros(2, dtype=np.float32), 0.0, 0.0


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

    def reset(self, seed=None):
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
    assert agent.actor.training is True
