from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.ppo import PPOAgent, RolloutBuffer
from training.on_policy_trainer import OnPolicyTrainer


def _one_step_advantage(*, terminated: bool, truncated: bool) -> float:
    buffer = RolloutBuffer(
        n_steps=2,
        obs_dim=1,
        action_dim=2,
        device=torch.device("cpu"),
    )
    buffer.add(
        obs=np.array([0.0], dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        reward=1.0,
        log_prob=0.0,
        value=0.0,
        terminated=terminated,
        truncated=truncated,
    )
    advantages, _ = buffer.compute_gae(next_value=10.0, gamma=0.9, gae_lambda=0.95)
    return float(advantages[0])


def test_true_termination_blocks_bootstrap() -> None:
    assert _one_step_advantage(terminated=True, truncated=False) == pytest.approx(1.0)


def test_time_limit_truncation_bootstraps_final_state() -> None:
    assert _one_step_advantage(terminated=False, truncated=True) == pytest.approx(10.0)


def test_ppo_update_scores_stored_actions_without_resampling() -> None:
    agent = PPOAgent(
        obs_dim=1,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        params={"n_steps": 2, "n_epochs": 1, "batch_size": 2, "hidden_dims": [4]},
    )
    stored_actions = (
        np.array([-0.25, 0.5], dtype=np.float32),
        np.array([0.75, -0.5], dtype=np.float32),
    )
    for index, action in enumerate(stored_actions):
        agent.buffer.add(
            obs=np.array([float(index)], dtype=np.float32),
            action=action,
            reward=float(index + 1),
            log_prob=0.0,
            value=0.0,
            terminated=index == 1,
            truncated=False,
        )

    evaluated_actions = []
    evaluate_actions = agent.actor.evaluate_actions

    def record_evaluated_actions(obs, actions):
        evaluated_actions.append(actions.detach().clone())
        return evaluate_actions(obs, actions)

    def reject_resampling(*_args, **_kwargs):
        raise AssertionError("PPO update must not sample replacement actions")

    agent.actor.evaluate_actions = record_evaluated_actions
    agent.actor.get_action = reject_resampling
    metrics = agent.update(next_value=0.0)

    assert "train/policy_loss" in metrics
    assert len(evaluated_actions) == 1
    assert {
        tuple(row.tolist()) for row in evaluated_actions[0]
    } == {tuple(action.tolist()) for action in stored_actions}


def test_ppo_updates_a_single_transition_rollout() -> None:
    agent = PPOAgent(
        obs_dim=1,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        params={"n_steps": 2, "n_epochs": 1, "batch_size": 2, "hidden_dims": [4]},
    )
    agent.buffer.add(
        obs=np.array([0.0], dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        reward=1.0,
        log_prob=0.0,
        value=0.0,
        terminated=True,
        truncated=False,
    )

    metrics = agent.update(next_value=0.0)

    assert "train/policy_loss" in metrics
    assert "train/value_loss" in metrics


class _RecordingBuffer:
    def __init__(self) -> None:
        self.transitions = []

    def clear(self) -> None:
        pass

    def add(self, *args, **kwargs) -> None:
        self.transitions.append((args, kwargs))

    def is_full(self) -> bool:
        return False


class _RecordingAgent:
    def __init__(self) -> None:
        self.buffer = _RecordingBuffer()
        self.next_values = []

    def act(self, obs):
        return np.zeros(2, dtype=np.float32), 0.0, float(obs[0])

    def update(self, next_value):
        self.next_values.append(float(next_value))
        return {}


class _OneStepTruncationEnv:
    map_name = "test_map"

    def __init__(self) -> None:
        self.agents = ["car_0"]

    def reset(self, options=None):
        self.agents = ["car_0"]
        return {"car_0": {"value": 0.0}}, {"car_0": {}}

    def step(self, _actions):
        self.agents = []
        return (
            {"car_0": {"value": 7.0}},
            {"car_0": 0.0},
            {"car_0": False},
            {"car_0": True},
            {"car_0": {"terminal_reason": "time_limit"}},
        )

    def get_global_state(self):
        return type(
            "State",
            (),
            {"vector": np.zeros(1, dtype=np.float32), "masks": {}},
        )()


class _ObservationComposer:
    def reset(self) -> None:
        pass

    def wrap(self, obs, _info):
        return np.array([obs["value"]], dtype=np.float32)

    def update_prev_action(self, _action) -> None:
        pass


class _RewardComposer:
    def reset(self) -> None:
        pass

    def compute(self, _context):
        return 1.0, {}


class _ActionComposer:
    def process(self, action):
        return np.asarray(action, dtype=np.float32)


def test_on_policy_trainer_bootstraps_a_truncated_final_observation() -> None:
    agent = _RecordingAgent()
    trainer = OnPolicyTrainer(
        env=_OneStepTruncationEnv(),
        rl_agent_id="car_0",
        agent=agent,
        other_agents={},
        obs_composer=_ObservationComposer(),
        reward_composer=_RewardComposer(),
        action_composer=_ActionComposer(),
    )

    trainer.train(n_episodes=1)

    assert agent.next_values == [7.0]
    _, lifecycle = agent.buffer.transitions[0]
    assert lifecycle["terminated"] is False
    assert lifecycle["truncated"] is True
