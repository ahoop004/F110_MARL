from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.mappo import MAPPORolloutBuffer
from agents.mappo import MAPPOAgent
from training.marl_trainer import MARLTrainer


def _one_step_advantage(*, terminated: bool, truncated: bool) -> float:
    buffer = MAPPORolloutBuffer(
        n_steps=2,
        obs_dim=1,
        global_state_dim=1,
        action_dim=2,
        device=torch.device("cpu"),
    )
    buffer.add(
        obs=np.array([0.0], dtype=np.float32),
        global_state=np.array([0.0], dtype=np.float32),
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


def test_mappo_update_accepts_single_step_agent_buffers() -> None:
    agent = MAPPOAgent(
        obs_dim=1,
        global_state_dim=1,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1"],
        params={"n_steps": 2, "n_epochs": 1, "batch_size": 2, "hidden_dims": [4]},
    )
    for aid in agent.agent_ids:
        agent.store(
            agent_id=aid,
            obs=np.array([0.0], dtype=np.float32),
            global_state=np.array([0.0], dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=1.0,
            log_prob=0.0,
            value=0.0,
            terminated=True,
            truncated=False,
        )

    evaluated_actions = []
    evaluate_actions = agent.actor.evaluate_actions

    def record_evaluated_actions(obs, actions):
        evaluated_actions.append(actions.detach().clone())
        return evaluate_actions(obs, actions)

    def reject_resampling(*_args, **_kwargs):
        raise AssertionError("MAPPO update must not sample replacement actions")

    agent.actor.evaluate_actions = record_evaluated_actions
    agent.actor.get_action = reject_resampling
    metrics = agent.update(next_global_state=np.array([0.0], dtype=np.float32))

    assert "train/policy_loss" in metrics
    assert "train/value_loss" in metrics
    assert len(evaluated_actions) == 1
    assert torch.equal(evaluated_actions[0], torch.zeros((2, 2)))


def test_mappo_rejects_pre_lifecycle_global_state_checkpoint(tmp_path) -> None:
    old_agent = MAPPOAgent(
        obs_dim=1,
        global_state_dim=7,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0"],
        params={"hidden_dims": [4]},
    )
    path = tmp_path / "old.pt"
    old_agent.save(str(path))
    lifecycle_agent = MAPPOAgent(
        obs_dim=1,
        global_state_dim=12,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0"],
        params={"hidden_dims": [4]},
    )
    with pytest.raises(ValueError, match="checkpoint contract"):
        lifecycle_agent.load(str(path))


def test_mappo_rejects_checkpoint_from_different_reward_critic_contract(tmp_path) -> None:
    team_agent = MAPPOAgent(
        obs_dim=1,
        global_state_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1"],
        params={
            "hidden_dims": [4],
            "critic_mode": "shared_team",
            "reward_mode": "team_shared",
        },
    )
    path = tmp_path / "team.pt"
    team_agent.save(str(path))
    individual_agent = MAPPOAgent(
        obs_dim=1,
        global_state_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1"],
        params={
            "hidden_dims": [4],
            "critic_mode": "agent_conditioned",
            "reward_mode": "individual",
        },
    )

    with pytest.raises(ValueError, match="checkpoint contract"):
        individual_agent.load(str(path))


class _ObservationComposer:
    def reset(self) -> None:
        pass

    def wrap(self, obs, _info) -> np.ndarray:
        return np.array([obs["value"]], dtype=np.float32)

    def update_prev_action(self, _action) -> None:
        pass


class _RewardComposer:
    def __init__(self, reward: float = 1.0) -> None:
        self.reward = float(reward)

    def reset(self) -> None:
        pass

    def compute(self, _step_info):
        return self.reward, {"unit": self.reward}


class _ActionComposer:
    def process(self, action) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).copy()


class _RecordingHook:
    def __init__(self) -> None:
        self.records = []

    def on_step(self, record) -> None:
        self.records.append(record)

    def on_update(self, _metrics) -> None:
        pass

    def on_episode_end(self, _episode, _reward, _info, _metrics) -> None:
        pass

    def on_training_end(self) -> None:
        pass


class _FakeMAPPOAgent:
    def __init__(self) -> None:
        self.stored = []

    def clear_buffers(self) -> None:
        pass

    def act(self, _obs):
        return np.zeros(2, dtype=np.float32), 0.0

    def evaluate_state(self, _global_state, _agent_id=None) -> float:
        return 0.0

    def store(self, **transition) -> None:
        self.stored.append(transition)

    def any_buffer_full(self) -> bool:
        return False

    def update(self, *, next_global_state):
        return {"updated": float(len(next_global_state))}


class _MixedEndingEnv:
    possible_agents = ["car_0", "car_1"]
    map_name = "test_map"

    def __init__(self) -> None:
        self.actions_seen = []
        self._step = 0
        self.agents = []
        self.episode_done = False

    def reset(self):
        self._step = 0
        self.agents = self.possible_agents.copy()
        self.episode_done = False
        return self._obs(), {aid: {} for aid in self.possible_agents}

    def _obs(self):
        return {
            aid: {"value": float(self._step)}
            for aid in self.possible_agents
        }

    def get_global_state(self):
        return SimpleNamespace(vector=np.array([float(self._step)], dtype=np.float32))

    def step(self, actions):
        self.actions_seen.append(set(actions))
        self._step += 1
        if self._step == 1:
            terminations = {"car_0": False, "car_1": True}
            truncations = {"car_0": False, "car_1": False}
            self.agents = ["car_0"]
        else:
            terminations = {"car_0": False, "car_1": False}
            truncations = {"car_0": True, "car_1": False}
            self.agents = []
            self.episode_done = True
        infos = {aid: {} for aid in self.possible_agents}
        return self._obs(), {aid: 0.0 for aid in self.possible_agents}, terminations, truncations, infos


def test_trainer_stops_collecting_after_individual_agent_termination() -> None:
    env = _MixedEndingEnv()
    agent = _FakeMAPPOAgent()
    hook = _RecordingHook()
    trainer = MARLTrainer(
        env=env,
        agent=agent,
        trainable_ids=["car_0", "car_1"],
        other_agents={},
        obs_composers={aid: _ObservationComposer() for aid in env.possible_agents},
        reward_composers={aid: _RewardComposer() for aid in env.possible_agents},
        action_composer=_ActionComposer(),
        action_repeat=3,
        hooks=[hook],
        focal_agent_id="car_0",
        run_id="terminal-test",
    )

    trainer.train(n_episodes=1)

    assert env.actions_seen == [{"car_0", "car_1"}, {"car_0"}]
    assert [record.agent_id for record in hook.records] == ["car_0", "car_1", "car_0"]
    car_1_records = [record for record in hook.records if record.agent_id == "car_1"]
    assert len(car_1_records) == 1
    assert car_1_records[0].terminated is True
    assert car_1_records[0].truncated is False
    assert hook.records[-1].agent_id == "car_0"
    assert hook.records[-1].terminated is False
    assert hook.records[-1].truncated is True
    assert hook.records[-1].episode_id == "terminal-test_ep000000"


def test_team_reward_mean_uses_fixed_configured_team_size() -> None:
    env = _MixedEndingEnv()
    trainer = MARLTrainer(
        env=env,
        agent=_FakeMAPPOAgent(),
        trainable_ids=["car_0", "car_1"],
        other_agents={},
        obs_composers={aid: _ObservationComposer() for aid in env.possible_agents},
        reward_composers={aid: _RewardComposer() for aid in env.possible_agents},
        action_composer=_ActionComposer(),
        reward_mode="team_shared",
        team_reward_reduction="mean",
    )

    assert trainer._learning_rewards({"car_0": 2.0, "car_1": 6.0}) == {
        "car_0": 4.0,
        "car_1": 4.0,
    }
    # car_1 is no longer active, but the denominator remains two.
    assert trainer._learning_rewards({"car_0": 2.0}) == {"car_0": 1.0}


def test_team_shared_training_records_learning_and_individual_rewards() -> None:
    env = _MixedEndingEnv()
    agent = _FakeMAPPOAgent()
    hook = _RecordingHook()
    trainer = MARLTrainer(
        env=env,
        agent=agent,
        trainable_ids=["car_0", "car_1"],
        other_agents={},
        obs_composers={aid: _ObservationComposer() for aid in env.possible_agents},
        reward_composers={
            "car_0": _RewardComposer(2.0),
            "car_1": _RewardComposer(6.0),
        },
        action_composer=_ActionComposer(),
        hooks=[hook],
        reward_mode="team_shared",
        team_reward_reduction="mean",
    )

    trainer.train(n_episodes=1)

    assert [item["reward"] for item in agent.stored] == [4.0, 4.0, 1.0]
    assert [record.reward for record in hook.records] == [4.0, 4.0, 1.0]
    assert [record.info["individual_reward"] for record in hook.records] == [
        2.0,
        6.0,
        2.0,
    ]
    assert all(record.info["reward_mode"] == "team_shared" for record in hook.records)


def test_agent_conditioned_critic_inputs_identify_focal_agent() -> None:
    agent = MAPPOAgent(
        obs_dim=1,
        global_state_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1"],
        params={"hidden_dims": [4], "critic_mode": "agent_conditioned"},
    )
    state = np.array([3.0, 4.0], dtype=np.float32)

    assert agent._critic_input(state, "car_0").tolist() == [3.0, 4.0, 1.0, 0.0]
    assert agent._critic_input(state, "car_1").tolist() == [3.0, 4.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="known agent_id"):
        agent.evaluate_state(state)


def test_shared_team_critic_uses_unmodified_global_state() -> None:
    agent = MAPPOAgent(
        obs_dim=1,
        global_state_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1"],
        params={"hidden_dims": [4], "critic_mode": "shared_team"},
    )
    state = np.array([3.0, 4.0], dtype=np.float32)

    assert agent.critic_input_dim == 2
    assert agent._critic_input(state, "car_0").tolist() == [3.0, 4.0]
    assert agent._critic_input(state, "car_1").tolist() == [3.0, 4.0]
