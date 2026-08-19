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

    metrics = agent.update(next_global_state=np.array([0.0], dtype=np.float32))

    assert "train/policy_loss" in metrics
    assert "train/value_loss" in metrics


class _ObservationComposer:
    def reset(self) -> None:
        pass

    def wrap(self, obs, _info) -> np.ndarray:
        return np.array([obs["value"]], dtype=np.float32)

    def update_prev_action(self, _action) -> None:
        pass


class _RewardComposer:
    def reset(self) -> None:
        pass

    def compute(self, _step_info):
        return 1.0, {"unit": 1.0}


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

    def evaluate_state(self, _global_state) -> float:
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
