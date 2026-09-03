from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.mappo import MAPPORolloutBuffer
from agents.mappo import MAPPOAgent
from src.replay.dataset_writer import DatasetHook, DatasetWriter
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
        self.act_batch_calls = []
        self.value_batch_calls = []

    def clear_buffers(self) -> None:
        pass

    def act(self, _obs):
        return np.zeros(2, dtype=np.float32), 0.0

    def act_batch(self, agent_ids, observations, deterministic=False):
        assert len(agent_ids) == len(observations)
        self.act_batch_calls.append(tuple(agent_ids))
        return (
            {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids},
            {aid: 0.0 for aid in agent_ids},
        )

    def evaluate_state(self, _global_state, _agent_id=None) -> float:
        return 0.0

    def evaluate_states(self, _global_state, agent_ids):
        self.value_batch_calls.append(tuple(agent_ids))
        return {aid: 0.0 for aid in agent_ids}

    def store(self, **transition) -> None:
        self.stored.append(transition)

    def store_batch(
        self,
        agent_ids,
        *,
        observations,
        global_state,
        actions,
        rewards,
        log_probs,
        values,
        terminated,
        truncated,
    ) -> None:
        for aid in agent_ids:
            self.store(
                agent_id=aid,
                obs=observations[aid],
                global_state=global_state,
                action=actions[aid],
                reward=rewards[aid],
                log_prob=log_probs[aid],
                value=values[aid],
                terminated=terminated[aid],
                truncated=truncated[aid],
            )

    def any_buffer_full(self) -> bool:
        return False

    def update(self, *, next_global_state):
        return {"updated": float(len(next_global_state))}


class _MixedEndingEnv:
    possible_agents = ["car_0", "car_1"]
    map_name = "test_map"

    def __init__(self) -> None:
        self.actions_seen = []
        self.global_state_calls = 0
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
        self.global_state_calls += 1
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
    assert env.global_state_calls == 3  # initial state plus one per environment step
    assert agent.act_batch_calls == [("car_0", "car_1"), ("car_0",)]
    assert agent.value_batch_calls == [("car_0", "car_1"), ("car_0",)]
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
        params={
            "hidden_dims": [4],
            "critic_mode": "shared_team",
            "reward_mode": "team_shared",
        },
    )
    state = np.array([3.0, 4.0], dtype=np.float32)

    assert agent.critic_input_dim == 2
    assert agent._critic_input(state, "car_0").tolist() == [3.0, 4.0]
    assert agent._critic_input(state, "car_1").tolist() == [3.0, 4.0]


def test_mappo_rejects_duplicate_agent_ids() -> None:
    with pytest.raises(ValueError, match="unique and ordered"):
        MAPPOAgent(
            obs_dim=1,
            global_state_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
            agent_ids=["car_0", "car_0"],
            params={"hidden_dims": [4]},
        )


def test_mappo_rejects_reordered_agent_conditioned_checkpoint(tmp_path) -> None:
    original = MAPPOAgent(
        obs_dim=1,
        global_state_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1"],
        params={"hidden_dims": [4]},
    )
    path = tmp_path / "ordered.pt"
    original.save(str(path))
    reordered = MAPPOAgent(
        obs_dim=1,
        global_state_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_1", "car_0"],
        params={"hidden_dims": [4]},
    )

    with pytest.raises(ValueError, match="checkpoint contract"):
        reordered.load(str(path))


def test_mappo_rejects_checkpoint_with_different_action_bounds(tmp_path) -> None:
    original = MAPPOAgent(
        obs_dim=1,
        global_state_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0"],
        params={"hidden_dims": [4]},
    )
    path = tmp_path / "bounds.pt"
    original.save(str(path))
    incompatible = MAPPOAgent(
        obs_dim=1,
        global_state_dim=2,
        action_low=np.array([-0.5, -1.0], dtype=np.float32),
        action_high=np.array([0.5, 1.0], dtype=np.float32),
        agent_ids=["car_0"],
        params={"hidden_dims": [4]},
    )

    with pytest.raises(ValueError, match="action bounds differ"):
        incompatible.load(str(path))


def test_batched_deterministic_inference_matches_scalar_and_forwards_once() -> None:
    agent = MAPPOAgent(
        obs_dim=3,
        global_state_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1", "car_2"],
        params={"hidden_dims": [4], "critic_mode": "agent_conditioned"},
    )
    ordered_ids = ["car_2", "car_0"]
    observations = np.array(
        [[2.0, 2.5, 3.0], [0.0, 0.5, 1.0]], dtype=np.float32
    )
    global_state = np.array([4.0, 5.0], dtype=np.float32)
    scalar_actions = {
        aid: agent.act(obs, deterministic=True)[0]
        for aid, obs in zip(ordered_ids, observations)
    }
    scalar_values = {
        aid: agent.evaluate_state(global_state, aid) for aid in ordered_ids
    }

    actor_calls = 0
    critic_calls = 0
    actor_inputs = []

    def count_actor(_module, args, _output):
        nonlocal actor_calls
        actor_calls += 1
        actor_inputs.append(args[0].detach().cpu().numpy().copy())

    def count_critic(_module, _args, _output):
        nonlocal critic_calls
        critic_calls += 1

    actor_handle = agent.actor.register_forward_hook(count_actor)
    critic_handle = agent.critic.register_forward_hook(count_critic)
    try:
        actions, log_probs = agent.act_batch(
            ordered_ids, observations, deterministic=True
        )
        values = agent.evaluate_states(global_state, ordered_ids)
    finally:
        actor_handle.remove()
        critic_handle.remove()

    assert list(actions) == ordered_ids
    assert list(log_probs) == ordered_ids
    assert list(values) == ordered_ids
    assert actor_calls == 1
    assert critic_calls == 1
    np.testing.assert_array_equal(actor_inputs[0], observations)
    assert actor_inputs[0].shape[1] == agent.obs_dim
    for aid in ordered_ids:
        np.testing.assert_allclose(actions[aid], scalar_actions[aid], rtol=0, atol=1e-7)
        assert log_probs[aid] == 0.0
        assert values[aid] == pytest.approx(scalar_values[aid], abs=1e-7)


def test_batched_stochastic_inference_preserves_controlled_rng_order() -> None:
    agent = MAPPOAgent(
        obs_dim=2,
        global_state_dim=1,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1", "car_2"],
        params={"hidden_dims": [4]},
    )
    ordered_ids = ["car_0", "car_1", "car_2"]
    observations = np.array(
        [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]], dtype=np.float32
    )

    torch.manual_seed(1234)
    scalar = [agent.act(obs) for obs in observations]
    torch.manual_seed(1234)
    actions, log_probs = agent.act_batch(ordered_ids, observations)

    for index, agent_id in enumerate(ordered_ids):
        np.testing.assert_allclose(
            actions[agent_id], scalar[index][0], rtol=0, atol=1e-6
        )
        assert log_probs[agent_id] == pytest.approx(scalar[index][1], abs=1e-6)


def test_batched_inference_handles_empty_single_and_invalid_agent_sets() -> None:
    agent = MAPPOAgent(
        obs_dim=2,
        global_state_dim=1,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1"],
        params={"hidden_dims": [4]},
    )
    assert agent.act_batch([], np.empty((0, 2), dtype=np.float32)) == ({}, {})
    assert agent.evaluate_states(np.array([0.0], dtype=np.float32), []) == {}

    actions, log_probs = agent.act_batch(
        ["car_1"], np.array([[1.0, 2.0]], dtype=np.float32)
    )
    assert list(actions) == ["car_1"]
    assert list(log_probs) == ["car_1"]
    assert list(agent.evaluate_states(np.array([0.0], dtype=np.float32), ["car_1"])) == [
        "car_1"
    ]
    with pytest.raises(ValueError, match="duplicate"):
        agent.act_batch(
            ["car_0", "car_0"], np.zeros((2, 2), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="unknown"):
        agent.evaluate_states(np.array([0.0], dtype=np.float32), ["car_9"])
    with pytest.raises(ValueError, match="shape"):
        agent.act_batch(["car_0"], np.zeros((1, 3), dtype=np.float32))


def test_batched_rollout_storage_matches_scalar_buffers(monkeypatch) -> None:
    def make_agent() -> MAPPOAgent:
        return MAPPOAgent(
            obs_dim=2,
            global_state_dim=3,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
            agent_ids=["car_0", "car_1", "car_2"],
            params={"hidden_dims": [4], "n_steps": 4},
        )

    scalar_agent = make_agent()
    batch_agent = make_agent()
    ordered_ids = ["car_2", "car_0"]
    observations = {
        "car_2": np.array([2.0, 2.5], dtype=np.float32),
        "car_0": np.array([0.0, 0.5], dtype=np.float32),
    }
    state = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    actions = {
        "car_2": np.array([0.2, -0.2], dtype=np.float32),
        "car_0": np.array([0.1, -0.1], dtype=np.float32),
    }
    rewards = {"car_2": 2.0, "car_0": 1.0}
    log_probs = {"car_2": -0.2, "car_0": -0.1}
    values = {"car_2": 12.0, "car_0": 10.0}
    terminated = {"car_2": False, "car_0": True}
    truncated = {"car_2": True, "car_0": False}
    for aid in ordered_ids:
        scalar_agent.store(
            agent_id=aid,
            obs=observations[aid],
            global_state=state,
            action=actions[aid],
            reward=rewards[aid],
            log_prob=log_probs[aid],
            value=values[aid],
            terminated=terminated[aid],
            truncated=truncated[aid],
        )

    # Populate the dynamic active-set index cache outside the measured hot path.
    batch_agent.store_batch(
        ordered_ids,
        observations=observations,
        global_state=state,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        terminated=terminated,
        truncated=truncated,
    )
    batch_agent.clear_buffers()

    import agents.mappo as mappo_module

    original_as_tensor = mappo_module.torch.as_tensor
    conversion_calls = 0

    def counted_as_tensor(*args, **kwargs):
        nonlocal conversion_calls
        conversion_calls += 1
        return original_as_tensor(*args, **kwargs)

    monkeypatch.setattr(mappo_module.torch, "as_tensor", counted_as_tensor)
    batch_agent.store_batch(
        ordered_ids,
        observations=observations,
        global_state=state,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        terminated=terminated,
        truncated=truncated,
    )

    assert conversion_calls == 1
    for aid in scalar_agent.agent_ids:
        expected = scalar_agent.buffers[aid]
        actual = batch_agent.buffers[aid]
        assert actual.ptr == expected.ptr
        for field in (
            "obs",
            "global_states",
            "actions",
            "rewards",
            "log_probs",
            "values",
            "terminated",
            "truncated",
        ):
            assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_marl_trainer_skips_transition_construction_without_step_hooks(
    monkeypatch,
) -> None:
    import training.marl_trainer as trainer_module

    def reject_transition(*_args, **_kwargs):
        raise AssertionError("no hook consumes a TransitionRecord")

    monkeypatch.setattr(trainer_module, "TransitionRecord", reject_transition)
    monkeypatch.setattr(
        trainer_module,
        "transition_lifecycle_fields",
        reject_transition,
    )
    env = _MixedEndingEnv()
    agent = _FakeMAPPOAgent()
    trainer = MARLTrainer(
        env=env,
        agent=agent,
        trainable_ids=["car_0", "car_1"],
        other_agents={},
        obs_composers={aid: _ObservationComposer() for aid in env.possible_agents},
        reward_composers={aid: _RewardComposer() for aid in env.possible_agents},
        action_composer=_ActionComposer(),
        hooks=[],
    )

    trainer.train(n_episodes=1)
    assert len(agent.stored) == 3


def test_marl_dataset_hook_retains_complete_per_agent_transitions(tmp_path) -> None:
    env = _MixedEndingEnv()
    agent = _FakeMAPPOAgent()
    writer = DatasetWriter(tmp_path / "dataset", chunk_size=10)
    trainer = MARLTrainer(
        env=env,
        agent=agent,
        trainable_ids=["car_0", "car_1"],
        other_agents={},
        obs_composers={aid: _ObservationComposer() for aid in env.possible_agents},
        reward_composers={aid: _RewardComposer() for aid in env.possible_agents},
        action_composer=_ActionComposer(),
        hooks=[DatasetHook(writer)],
        run_id="dataset-test",
    )

    trainer.train(n_episodes=1)
    chunk = np.load(tmp_path / "dataset" / "transitions_000000.npz", allow_pickle=True)
    assert chunk["agent_id"].tolist() == ["car_0", "car_1", "car_0"]
    assert chunk["episode_id"].tolist() == [
        "dataset-test_ep000000",
        "dataset-test_ep000000",
        "dataset-test_ep000000",
    ]
    assert chunk["terminated"].tolist() == [False, True, False]
    assert chunk["truncated"].tolist() == [False, False, True]
    assert chunk["global_state"].shape == (3, 1)
