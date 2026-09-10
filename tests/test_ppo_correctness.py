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
    def __init__(self) -> None:
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.actions_seen_while_wrapping = []

    def reset(self) -> None:
        self.previous_action.fill(0.0)

    def wrap(self, obs, _info):
        self.actions_seen_while_wrapping.append(self.previous_action.copy())
        return np.array([obs["value"]], dtype=np.float32)

    def update_prev_action(self, action) -> None:
        self.previous_action[:] = np.asarray(action, dtype=np.float32)


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
    composer = _ObservationComposer()
    trainer = OnPolicyTrainer(
        env=_OneStepTruncationEnv(),
        rl_agent_id="car_0",
        agent=agent,
        other_agents={},
        obs_composer=composer,
        reward_composer=_RewardComposer(),
        action_composer=_ActionComposer(),
    )

    trainer.train(n_episodes=1)

    assert agent.next_values == [7.0]
    _, lifecycle = agent.buffer.transitions[0]
    assert lifecycle["terminated"] is False
    assert lifecycle["truncated"] is True


def test_on_policy_next_observation_uses_current_previous_action() -> None:
    class FixedActionAgent(_RecordingAgent):
        ACTION = np.array([0.25, -0.5], dtype=np.float32)

        def act(self, obs):
            return self.ACTION.copy(), 0.0, float(obs[0])

    agent = FixedActionAgent()
    composer = _ObservationComposer()
    trainer = OnPolicyTrainer(
        env=_OneStepTruncationEnv(),
        rl_agent_id="car_0",
        agent=agent,
        other_agents={},
        obs_composer=composer,
        reward_composer=_RewardComposer(),
        action_composer=_ActionComposer(),
    )

    trainer.train(n_episodes=1)

    np.testing.assert_array_equal(
        composer.actions_seen_while_wrapping[0], np.zeros(2, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        composer.actions_seen_while_wrapping[1], FixedActionAgent.ACTION
    )


def test_batched_ppo_inference_and_independent_worker_bootstraps():
    from training.on_policy_trainer import _RemotePolicy

    agent = PPOAgent(1, -np.ones(2), np.ones(2), {"hidden_dims": [4], "device": "cpu"})
    observations = np.array([[0.0], [1.0]], dtype=np.float32)
    actions, log_probs, values = agent.act_batch(observations, deterministic=True)
    assert actions.shape == (2, 2)
    assert log_probs.shape == values.shape == (2,)
    for i, obs in enumerate(observations):
        action, log_prob, value = agent.act(obs, deterministic=True)
        np.testing.assert_allclose(actions[i], action, atol=1e-7)
        assert values[i] == pytest.approx(value)
        assert log_probs[i] == log_prob

    class Connection:
        def send(self, message):
            self.message = message

        def recv(self):
            return {}

    rollouts = []
    for reward, terminal in [(1.0, False), (100.0, True)]:
        connection = Connection()
        worker = _RemotePolicy(connection, 2, 1, 2, 0.9, 0.95)
        worker.buffer.add(np.zeros(1), np.zeros(2), reward, 0.0, 0.0, terminal, not terminal)
        worker.update(next_value=10.0)
        assert connection.message[0] == "rollout"
        rollouts.append(connection.message[1])
    captured = []
    agent._update = lambda *tensors: captured.extend(tensors) or {}
    agent.update_rollouts(rollouts)
    # Each environment bootstraps independently before pooling; a terminal
    # transition cannot contribute its reward/value to another worker's GAE.
    torch.testing.assert_close(captured[3], torch.tensor([10.0, 100.0]))
    torch.testing.assert_close(captured[4], torch.tensor([10.0, 100.0]))


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
)])
def test_ppo_prediction_matches_actions_without_critic_or_rng_changes(device, monkeypatch):
    agent = PPOAgent(115, -np.ones(2), np.ones(2), {"hidden_dims": [256, 256], "device": device})
    observations = np.random.default_rng(42).normal(size=(8, 115)).astype(np.float32)
    expected = [agent.act(obs, deterministic=True)[0] for obs in observations]
    cpu_rng = torch.get_rng_state()
    device_rng = torch.cuda.get_rng_state() if device == "cuda" else None

    def unexpected_critic(*args):
        raise AssertionError("Evaluation must not run the critic")

    monkeypatch.setattr(agent.critic, "forward", unexpected_critic)
    for obs, action in zip(observations, expected):
        np.testing.assert_array_equal(agent.predict(obs), action)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if device_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), device_rng)


def _parallel_test_setup(device="cpu"):
    from pathlib import Path
    from core.scenario import load_and_expand_scenario
    from core.setup import create_training_setup
    from run import build_obs_composer, build_reward_composer
    from wrappers.actions.composer import ActionComposer

    path = Path("scenarios/ppo_lap_completion_pretrain.yaml").resolve()
    scenario = load_and_expand_scenario(str(path))
    scenario["experiment"].update(num_envs=2, episodes=3, seed=42)
    scenario["environment"].update(max_steps=4)
    for key in ("map_bundles", "map_bundles_train", "map_bundles_eval"):
        scenario["environment"][key] = ["circle_map"]
    cfg = scenario["agents"]["car_0"]
    cfg["params"].update(device=device, n_steps=8, n_epochs=1, batch_size=4, hidden_dims=[4])
    env, opponents, _ = create_training_setup(scenario, scenario_dir=path.parent)
    space = env.action_spaces["car_0"]
    obs = build_obs_composer(cfg, scenario["environment"], path.parent)
    agent = PPOAgent(obs.obs_dim, space.low, space.high, cfg["params"])
    trainer = OnPolicyTrainer(
        env, "car_0", agent, opponents, obs, build_reward_composer(cfg, path.parent),
        ActionComposer.from_config(space.low, space.high, cfg["action_constraints"]),
        run_id="parallel-test",
    )
    return trainer, scenario, path.parent


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
)])
def test_parallel_ppo_collects_exact_episodes_and_is_repeatable(device):
    from collections import defaultdict
    from types import SimpleNamespace
    from training.hooks import TrainingHook, WandbHook

    class Capture(TrainingHook):
        def __init__(self):
            self.records, self.episodes, self.updates, self.ends = [], [], [], 0

        def on_step(self, record):
            self.records.append(record)

        def on_episode_end(self, episode, reward, info, metrics):
            self.episodes.append((episode, reward, info, metrics))

        def on_update(self, metrics):
            self.updates.append(metrics)

        def on_training_end(self):
            self.ends += 1

    runs = []
    for _ in range(2):
        trainer, scenario, directory = _parallel_test_setup(device)
        capture = Capture()
        logged = []
        wandb = WandbHook(SimpleNamespace(log_metrics=logged.append))
        trainer.hooks = trainer._transition_hooks = [capture, wandb]
        try:
            trainer.train_parallel(scenario, directory, num_envs=2, n_episodes=3)
        finally:
            trainer.env.close()
        assert len(capture.records) == 12
        episode_logs = [row for row in logged if "episode/number" in row]
        assert len(episode_logs) == 3
        for row, (episode, reward, info, metrics) in zip(episode_logs, capture.episodes):
            assert row["episode/number"] == episode
            assert row["episode/reward"] == reward
            assert row["episode/worker_id"] == info["worker_id"]
            assert row["episode/steps"] == metrics["episode_steps"]
        assert not wandb._episodes
        assert [episode[0] for episode in capture.episodes] == [0, 1, 2]
        assert len(capture.updates) == 2
        assert capture.ends == 1
        episodes = defaultdict(list)
        for record in capture.records:
            episodes[record.episode_id].append(record)
            assert record.info["worker_seed"] == 42 + record.info["worker_id"]
            np.testing.assert_array_equal(record.next_obs[-2:], record.action_norm)
        assert len(episodes) == 3
        for records in episodes.values():
            assert [record.step_idx for record in records] == list(range(4))
            np.testing.assert_array_equal(records[0].obs[-2:], np.zeros(2))
            assert records[-1].truncated and not records[-1].terminated
            for previous, current in zip(records, records[1:]):
                np.testing.assert_array_equal(previous.next_obs, current.obs)
        runs.append(capture)
    for first, second in zip(runs[0].records, runs[1].records):
        assert first.episode_id == second.episode_id
        np.testing.assert_array_equal(first.action_norm, second.action_norm)
        assert first.reward == second.reward


def test_parallel_worker_failure_is_reported_and_children_are_reaped():
    import multiprocessing as mp

    existing = {child.pid for child in mp.active_children()}
    trainer, scenario, directory = _parallel_test_setup()
    scenario["agents"]["car_0"]["observation"] = "/nonexistent/ppo-observation.yaml"
    try:
        with pytest.raises(RuntimeError, match="PPO worker .* failed"):
            trainer.train_parallel(scenario, directory, num_envs=2, n_episodes=3)
    finally:
        trainer.env.close()
    assert {child.pid for child in mp.active_children()} == existing


def test_cli_evaluates_ppo_checkpoint_with_batched_inference_available(tmp_path, monkeypatch):
    import sys
    import run

    trainer, scenario, directory = _parallel_test_setup()
    checkpoint = tmp_path / "ppo.pt"
    trainer.agent.save(str(checkpoint))
    trainer.env.close()
    monkeypatch.setattr(run, "load_and_expand_scenario", lambda _: scenario)
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--scenario", str(directory / "ppo_lap_completion_pretrain.yaml"),
        "--eval", "--checkpoint", str(checkpoint), "--eval-episodes", "1",
        "--output-dir", str(tmp_path / "eval"), "--run-id", "ppo-eval-test",
        "--no-render", "--no-wandb", "--quiet",
    ])
    run.main()
