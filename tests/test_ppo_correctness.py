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

    def record_evaluated_actions(obs, actions, raw_actions=None):
        evaluated_actions.append(actions.detach().clone())
        return evaluate_actions(obs, actions, raw_actions)

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


@pytest.mark.parametrize("mean", [0.0, 12.0, -12.0])
def test_saturated_policy_retains_unit_importance_ratios(mean):
    from agents.common import Actor

    torch.manual_seed(42)
    actor = Actor(1, 2, [])
    with torch.no_grad():
        actor.net[0].weight.zero_()
        actor.net[0].bias.fill_(mean)
        actor.log_std.fill_(2.0)
    observations = torch.zeros(1024, 1)
    actions, old_lp, raw = actor.get_action(observations, return_raw=True)
    assert (actions.abs() >= 1 - 1e-6).any()
    new_lp, entropy = actor.evaluate_actions(observations, actions, raw.detach())
    torch.testing.assert_close((new_lp - old_lp).exp(), torch.ones(1024))
    assert torch.isfinite(new_lp).all() and torch.isfinite(entropy).all()
    (-new_lp.mean() - .001 * entropy.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters())


def test_bounded_entropy_discourages_excessive_gaussian_variance():
    from agents.common import Actor

    actor = Actor(1, 2, [])
    with torch.no_grad():
        actor.net[0].weight.zero_()
        actor.net[0].bias.zero_()
        actor.log_std.fill_(2.0)
    _, entropy = actor.evaluate_actions(torch.zeros(1, 1), torch.zeros(1, 2))
    entropy.sum().backward()
    assert (actor.log_std.grad < 0).all()
    # The quadrature estimate agrees with an independent Monte Carlo integral.
    torch.manual_seed(42)
    raw = torch.randn(100000, 2) * np.exp(2.0)
    expected = (torch.distributions.Normal(0.0, np.exp(2.0)).entropy()
                + 2 * (np.log(2) - raw - torch.nn.functional.softplus(-2 * raw))).sum(-1).mean()
    assert entropy.item() == pytest.approx(expected.item(), abs=0.2)


def test_ppo_pools_owned_fragments_without_cross_episode_bootstrap():
    agent = PPOAgent(1, -np.ones(2), np.ones(2), {
        "hidden_dims": [4], "n_steps": 2, "min_rollout_steps": 2, "gamma": .9,
    })
    captured = []
    agent._update = lambda *tensors: captured.extend(tensors) or {"updated": 1}
    agent.buffer.add(np.zeros(1), np.zeros(2), 1., 0., 0., False, True)
    assert agent.update(10.) == {}
    agent.buffer.clear()
    agent.buffer.add(np.ones(1), np.zeros(2), 100., 0., 0., True, False)
    assert agent.update(999.) == {"updated": 1}
    torch.testing.assert_close(captured[0], torch.tensor([[0.], [1.]]))
    torch.testing.assert_close(captured[3], torch.tensor([10., 100.]))
    assert agent.flush_pending_update() == {}


def test_ppo_kl_limit_stops_further_optimizer_steps():
    torch.manual_seed(42)
    agent = PPOAgent(1, -np.ones(2), np.ones(2), {
        "hidden_dims": [4], "learning_rate": 1., "target_kl": 1e-5,
        "n_epochs": 10, "batch_size": 64,
    })
    obs = np.zeros((64, 1), dtype=np.float32)
    actions, lp, _ = agent.act_batch(obs)
    metrics = agent._update(
        torch.from_numpy(obs), torch.from_numpy(actions), torch.from_numpy(lp),
        torch.from_numpy(actions[:, 0].copy()), torch.ones(64),
        torch.from_numpy(agent.last_raw_actions),
    )
    assert 1 <= metrics["train/optimizer_steps"] < 10
    assert metrics["train/kl_early_stop"] == 1
    assert metrics["train/last_checked_kl"] > agent.target_kl


@pytest.mark.parametrize("params", [
    {"target_kl": 0}, {"target_kl": float("nan")},
    {"min_rollout_steps": 0}, {"log_std_init": 3},
])
def test_invalid_stability_parameters_are_rejected(params):
    with pytest.raises(ValueError):
        PPOAgent(1, -np.ones(2), np.ones(2), {"hidden_dims": [4], **params})


def test_circle_stability_scenario_preserves_physics_and_rewards_slow_progress():
    from pathlib import Path
    from core.scenario import load_and_expand_scenario
    from run import build_reward_composer, resolve_training_params

    baseline = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain_combined_slip.yaml")
    stable = load_and_expand_scenario("scenarios/ppo_combined_slip_circle_stable.yaml")
    assert stable["environment"]["vehicle_params"] == baseline["environment"]["vehicle_params"]
    assert stable["environment"]["episode_termination"]["lap_completion"]
    assert stable["experiment"]["total_steps"] is None
    for key in ("observation", "action_constraints"):
        assert stable["agents"]["car_0"][key] == baseline["agents"]["car_0"][key]
    rewards = [build_reward_composer(s["agents"]["car_0"], Path("scenarios"))
               for s in (baseline, stable)]
    # A 0.25 m/s forward decision on a ~350 m circle becomes worth exploring.
    step = {"track_length": 350., "info": {"track_limits": {"exceeded": False},
            "centerline": {"progress_delta": .25 * stable["environment"]["timestep"] / 350}}}
    assert rewards[0].compute(step)[0] > 0
    assert rewards[1].compute(step)[0] > 0
    params = resolve_training_params(stable["agents"]["car_0"], stable)
    agent = PPOAgent(158, -np.ones(2), np.ones(2), {**params, "device": "cpu"})
    assert agent.min_rollout_steps == agent.n_steps
    assert agent.gamma > resolve_training_params(baseline["agents"]["car_0"], baseline)["gamma"]


def test_pretraining_networks_and_physical_discount_contract(tmp_path):
    from core.scenario import load_and_expand_scenario
    from run import resolve_training_params
    from agents.mappo import MAPPOAgent

    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    params = resolve_training_params(scenario["agents"]["car_0"], scenario)
    agent = PPOAgent(50, -np.ones(2), np.ones(2), {**params, "device": "cpu"})
    for net, expected in ((agent.actor.net, [256, 256, 2]),
                          (agent.critic.net, [512, 512, 1])):
        assert [layer.out_features for layer in net if isinstance(layer, torch.nn.Linear)] == expected
        activations = [layer for layer in net if isinstance(layer, torch.nn.LeakyReLU)]
        assert len(activations) == 2
        for activation in activations:
            torch.testing.assert_close(activation(torch.tensor([-1., 1.])), torch.tensor([-.2, 1.]))
    assert agent.batch_size == 1024
    dt = scenario["environment"]["timestep"] * scenario["environment"]["action_repeat"]
    assert agent.gamma ** (0.05 / dt) == pytest.approx(0.99, abs=1e-12)

    checkpoint = tmp_path / "leaky_ppo.pt"
    agent.save(str(checkpoint))
    mappo_params = {
        "pi_hidden_dims": [256, 256], "vf_hidden_dims": [8], "device": "cpu",
        "_physics_contract": params["_physics_contract"],
        "_action_contract": params["_action_contract"],
        "activation": "leaky_relu", "critic_mode": "shared_team", "reward_mode": "team_shared",
    }
    recipient = MAPPOAgent(50, 12, -np.ones(2), np.ones(2), ["car_0", "car_1"], mappo_params)
    recipient.load_pretrained_actor(str(checkpoint))
    observations = torch.ones(2, 50)
    torch.testing.assert_close(agent.actor.net(observations), recipient.actor.net(observations))
    incompatible = MAPPOAgent(
        50, 12, -np.ones(2), np.ones(2), ["car_0", "car_1"],
        {**mappo_params, "activation": "tanh"},
    )
    with pytest.raises(ValueError, match="activation"):
        incompatible.load_pretrained_actor(str(checkpoint))


@pytest.mark.parametrize("params", [
    {"lr_schedule": "unknown"},
    {"lr_schedule": "linear"},
    {"learning_rate": float("nan")},
    {"lr_schedule": "linear", "learning_rate_end": -1.0},
    {"learning_rate_end": 1e-4},
])
def test_ppo_rejects_invalid_learning_rate_schedules(params):
    with pytest.raises(ValueError, match="learning_rate|learning rate|lr_schedule"):
        PPOAgent(1, -np.ones(2), np.ones(2), params)


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


@pytest.mark.parametrize("schedule", ["constant", "linear"])
def test_single_env_learning_rate_tracks_completed_episodes(schedule, tmp_path):
    from training.hooks import TrainingHook

    class Capture(TrainingHook):
        def __init__(self):
            self.rates = []

        def on_update(self, metrics):
            self.rates.append(metrics["train/learning_rate"])

    params = {"hidden_dims": [4], "n_steps": 2, "n_epochs": 1, "learning_rate": 0.001}
    if schedule == "linear":
        params.update(lr_schedule="linear", learning_rate_end=0.0001)
    agent = PPOAgent(1, -np.ones(2), np.ones(2), params)
    capture = Capture()
    trainer = OnPolicyTrainer(
        _OneStepTruncationEnv(), "car_0", agent, {}, _ObservationComposer(),
        _RewardComposer(), _ActionComposer(), hooks=[capture],
    )
    trainer.train(4)
    expected = [0.001, 0.000775, 0.00055, 0.000325] if schedule == "linear" else [0.001] * 4
    assert capture.rates == pytest.approx(expected)
    endpoint = 0.0001 if schedule == "linear" else 0.001
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(endpoint)
    checkpoint = tmp_path / "ppo.pt"
    agent.save(str(checkpoint))
    loaded = PPOAgent(1, -np.ones(2), np.ones(2), params)
    loaded.load(str(checkpoint))
    assert loaded.optimizer.param_groups[0]["lr"] == pytest.approx(endpoint)


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


def test_trainer_preserves_raw_samples_and_flushes_final_partial_pool():
    agent = PPOAgent(1, -np.ones(2), np.ones(2), {
        "hidden_dims": [4], "n_steps": 2, "min_rollout_steps": 2,
    })
    captured = []
    def update(*tensors):
        captured.append(tensors)
        return {"train/rollout_steps": len(tensors[0])}
    agent._update = update
    OnPolicyTrainer(
        _OneStepTruncationEnv(), "car_0", agent, {}, _ObservationComposer(),
        _RewardComposer(), _ActionComposer(),
    ).train(3)
    assert [len(row[0]) for row in captured] == [2, 1]
    for obs, actions, old_lp, _, _, raw in captured:
        assert torch.isfinite(raw).all()
        torch.testing.assert_close(raw.tanh(), actions)
        new_lp, _ = agent.actor.evaluate_actions(obs, actions, raw)
        torch.testing.assert_close(new_lp, old_lp)


def test_ppo_dataset_state_precedes_repeated_action(tmp_path):
    from types import SimpleNamespace
    from src.replay.dataset_writer import DatasetHook, DatasetWriter

    class Env(_OneStepTruncationEnv):
        def reset(self, options=None):
            self.state = np.array([0.0], dtype=np.float32)
            return super().reset(options)

        def get_global_state(self):
            return SimpleNamespace(vector=self.state, masks={})

        def step(self, actions):
            self.state[0] += 1
            done = self.state[0] == 4
            self.agents = [] if done else ["car_0"]
            return ({"car_0": {"value": float(self.state[0])}}, {},
                    {"car_0": False}, {"car_0": done}, {"car_0": {}})

    writer = DatasetWriter(tmp_path / "dataset")
    trainer = OnPolicyTrainer(
        Env(), "car_0", _RecordingAgent(), {}, _ObservationComposer(),
        _RewardComposer(), _ActionComposer(), action_repeat=2,
        hooks=[DatasetHook(writer)],
    )
    trainer.train(1)
    with np.load(tmp_path / "dataset/transitions_000000.npz") as chunk:
        np.testing.assert_array_equal(chunk["obs"], [[0], [2]])
        np.testing.assert_array_equal(chunk["global_state"], chunk["obs"])
        np.testing.assert_array_equal(chunk["next_obs"], [[2], [4]])
        assert chunk["truncated"].tolist() == [False, True]


def test_ppo_resets_fixed_opponents_before_each_episode():
    class Opponent:
        calls = 9
        history = []

        def reset(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            self.history.append(self.calls)
            return np.zeros(2)

    class Env(_OneStepTruncationEnv):
        def reset(self, options=None):
            obs, info = super().reset(options)
            self.agents.append("opponent")
            obs["opponent"] = {}
            return obs, info

    opponent = Opponent()
    OnPolicyTrainer(
        Env(), "car_0", _RecordingAgent(), {"opponent": opponent},
        _ObservationComposer(), _RewardComposer(), _ActionComposer(),
    ).train(2)
    assert opponent.history == [1, 1]


def test_on_policy_resets_integrated_speed_each_episode():
    from wrappers.actions.composer import ActionComposer

    class Env(_OneStepTruncationEnv):
        def __init__(self):
            super().__init__()
            self.speeds = []

        def step(self, actions):
            self.speeds.append(float(actions['car_0'][1]))
            return super().step(actions)

    agent = _RecordingAgent()
    agent.act = lambda obs: (np.array([0., 1.], dtype=np.float32), 0., 0.)
    actions = ActionComposer.from_config(-np.ones(2), np.ones(2),
        dict(speed_control='acceleration', max_acceleration=5, max_deceleration=5,
             prevent_reverse=True), decision_dt=.01)
    actions.process([0, 1])  # A previous episode must not leak into this run.
    env = Env()
    trainer = OnPolicyTrainer(env, 'car_0', agent, {}, _ObservationComposer(), _RewardComposer(), actions)
    trainer.train(2)
    assert env.speeds == pytest.approx([.05, .05])


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


def _parallel_test_setup(device="cpu", accelerated=False):
    from pathlib import Path
    from core.scenario import load_and_expand_scenario
    from core.setup import create_training_setup
    from run import build_obs_composer, build_reward_composer, resolve_training_params
    from wrappers.actions.composer import ActionComposer

    path = Path("scenarios/ppo_lap_completion_pretrain.yaml").resolve()
    scenario = load_and_expand_scenario(str(path))
    scenario["experiment"].update(num_envs=2, episodes=3, seed=42, total_steps=None)
    scenario["environment"].pop("track_limits", None)
    scenario["environment"]["episode_termination"]["lap_completion"] = True
    scenario["environment"]["terminate_on_collision"] = True
    scenario["agents"]["car_0"]["reward"] = "../configs/reward/tasks/lap_completion_circle_stable.yaml"
    scenario["evaluation"].pop("every_steps", None)
    scenario["evaluation"]["max_steps"] = 4
    scenario["evaluation"]["selection_strategy"] = "completion_safety"
    # Exercise the retained legacy direct/chassis-acceleration trainer contracts.
    from env.f110ParallelEnv import _default_vehicle_params
    scenario["environment"].update(max_steps=4, timestep=.01, vehicle_params=_default_vehicle_params())
    scenario["environment"].pop("friction", None)
    scenario["environment"].pop("spawn", None)
    scenario["agents"]["car_0"]["observation"] = "../configs/observations/rl_racer.yaml"
    scenario["agents"]["car_0"]["action_constraints"] = {"prevent_reverse": True, "speed_index": 1}
    for key in ("map_bundles", "map_bundles_train", "map_bundles_eval"):
        scenario["environment"][key] = ["circle_map"]
    cfg = scenario["agents"]["car_0"]
    cfg["params"].update(
        device=device, n_steps=8, n_epochs=1, batch_size=4,
        pi_hidden_dims=[4], vf_hidden_dims=[4],
    )
    if accelerated:
        cfg["action_constraints"].update(
            speed_control="acceleration", max_acceleration=5.0, max_deceleration=5.0,
        )
    env, opponents, _ = create_training_setup(scenario, scenario_dir=path.parent)
    space = env.action_spaces["car_0"]
    obs = build_obs_composer(cfg, scenario["environment"], path.parent)
    agent = PPOAgent(obs.obs_dim, space.low, space.high, resolve_training_params(cfg, scenario))
    trainer = OnPolicyTrainer(
        env, "car_0", agent, opponents, obs, build_reward_composer(cfg, path.parent),
        ActionComposer.from_config(space.low, space.high, cfg["action_constraints"], decision_dt=.01),
        run_id="parallel-test",
    )
    return trainer, scenario, path.parent


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
)])
@pytest.mark.parametrize("accelerated", [False, True])
def test_parallel_ppo_collects_exact_episodes_and_is_repeatable(device, accelerated):
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
        trainer, scenario, directory = _parallel_test_setup(device, accelerated)
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
        # Two of three global episodes finish between the pooled updates.
        assert [row["train/learning_rate"] for row in capture.updates] == pytest.approx(
            [0.001, 0.0004]
        )
        assert trainer.agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
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
            if accelerated:
                speed_reference = 0.0
                for record in records:
                    speed_reference = np.clip(speed_reference + float(record.action_norm[1]) * .05, 0, 20)
                    assert record.action_phys[1] == pytest.approx(speed_reference, abs=1e-6)
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


def test_parallel_pooling_keeps_raw_samples_and_flushes_at_shutdown():
    trainer, scenario, directory = _parallel_test_setup()
    trainer.agent.min_rollout_steps = 16  # Entire 12-decision run is a partial pool.
    captured = []
    def update(*tensors):
        captured.append(tensors)
        return {"train/rollout_steps": len(tensors[0])}
    trainer.agent._update = update
    try:
        trainer.train_parallel(scenario, directory, num_envs=2, n_episodes=3)
    finally:
        trainer.env.close()
    assert len(captured) == 1
    obs, actions, old_lp, _, _, raw = captured[0]
    assert len(obs) == 12 and torch.isfinite(raw).all()
    torch.testing.assert_close(raw.tanh(), actions)
    new_lp, _ = trainer.agent.actor.evaluate_actions(obs, actions, raw)
    torch.testing.assert_close(new_lp, old_lp)


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("protocol", [None, "selection", "final"])
def test_cli_evaluates_ppo_checkpoint_with_batched_inference_available(tmp_path, monkeypatch, accelerated, protocol):
    import hashlib
    import json
    import sys
    import run
    from core.provenance import build_run_provenance

    trainer, scenario, directory = _parallel_test_setup(accelerated=accelerated)
    scenario["wandb"]["enabled"] = False
    scenario["evaluation"]["episodes"] = 2
    scenario["evaluation"]["final_test"]["episodes"] = 3
    checkpoint = tmp_path / "best_model.pt"
    trainer.agent.save(str(checkpoint))
    trainer.env.close()
    if protocol == "final":
        import os
        scenario["experiment"]["checkpoint"] = os.path.relpath(checkpoint.parent, directory)
    payload = torch.load(checkpoint, weights_only=False)
    payload["provenance"] = build_run_provenance(
        scenario, scenario_path=directory / "ppo_lap_completion_pretrain.yaml",
        run_id="test", algorithm="ppo", trainable_agents=["car_0"],
    )
    torch.save(payload, checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(run, "load_and_expand_scenario", lambda _: scenario)
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--scenario", str(directory / "ppo_lap_completion_pretrain.yaml"),
        "--eval", *([] if protocol == "final" else ["--checkpoint", str(checkpoint)]),
        *(["--eval-protocol", protocol] if protocol else ["--eval-episodes", "1"]),
        "--output-dir", str(tmp_path / "eval"), "--run-id", "ppo-eval-test",
        "--no-render", "--no-wandb", "--quiet",
    ])
    run.main()
    report = json.loads((tmp_path / "eval" / "evaluation_report.json").read_text())
    expected_seeds = {None: [42], "selection": [10042, 10043], "final": [20042, 20043, 20044]}[protocol]
    assert report["protocol"] == (protocol or "custom")
    assert report["seeds"] == expected_seeds
    assert [row["seed"] for row in report["episode_results"]] == expected_seeds
    assert report["summary"]["episodes"] == len(expected_seeds)
    assert report["summary"]["mean_episode_length"] == 4
    assert [row['map_bundle'] for row in report['episode_results']] == ['circle_map'] * len(expected_seeds)
    assert report['per_map']['circle_map']['episodes'] == len(expected_seeds)
    assert report["horizon_s"] == pytest.approx(0.04)
    assert report["provenance_mismatches"] == []
    assert report["checkpoint_sha256"] == checkpoint_hash
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    assert scenario["experiment"]["seed"] == 42
    assert scenario["experiment"]["episodes"] == 3


@pytest.mark.parametrize("key,value", [
    ("algorithm", "mappo"), ("obs_dim", 2), ("action_dim", 3),
    ("actor_hidden_dims", [8]), ("critic_hidden_dims", [8]),
    ("activation", "relu"), ("action_low", [-2.0, -1.0]),
    ("action_contract", {"speed_control": "acceleration"}),
])
def test_ppo_transfer_rejects_incompatible_checkpoint(tmp_path, key, value):
    agent = PPOAgent(1, -np.ones(2), np.ones(2), {"hidden_dims": [4]})
    checkpoint = tmp_path / "best_model.pt"
    agent.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    payload[key] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="Incompatible PPO checkpoint"):
        agent.load(str(checkpoint), load_optimizer=False)


@pytest.mark.parametrize("checkpoint_source", ["cli", "yaml", "override"])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_cli_initializes_ppo_training_from_best_checkpoint(tmp_path, monkeypatch, num_envs, checkpoint_source):
    import hashlib
    import json
    import sys
    import run

    source, scenario, directory = _parallel_test_setup()
    source.env.close()
    # Give the source optimizer moments and a different LR to detect accidental restoration.
    for parameter in source.agent._optim_parameters:
        parameter.grad = torch.ones_like(parameter)
    source.agent.optimizer.step()
    source.agent.set_training_progress(1.0)
    checkpoint = tmp_path / "source" / "best_model.pt"
    source.agent.save(str(checkpoint))
    # Pre-existing checkpoints do not have critic_hidden_dims; their tensor shapes
    # still enforce the critic architecture when loading.
    payload = torch.load(checkpoint, weights_only=False)
    payload.pop("critic_hidden_dims")
    torch.save(payload, checkpoint)
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    scenario["experiment"].update(num_envs=num_envs, episodes=2)
    scenario["evaluation"].update(every_episodes=1, episodes=1)
    for key in ("map_bundles", "map_bundles_train", "map_bundles_eval"):
        scenario["environment"][key] = ["Budapest_map"]
    method_name = "train_parallel" if num_envs > 1 else "train"
    original_train = getattr(OnPolicyTrainer, method_name)
    initialized = []

    def verify_initialization(trainer, *args, **kwargs):
        for name in ("actor", "critic"):
            expected = getattr(source.agent, name).state_dict()
            for key, value in getattr(trainer.agent, name).state_dict().items():
                assert torch.equal(value, expected[key])
        assert not trainer.agent.optimizer.state
        assert trainer.agent.optimizer.param_groups[0]["lr"] == pytest.approx(.001)
        assert trainer.agent.buffer.size() == 0
        initialized.append(True)
        return original_train(trainer, *args, **kwargs)

    monkeypatch.setattr(OnPolicyTrainer, method_name, verify_initialization)
    monkeypatch.setattr(run, "load_and_expand_scenario", lambda _: scenario)
    if checkpoint_source == "yaml":
        import os
        scenario["experiment"]["checkpoint"] = (
            os.path.relpath(checkpoint.parent, directory) if num_envs == 2 else str(checkpoint)
        )
    elif checkpoint_source == "override":
        scenario["experiment"]["checkpoint"] = "/nonexistent/ignored-checkpoint.pt"
    output = tmp_path / "transfer"
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--scenario", str(directory / "ppo_lap_completion_transfer.yaml"),
        *([] if checkpoint_source == "yaml" else [
            "--checkpoint", str(checkpoint.parent if num_envs == 2 else checkpoint),
        ]),
        "--output-dir", str(output), "--run-id", "ppo-transfer-test",
        "--no-render", "--no-wandb", "--quiet",
    ])
    run.main()
    assert initialized == [True]
    provenance = json.loads((output / "config_snapshot.json").read_text())["provenance"]
    assert provenance["initial_checkpoint"] == {
        "path": str(checkpoint), "sha256": source_hash,
        "load_scope": "actor_and_critic", "optimizer_restored": False,
        "training_progress_restored": False,
    }
    assert provenance["map_split"] == {"train": ["Budapest_map"], "eval": ["Budapest_map"]}
    saved = torch.load(output / "best_model.pt", weights_only=False)
    assert saved["provenance"]["initial_checkpoint"] == provenance["initial_checkpoint"]
    assert any(not torch.equal(value, saved["actor"][key]) for key, value in payload["actor"].items())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash


def test_checkpoint_directory_requires_best_model(tmp_path):
    from run import resolve_checkpoint_path

    (tmp_path / "checkpoint_ep000000.pt").touch()
    with pytest.raises(FileNotFoundError, match="best_model.pt"):
        resolve_checkpoint_path(str(tmp_path))


def test_transfer_scenario_preserves_pretraining_contract():
    from core.scenario import load_and_expand_scenario

    source = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    transfer = load_and_expand_scenario("scenarios/ppo_lap_completion_transfer.yaml")
    from core.provenance import physics_contract
    from run import build_obs_composer
    from pathlib import Path
    assert physics_contract(source['environment']) == physics_contract(transfer['environment'])
    contracts = [build_obs_composer(s['agents']['car_0'], s['environment'], Path('scenarios')).contract
                 for s in (source, transfer)]
    assert contracts[0] == contracts[1]
    assert source['agents']['car_0']['action_constraints'] == transfer['agents']['car_0']['action_constraints']
    assert transfer['experiment']['checkpoint'] is None
    assert transfer['agents']['car_0']['params']['n_steps'] == transfer['experiment']['num_envs'] * 1024
    assert source["experiment"]["name"] != transfer["experiment"]["name"]
    # The transfer track is user-selectable; all three lists must agree.
    maps = transfer["environment"]["map_bundles"]
    assert len(maps) == 1
    for key in ("map_bundles_train", "map_bundles_eval"):
        assert transfer["environment"][key] == maps
    for key in ("vehicle_params", "timestep", "action_repeat"):
        assert transfer["environment"][key] == source["environment"][key]


@pytest.mark.parametrize("scenario_name", ["mappo_gaplock", "calibration/hybrid_pp_ftg_1lap"])
def test_training_checkpoint_rejects_unsupported_roles(monkeypatch, scenario_name):
    import sys
    import run

    monkeypatch.setattr(sys, "argv", [
        "run.py", "--scenario", f"scenarios/{scenario_name}.yaml",
        "--checkpoint", "/nonexistent/model.pt", "--no-wandb", "--quiet",
    ])
    with pytest.raises(ValueError, match="one PPO learner"):
        run.main()


def test_transfer_rejects_source_output_directory_before_setup(tmp_path, monkeypatch):
    import sys
    import run

    (tmp_path / "best_model.pt").touch()
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--scenario", "scenarios/ppo_lap_completion_transfer.yaml",
        "--checkpoint", str(tmp_path), "--output-dir", str(tmp_path),
        "--no-wandb", "--quiet",
    ])
    with pytest.raises(ValueError, match="new --output-dir"):
        run.main()
    assert list(tmp_path.iterdir()) == [tmp_path / "best_model.pt"]


@pytest.mark.parametrize("value", [True, 123, [], {}, "", "   "])
def test_scenario_rejects_invalid_checkpoint_path(value):
    from core.scenario import ScenarioError, load_and_expand_scenario, validate_scenario

    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_transfer.yaml")
    scenario["experiment"]["checkpoint"] = value
    with pytest.raises(ScenarioError, match="experiment.checkpoint"):
        validate_scenario(scenario)


def test_bounded_smoke_cli_caps_training_and_evaluation():
    from argparse import Namespace
    from run import apply_cli_overrides
    args = Namespace(seed=None, episodes=None, wandb=False, no_wandb=True,
                     render=False, no_render=True, max_steps=64)
    config = apply_cli_overrides({'environment': {'max_steps': 16000},
                                 'evaluation': {'enabled': True, 'max_steps': 32000}}, args)
    assert config['environment']['max_steps'] == config['evaluation']['max_steps'] == 64
    args.max_steps = 0
    with pytest.raises(ValueError, match='positive'):
        apply_cli_overrides(config, args)


def test_paper_buffer_internal_truncation_uses_final_state_not_reset_state():
    buffer = RolloutBuffer(3, 1, 2, torch.device('cpu'))
    for value, terminal, truncated, final in [(10., True, False, None), (2., False, True, 7.), (100., True, False, None)]:
        buffer.add(np.zeros(1), np.zeros(2), 1., 0., value, terminal, truncated, final_value=final)
    _, returns = buffer.compute_gae(123., .9, .95)
    torch.testing.assert_close(returns, torch.tensor([1., 7.3, 1.]))


@pytest.mark.parametrize('terminal', [True, False])
def test_paper_fixed_rollouts_cross_resets_and_stop_at_exact_budget(terminal):
    from training.hooks import TrainingHook

    class Env(_OneStepTruncationEnv):
        def step(self, actions):
            obs, rewards, terms, truncs, info = super().step(actions)
            terms['car_0'], truncs['car_0'] = terminal, not terminal
            return obs, rewards, terms, truncs, info

    class Capture(TrainingHook):
        def __init__(self):
            self.metrics = []
        def on_update(self, metrics):
            self.metrics.append(dict(metrics))

    agent = PPOAgent(1, -np.ones(2), np.ones(2), {'hidden_dims': [4], 'n_steps': 4,
        'n_epochs': 1, 'batch_size': 4, 'learning_rate': .001, 'lr_schedule': 'linear', 'learning_rate_end': .0001})
    sizes = []
    update = agent.update
    def capture_update(next_value):
        sizes.append(agent.buffer.size())
        return update(next_value)
    agent.update = capture_update
    capture = Capture()
    trainer = OnPolicyTrainer(Env(), 'car_0', agent, {}, _ObservationComposer(),
                               _RewardComposer(), _ActionComposer(), hooks=[capture])
    trainer.train(total_steps=10)
    assert sizes == [4, 4, 2]
    assert [m['train/environment_steps'] for m in capture.metrics] == [4, 8, 10]
    assert [m['train/learning_rate'] for m in capture.metrics] == pytest.approx([.00064, .00028, .0001])


def test_paper_budget_cut_bootstraps_without_marking_environment_terminal():
    class Env(_OneStepTruncationEnv):
        def step(self, actions):
            obs, rewards, terms, truncs, info = super().step(actions)
            self.agents = ['car_0']
            truncs['car_0'] = False
            return obs, rewards, terms, truncs, info
    agent = PPOAgent(1, -np.ones(2), np.ones(2), {'hidden_dims': [4], 'n_steps': 4})
    captured = []
    def update(next_value):
        captured.append((agent.buffer.size(), agent.buffer.terminated[:2].clone(),
                         agent.buffer.truncated[:2].clone(), next_value))
        return {}
    agent.update = update
    agent.value = lambda obs: 7.
    trainer = OnPolicyTrainer(Env(), 'car_0', agent, {}, _ObservationComposer(), _RewardComposer(), _ActionComposer())
    trainer.train(total_steps=2)
    size, terms, truncs, bootstrap = captured[0]
    assert size == 2 and bootstrap == 7.
    assert not terms.any() and not truncs.any()


def test_paper_configuration_declares_transition_budget_and_continuous_task():
    from core.scenario import load_and_expand_scenario, validate_scenario
    scenario = load_and_expand_scenario('scenarios/ppo_lap_completion_pretrain.yaml')
    validate_scenario(scenario)
    assert scenario['experiment']['total_steps'] == 120000000
    assert scenario['experiment']['num_envs'] == 400
    assert scenario['agents']['car_0']['params']['n_steps'] == 400 * 1024
    assert scenario['environment']['max_steps'] == 0
    assert scenario['environment']['episode_termination']['lap_completion'] is False
    assert scenario['evaluation']['target_laps'] == 20


@pytest.mark.parametrize('safety', [False, True])
def test_evaluation_can_enforce_downstream_track_limits(safety, monkeypatch):
    from pathlib import Path
    from core.scenario import load_and_expand_scenario
    from core.setup import create_training_setup
    filename = 'ppo_lap_completion_validate.yaml' if safety else 'ppo_lap_completion_pretrain.yaml'
    scenario = load_and_expand_scenario(str(Path('scenarios') / filename))
    env, _, _ = create_training_setup(scenario, mode='eval', scenario_dir=Path('scenarios'))
    try:
        env.reset(seed=42)
        assert env.track_limits_enabled
        assert env.terminate_on_track_boundary is safety
        assert env.terminate_on_collision['car_0'] is safety
        assert env.sim.wall_collision_response is safety
        # Step must honor the mode, not only retain a configuration flag.
        original = env._inject_track_previews
        def outside(infos):
            original(infos)
            infos['car_0']['track_limits'].update(exceeded=True, offtrack_distance=.1)
        monkeypatch.setattr(env, '_inject_track_previews', outside)
        _, _, terminated, _, _ = env.step({'car_0': np.zeros(2, dtype=np.float32)})
        assert terminated['car_0'] is safety
        monkeypatch.setattr(env, '_inject_track_previews', original)
        env.reset(seed=42)
        car = env.sim.agents[0]
        scan = car.compute_scan
        def collision_scan():
            result = scan()
            car.in_collision = True
            return result
        monkeypatch.setattr(car, 'compute_scan', collision_scan)
        _, _, terminated, _, infos = env.step({'car_0': np.zeros(2, dtype=np.float32)})
        assert terminated['car_0'] is safety
        if safety:
            assert infos['car_0']['terminal_reason'] == 'collision'
    finally:
        env.close()


def test_paper_parallel_budget_and_value_requests_with_mock_collectors(monkeypatch):
    """Exercise the parent protocol without spawning processes or a simulator."""
    import multiprocessing
    from collections import deque
    from types import SimpleNamespace
    from training.hooks import TrainingHook

    def rollout(n):
        return (np.zeros((n, 1)), np.zeros((n, 2)), np.zeros(n), np.ones(n), np.ones(n), np.zeros((n, 2)))
    pipes, processes = [], []
    class Connection:
        def __init__(self):
            self.messages = deque()
            self.sent = []
        def poll(self, timeout): return bool(self.messages)
        def recv(self): return self.messages.popleft()
        def send(self, message): self.sent.append(message)
        def close(self): pass
    class Process:
        def __init__(self, **kwargs):
            args = kwargs['args']
            self.budget = args[-1]
            pipe = args[0]
            pipe.messages.append(('ready', (1, -np.ones(2), np.ones(2))))
            # Bootstrap values and sampled actions use separate requests.
            pipe.messages.extend([('value', np.zeros(1)), ('act', np.zeros(1))])
            remaining = self.budget
            while remaining:
                n = min(args[6], remaining)
                pipe.messages.append(('rollout', rollout(n)))
                remaining -= n
            pipe.messages.append(('done', None))
            processes.append(self)
        def start(self): pass
        def join(self, **kwargs): pass
        def is_alive(self): return False
    def pipe():
        connection = Connection()
        pipes.append(connection)
        return connection, connection
    monkeypatch.setattr(multiprocessing, 'get_context', lambda *_: SimpleNamespace(Pipe=pipe, Process=Process))

    class Capture(TrainingHook):
        def __init__(self): self.steps = []
        def on_update(self, metrics): self.steps.append(metrics['train/environment_steps'])
    agent = PPOAgent(1, -np.ones(2), np.ones(2), {'n_steps': 8, 'hidden_dims': [4],
        'learning_rate': .001, 'lr_schedule': 'linear', 'learning_rate_end': .0001})
    sizes = []
    def update(rollouts):
        sizes.append([len(r[0]) for r in rollouts])
        return {'train/learning_rate': agent.optimizer.param_groups[0]['lr']}
    agent.update_rollouts = update
    capture = Capture()
    trainer = OnPolicyTrainer(None, 'car_0', agent, {}, None, _RewardComposer(), None, hooks=[capture])
    trainer.train_parallel({}, '.', 2, 0, total_steps=11)
    assert [p.budget for p in processes] == [6, 5]
    assert sizes == [[4, 4], [2, 1]]
    assert capture.steps == [8, 11]
    assert agent.optimizer.param_groups[0]['lr'] == pytest.approx(.0001)
    assert all(p.sent[0] == ("start", None) for p in pipes)
    assert all(isinstance(p.sent[1], float) for p in pipes)


def test_paper_evaluation_metrics_exclude_partial_start_and_count_excursions():
    from metrics.racing_eval import create_episode_facts, update_agent_step_facts, aggregate_eval_episodes
    facts = create_episode_facts(episode=0, agent_ids=['car_0'], trainable_ids=['car_0'], opponent_ids=[])
    for step, start, duration, crossed, distance in [
        (1, None, None, False, .5), # random-spawn approach is not a full lap
        (2, 1, None, False, 0.),   # initial forward crossing starts timing
        (3, 1, None, False, .2),
        (4, 3, 2, True, 0.),      # first full lap had an excursion
        (5, 3, None, False, 0.),
        (6, 5, 2, True, 0.),      # second lap stayed in bounds
    ]:
        update_agent_step_facts(facts, step_idx=step, infos={'car_0': {
            'lap_start_step': start, 'lap_time_steps': duration, 'lap_crossed': crossed,
            'track_limits': {'offtrack_distance': distance, 'exceeded': distance > 0},
        }})
    summary = aggregate_eval_episodes([facts], timestep=.05)
    assert summary['measured_laps'] == 2 and summary['valid_laps'] == 1
    assert summary['fastest_valid_lap_s'] == pytest.approx(.1)
    assert summary['mean_lap_time_s'] == pytest.approx(.1)
    assert summary['offtrack_error_m_s_per_lap'] == pytest.approx(.005)
    assert summary['boundary_violation_lap_rate'] == .5


def test_paper_checkpoint_and_evaluation_cadence_uses_transitions(tmp_path):
    from training.hooks import CheckpointHook, EvaluationCheckpointHook
    from types import SimpleNamespace
    saved, evaluations = [], []
    hook = CheckpointHook(None, tmp_path, save_every_steps=8)
    hook._save = lambda path, metadata=None: saved.append((path.name, metadata))
    hook.on_episode_end(0, 0, {}, {})
    for steps in (4, 8, 10): hook.on_update({'train/environment_steps': steps})
    hook.on_training_end()
    assert [x[0] for x in saved] == ['checkpoint_step000000008.pt', 'final_model.pt']
    assert saved[-1][1]['environment_steps'] == 10
    evaluator = SimpleNamespace(evaluate=lambda: evaluations.append(True) or {
        'completion_rate': 1., 'valid_laps': 20, 'fastest_valid_lap_s': 5.6,
        'offtrack_error_m_s_per_lap': .001,
    })
    selection = EvaluationCheckpointHook(None, tmp_path, evaluator, evaluate_every=1,
                                        evaluate_every_steps=8, selection_strategy='lap_time')
    selection._save = lambda *args, **kwargs: None
    selection.on_episode_end(0, 0, {}, {})
    for steps in (4, 8, 10): selection.on_update({'train/environment_steps': steps})
    assert len(evaluations) == 1


@pytest.mark.parametrize('failure', [None, 'timeout', 'error'])
def test_collector_startup_batches_and_preserves_first_failure(monkeypatch, failure):
    """No processes/simulator: verify launch order, barrier and failure cleanup."""
    import multiprocessing
    from collections import deque
    from types import SimpleNamespace

    events, pipes = [], []
    class Connection:
        def __init__(self):
            self.messages = deque()
            self.worker = len(pipes)
            self.sent = []
            pipes.append(self)
        def poll(self, timeout):
            events.append(('poll', self.worker, timeout))
            return bool(self.messages)
        def recv(self):
            result = self.messages.popleft()
            events.append(('receive', self.worker, result[0]))
            return result
        def send(self, message):
            self.sent.append(message)
            if message == ('start', None):
                events.append(('start', self.worker))
        def close(self): pass
    class Process:
        pid, exitcode = 123, None
        def __init__(self, **kwargs):
            args = kwargs['args']
            self.pipe, self.worker = args[0], args[4]
            self.alive = False
        def start(self):
            self.alive = True
            events.append(('launch', self.worker))
            if self.worker == 0 and failure == 'timeout': return
            if self.worker == 0 and failure == 'error':
                self.pipe.messages.append(('error', 'original setup failure'))
                return
            self.pipe.messages.append(('ready', (1, -np.ones(2), np.ones(2))))
            self.pipe.messages.append(('rollout', (np.zeros((1, 1)),)))
            self.pipe.messages.append(('done', None))
        def is_alive(self): return self.alive
        def terminate(self):
            events.append(('terminate', self.worker))
            self.alive = False
        def join(self, **kwargs):
            events.append(('join', self.worker))
            self.alive = False
        def kill(self): self.alive = False
    def pipe():
        connection = Connection()
        return connection, connection
    monkeypatch.setattr(multiprocessing, 'get_context', lambda *_: SimpleNamespace(Pipe=pipe, Process=Process))
    agent = SimpleNamespace(n_steps=10, obs_dim=1, action_low=-np.ones(2), action_high=np.ones(2),
                            gamma=.99, gae_lambda=.95, update_rollouts=lambda _: {'updated': 1})
    trainer = OnPolicyTrainer(None, 'car_0', agent, {}, None, _RewardComposer(), None)
    scenario = {'experiment': {'worker_startup_batch_size': 2, 'worker_startup_timeout_s': 17,
                               'worker_response_timeout_s': 3}}
    if failure:
        expected = 'startup after 17s' if failure == 'timeout' else 'original setup failure'
        with pytest.raises(RuntimeError, match=expected):
            trainer.train_parallel(scenario, '.', 5, 0, total_steps=5)
        assert [e[1] for e in events if e[0] == 'launch'] == [0, 1]
        assert not any(e[0] == 'start' for e in events)
        assert max(i for i, e in enumerate(events) if e[0] == 'terminate') < min(
            i for i, e in enumerate(events) if e[0] == 'join')
    else:
        trainer.train_parallel(scenario, '.', 5, 0, total_steps=5)
        assert events.index(('receive', 1, 'ready')) < events.index(('launch', 2))
        assert events.index(('receive', 3, 'ready')) < events.index(('launch', 4))
        assert events.index(('receive', 4, 'ready')) < events.index(('start', 0))
        assert ('poll', 0, 17) in events and ('poll', 0, 3) in events
        assert trainer.collected_steps == 5


@pytest.mark.parametrize('disconnect_at', ['ready', 'start'])
def test_collector_parent_disconnect_does_not_send_secondary_error(monkeypatch, disconnect_at):
    from types import SimpleNamespace
    import training.on_policy_trainer as module
    import core.setup as setup
    import run
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('PYGLET_HEADLESS', 'true')
    monkeypatch.setattr(torch, 'set_num_threads', lambda *_: None)
    closed, sent = [], []
    space = SimpleNamespace(low=-np.ones(2), high=np.ones(2), n=2)
    env = SimpleNamespace(action_spaces={'car_0': space}, close=lambda: closed.append('env'))
    monkeypatch.setattr(setup, 'create_training_setup', lambda *a, **kw: (env, {}, {}))
    monkeypatch.setattr(run, 'build_obs_composer', lambda *a: SimpleNamespace(obs_dim=1))
    monkeypatch.setattr(run, 'build_reward_composer', lambda *a: _RewardComposer())
    monkeypatch.setattr(module, '_RemotePolicy', lambda *a: None)
    monkeypatch.setattr(module, 'OnPolicyTrainer', lambda *a, **kw: None)
    class Connection:
        def send(self, message):
            sent.append(message[0])
            if disconnect_at == 'ready': raise BrokenPipeError('parent exited')
        def recv(self): raise EOFError('parent exited')
        def close(self): closed.append('pipe')
    module._collect_ppo_worker(Connection(), {'experiment': {'seed': 42}, 'environment': {},
        'agents': {'car_0': {}}}, '.', 'car_0', 0, 1, 2, 'test', .99, .95, False, False)
    assert sent == ['ready']
    assert closed == ['env', 'pipe']


def test_collector_error_reporting_preserves_original_when_parent_disappears(capsys):
    from training.on_policy_trainer import _report_worker_error
    class ClosedConnection:
        def send(self, _): raise BrokenPipeError('secondary pipe failure')
    try:
        raise ValueError('original calibration failure')
    except ValueError:
        _report_worker_error(ClosedConnection())
    stderr = capsys.readouterr().err
    assert 'ValueError: original calibration failure' in stderr
    assert 'secondary pipe failure' not in stderr


@pytest.mark.parametrize('field', ['worker_startup_batch_size', 'worker_startup_timeout_s', 'worker_response_timeout_s'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_collector_startup_settings_fail_before_launch(field, value):
    from training.on_policy_trainer import _worker_startup_settings
    from core.scenario import load_and_expand_scenario, validate_scenario, ScenarioError
    scenario = load_and_expand_scenario('scenarios/ppo_lap_completion_pretrain.yaml')
    scenario['experiment'][field] = value
    with pytest.raises(ValueError, match=field): _worker_startup_settings(scenario)
    with pytest.raises(ScenarioError, match=field): validate_scenario(scenario)
