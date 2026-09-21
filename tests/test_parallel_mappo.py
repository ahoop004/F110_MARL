from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.mappo import MAPPOAgent
from training.parallel_mappo import CollectorAgent, infer_requests
from training.hooks import TrainingHook, WandbHook


def contract(agent):
    return {name: getattr(agent, name) for name in (
        "agent_ids", "obs_dim", "global_state_dim", "global_state_contract_version",
        "action_dim", "action_low", "action_high", "observation_contract", "gamma",
        "gae_lambda", "critic_mode", "reward_mode", "team_return_mode", "team_reward_reduction",
    )}


def small_agent(mode="joint"):
    return MAPPOAgent(2, 3, -np.ones(2), np.ones(2), ["a", "b"], dict(
        hidden_dims=[4], n_steps=4, n_epochs=1, batch_size=8, gamma=.9, gae_lambda=1.,
        team_return_mode=mode, reward_mode="team_shared",
        critic_mode="shared_team" if mode == "joint" else "agent_conditioned",
    ))


@pytest.mark.parametrize("mode", ["joint", "per_agent"])
def test_multi_environment_inference_matches_scalar_values_and_preserves_raw_actions(mode):
    agent = small_agent(mode)
    requests = {
        (0, 0): ("act", (["a", "b"], np.ones((2, 2)), np.zeros(3))),
        (0, 1): ("act", (["b"], np.zeros((1, 2)), np.ones(3))),
        (1, 2): ("act", ([], np.empty((0, 2)), np.ones(3))),
        (1, 3): ("value", np.full(3, 2.)),
    }
    result = infer_requests(agent, requests)
    for key, (kind, payload) in requests.items():
        ids, state = (payload[0], payload[2]) if kind == "act" else (agent.agent_ids, payload)
        values = result[key][2] if kind == "act" else result[key]
        assert values == pytest.approx(agent.evaluate_states(state, ids))
        if kind == "act":
            assert set(result[key][0]) == set(ids)
            for aid in ids:
                np.testing.assert_allclose(result[key][0][aid], np.tanh(result[key][3][aid]), atol=1e-7)


def store(collector, ids, reward, terminal):
    collector.store_batch(ids, observations={a: np.zeros(2) for a in ids},
        global_state=np.zeros(3), actions={a: np.zeros(2) for a in ids},
        rewards={a: reward for a in ids}, log_probs={a: 0. for a in ids},
        values={a: 0. for a in ids}, terminated=terminal,
        truncated={a: False for a in ids}, raw_actions={a: np.zeros(2) for a in ids})
    if collector.team_return_mode == "joint":
        collector.store_team_step(ids, reward=reward, value=0., terminal=all(terminal.values()))


def test_joint_fragments_keep_delayed_credit_and_do_not_cross_reset_or_environment():
    collector = CollectorAgent(contract(small_agent()), 4)
    store(collector, ["a", "b"], 1., {"a": True, "b": False})
    store(collector, ["b"], 10., {"b": True})
    collector.finish_fragment({"a": 999., "b": 999.})
    first = collector.take_fragments()
    assert np.concatenate([f[0][:, -1] for f in first]).tolist() == pytest.approx([10., 10., 10.])
    # The next episode gets a rollout cut, which bootstraps but cannot leak backward.
    store(collector, ["a", "b"], 2., {"a": False, "b": False})
    collector.finish_fragment({"a": 5., "b": 5.})
    second = collector.take_fragments()
    assert np.concatenate([f[0][:, -1] for f in second]).tolist() == pytest.approx([6.5, 6.5])
    assert np.concatenate([f[0][:, -1] for f in first]).tolist() == pytest.approx([10., 10., 10.])
    assert not hasattr(collector, "actor")


@pytest.mark.parametrize("mode", ["joint", "per_agent"])
def test_parallel_pooled_update_matches_serial_update(mode):
    torch.manual_seed(3)
    serial = small_agent(mode)
    parallel = small_agent(mode)
    parallel.actor.load_state_dict(serial.actor.state_dict())
    parallel.critic.load_state_dict(serial.critic.state_dict())
    collector = CollectorAgent(contract(serial), 4)
    for owner in (serial, collector):
        store(owner, ["a", "b"], 1., {"a": True, "b": False})
        store(owner, ["b"], 3., {"b": True})
    state = np.zeros(3)
    collector.finish_fragment(serial.evaluate_states(state, serial.agent_ids))
    rng = torch.get_rng_state()
    expected = serial.update(state)
    torch.set_rng_state(rng)
    actual = parallel.update_rollouts(collector.take_fragments())
    assert actual == pytest.approx(expected, abs=1e-6)
    for a, b in zip(serial._optim_parameters, parallel._optim_parameters):
        torch.testing.assert_close(a, b)


def setup(workers, horizon):
    from core.scenario import load_and_expand_scenario, resolve_mappo_config
    from core.setup import create_training_setup
    from run import build_obs_composers, build_reward_composers, resolve_training_params
    from training.marl_trainer import MARLTrainer
    from wrappers.actions.composer import ActionComposer

    path = Path("scenarios/mappo_2v2_penalties_scratch.yaml").resolve()
    scenario = load_and_expand_scenario(str(path))
    scenario["experiment"].update(num_envs=3, num_workers=workers, episodes=5, seed=42)
    scenario["training_defaults"].update(rollout_steps_per_env=horizon, device="cpu")
    scenario["environment"].update(max_steps=3, terminate_on_collision=False)
    scenario["evaluation"]["enabled"] = False
    ids = ["car_0", "car_1"]
    for aid in ids:
        scenario["agents"][aid]["params"].update(
            device="cpu", n_steps=4, n_epochs=1, batch_size=16,
            pi_hidden_dims=[4], vf_hidden_dims=[4],
        )
    env, opponents, _ = create_training_setup(scenario, scenario_dir=path.parent)
    obs = build_obs_composers(scenario["agents"], ids, scenario["environment"], path.parent)
    rewards = build_reward_composers(scenario["agents"], ids, path.parent)
    space = env.action_spaces[ids[0]]
    snapshot = env.get_global_state()
    params = {**resolve_training_params(scenario["agents"][ids[0]], scenario),
              **resolve_mappo_config(scenario), "_observation_contract": obs[ids[0]].contract,
              "_global_state_contract_version": snapshot.metadata["vector_contract_version"]}
    agent = MAPPOAgent(obs[ids[0]].obs_dim, len(snapshot.vector), space.low, space.high, ids, params)
    actions = ActionComposer.from_config(space.low, space.high,
        scenario["agents"][ids[0]]["action_constraints"], decision_dt=.05)
    trainer = MARLTrainer(env, agent, ids, opponents, obs, rewards, actions,
                         reward_mode="team_shared", run_id="parallel-mappo-test")
    return trainer, scenario, path.parent


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


@pytest.mark.parametrize("workers,horizon", [(1, 2), (2, 5)])
def test_spawned_grouped_collectors_count_steps_resets_and_unequal_episode_budgets(workers, horizon):
    trainer, scenario, directory = setup(workers, horizon)
    capture = Capture()
    logs = []
    wandb = WandbHook(SimpleNamespace(log_metrics=logs.append))
    trainer.hooks = trainer._transition_hooks = [capture, wandb]
    before = [p.detach().clone() for p in trainer.agent._optim_parameters]
    try:
        trainer.train_parallel(scenario, directory, num_envs=3, n_episodes=5)
    finally:
        trainer.env.close()
    assert len(capture.episodes) == 5
    assert len(capture.records) == 30
    assert capture.ends == 1
    assert capture.updates[-1]["train/environment_steps"] == 15
    assert capture.updates[-1]["train/agent_steps"] == 30
    assert len({r.episode_id for r in capture.records}) == 5
    assert {r.info["worker_id"] for r in capture.records} == {0, 1, 2}
    assert {r.info["worker_seed"] for r in capture.records} == {42, 43, 44}
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.agent._optim_parameters))
    assert len([row for row in logs if "episode/number" in row]) == 5
    assert all(np.isfinite(m["train/policy_loss"]) for m in capture.updates
               if m["train/rollout_agent_samples"])


def test_parallel_mappo_validates_horizon_and_worker_count():
    from core.scenario import load_and_expand_scenario, validate_scenario, ScenarioError
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_penalties_scratch.yaml")
    for field, value in [("rollout_steps_per_env", 0), ("rollout_steps_per_env", True)]:
        scenario["training_defaults"][field] = value
        with pytest.raises(ScenarioError, match="positive integer"):
            validate_scenario(scenario)
    scenario["training_defaults"]["rollout_steps_per_env"] = 256
    scenario["experiment"]["num_workers"] = 0
    with pytest.raises(ScenarioError, match="positive integer"):
        validate_scenario(scenario)


def test_worker_startup_error_is_reported_and_processes_are_reaped():
    import multiprocessing as mp
    import os

    trainer, scenario, directory = setup(2, 2)
    before = {p.pid for p in mp.active_children()}
    threads_before = os.environ.get("OMP_NUM_THREADS")
    scenario["agents"]["car_0"]["observation"] = "/missing/mappo-observation.yaml"
    try:
        with pytest.raises(RuntimeError, match="MAPPO worker .* failed"):
            trainer.train_parallel(scenario, directory, num_envs=3, n_episodes=5)
    finally:
        trainer.env.close()
    assert {p.pid for p in mp.active_children()} == before
    assert os.environ.get("OMP_NUM_THREADS") == threads_before


@pytest.mark.parametrize("workers,horizon", [(1, 2), (2, 5)])
def test_step_budget_collects_exact_remainder_across_resets(workers, horizon):
    trainer, scenario, directory = setup(workers, horizon)
    capture = Capture()
    trainer.hooks = trainer._transition_hooks = [capture]
    try:
        trainer.train_parallel(scenario, directory, num_envs=3, total_steps=17)
    finally:
        trainer.env.close()
    assert trainer._environment_steps == 17
    assert len(capture.records) == 34
    # Quotas 6, 6, 5: five real three-step episodes; one partial episode is not logged.
    assert len(capture.episodes) == 5
    assert sum(m["train/rollout_agent_samples"] for m in capture.updates) == 34
    assert capture.updates[-1]["train/agent_steps"] == 34
    assert capture.ends == 1
    # The budget cut is not falsely recorded as a terminal or truncation.
    partial = [r for r in capture.records if r.info["worker_id"] == 2][-2:]
    assert all(not r.terminated and not r.truncated for r in partial)


def test_serial_step_budget_stops_mid_episode_without_fabricating_outcome():
    trainer, _, _ = setup(1, 2)
    capture = Capture()
    trainer.hooks = trainer._transition_hooks = [capture]
    try:
        trainer.train(total_steps=5)
    finally:
        trainer.env.close()
    assert trainer._environment_steps == 5
    assert len(capture.records) == 10
    assert len(capture.episodes) == 1
    assert capture.ends == 1
    assert all(not r.terminated and not r.truncated for r in capture.records[-2:])
