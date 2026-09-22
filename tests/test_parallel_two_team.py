from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.mappo import MAPPOAgent
from core.scenario import load_and_expand_scenario, validate_scenario, ScenarioError
from training.parallel_two_team import CONTRACT_FIELDS, TeamCollector, infer_team_requests, train_parallel


TEAMS = {"team_a": ["car_0", "car_1"], "team_b": ["car_2", "car_3"]}


def small_agents():
    return {team: MAPPOAgent(2, 3, -np.ones(2), np.ones(2), ids, dict(
        hidden_dims=[4], device="cpu", n_steps=4, n_epochs=1, batch_size=8,
        gamma=.9, gae_lambda=1., team_return_mode="joint", reward_mode="team_shared",
        critic_mode="shared_team", team_reward_reduction="sum")) for team, ids in TEAMS.items()}


def contract(agent):
    return {key: getattr(agent, key) for key in CONTRACT_FIELDS}


def store(owner, ids, reward, terminal=False):
    owner.store_batch(ids, observations={aid: np.zeros(2) for aid in ids}, global_state=np.zeros(3),
        actions={aid: np.zeros(2) for aid in ids}, rewards={aid: reward for aid in ids},
        log_probs={aid: 0. for aid in ids}, values={aid: 0. for aid in ids},
        terminated={aid: terminal for aid in ids}, truncated={aid: False for aid in ids},
        raw_actions={aid: np.zeros(2) for aid in ids})
    owner.store_team_step(ids, reward=reward, value=0., terminal=terminal)


def test_team_inference_routes_repeated_ids_and_values_for_eliminated_teams():
    agents = small_agents()
    requests = {
        (0, 0): ("act", ({aid: np.zeros(2) for ids in TEAMS.values() for aid in ids}, np.zeros(3))),
        (0, 1): ("act", ({"car_0": np.ones(2)}, np.ones(3))),
        (1, 2): ("cut", np.full(3, 2.)),
    }
    responses = infer_team_requests(agents, requests)
    for key, (kind, payload) in requests.items():
        state = payload[1] if kind == "act" else payload
        values = responses[key][4] if kind == "act" else responses[key]
        assert values == pytest.approx({team: agent.evaluate_state(state) for team, agent in agents.items()}, abs=1e-7)
        if kind == "act":
            assert responses[key][0].keys() == payload[0].keys()
            for aid, action in responses[key][0].items():
                np.testing.assert_allclose(action, np.tanh(responses[key][3][aid]), atol=1e-7)
    assert set(responses[(0, 1)][0]) == {"car_0"}
    assert set(responses[(0, 1)][4]) == set(TEAMS)


def test_fragments_keep_delayed_credit_and_inactive_critic_targets_without_reset_leakage():
    agents = small_agents()
    collector = TeamCollector(contract(agents["team_a"]), 4)
    store(collector, ["car_0", "car_1"], 1.)
    store(collector, ["car_1"], 2.)
    store(collector, [], 8., terminal=True)
    collector.finish_team_fragment(999., [np.zeros(3)] * 3)
    actor = collector.take_fragments()
    value = collector.take_value_fragments()
    assert np.concatenate([row[0][:, -1] for row in actor]).tolist() == pytest.approx([9.28, 9.28, 9.2])
    assert value[0][:, -1].tolist() == [8.]
    assert not collector._team_rollout
    # A separate, entirely inactive fragment still bootstraps and clears.
    store(collector, [], 4.)
    collector.finish_team_fragment(2., [np.zeros(3)])
    assert not collector.take_fragments()
    assert collector.take_value_fragments()[0][:, -1].tolist() == pytest.approx([5.8])
    assert not collector._team_rollout
    assert not hasattr(collector, "actor") and not hasattr(collector, "optimizer")


def test_pooled_team_actor_updates_match_serial_fragment_updates():
    serial = small_agents()["team_a"]
    parallel = small_agents()["team_a"]
    parallel.actor.load_state_dict(serial.actor.state_dict())
    parallel.critic.load_state_dict(serial.critic.state_dict())
    collector = TeamCollector(contract(serial), 4)
    for owner in (serial, collector):
        store(owner, owner.agent_ids, 1.)
        store(owner, [owner.agent_ids[1]], 3., terminal=True)
    state = np.zeros(3)
    collector.finish_team_fragment(serial.evaluate_state(state), [state] * 2)
    rng = torch.random.get_rng_state()
    expected = serial.update(state)
    torch.random.set_rng_state(rng)
    actual = parallel.update_rollouts(collector.take_fragments())
    assert actual == pytest.approx(expected, abs=1e-6)
    for a, b in zip(serial._optim_parameters, parallel._optim_parameters):
        torch.testing.assert_close(a, b)


def scenario_config(workers=2, *, episodes=False):
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_selfplay.yaml")
    scenario["experiment"].update(num_envs=3, num_workers=workers, seed=42,
        total_steps=11, worker_startup_timeout_s=60, worker_response_timeout_s=30)
    if episodes:
        scenario["experiment"].pop("total_steps")
        scenario["experiment"]["episodes"] = 5
    scenario["environment"]["max_steps"] = 3
    scenario["training_defaults"].update(device="cpu", pi_hidden_dims=[4], vf_hidden_dims=[4],
        rollout_steps_per_env=2, n_steps=4, n_epochs=1, batch_size=16, checkpoint_every_steps=5)
    scenario["evaluation"].update(max_steps=3, episodes=1, every_steps=5)
    scenario["wandb"]["enabled"] = False
    validate_scenario(scenario)
    return scenario


@pytest.mark.parametrize("workers,episodes", [(1, False), (2, False), (2, True)])
def test_spawned_workers_exact_budgets_resets_updates_checkpoints_and_cleanup(tmp_path, monkeypatch, workers, episodes):
    import run

    scenario = scenario_config(workers, episodes=episodes)
    monkeypatch.setattr("sys.argv", ["run.py", "--scenario", "scenarios/mappo_2v2_selfplay.yaml",
                                    "--output-dir", str(tmp_path), "--no-wandb"])
    settings = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run._run_two_team(scenario, run.parse_args(), run.ConsoleLogger(verbose=False), Path("scenarios").resolve())
    finally:
        torch.set_num_threads(threads)
    assert not any(p.name.startswith("two-team-collector-") for p in mp.active_children())
    assert {key: os.environ.get(key) for key in settings} == settings

    def rows(name):
        return [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]

    total = 15 if episodes else 11
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["environment_steps"] == total
    assert summary["agent_steps"] == 4 * total
    races = rows("team_metrics.jsonl")
    assert {row["selfplay/environment_id"] for row in races} == {0, 1, 2}
    assert {row["selfplay/seed"] for row in races} == {42, 43, 44}
    assert sum(row["selfplay/steps"] for row in races) == total
    assert sum(row["selfplay/completed"] for row in races) == (5 if episodes else 3)
    assert sum(row["selfplay/budget_cut"] for row in races) == (0 if episodes else 2)
    updates = rows("updates.jsonl")
    assert [row["train/environment_steps"] for row in updates] == ([6, 11, 15] if episodes else [6, 11])
    assert sum(row["train/team_a/rollout_agent_samples"] for row in updates) == 2 * total
    assert sum(row["train/team_b/rollout_agent_samples"] for row in updates) == 2 * total
    assert all(np.isfinite(row["train/team_a/policy_loss"]) and np.isfinite(row["train/team_b/policy_loss"]) for row in updates)
    assert all(row["perf/collector_peak_rss_mib"] > 0 for row in updates)
    assert [row["selfplay_eval/environment_steps"] for row in rows("evaluation_metrics.jsonl")] == (
        [6, 11, 15] if episodes else [6, 11])
    # Eval steps never enter the training budget, and both policies checkpoint.
    final_dir = tmp_path / "final_pair"
    pair = json.loads((final_dir / "pair.json").read_text())
    assert pair["environment_steps"] == total
    assert set(pair["files"]) == set(TEAMS)
    for file in pair["files"].values():
        checkpoint = torch.load(final_dir / file, weights_only=False, map_location="cpu")
        assert checkpoint["optimizer"]["state"]


def test_worker_failure_is_forwarded_and_processes_are_cleaned_up(monkeypatch):
    from types import SimpleNamespace

    scenario = scenario_config()
    # Deliberately incompatible network contracts fail during worker startup.
    trainer = SimpleNamespace(agents=small_agents(), teams=TEAMS)
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="global-state contract mismatch"):
        train_parallel(trainer, scenario, Path("scenarios").resolve(), on_episode=lambda row: None)
    assert not any(p.name.startswith("two-team-collector-") for p in mp.active_children())
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        assert os.environ.get(key) == before.get(key)


def test_parallel_selfplay_configuration_rejects_bad_counts_and_horizon():
    config = load_and_expand_scenario("scenarios/mappo_2v2_selfplay.yaml")
    assert config["experiment"]["num_envs"] == 400
    assert config["experiment"]["num_workers"] == 100
    for block, key, value in (("experiment", "num_workers", 0),
                              ("training_defaults", "rollout_steps_per_env", 0),
                              ("experiment", "total_steps", 399)):
        invalid = deepcopy(config)
        invalid[block][key] = value
        with pytest.raises(ScenarioError):
            validate_scenario(invalid)
