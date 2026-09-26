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
        # Different GEMM batch sizes can differ by float32 roundoff near zero.
        assert values == pytest.approx(agent.evaluate_states(state, ids), abs=1e-7)
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


def make_setup(workers, horizon):
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
    trainer, scenario, directory = make_setup(workers, horizon)
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
    assert capture.updates[-1]["train/physics_steps"] == 15
    races = [row[3]["race_record"] for row in capture.episodes]
    assert len({r["episode_id"] for r in races}) == 5
    assert {r["environment_id"] for r in races} == {0, 1, 2}
    assert all(r["run_id"] == trainer.run_id for r in races)
    assert all(len(r["agents"]) == 4 for r in races)
    assert all(r["policy_version_start"] <= r["policy_version_end"] for r in races)
    assert all(r["environment_decisions"] == 3 for r in races)
    assert len({r.episode_id for r in capture.records}) == 5
    assert {r.info["worker_id"] for r in capture.records} == {0, 1, 2}
    assert {r.info["worker_seed"] for r in capture.records} == {42, 43, 44}
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.agent._optim_parameters))
    assert len([row for row in logs if "episode/number" in row]) == 5
    progress = [row for row in logs if "collector/phase" in row]
    assert progress[0]["collector/phase"] == "startup"
    assert progress[0]["collector/updates"] == 0
    assert progress[-1]["collector/updated_environment_steps"] == 15
    assert progress[-1]["collector/actions_dispatched"] == 15
    assert {row["collector/phase"] for row in progress} >= {"collecting", "updating"}
    # Collector telemetry must not masquerade as a learning update.
    assert len([row for row in logs if "train/update" in row]) == len(capture.updates)
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

    trainer, scenario, directory = make_setup(2, 2)
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
    trainer, scenario, directory = make_setup(workers, horizon)
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


def test_parallel_selective_recording_keeps_all_cars_and_budget_cut(tmp_path):
    from replay.dataset_writer import RaceDatasetWriter, RaceDatasetHook
    from replay.race_reader import iter_race_frames, load_clips
    trainer, scenario, directory = make_setup(2, 2)
    writer = RaceDatasetWriter(tmp_path/'races', config=dict(sample_probability=1., chunk_frames=4))
    hook = RaceDatasetHook(writer)
    trainer.hooks = [hook]
    trainer._transition_hooks = []
    try:
        trainer.train_parallel(scenario, directory, num_envs=3, total_steps=17)
    finally:
        trainer.env.close()
        writer.close(complete=False)
    frames = list(iter_race_frames(tmp_path/'races'))
    assert len(frames) == 17
    assert len({(f['episode_id'], f['physics_index']) for f in frames}) == 17
    assert {f['environment_id'] for f in frames} == {0, 1, 2}
    assert all(len(f['pre_state']) == len(f['post_state']) == len(f['commands']) == 4 for f in frames)
    assert all(f['commands']['car_2']['applied'] is not None for f in frames)
    assert all(f['learners']['car_0']['reward_components'] for f in frames)
    samples = [c for c in load_clips(tmp_path/'races') if c['kind'] == 'representative_race']
    assert len(samples) == 6 and sum(c['episode_complete'] for c in samples) == 5
    assert sum(c['end_reason'] == 'budget_cut' for c in samples) == 1
    assert all(c['spawn']['initial_states'] for c in samples)


def test_recording_covers_opponent_only_tail(tmp_path):
    from replay.dataset_writer import RaceDatasetWriter, RaceDatasetHook
    from replay.race_recorder import RaceRecorder
    from replay.race_reader import iter_race_frames, load_clips
    trainer, scenario, directory = make_setup(1, 4)
    env = trainer.env
    original_step = env.step
    def step(actions):
        obs, rewards, terms, truncs, infos = original_step(actions)
        if env._elapsed_steps == 1:
            for aid in trainer.trainable_ids:
                env.lifecycle.record_collision(aid, step=0)
                infos[aid].update(status='crashed', terminal_reason='collision', terminal_step=0)
                terms[aid] = True
            env.agents = [a for a in env.agents if a not in trainer.trainable_ids]
        return obs, rewards, terms, truncs, infos
    env.step = step
    writer = RaceDatasetWriter(tmp_path/'races', config=dict(sample_probability=1., events=False))
    hook = RaceDatasetHook(writer)
    trainer.hooks = [hook]
    trainer._transition_hooks = []
    trainer.race_recorder = RaceRecorder(hook.recording_config, hook.on_race_record, run_id='tail')
    try:
        trainer.train(1)
    finally:
        env.close()
        writer.close(complete=False)
    frames = list(iter_race_frames(tmp_path/'races'))
    assert len(frames) == 3
    assert set(frames[0]['learners']) == set(trainer.trainable_ids)
    assert frames[1]['learners'] == frames[2]['learners'] == {}
    assert frames[1]['commands']['car_0']['requested'] is None
    assert frames[1]['commands']['car_0']['applied'] is not None  # Terminal control, not an invented actor command.
    assert frames[-1]['post_state']['car_0']['terminal_reason'] == 'collision'
    assert load_clips(tmp_path/'races')[0]['all_cars_terminal']


def test_serial_step_budget_stops_mid_episode_without_fabricating_outcome():
    trainer, _, _ = make_setup(1, 2)
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


@pytest.mark.parametrize('parallel', [False, True])
def test_recording_window_reserves_late_frames_for_fixed_opponents(tmp_path, parallel):
    from replay.dataset_writer import RaceDatasetWriter, RaceDatasetHook
    from replay.race_recorder import RaceRecorder
    from replay.race_reader import iter_race_frames
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    trainer, scenario, directory = make_setup(2, 2)
    cfg = dict(sample_probability=1., max_frames=5, max_bytes=40000000,
        windows=[dict(start_step=0, end_step=6, max_frames=2, max_bytes=20000000),
                 dict(start_step=6, end_step=20, max_frames=3, max_bytes=20000000)])
    writer = RaceDatasetWriter(tmp_path, config=cfg)
    hook = RaceDatasetHook(writer)
    trainer.hooks, trainer._transition_hooks = [hook], []
    if not parallel:
        trainer.race_recorder = RaceRecorder(cfg, writer.add_event, run_id='fixed-window')
    try:
        if parallel:
            trainer.train_parallel(scenario, directory, num_envs=3, total_steps=17)
        else:
            trainer.train(total_steps=17)
    finally:
        trainer.env.close()
        writer.close()
        torch.set_num_threads(threads)
    frames = list(iter_race_frames(tmp_path))
    assert trainer._environment_steps == 17
    assert len(frames) == 5
    assert [w['frames'] for w in writer.window_usage] == [2, 3]
    late = [f for f in frames if f['recording_window_index'] == 1]
    assert all(f['recording_progress'] >= 6 for f in late)
    assert all(f['commands']['car_2']['applied'] is not None for f in frames)


def test_ready_mappo_dispatch_keeps_exact_budget_and_policy_updates():
    trainer, scenario, directory = make_setup(2, 2)
    scenario['experiment']['collector_scheduling'] = 'ready'
    capture = Capture()
    trainer.hooks = trainer._transition_hooks = [capture]
    try:
        trainer.train_parallel(scenario, directory, num_envs=3, total_steps=17)
    finally:
        trainer.env.close()
    assert trainer._environment_steps == 17
    assert len(capture.records) == 34
    assert sum(m['train/rollout_agent_samples'] for m in capture.updates) == 34
    assert capture.ends == 1
