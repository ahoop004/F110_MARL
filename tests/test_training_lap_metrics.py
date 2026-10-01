"""Episode lap metrics remain meaningful before and after a successful lap."""

import csv
import json
from types import SimpleNamespace

from loggers.csv_logger import CSVLogger
from training.hooks import ConsoleHook, CSVHook, WandbHook


def test_mappo_update_reports_timings_without_completed_episodes():
    from training.hooks import MAPPOConsoleHook
    lines = []
    hook = MAPPOConsoleHook(SimpleNamespace(print_info=lines.append), every_updates=1)
    hook.on_update({'train/updates': 1, 'train/environment_steps': 102400,
                    'perf/end_to_end_env_steps_per_second': 700.,
                    'perf/collection_seconds': 50., 'perf/update_seconds': 14.,
                    'perf/round_env_steps_per_second': 1600.,
                    'perf/inference_seconds': 10., 'perf/worker_receive_seconds': 2.,
                    'perf/worker_wait_seconds': 36.})
    for expected in ('completed_total=0', 'env_steps/s=700.0', 'round_steps/s=1600.0',
                     'collect_s=50.00', 'update_s=14.00', 'infer_s=10.00',
                     'receive_s=2.00', 'wait_s=36.00'):
        assert expected in lines[-1]


def test_episode_lap_outputs_handle_completion_and_reset(tmp_path):
    lines, payloads = [], []
    console = ConsoleHook(SimpleNamespace(print_info=lines.append))
    wandb = WandbHook(SimpleNamespace(log_metrics=payloads.append))
    csv_hook = CSVHook(CSVLogger(str(tmp_path)))

    for episode, (count, time_s) in enumerate([(0, None), (2, 4.1), (0, None)]):
        info = {"outcome": "track_boundary", "lap_count": count}
        metrics = {"episode_steps": 100, "lap_time_s": time_s}
        for hook in (console, wandb, csv_hook):
            hook.on_episode_end(episode, 16.26, info, metrics)
    csv_hook.on_training_end()

    episode_lines = [line for line in lines if line.startswith("ep ")]
    assert "laps=0  lap_time=n/a" in episode_lines[0]
    assert "laps=2  lap_time=4.10s" in episode_lines[1]
    assert "laps=0  lap_time=n/a" in episode_lines[2]
    assert [row["episode/lap_count"] for row in payloads] == [0, 2, 0]
    assert "episode/lap_time_s" not in payloads[0]
    assert payloads[1]["episode/lap_time_s"] == 4.1
    assert "episode/lap_time_s" not in payloads[2]
    with (tmp_path / "episode_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["lap_count"] for row in rows] == ["0", "2", "0"]
    # The header must include timing even when the first episode has no lap.
    assert [row["lap_time_s"] for row in rows] == ["", "4.1", ""]


def test_csv_keeps_late_metrics_and_update_diagnostics(tmp_path):
    logger = CSVLogger(str(tmp_path))
    logger.log_training_episode(0, 1., {}, {"episode_steps": 1})
    logger.log_training_episode(1, 2., {}, {"episode_steps": 2, "late_incident": 3})
    logger.log_update({"train/environment_steps": 1})
    logger.log_update({"train/environment_steps": 2, "train/approx_kl": .05})
    logger.close()
    with (tmp_path / "episode_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["late_incident"] for r in rows] == ["", "3"]
    with (tmp_path / "update_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["train/approx_kl"] for r in rows] == ["", "0.05"]


def test_race_record_preserves_finish_and_continuous_missingness(tmp_path):
    from metrics.racing_eval import (create_episode_facts, update_agent_step_facts,
                                    finalize_episode_facts, episode_race_record, aggregate_eval_episodes)
    from training.hooks import MAPPOConsoleHook
    facts = create_episode_facts(episode=0, agent_ids=["a", "b", "c", "d"],
                                trainable_ids=["a", "b"], opponent_ids=["c", "d"])
    update_agent_step_facts(facts, step_idx=10, infos={
        "a": {"race_completed": True, "terminal_reason": "race_complete", "terminal_step": 9,
              "finish_position": 1, "centerline": {"progress_delta": .2}},
        "b": {"terminal_reason": "track_boundary", "terminal_step": 9}},
        terminations={"a": True, "b": True})
    # Parked finishers retain their finish even if subsequent telemetry says collision.
    update_agent_step_facts(facts, step_idx=11, infos={
        "a": {"terminal_reason": "collision", "terminal_step": 10}}, terminations={"a": True})
    finalize_episode_facts(facts)
    finite = episode_race_record(facts, timestep=.05)
    assert finite["first_place"] == 1 and finite["both_finished"] == 0
    assert finite["at_least_one_finished"] is True
    assert finite["agents"]["a"]["clean_finish_time_s"] == .5
    assert not finite["agents"]["a"]["collision_dnf"]
    assert finite["agents"]["b"]["boundary_dnf"]
    assert finite["agents"]["b"]["clean_finish_time_s"] is None
    summary = aggregate_eval_episodes([facts], timestep=.05)
    assert summary["first_place_count"] == 1 and summary["race_count"] == 1
    assert summary["per_car"]["b"]["boundary_dnf_count"] == 1
    continuous = episode_race_record(facts, timestep=.05, finite_race=False)
    assert all(continuous[key] is None for key in ("both_finished", "first_place", "sweep", "rank_score"))
    assert continuous["agents"]["a"]["finished"] is None
    finite.update(episode_id="run_env0_ep0", spawn_ids={}, training_return=2.)
    logger = CSVLogger(str(tmp_path), scenario_config={"logging": {"csv_exports": True}})
    logger.log_training_episode(0, 2., {}, {"race_record": finite})
    logger.close()
    assert json.loads((tmp_path / "race_metrics.jsonl").read_text())["agents"]["b"]["boundary_dnf"]
    with (tmp_path / "agent_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and {row["episode_id"] for row in rows} == {"run_env0_ep0"}
    assert rows[2]["reward"] == ""  # Opponent rewards are unavailable.
    lines = []
    monitor = MAPPOConsoleHook(SimpleNamespace(print_info=lines.append), window=1, every_updates=1)
    monitor.on_episode_end(0, 2., {}, {"race_record": finite})
    monitor.on_update({"train/updates": 1, "train/environment_steps": 11})
    assert "completed_window=1" in lines[-1] and "first_place=100.0%" in lines[-1]
    continuous["training_return"] = 1.
    monitor.on_episode_end(1, 1., {}, {"race_record": continuous})
    monitor.on_update({"train/updates": 2, "train/environment_steps": 20})
    assert "first_place" not in lines[-1] and "progress_laps=" in lines[-1]


def test_default_race_logging_keeps_one_source_and_update_clock(tmp_path):
    race = {"episode_id": "run_ep0", "agents": {"car_0": {"team": "trainable"}},
            "training_return": 2., "attack_successes": 1}
    logger = CSVLogger(str(tmp_path))
    logger.log_training_episode(0, 2., {}, {"race_record": race, "train/policy_loss": .1})
    logger.log_update({"train/policy_loss": .1, "train/environment_steps": 100})
    logger.log_collector_progress({"collector/phase": "updating"})
    logger.close()
    logger.close()  # Training hook and CLI cleanup both close the logger.
    assert json.loads((tmp_path / "race_metrics.jsonl").read_text()) == race
    assert not (tmp_path / "episode_metrics.csv").exists()
    assert not (tmp_path / "agent_metrics.csv").exists()
    assert not (tmp_path / "collector_progress.csv").exists()
    assert list(csv.DictReader((tmp_path / "update_metrics.csv").open())) == [
        {"train/policy_loss": "0.1", "train/environment_steps": "100"}]


def test_csv_batches_schema_changes_and_flushes_on_interval(tmp_path, monkeypatch):
    import loggers.csv_logger as module
    now = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    logger = CSVLogger(str(tmp_path), scenario_config={"logging": {
        "flush_every": 3, "flush_interval_seconds": 10., "collector_progress": True}})
    logger.log_update({"step": 1})
    logger.log_update({"step": 2, "late": 4})
    path = tmp_path / "update_metrics.csv"
    assert not path.exists()
    logger.log_update({"step": 3, "late": 5})
    assert len(list(csv.DictReader(path.open()))) == 3
    logger.log_update({"step": 4, "another": 6})
    now[0] = 11.
    logger.log_collector_progress({"collector/phase": "updating"})
    rows = list(csv.DictReader(path.open()))
    assert [r["another"] for r in rows] == ["", "", "", "6"]
    assert next(csv.DictReader((tmp_path / "collector_progress.csv").open())) == {
        "collector/phase": "updating"}
    logger.log_update({"step": 5})
    logger.close()
    assert len(list(csv.DictReader(path.open()))) == 5


def test_heuristic_agent_facts_are_kept_without_a_race_record(tmp_path):
    logger = CSVLogger(str(tmp_path))
    logger.log_training_episode(0, 0., {}, {
        'agent_outcomes': {'a': 'finished', 'b': 'self_crash'}, 'episode_steps': 10})
    logger.close()
    rows = list(csv.DictReader((tmp_path / 'agent_metrics.csv').open()))
    assert {r['agent_id']: r['outcome'] for r in rows} == {'a': 'finished', 'b': 'self_crash'}


def test_completion_console_and_wandb_share_learner_laps_without_finish_time_aliases():
    from copy import deepcopy
    from training.hooks import MAPPOConsoleHook
    lines, payloads = [], []
    logger = SimpleNamespace(print_info=lines.append)
    console = ConsoleHook(logger, lap_completion=True)
    monitor = MAPPOConsoleHook(logger, lap_completion=True, every_updates=1, diagnostic_every=0)
    wandb = WandbHook(SimpleNamespace(log_metrics=payloads.append))
    metrics = {'agent_rewards': {'a': 1., 'b': 2.}, 'reward_mode': 'team_shared',
               'agent_individual_rewards': {'a': 1., 'b': 2.},
               'agent_outcomes': {'a': 'finished', 'b': 'self_crash'},
               'agent_terminal_reasons': {'a': 'race_complete', 'b': 'collision'},
               'race_record': {'training_return': 3., 'mean_learner_laps': 2., 'both_finished': False,
                               'agents': {}}}
    for aid, team, count, time in [('a', 'trainable', 1, 10.), ('b', 'trainable', 3, 20.),
                                  ('opponent', 'opponent', 9, 100.)]:
        metrics['race_record']['agents'][aid] = dict(team=team, valid_lap_count=count,
            mean_valid_lap_time_s=time, clean_finish_time_s=90.,
            collision_dnf=aid == 'b', boundary_dnf=False, timeout=False)
    original = deepcopy(metrics)
    console.on_episode_end(1, 3., {}, metrics)
    wandb.on_episode_end(1, 3., {}, metrics)
    monitor.on_episode_end(1, 3., {}, metrics)
    monitor.on_update({'train/updates': 1, 'train/environment_steps': 100,
                       'perf/collection_seconds': 25., 'perf/inference_seconds': 10.})
    assert 'reward=+3.00  mean=+3.00  laps=2  lap_time=17.50s' in lines[0]
    assert 'outcome=a:finished b:self_crash' in lines[0]
    assert 'reward=+3.00 mean=3.00 laps=2.00 lap_time=17.50s finished=0.0% failed=50.0%' in lines[1]
    assert all(token not in line for line in lines for token in
               ('terminal reasons', 'individual reward', 'first_place', 'sweep=', 'collect_s=', 'infer_s='))
    assert payloads[0]['episode/lap_count'] == 2.
    assert payloads[0]['episode/lap_time_s'] == 17.5
    assert metrics == original
    for agent in metrics['race_record']['agents'].values():
        agent.update(mean_valid_lap_time_s=None, valid_lap_count=0)
    console.on_episode_end(2, 3., {}, metrics)
    wandb.on_episode_end(2, 3., {}, metrics)
    assert 'lap_time=n/a' in lines[-1]
    assert 'episode/lap_time_s' not in payloads[-1]
