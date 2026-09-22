"""Episode lap metrics remain meaningful before and after a successful lap."""

import csv
import json
from types import SimpleNamespace

from loggers.csv_logger import CSVLogger
from training.hooks import ConsoleHook, CSVHook, WandbHook


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
    logger = CSVLogger(str(tmp_path))
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
