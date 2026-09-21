"""Episode lap metrics remain meaningful before and after a successful lap."""

import csv
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
