import csv
import json

import torch

from core.provenance import build_run_provenance, provenance_mismatches
from loggers.csv_logger import CSVLogger
from training.hooks import CSVHook, CheckpointHook


class _CheckpointAgent:
    def save(self, path: str) -> None:
        torch.save({"weights": torch.tensor([1.0])}, path)


def test_checkpoint_hook_attaches_run_provenance(tmp_path) -> None:
    provenance = {"scenario_source_sha256": "abc", "resolved_config_sha256": "def"}
    hook = CheckpointHook(
        _CheckpointAgent(),
        output_dir=str(tmp_path),
        save_every=1,
        provenance=provenance,
    )

    hook.on_episode_end(1, 1.0, {}, {})

    checkpoint = torch.load(tmp_path / "checkpoint_ep000001.pt", weights_only=False)
    assert checkpoint["provenance"] == provenance
    assert torch.equal(checkpoint["weights"], torch.tensor([1.0]))


def test_csv_hook_writes_snapshot_and_episode_metrics(tmp_path) -> None:
    logger = CSVLogger(
        str(tmp_path),
        scenario_config={"experiment": {"name": "test"}},
        provenance={"run_id": "run-1"},
    )
    hook = CSVHook(logger)

    hook.on_episode_end(
        0,
        3.5,
        {"outcome": "race_complete", "map_bundle": "circle_map", "lap_count": 3},
        {"episode_steps": 123, "train/loss": 0.25},
    )
    hook.on_training_end()

    snapshot = json.loads((tmp_path / "config_snapshot.json").read_text())
    assert snapshot["provenance"]["run_id"] == "run-1"
    row = next(csv.DictReader((tmp_path / "episode_metrics.csv").open()))
    assert row["episode_steps"] == "123"
    assert row["map_bundle"] == "circle_map"
    assert row["train_loss"] == "0.25"


def test_run_provenance_hashes_source_and_resolved_config(tmp_path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("experiment:\n  name: test\n")
    scenario = {
        "experiment": {"name": "test", "seed": 7},
        "environment": {"maps": []},
        "agents": {},
    }

    provenance = build_run_provenance(
        scenario,
        scenario_path=scenario_path,
        run_id="test-run",
        algorithm="ppo",
        trainable_agents=["car_0"],
    )

    assert len(provenance["scenario_source_sha256"]) == 64
    assert len(provenance["resolved_config_sha256"]) == 64
    assert provenance["seed"] == 7
    assert provenance["trainable_agents"] == ["car_0"]


def test_provenance_comparison_detects_config_and_map_changes() -> None:
    stored = {
        "algorithm": "ppo",
        "scenario_source_sha256": "source",
        "resolved_config_sha256": "config-a",
        "map_protocols": {"track": {"yaml_sha256": "map-a", "yaml_path": "/old"}},
    }
    current = {
        **stored,
        "resolved_config_sha256": "config-b",
        "map_protocols": {"track": {"yaml_sha256": "map-b", "yaml_path": "/new"}},
    }

    mismatches = provenance_mismatches(stored, current)

    assert any("resolved_config_sha256" in mismatch for mismatch in mismatches)
    assert any("map_protocols" in mismatch for mismatch in mismatches)
