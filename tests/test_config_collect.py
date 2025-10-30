from pathlib import Path

import pytest

from f110x.utils.config_models import ExperimentConfig, MainConfig


def test_collect_defaults():
    cfg = MainConfig.from_dict({})
    settings = cfg.collect_settings
    assert settings["collect_workers"] == 1
    assert settings["collect_prefetch"] == 2
    assert settings["collect_seed_stride"] == 1


def test_collect_overrides(tmp_path):
    data = {
        "main": {
            "collect_workers": 3,
            "collect_prefetch": 5,
            "collect_seed_stride": 7,
        }
    }
    config = ExperimentConfig.from_dict(data)
    settings = config.main.collect_settings
    assert settings == {
        "collect_workers": 3,
        "collect_prefetch": 5,
        "collect_seed_stride": 7,
    }

    # ensure metadata copied into runner context
    scenario = tmp_path / "cfg.yaml"
    scenario.write_text("main:\n  collect_workers: 4\n")
    loaded = ExperimentConfig.load(scenario, experiment=None)
    assert loaded.main.collect_workers == 4


def test_collect_invalid(monkeypatch):
    with pytest.raises(ValueError):
        MainConfig.from_dict({"collect_workers": 0})
