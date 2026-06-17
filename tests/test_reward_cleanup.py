from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wrappers.rewards.composer import (  # noqa: E402
    COMPONENT_ALIASES,
    COMPONENT_REGISTRY,
    RewardComposer,
    canonical_component_key,
)
from training.reward_context import build_reward_context  # noqa: E402


def test_stale_src_rewards_directory_removed() -> None:
    assert not (ROOT / "src" / "rewards").exists()


def test_task_configs_are_scaffolded_and_scenarios_use_tasks() -> None:
    task_dir = ROOT / "configs" / "reward" / "tasks"
    expected_tasks = [
        "attacker_racing.yaml",
        "centerline_racing.yaml",
        "gaplock_attack.yaml",
        "gaplock_attacker.yaml",
        "gaplock_centerline_pressure.yaml",
        "race_1v1.yaml",
        "race_1v1_completion.yaml",
        "race_team_2v2.yaml",
        "race_team_2v2_completion.yaml",
        "time_trial.yaml",
    ]
    assert task_dir.is_dir()
    assert sorted(path.name for path in task_dir.glob("*.yaml")) == expected_tasks

    scenario_task_refs = []
    for path in sorted((ROOT / "scenarios").glob("*.yaml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("reward:"):
                continue
            reward_ref = stripped.split(":", 1)[1].strip()
            if "configs/reward/tasks/" in reward_ref:
                scenario_task_refs.append(Path(reward_ref).name)

    assert scenario_task_refs
    assert sorted(set(scenario_task_refs)) == [
        "attacker_racing.yaml",
        "centerline_racing.yaml",
        "gaplock_attack.yaml",
        "gaplock_attacker.yaml",
        "race_1v1.yaml",
        "race_team_2v2.yaml",
    ]


def test_reward_config_filenames_are_component_bundles() -> None:
    reserved_task_terms = ("gaplock", "racing", "attacker", "defender", "overtake", "block", "draft")
    offenders = [
        path.name
        for path in (ROOT / "configs" / "reward").glob("*.yaml")
        if any(term in path.stem for term in reserved_task_terms)
    ]

    assert offenders == []


def test_scenario_reward_paths_resolve_to_current_configs() -> None:
    missing = []
    direct_bundle_paths = []
    for path in sorted((ROOT / "scenarios").glob("*.yaml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("reward:"):
                continue
            reward_ref = stripped.split(":", 1)[1].strip()
            if not reward_ref:
                continue
            if "configs/reward/tasks/" not in reward_ref:
                direct_bundle_paths.append((path.relative_to(ROOT).as_posix(), reward_ref))
            resolved = (path.parent / reward_ref).resolve()
            if not resolved.exists():
                missing.append((path.relative_to(ROOT).as_posix(), reward_ref))

    assert direct_bundle_paths == []
    assert missing == []


def test_task_configs_are_include_based() -> None:
    direct_reward_blocks = []
    for path in sorted((ROOT / "configs" / "reward" / "tasks").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if "\nreward:" in text or text.startswith("reward:"):
            direct_reward_blocks.append(path.name)
        assert "includes:" in text

    assert direct_reward_blocks == []


def test_scenario_facing_reward_bundles_are_include_based() -> None:
    direct_reward_blocks = []
    for path in sorted((ROOT / "configs" / "reward").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if "\nreward:" in text or text.startswith("reward:"):
            direct_reward_blocks.append(path.name)
        assert "includes:" in text

    assert direct_reward_blocks == []


def test_no_active_imports_use_old_rewards_package() -> None:
    bad_patterns = (
        "src." + "rewards",
        "from " + "rewards",
        "import " + "rewards",
    )
    search_roots = (
        ROOT / "run.py",
        ROOT / "src",
        ROOT / "configs",
        ROOT / "scenarios",
        ROOT / "tests",
    )

    offenders = []
    for root in search_roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in bad_patterns:
                if pattern in text:
                    offenders.append((path.relative_to(ROOT).as_posix(), pattern))

    assert offenders == []


def test_legacy_aliases_map_to_generic_component_classes() -> None:
    expected = {
        "gaplock_pressure": "target_proximity",
        "gaplock_forcing": "target_edge_pressure",
        "terminal_success": "target_crash_bonus",
        "terminal_timeout": "timeout_penalty",
        "terminal_self_crash": "self_crash_penalty",
    }

    assert COMPONENT_ALIASES == expected
    for legacy_key, generic_key in expected.items():
        assert canonical_component_key(legacy_key) == generic_key
        assert COMPONENT_REGISTRY[generic_key] is COMPONENT_REGISTRY[canonical_component_key(legacy_key)]


@pytest.mark.parametrize(
    "config_path",
    sorted((ROOT / "configs" / "reward").rglob("*.yaml")),
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_all_reward_configs_load(config_path: Path) -> None:
    RewardComposer.from_file(str(config_path))


def test_legacy_alias_outputs_use_generic_breakdown_names() -> None:
    composer = RewardComposer.from_config(
        {
            "reward": {
                "gaplock_pressure": {
                    "enabled": True,
                    "weight": 2.0,
                    "preferred_distance": 1.0,
                    "distance_tolerance": 1.0,
                },
                "gaplock_forcing": {
                    "enabled": True,
                    "weight": 0.5,
                },
            }
        }
    )

    total, breakdown = composer.compute(
        {
            "next_obs": {
                "pose": [0.0, 0.0, 0.0],
                "target_pose": [1.0, 0.0, 0.0],
            },
            "info": {
                "forcing_reward": 4.0,
            },
        }
    )

    assert total == pytest.approx(4.0)
    assert breakdown == {
        "target_proximity/bonus": pytest.approx(2.0),
        "target_edge_pressure/bonus": pytest.approx(2.0),
    }
    assert not any(name.startswith("gaplock/") for name in breakdown)


def test_atomic_centerline_terms_match_composite_total() -> None:
    composite = RewardComposer.from_config(
        {
            "reward": {
                "centerline": {
                    "enabled": True,
                    "vs_weight": 1.0,
                    "vd_weight": 0.005,
                    "d_weight": 0.01,
                    "steer_weight": 0.03,
                    "normalize_by_track_length": True,
                    "reference_length": 400.0,
                }
            }
        }
    )
    atomic = RewardComposer.from_config(
        {
            "reward": {
                "centerline_progress": {
                    "enabled": True,
                    "weight": 1.0,
                    "normalize_by_track_length": True,
                    "reference_length": 400.0,
                },
                "centerline_lateral_velocity_penalty": {
                    "enabled": True,
                    "weight": 0.005,
                    "normalize_by_track_length": True,
                    "reference_length": 400.0,
                },
                "centerline_deviation_penalty": {
                    "enabled": True,
                    "weight": 0.01,
                },
                "steering_penalty": {
                    "enabled": True,
                    "weight": 0.03,
                },
            }
        }
    )
    step_info = {
        "info": {
            "centerline": {
                "vs": 2.0,
                "vd": -0.4,
                "d": 0.25,
            }
        },
        "action": [0.2, 0.8],
        "track_length": 200.0,
    }

    composite_total, composite_breakdown = composite.compute(step_info)
    atomic_total, atomic_breakdown = atomic.compute(step_info)

    assert composite_breakdown == {"centerline/total": pytest.approx(composite_total)}
    assert atomic_total == pytest.approx(composite_total)
    assert atomic_breakdown == {
        "centerline_progress/bonus": pytest.approx(4.0),
        "centerline_lateral_velocity/penalty": pytest.approx(-0.004),
        "centerline_deviation/penalty": pytest.approx(-0.0025),
        "steering/penalty": pytest.approx(-0.006),
    }


def test_atomic_progress_safety_terms_match_composite_total() -> None:
    composite = RewardComposer.from_config(
        {
            "reward": {
                "progress_safety": {
                    "enabled": True,
                    "wrong_way_penalty": -2.0,
                    "reverse_progress_weight": 5.0,
                    "max_abs_d": 1.5,
                    "offtrack_penalty": -1.0,
                }
            }
        }
    )
    atomic = RewardComposer.from_config(
        {
            "reward": {
                "wrong_way_penalty": {
                    "enabled": True,
                    "penalty": -2.0,
                },
                "reverse_progress_penalty": {
                    "enabled": True,
                    "weight": 5.0,
                },
                "offtrack_penalty": {
                    "enabled": True,
                    "max_abs_d": 1.5,
                    "penalty": -1.0,
                },
            }
        }
    )
    step_info = {
        "info": {
            "centerline": {
                "wrong_way": True,
                "progress_delta": -0.4,
                "d": -1.6,
            }
        }
    }

    composite_total, composite_breakdown = composite.compute(step_info)
    atomic_total, atomic_breakdown = atomic.compute(step_info)

    assert composite_breakdown == {
        "progress_safety/wrong_way": pytest.approx(-2.0),
        "progress_safety/reverse_progress": pytest.approx(-2.0),
        "progress_safety/offtrack": pytest.approx(-1.0),
    }
    assert atomic_total == pytest.approx(composite_total)
    assert atomic_breakdown == {
        "wrong_way/penalty": pytest.approx(-2.0),
        "reverse_progress/penalty": pytest.approx(-2.0),
        "offtrack/penalty": pytest.approx(-1.0),
    }


def test_completion_components_use_generic_breakdown_names() -> None:
    composer = RewardComposer.from_config(
        {
            "reward": {
                "progress_delta_bonus": {
                    "enabled": True,
                    "weight": 100.0,
                    "positive_only": True,
                },
                "step_time_penalty": {
                    "enabled": True,
                    "penalty": -0.01,
                },
            }
        }
    )

    total, breakdown = composer.compute(
        {
            "info": {
                "centerline": {
                    "progress_delta": 0.02,
                }
            },
            "done": False,
            "terminated": False,
            "truncated": False,
        }
    )

    assert total == pytest.approx(1.99)
    assert breakdown == {
        "progress_delta/bonus": pytest.approx(2.0),
        "step_time/penalty": pytest.approx(-0.01),
    }


def test_progress_delta_bonus_ignores_reverse_progress_by_default() -> None:
    composer = RewardComposer.from_config(
        {
            "reward": {
                "progress_delta_bonus": {
                    "enabled": True,
                    "weight": 100.0,
                }
            }
        }
    )

    total, breakdown = composer.compute(
        {
            "info": {
                "centerline": {
                    "progress_delta": -0.02,
                }
            }
        }
    )

    assert total == pytest.approx(0.0)
    assert breakdown == {"progress_delta/bonus": pytest.approx(0.0)}


def test_relative_progress_uses_configured_target_progress_delta() -> None:
    composer = RewardComposer.from_config(
        {
            "reward": {
                "relative_progress_bonus": {
                    "enabled": True,
                    "weight": 100.0,
                }
            }
        }
    )

    total, breakdown = composer.compute(
        {
            "agent_id": "car_0",
            "target_id": "car_1",
            "info": {
                "centerline": {
                    "progress_delta": 0.03,
                }
            },
            "all_infos": {
                "car_0": {
                    "centerline": {
                        "progress_delta": 0.03,
                    }
                },
                "car_1": {
                    "centerline": {
                        "progress_delta": 0.01,
                    }
                },
                "car_2": {
                    "centerline": {
                        "progress_delta": 0.05,
                    }
                },
            },
        }
    )

    assert total == pytest.approx(2.0)
    assert breakdown == {"relative_progress/bonus": pytest.approx(2.0)}


def test_relative_progress_returns_empty_without_target_progress() -> None:
    composer = RewardComposer.from_config(
        {
            "reward": {
                "relative_progress_bonus": {
                    "enabled": True,
                    "weight": 100.0,
                }
            }
        }
    )

    total, breakdown = composer.compute(
        {
            "agent_id": "car_0",
            "info": {
                "centerline": {
                    "progress_delta": 0.03,
                }
            },
            "all_infos": {
                "car_0": {
                    "centerline": {
                        "progress_delta": 0.03,
                    }
                },
                "car_1": {},
                "car_2": {},
            },
        }
    )

    assert total == pytest.approx(0.0)
    assert breakdown == {}


def test_finish_ahead_bonus_requires_ego_finish_before_target() -> None:
    composer = RewardComposer.from_config(
        {
            "reward": {
                "finish_ahead_bonus": {
                    "enabled": True,
                    "bonus": 100.0,
                    "require_clean": True,
                }
            }
        }
    )

    total, breakdown = composer.compute(
        {
            "terminated": True,
            "info": {
                "finish_line": True,
                "target_finished": False,
                "collision": False,
            },
        }
    )

    assert total == pytest.approx(100.0)
    assert breakdown == {"finish_ahead/bonus": pytest.approx(100.0)}

    total, breakdown = composer.compute(
        {
            "terminated": True,
            "info": {
                "finish_line": True,
                "target_finished": True,
                "collision": False,
            },
        }
    )

    assert total == pytest.approx(0.0)
    assert breakdown == {}


def test_team_progress_components_use_reward_context_groups() -> None:
    composer = RewardComposer.from_config(
        {
            "reward": {
                "team_progress_bonus": {
                    "enabled": True,
                    "weight": 100.0,
                    "aggregation": "mean",
                },
                "team_relative_progress_bonus": {
                    "enabled": True,
                    "weight": 100.0,
                    "aggregation": "mean",
                },
            }
        }
    )

    total, breakdown = composer.compute(
        {
            "agent_id": "car_0",
            "trainable_agent_ids": ["car_0", "car_1"],
            "opponent_agent_ids": ["car_2", "car_3"],
            "info": {
                "centerline": {
                    "progress_delta": 0.02,
                }
            },
            "all_infos": {
                "car_0": {
                    "centerline": {
                        "progress_delta": 0.02,
                    }
                },
                "car_1": {
                    "centerline": {
                        "progress_delta": 0.04,
                    }
                },
                "car_2": {
                    "centerline": {
                        "progress_delta": 0.01,
                    }
                },
                "car_3": {
                    "centerline": {
                        "progress_delta": 0.02,
                    }
                },
            },
        }
    )

    assert total == pytest.approx(4.5)
    assert breakdown == {
        "team_progress/bonus": pytest.approx(3.0),
        "team_relative_progress/bonus": pytest.approx(1.5),
    }


def test_reward_context_exposes_target_and_team_groups() -> None:
    class DummyEnv:
        trainable_agents = ["car_0", "car_1"]
        fixed_policy_agents = ["car_2", "car_3"]
        last_step_facts = None

        def get_global_state(self):
            raise RuntimeError("global state unavailable in this unit test")

        def get_target_id(self, agent_id: str):
            return {"car_0": "car_2"}.get(agent_id)

    context = build_reward_context(
        env=DummyEnv(),
        agent_id="car_0",
        info_dict={"car_0": {}, "car_1": {}, "car_2": {}, "car_3": {}},
        obs_dict={},
        actions={},
    )

    assert context["target_id"] == "car_2"
    assert context["trainable_agent_ids"] == ["car_0", "car_1"]
    assert context["teammate_ids"] == ["car_1"]
    assert context["opponent_agent_ids"] == ["car_2", "car_3"]
    assert context["global_state"].shape == (0,)


@pytest.mark.parametrize(
    ("bundle_path", "task_path"),
    [
        ("configs/reward/target_pressure_centerline.yaml", "configs/reward/tasks/gaplock_attack.yaml"),
        ("configs/reward/target_pressure.yaml", "configs/reward/tasks/gaplock_attacker.yaml"),
        (
            "configs/reward/target_proximity_centerline.yaml",
            "configs/reward/tasks/gaplock_centerline_pressure.yaml",
        ),
        ("configs/reward/centerline_progress.yaml", "configs/reward/tasks/centerline_racing.yaml"),
        ("configs/reward/centerline_target_crash.yaml", "configs/reward/tasks/attacker_racing.yaml"),
        ("configs/reward/centerline_lap_target_finish.yaml", "configs/reward/tasks/race_1v1.yaml"),
        (
            "configs/reward/centerline_lap_target_finish_conservative.yaml",
            "configs/reward/tasks/race_team_2v2.yaml",
        ),
    ],
)
def test_task_configs_match_current_reward_bundles(bundle_path: str, task_path: str) -> None:
    bundle = RewardComposer.from_file(str(ROOT / bundle_path))
    task = RewardComposer.from_file(str(ROOT / task_path))
    step_info = {
        "info": {
            "centerline": {
                "vs": 2.0,
                "vd": -0.4,
                "d": 0.25,
                "progress_delta": 0.01,
                "wrong_way": False,
            },
            "forcing_reward": 1.5,
            "collision": False,
            "target_collision": False,
            "target_finished": False,
            "finish_line": False,
        },
        "next_obs": {
            "pose": [0.0, 0.0, 0.0],
            "target_pose": [1.0, 0.0, 0.0],
        },
        "action": [0.2, 0.8],
        "track_length": 200.0,
        "done": False,
        "terminated": False,
        "truncated": False,
    }

    bundle_total, _ = bundle.compute(step_info)
    task_total, _ = task.compute(step_info)

    assert task_total == pytest.approx(bundle_total)
