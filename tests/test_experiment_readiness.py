from types import SimpleNamespace
from pathlib import Path
from itertools import product
import copy
import sys

import pytest
import yaml

from core.scenario import ScenarioError, load_and_expand_scenario, validate_scenario
from core.agent_builder import get_trainable_agent_ids, is_trainable_agent
from training.reward_context import build_reward_context
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


def test_shared_yaml_loader_preserves_include_precedence_and_isolation(tmp_path):
    from core.scenario import load_scenario, load_yaml_config
    from core.feature_requirements import _resolve_config

    (tmp_path / "base.yaml").write_text("reward:\n  collision: {enabled: true, penalty: -1}\nitems: [1, 2]\n")
    (tmp_path / "left.yaml").write_text("includes: base.yaml\nreward:\n  collision: {penalty: -2}\n")
    (tmp_path / "right.yaml").write_text("includes: base.yaml\nreward:\n  collision: {penalty: -3}\n")
    path = tmp_path / "task.yaml"
    path.write_text("includes: [left.yaml, right.yaml]\nreward:\n  collision: {penalty: -4}\nitems: [3]\n")
    expected = {"reward": {"collision": {"enabled": True, "penalty": -4}}, "items": [3]}
    assert load_scenario(str(path)) == _resolve_config(path.name, tmp_path) == expected
    assert RewardComposer.from_file(str(path)).compute({"info": {"terminal_reason": "collision"}})[0] == -4
    loaded = load_yaml_config(path)
    loaded["reward"]["collision"]["penalty"] = 999
    loaded["items"].append(4)
    assert load_yaml_config(path) == expected


@pytest.mark.parametrize("content", ["includes: cycle.yaml", "includes: [42]", "[1, 2]", "reward: ["])
def test_shared_yaml_loader_rejects_invalid_configs(tmp_path, content):
    from core.scenario import load_scenario, load_yaml_config
    path = tmp_path / "cycle.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_yaml_config(path)
    with pytest.raises(ScenarioError):
        load_scenario(str(path))
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "missing.yaml")
    with pytest.raises(ScenarioError):
        load_scenario(str(tmp_path / "missing.yaml"))


def test_builtin_factory_and_training_setup_contract():
    from src.core.config import AgentFactory, register_builtin_agents
    from src.core.setup import create_training_setup
    register_builtin_agents()
    assert {"ftg", "pure_pursuit", "stanley", "hybrid_pp_ftg"} <= set(AgentFactory.available_agents())
    env, agents, reward_strategies = create_training_setup(load_and_expand_scenario("scenarios/ppo.yaml"))
    try:
        assert set(agents) == {"car_1"}
        assert reward_strategies == {}
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ["a2c", "ddpg", "sac", "td3", "dqn", "qrdqn", "tqc", "typo"])
@pytest.mark.parametrize("explicit", [None, True, False])
def test_unsupported_algorithms_cannot_become_fixed_opponents(algorithm, explicit) -> None:
    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    agent = scenario["agents"]["car_0"]
    agent["algorithm"] = algorithm
    if explicit is not None:
        agent["trainable"] = explicit
    with pytest.raises(ScenarioError, match="unknown algorithm"):
        validate_scenario(scenario)
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        is_trainable_agent(agent)


@pytest.mark.parametrize("second_algorithm", ["ppo", "mappo"])
def test_unsupported_trainable_teams_fail_during_validation(second_algorithm) -> None:
    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    scenario["agents"]["car_2"] = copy.deepcopy(scenario["agents"]["car_0"])
    scenario["agents"]["car_2"]["algorithm"] = second_algorithm
    with pytest.raises(ScenarioError, match="exactly one|Mixed trainable"):
        validate_scenario(scenario)


@pytest.mark.parametrize("algorithm,trainable", [("ppo", False), ("ftg", True), ("ppo", "false")])
def test_unsupported_explicit_roles_are_rejected(algorithm, trainable) -> None:
    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    scenario["agents"]["car_0"].update(algorithm=algorithm, trainable=trainable)
    with pytest.raises(ScenarioError, match="trainable"):
        validate_scenario(scenario)


def test_ignored_training_options_are_rejected() -> None:
    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    scenario["experiment"]["total_steps"] = 10
    with pytest.raises(ScenarioError, match="episodes"):
        validate_scenario(scenario)
    scenario = load_and_expand_scenario("scenarios/mappo_gaplock.yaml")
    scenario["curriculum"] = {"phases": [{"name": "first"}]}
    with pytest.raises(ScenarioError, match="curriculum"):
        validate_scenario(scenario)


def test_unsupported_algorithm_exits_before_setup_or_logging(tmp_path, monkeypatch) -> None:
    import run

    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    scenario["agents"]["car_0"]["algorithm"] = "sac"
    path = tmp_path / "unsupported.yaml"
    path.write_text(yaml.safe_dump(scenario))
    monkeypatch.setattr(sys, "argv", ["run.py", "--scenario", str(path), "--no-wandb"])

    def reject_side_effect(*args, **kwargs):
        pytest.fail("invalid scenarios must be rejected before setup/logging")

    for name in ("create_training_setup", "CSVLogger", "WandbLogger"):
        monkeypatch.setattr(run, name, reject_side_effect)
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 1


@pytest.mark.parametrize("path", sorted(Path("scenarios").rglob("*.yaml")), ids=str)
def test_retained_scenarios_and_resource_paths(path) -> None:
    # Preserve historical planning templates without silently changing them to MAPPO.
    if path.name in {"circle_attacker.yaml", "circle_defender.yaml", "marl_attacker.yaml"}:
        with pytest.raises(ScenarioError, match="exactly one trainable"):
            load_and_expand_scenario(str(path))
        return
    scenario = load_and_expand_scenario(str(path))
    for agent in scenario["agents"].values():
        for field in ("observation", "reward"):
            resource = agent.get(field)
            if isinstance(resource, str):
                assert (path.parent / resource).is_file(), (path, field, resource)


@pytest.mark.parametrize("path", sorted(Path("sweeps").glob("*.yaml")), ids=str)
def test_sweep_parameters_reach_supported_cli_options(path, monkeypatch) -> None:
    import run

    sweep = yaml.safe_load(path.read_text())
    parameters = sweep["parameters"]
    choices = [spec.get("values", [spec.get("value")]) for spec in parameters.values()]
    assert "${args}" in sweep["command"]
    for values in product(*choices):
        arguments = dict(zip(parameters, values))
        argv = []
        for token in sweep["command"]:
            if token == "${args}":
                argv.extend(f"--{key}={value}" for key, value in arguments.items())
            elif token not in {"${env}", "python3", "${program}"}:
                argv.append(token)
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        args = run.parse_args()
        scenario = run.apply_cli_overrides(load_and_expand_scenario(args.scenario), args)
        assert scenario["experiment"]["seed"] == arguments["seed"]
        assert scenario["wandb"]["enabled"] is True
        assert get_trainable_agent_ids(scenario["agents"])


def test_complete_4_has_consistent_full_circuit_contract_and_held_out_maps() -> None:
    scenario = load_and_expand_scenario("scenarios/complete_4.yaml")
    environment = scenario["environment"]

    assert environment["target_laps"] == 1
    assert environment["lap_counting"]["count_initial_crossing_as_lap"] is False
    assert environment["max_steps"] == 250000
    assert environment["map_bundles_eval"]
    assert set(environment["map_bundles_train"]).isdisjoint(
        environment["map_bundles_eval"]
    )
    assert scenario["wandb"]["enabled"] is True
    assert scenario["wandb"]["group"] == "complete4-full-circuit-v1"
    assert scenario["wandb"]["job_type"] == "train-individual"
    assert "one-full-circuit" in scenario["wandb"]["notes"].lower()


def test_duration_calibration_scenarios_cover_all_maps() -> None:
    one_lap = load_and_expand_scenario(
        "scenarios/calibration/hybrid_pp_ftg_1lap.yaml"
    )
    three_lap = load_and_expand_scenario(
        "scenarios/calibration/pure_pursuit_3lap.yaml"
    )

    assert one_lap["environment"]["target_laps"] == 1
    assert three_lap["environment"]["target_laps"] == 3
    assert one_lap["environment"]["map_bundles_train"] == three_lap["environment"][
        "map_bundles_train"
    ]
    assert one_lap["experiment"]["episodes"] == len(
        one_lap["environment"]["map_bundles_train"]
    )


def test_complete_4_reward_is_lap_normalized_and_penalizes_reverse_progress() -> None:
    composer = RewardComposer.from_file(
        "configs/reward/tasks/complete_4_lap_completion.yaml"
    )
    component_names = {type(component).__name__ for component in composer._components}

    assert "CenterlineProgressComponent" not in component_names
    assert "ProgressDeltaBonusComponent" in component_names
    assert "WrongWayPenaltyComponent" in component_names

    base_info = {
        "collision": False,
        "lap_crossed": False,
        "race_completed": False,
        "terminal_reason": None,
        "time_limit": False,
    }
    forward_total, forward = composer.compute(
        {
            "info": {
                **base_info,
                "centerline": {"progress_delta": 0.001, "wrong_way": False},
            }
        }
    )
    reverse_total, reverse = composer.compute(
        {
            "info": {
                **base_info,
                "centerline": {"progress_delta": -0.001, "wrong_way": True},
            }
        }
    )

    assert forward["progress_delta/bonus"] == pytest.approx(0.1)
    assert reverse["progress_delta/bonus"] == pytest.approx(-0.1)
    assert reverse["wrong_way/penalty"] == pytest.approx(-0.05)
    assert forward_total == pytest.approx(0.095)
    assert reverse_total == pytest.approx(-0.155)


def test_signed_progress_delta_clamps_projection_jumps_symmetrically() -> None:
    composer = RewardComposer.from_file(
        "configs/reward/tasks/complete_4_lap_completion.yaml"
    )
    common = {
        "collision": False,
        "lap_crossed": False,
        "race_completed": False,
        "terminal_reason": None,
        "time_limit": False,
        "wrong_way": False,
    }
    _, positive = composer.compute(
        {"info": {**common, "centerline": {"progress_delta": 0.2}}}
    )
    _, negative = composer.compute(
        {"info": {**common, "centerline": {"progress_delta": -0.2}}}
    )

    assert positive["progress_delta/bonus"] == pytest.approx(2.5)
    assert negative["progress_delta/bonus"] == pytest.approx(-2.5)


def test_ppo_pretraining_uses_lap_normalized_progress_reward() -> None:
    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    assert scenario["agents"]["car_0"]["reward"].endswith(
        "configs/reward/tasks/lap_completion_normalized_progress.yaml"
    )

    composer = RewardComposer.from_file(
        "configs/reward/tasks/lap_completion_normalized_progress.yaml"
    )
    component_names = {type(component).__name__ for component in composer._components}
    assert "ReverseVelocityPenaltyComponent" not in component_names
    assert "WrongWayPenaltyComponent" not in component_names

    active_info = {
        "terminal_reason": None,
        "lap_crossed": False,
        "race_completed": False,
        "collision": False,
    }
    forward_total, forward = composer.compute(
        {"info": {**active_info, "centerline": {"progress_delta": 0.01}}}
    )
    reverse_total, reverse = composer.compute(
        {"info": {**active_info, "centerline": {"progress_delta": -0.01}}}
    )
    crash_total, crash = composer.compute(
        {
            "done": True,
            "terminated": True,
            "info": {
                **active_info,
                "terminal_reason": "collision",
                "collision": True,
                "centerline": {"progress_delta": 0.0},
            },
        }
    )

    assert forward["progress_delta/bonus"] == pytest.approx(0.01)
    assert reverse["progress_delta/bonus"] == pytest.approx(-0.01)
    assert forward_total == pytest.approx(0.01 - 1.0 / 80_000)
    assert reverse_total == pytest.approx(-0.01 - 1.0 / 80_000)
    assert crash["collision/penalty"] == pytest.approx(-1.0)
    assert crash_total == pytest.approx(-1.0 - 1.0 / 80_000)


def test_circle_mappo_scenario_uses_ppo_actor_contract() -> None:
    scenario = load_and_expand_scenario(
        "scenarios/mappo_4car_1lap_circle_ppo_pretrained.yaml"
    )
    environment = scenario["environment"]

    assert environment["map_bundles"] == ["circle_map"]
    assert environment["map_bundles_train"] == ["circle_map"]
    assert environment["map_bundles_eval"] == ["circle_map"]
    assert environment["target_laps"] == 1
    assert environment["action_repeat"] == 1
    assert scenario["mappo"] == {
        "reward_mode": "individual",
        "critic_mode": "agent_conditioned",
        "team_reward_reduction": "mean",
    }
    assert len(scenario["agents"]) == 4
    assert all(agent["algorithm"] == "mappo" for agent in scenario["agents"].values())
    assert all(
        agent["action_constraints"]["prevent_reverse"] is True
        for agent in scenario["agents"].values()
    )
    assert scenario["training_defaults"]["pretrained_actor_checkpoint"].endswith(
        "/best_model.pt"
    )


@pytest.mark.parametrize("objective", ["combined", "first_place", "sweep"])
def test_frenet_team_variants_only_change_reward_and_experiment_labels(objective):
    baseline = load_and_expand_scenario("scenarios/mappo_2v2_frenet_ppo_pretrained.yaml")
    variant = load_and_expand_scenario(f"scenarios/mappo_2v2_frenet_ppo_pretrained_{objective}.yaml")
    assert variant["experiment"]["name"] != baseline["experiment"]["name"]
    assert variant["environment"] == baseline["environment"]
    assert variant["training_defaults"] == baseline["training_defaults"]
    assert variant["training_defaults"]["team_return_mode"] == "joint"
    for aid in ("car_0", "car_1"):
        assert variant["agents"][aid]["reward"].endswith(f"race_team_2v2_{objective}.yaml")
        variant["agents"][aid]["reward"] = baseline["agents"][aid]["reward"]
    variant["experiment"]["name"] = baseline["experiment"]["name"]
    variant["wandb"] = baseline["wandb"]
    assert variant == baseline
    variant["mappo"]["critic_mode"] = "agent_conditioned"
    with pytest.raises(ScenarioError, match="Joint team returns"):
        validate_scenario(variant)


def test_reward_context_exposes_active_centerline_track_length() -> None:
    env = SimpleNamespace(
        centerline_track_length=402.5,
        trainable_agents=["car_0"],
        fixed_policy_agents=[],
        last_step_facts=None,
    )
    context = build_reward_context(
        env=env,
        agent_id="car_0",
        info_dict={"car_0": {}},
        obs_dict={"car_0": {}},
        actions={"car_0": []},
    )

    assert context["track_length"] == pytest.approx(402.5)
    context["info"] = {"centerline": {"vs": 5.0}}
    composer = RewardComposer.from_config(
        {
            "reward": {
                "centerline_progress": {
                    "enabled": True,
                    "weight": 0.02,
                    "normalize_by_track_length": True,
                    "reference_length": 400.0,
                }
            }
        }
    )
    _, breakdown = composer.compute(context)
    assert breakdown["centerline_progress/bonus"] == pytest.approx(
        0.02 * 5.0 * 400.0 / 402.5
    )


def test_frenet_observations_are_bounded_for_experiment_configs() -> None:
    for path, expected_dim in (
        ("configs/observations/rl_racer_vehicle_track_frenet.yaml", 158),
        ("configs/observations/rl_racer_vehicle_track_frenet_neighbors.yaml", 173),
    ):
        composer = ObservationComposer.from_file(
            path,
            {"lidar_beams": 108, "lidar_range": 10.0},
        )
        assert composer.obs_dim == expected_dim
        assert all(
            getattr(component, "clip", True) for component in composer.components
        )


@pytest.mark.parametrize("field,value", [("num_envs", 0), ("num_envs", True),
                                         ("torch_threads", 0), ("torch_threads", 1.5)])
def test_parallel_settings_require_positive_integers(field, value):
    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    scenario["experiment"][field] = value
    with pytest.raises(ScenarioError, match="positive integer"):
        validate_scenario(scenario)


@pytest.mark.parametrize("change,match", [
    ({"environment": {"render": True}}, "headless"),
    ({"experiment": {"seed": None}}, "seeds"),
    ({"experiment": {"episodes": 1}}, "total episodes"),
    ({"training_defaults": {"n_steps": 3}}, "multiple"),
    ({"curriculum": {"phases": [{}]}}, "curriculum"),
])
def test_parallel_ppo_rejects_unsupported_contracts(change, match):
    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    scenario["experiment"]["num_envs"] = 2
    for key, fields in change.items():
        scenario.setdefault(key, {}).update(fields)
    with pytest.raises(ScenarioError, match=match):
        validate_scenario(scenario)


def test_parallel_mappo_is_explicitly_rejected():
    scenario = load_and_expand_scenario("scenarios/mappo_gaplock.yaml")
    scenario["experiment"]["num_envs"] = 2
    with pytest.raises(ScenarioError, match="PPO only"):
        validate_scenario(scenario)
