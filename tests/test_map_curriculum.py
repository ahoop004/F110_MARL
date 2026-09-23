from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario, resolve_evaluation_protocol, validate_scenario, ScenarioError
from env.map_schedule import MapScheduler
from training.hooks import EvaluationCheckpointHook
from training.map_curriculum import MapCurriculum, CurriculumEvaluator, MapCurriculumCheckpointHook


def report(**counts):
    return {"per_map": {name: {"episodes": 10, "strict_clean_finish_count": count} for name, count in counts.items()},
            "mean_clean_finish_time_s": 50}


def test_advancement_requires_retention_and_two_consecutive_passes():
    manager = MapCurriculum(["L", "circle", "hard"], "L")
    assert manager.observe(report(L=9, circle=8, hard=2)) is None
    assert manager.observe(report(L=8, circle=8, hard=2)) is None
    assert manager.observe(report(L=9, circle=8, hard=2)) is None
    assert manager.observe(report(L=9, circle=8, hard=2)) == "circle"
    assert manager.training_schedule() == ["circle", "L"]
    for _ in range(3):
        manager.observe(report(L=8, circle=10, hard=2))
    assert manager.active == ["L", "circle"]
    manager.observe(report(L=9, circle=9, hard=2))
    assert manager.observe(report(L=9, circle=9, hard=2)) == "hard"
    assert manager.training_schedule() == ["hard", "L", "hard", "circle"]


def test_zero_shot_bundle_pass_skips_other_training_stages_and_resets_on_failure():
    manager = MapCurriculum(["L", "circle"], "L")
    manager.observe(report(L=10, circle=10))
    assert not manager.complete
    manager.observe(report(L=8, circle=10))
    manager.observe(report(L=9, circle=9))
    assert not manager.complete
    manager.observe(report(L=9, circle=9))
    assert manager.complete
    assert manager.active == ["L"]


def test_missing_or_undersampled_maps_fail_closed():
    manager = MapCurriculum(["L", "circle"], "L")
    with pytest.raises(ValueError, match="circle"):
        manager.observe(report(L=10))
    summary = report(L=10, circle=10)
    summary["per_map"]["circle"]["episodes"] = 9
    with pytest.raises(ValueError, match="circle"):
        manager.observe(summary)
    assert manager.bundle_streak == 0


def test_checkpoint_selection_prioritizes_coverage_then_weakest_map():
    manager = MapCurriculum(["L", "circle", "hard"], "L")
    summaries = [report(L=10, circle=10, hard=0), report(L=9, circle=9, hard=7), report(L=9, circle=9, hard=9)]
    for summary in summaries:
        manager.observe(summary)
    scores = [EvaluationCheckpointHook.selection_score(s, "map_curriculum") for s in summaries]
    assert scores[0] < scores[1] < scores[2]


def test_runtime_schedule_preserves_evaluation_order_and_survives_reseed():
    scheduler = MapScheduler(dict(map_bundles=["L", "circle", "hard"], map_bundles_train=["L"],
        map_bundles_eval=["L", "circle", "hard"], map_cycle="per_episode", map_pick="round_robin"),
        rng=np.random.default_rng(0))
    scheduler.set_training_bundles(["hard", "L", "hard", "circle"])
    for seed in range(2):
        scheduler.reseed(np.random.default_rng(seed))
        assert [scheduler.select_next_bundle("train") for _ in range(4)] == ["hard", "L", "hard", "circle"]
        assert [scheduler.select_next_bundle("eval") for _ in range(3)] == ["L", "circle", "hard"]
    with pytest.raises(ValueError):
        scheduler.set_training_bundles(["unknown"])


def test_curriculum_hook_saves_passed_policy_and_stops_once(tmp_path):
    manager = MapCurriculum(["L", "circle"], "L")
    console = SimpleNamespace(print_info=lambda text: None)
    wrapped = CurriculumEvaluator(SimpleNamespace(evaluate=lambda: report(L=9, circle=9)), manager,
        SimpleNamespace(set_training_bundles=lambda maps: None), console)
    hook = MapCurriculumCheckpointHook(agent=None, output_dir=str(tmp_path), evaluator=wrapped,
        evaluate_every=1, evaluate_every_steps=10, selection_strategy="map_curriculum", console=console)
    saved = []
    hook._save = lambda path, **kw: saved.append(path.name)
    hook.on_update({"train/environment_steps": 10})
    assert not hook.should_stop
    hook.on_update({"train/environment_steps": 20})
    assert hook.should_stop
    hook.on_update({"train/environment_steps": 30})
    assert saved.count("curriculum_passed.pt") == 1
    assert len((tmp_path / "evaluation_history.jsonl").read_text().splitlines()) == 2


def test_scenario_and_independent_endurance_protocol():
    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_curriculum.yaml")
    selection = resolve_evaluation_protocol(scenario, "selection")
    final = resolve_evaluation_protocol(scenario, "final")
    assert scenario["evaluation"]["target_laps"] == 5
    assert selection["episodes"] == 90
    assert final["target_laps"] == 20
    assert final["max_steps"] == 64000
    assert selection["seed"] + selection["episodes"] <= final["seed"]
    assert final["episodes"] == 180
    for section, key, value in [("evaluation", "episodes", 8),
        ("evaluation", "terminate_on_track_limit", False), ("environment", "max_steps", 0),
        ("map_curriculum", "success_threshold", float("nan"))]:
        invalid = deepcopy(scenario)
        invalid[section][key] = value
        with pytest.raises(ScenarioError):
            validate_scenario(invalid)


@pytest.mark.parametrize("num_envs,advance", [(1, False), (2, False), (2, True)])
def test_cli_stops_at_bundle_gate_and_runs_frozen_endurance_validation(tmp_path, monkeypatch, num_envs, advance):
    import json
    import sys
    import run
    from training.ppo_evaluator import DeterministicPPOEvaluator

    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_curriculum.yaml")
    scenario["experiment"].update(total_steps=100, torch_threads=1, num_envs=num_envs)
    scenario["environment"]["max_steps"] = 2
    scenario["evaluation"].update(every_steps=2 * num_envs, episodes=9, max_steps=2)
    scenario["evaluation"]["final_test"].update(episodes=9, max_steps=2)
    scenario["map_curriculum"]["episodes_per_map"] = 1
    scenario["agents"]["car_0"]["params"].update(
        device="cpu", n_steps=2 * num_envs, batch_size=2, n_epochs=1, checkpoint_every_steps=2 * num_envs)
    calls = []
    original = DeterministicPPOEvaluator.evaluate
    from training.hooks import TrainingHook
    worker_maps = {}

    class CaptureMaps(TrainingHook):
        def on_episode_end(self, episode, reward, info, metrics):
            worker_maps.setdefault(info.get("worker_id", 0), []).append(info.get("map_bundle"))

    original_run = run._run_on_policy

    def run_with_capture(*args, **kwargs):
        args[10].append(CaptureMaps())
        return original_run(*args, **kwargs)

    monkeypatch.setattr(run, "_run_on_policy", run_with_capture)

    def evaluate(self, *args, **kwargs):
        calls.append((self.base_seed, self.env.target_laps, self.env.max_steps))
        if self.base_seed == scenario["evaluation"]["seed"]:
            # Force gate success to exercise the real trainer's stop/final path.
            return {"per_map": {name: {"episodes": 1, "strict_clean_finish_count": int(
                        not advance or len(calls) > 2 or name == "L_map")}
                    for name in scenario["environment"]["map_bundles_eval"]},
                    "mean_clean_finish_time_s": 50}
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DeterministicPPOEvaluator, "evaluate", evaluate)
    monkeypatch.setattr(run, "load_and_expand_scenario", lambda *args, **kwargs: deepcopy(scenario))
    monkeypatch.setattr(sys, "argv", ["run.py", "--scenario", "scenarios/ppo_lap_completion_curriculum.yaml",
        "--no-wandb", "--quiet", "--output-dir", str(tmp_path)])
    run.main()
    evaluations = 4 if advance else 2
    assert calls == [(10042, 5, 2)] * evaluations + [(20042, 20, 2)]
    if advance:
        assert set(worker_maps) == {0, 1}
        assert all("circle_map" in maps and "L_map" in maps for maps in worker_maps.values())
    state = json.loads((tmp_path / "curriculum_state.json").read_text())
    assert state["complete"] and state["environment_steps"] == evaluations * 2 * num_envs
    assert (tmp_path / "curriculum_passed.pt").exists()
    final = json.loads((tmp_path / "curriculum_final_evaluation.json").read_text())
    assert final["evaluation_protocol"]["name"] == "final"
    assert final["evaluation_protocol"]["target_laps"] == 20
    assert set(final["per_map"]) == set(scenario["environment"]["map_bundles_eval"])
    assert final["all_maps_passed"] is False
    assert not (tmp_path / f"checkpoint_step{(evaluations + 1) * 2 * num_envs:09d}.pt").exists()


def test_remote_policy_staggers_pool_without_resetting_unchanged_rotation():
    from training.on_policy_trainer import _RemotePolicy

    scheduler = MapScheduler(dict(map_bundles=["L", "circle", "hard"],
        map_bundles_train=["L"], map_cycle="per_episode", map_pick="round_robin"),
        rng=np.random.default_rng(0))
    control = {"stop": False, "training_bundles": ["hard", "L", "hard", "circle"]}
    connection = SimpleNamespace(send=lambda message: None,
        recv=lambda: {"metrics": {"train/updates": 1}, "collector_control": control})
    policy = _RemotePolicy(connection, 1, 1, 2, .99, .95, map_scheduler=scheduler, worker_id=1)
    policy.buffer.add(np.zeros(1), np.zeros(2), 0., 0., 0., True, False)
    assert policy.update(0.) == {"train/updates": 1}
    assert scheduler.select_next_bundle("train") == "L"
    policy.update(0.)
    assert scheduler.select_next_bundle("train") == "hard"
    policy.update(0.)
    assert scheduler.select_next_bundle("train") == "circle"
    control["stop"] = True
    policy.update(0.)
    assert policy.should_stop


def test_parallel_curriculum_default_rollout_and_divisibility():
    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_curriculum.yaml")
    assert scenario["experiment"]["num_envs"] == 400
    assert scenario["agents"]["car_0"]["params"]["n_steps"] == 400 * 1024
    assert scenario["evaluation"]["every_steps"] == 400 * 1024
    scenario["experiment"]["num_envs"] = 3
    with pytest.raises(ScenarioError, match="multiple of num_envs"):
        validate_scenario(scenario)
