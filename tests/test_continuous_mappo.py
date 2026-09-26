from copy import deepcopy
from pathlib import Path

import pytest

from core.scenario import load_and_expand_scenario, validate_scenario
from core.setup import create_training_setup
from wrappers.rewards.composer import RewardComposer


def test_continuous_reward_matches_ppo_metres_with_exclusive_collision_cost():
    ppo = RewardComposer.from_file("configs/reward/tasks/lap_completion_pretraining.yaml")
    team = RewardComposer.from_file("configs/reward/tasks/race_team_continuous_progress.yaml")
    assert not team.team_contract  # The trainer applies the fixed two-agent mean.
    for progress in (.01, -.02, .1):
        step = {"track_length": 100., "info": {"centerline": {"progress_delta": progress},
                "track_limits": {"exceeded": False}}, "timestep": .05}
        assert team.compute(step)[0] == pytest.approx(progress * 100.)
        assert team.compute(step)[0] == ppo.compute(step)[0]
        step["info"]["terminal_reason"] = "collision"
        assert team.compute(step) == (-1., {"progress_delta/collision": -1.})
    # No extra finish bonus, time penalty, or timeout penalty is added.
    for reason in ("race_complete", "time_limit"):
        step["info"].update(terminal_reason=reason, lap_crossed=True, race_completed=True)
        assert team.compute(step)[0] == pytest.approx(10.)


def test_base_pair_is_matched_and_penalty_task_keeps_finite_races():
    scratch = load_and_expand_scenario("scenarios/mappo_2v2_continuous.yaml")
    pretrained = load_and_expand_scenario("scenarios/mappo_2v2_continuous.yaml", overrides=['training_defaults.pretrained_actor_checkpoint="../outputs/L_map_pretrain/L_map_best_model.pt"',
         'experiment.name="mappo_2v2_base_pretrained"'])
    for s in (scratch, pretrained):
        assert s["experiment"]["total_steps"] == 120000000
        assert s["experiment"]["num_envs"] == 400
        assert s["environment"]["max_steps"] == 0
        assert s["environment"]["episode_termination"] == {
            "mode": "all_trainable", "lap_completion": False}
        assert s["evaluation"]["target_laps"] == 20
        assert s["evaluation"]["max_steps"] == 120000
        assert s["agents"]["car_0"]["reward"]["task"]["name"] == "race_team_continuous_progress"
        s["experiment"].pop("name")
        s["training_defaults"].pop("pretrained_actor_checkpoint")
    assert scratch == pretrained
    penalties = load_and_expand_scenario("scenarios/mappo_2v2_race.yaml", overrides=['training_defaults.update_version="parallel-mappo-grouped256-batch2048-v1"',
         'training_defaults.batch_size=2048',
         'training_defaults.rollout_steps_per_env=256',
         'training_defaults.checkpoint_every_steps=1024000',
         'wandb.group="mappo-2v2-penalties-current-physics"',
         'wandb.tags=["mappo","2v2","terminal-incidents-v1","current-pretrain-physics","racing-mpc-opponents"]',
         'wandb.notes="Matched physics, observations, racing MPC opponents and rewards. Both-finished '
         'rate, then rank plus recorded penalties select checkpoints; clean finish time breaks successful '
         'ties."',
         'experiment.name="mappo_2v2_penalties_scratch"',
         'experiment.num_envs=400',
         'experiment.num_workers=100',
         'experiment.worker_startup_batch_size=8',
         'experiment.worker_startup_timeout_s=600',
         'experiment.worker_response_timeout_s=120',
         'experiment.terminal_recent_episodes=100',
         'experiment.terminal_every_updates=10',
         'experiment.terminal_diagnostic_every_updates=100',
         'experiment.terminal_episode_detail=false',
         'evaluation.selection_strategy="team_combined_penalties"',
         'evaluation.every_steps=1024000',
         'agents.car_0.reward.task.name="race_team_2v2_penalties"',
         'agents.car_0.reward.task.description="Shared completion/placement reward with recorded terminal '
         'race penalties."',
         'agents.car_0.reward.reward.collision.enabled=false',
         'agents.car_0.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}',
         'agents.car_1.reward.task.name="race_team_2v2_penalties"',
         'agents.car_1.reward.task.description="Shared completion/placement reward with recorded terminal '
         'race penalties."',
         'agents.car_1.reward.reward.collision.enabled=false',
         'agents.car_1.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}'])
    assert penalties["experiment"].get("total_steps") is None
    assert penalties["environment"]["target_laps"] == 3
    assert penalties["environment"]["max_steps"] == 16000


def test_evaluation_restores_lap_termination_for_continuous_team_training():
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_continuous.yaml")
    # Make the override observable independently of the training metadata.
    scenario["environment"]["target_laps"] = 3
    before = deepcopy(scenario)
    for phase in ("train", "eval"):
        env, _, _ = create_training_setup(scenario, mode=phase, scenario_dir=Path("scenarios").resolve())
        try:
            assert env.target_laps == (3 if phase == "train" else 20)
            assert env.max_steps == (0 if phase == "train" else 120000)
            assert env.lifecycle.finish_on_laps == (phase == "eval")
            env.reset(seed=42)
            # Exercise the actual lifecycle: training crosses its nominal target
            # without finishing; evaluation finishes precisely at lap twenty.
            for lap in range(env.target_laps):
                env.lifecycle.record_lap_crossing("car_0", step=lap + 1)
            assert env.lifecycle.records["car_0"].is_active == (phase == "train")
        finally:
            env.close()
    assert scenario == before


def test_cli_step_budget_override_and_validation():
    from types import SimpleNamespace
    from run import apply_cli_overrides
    from core.scenario import ScenarioError
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_continuous.yaml")
    args = SimpleNamespace(seed=None, episodes=None, total_steps=17, wandb=False,
        no_wandb=False, render=False, no_render=False, num_envs=3, num_workers=2)
    apply_cli_overrides(scenario, args)
    validate_scenario(scenario)
    assert scenario["experiment"]["total_steps"] == 17
    scenario["experiment"]["total_steps"] = 2
    with pytest.raises(ScenarioError, match="at least num_envs"):
        validate_scenario(scenario)


@pytest.mark.parametrize("num_envs", [1, 2])
def test_ppo_step_budget_accepts_null_episode_budget(tmp_path, monkeypatch, num_envs):
    import sys
    import run
    from utils.torch_io import safe_load

    path = "scenarios/ppo_lap_completion_pretrain.yaml"
    scenario = load_and_expand_scenario(path)
    scenario["experiment"].update(total_steps=5, episodes=None, num_envs=num_envs,
                                  num_workers=1, torch_threads=1)
    scenario["environment"].update(max_steps=3)
    scenario["evaluation"]["enabled"] = False
    scenario["agents"]["car_0"]["params"].update(
        n_steps=4, n_epochs=1, batch_size=4, device="cpu")
    monkeypatch.setattr(run, "load_and_expand_scenario", lambda *_a, **_kw: deepcopy(scenario))
    monkeypatch.setattr(sys, "argv", ["run.py", "--scenario", path, "--no-wandb", "--quiet",
                                     "--output-dir", str(tmp_path)])
    run.main()
    checkpoint = safe_load(str(tmp_path / "final_model.pt"), map_location="cpu")
    assert checkpoint["environment_steps"] == 5
