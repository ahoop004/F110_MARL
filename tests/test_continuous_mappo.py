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
    scratch = load_and_expand_scenario("scenarios/mappo_2v2_base_scratch.yaml")
    pretrained = load_and_expand_scenario("scenarios/mappo_2v2_base_pretrained.yaml")
    for s in (scratch, pretrained):
        assert s["experiment"]["total_steps"] == 120000000
        assert s["experiment"]["num_envs"] == 400
        assert s["environment"]["max_steps"] == 0
        assert s["environment"]["episode_termination"] == {
            "mode": "all_trainable", "lap_completion": False}
        assert s["evaluation"]["target_laps"] == 20
        assert s["evaluation"]["max_steps"] == 120000
        assert s["agents"]["car_0"]["reward"].endswith("race_team_continuous_progress.yaml")
        s["experiment"].pop("name")
        s["training_defaults"].pop("pretrained_actor_checkpoint")
    assert scratch == pretrained
    penalties = load_and_expand_scenario("scenarios/mappo_2v2_penalties_scratch.yaml")
    assert penalties["experiment"].get("total_steps") is None
    assert penalties["environment"]["target_laps"] == 3
    assert penalties["environment"]["max_steps"] == 16000


def test_evaluation_restores_lap_termination_for_continuous_team_training():
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_base_scratch.yaml")
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
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_base_scratch.yaml")
    args = SimpleNamespace(seed=None, episodes=None, total_steps=17, wandb=False,
        no_wandb=False, render=False, no_render=False, num_envs=3, num_workers=2)
    apply_cli_overrides(scenario, args)
    validate_scenario(scenario)
    assert scenario["experiment"]["total_steps"] == 17
    scenario["experiment"]["total_steps"] = 2
    with pytest.raises(ScenarioError, match="at least num_envs"):
        validate_scenario(scenario)
