from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.env_builder import build_env_kwargs, create_environment
from core.map_selection import apply_map_split
from core.scenario import ScenarioError, load_and_expand_scenario, validate_scenario
from env.collision_state import RaceLifecycle
from env.state_buffer import TerminalAgentConfig, TerminalVehicleController
from env.state_views import build_global_state
from env.types import AgentRaceStatus, TransitionRecord
from metrics.racing_eval import (
    aggregate_eval_episodes,
    create_episode_facts,
    finalize_episode_facts,
    update_agent_step_facts,
)
from src.replay.dataset_writer import DatasetWriter, detect_dataset_schema
from wrappers.rewards.collision import CollisionRewardComponent
from wrappers.rewards.lap_completion import LapCompletionComponent, PerLapBonusComponent
from wrappers.rewards.target_finish import TargetFinishComponent
from wrappers.rewards.timeout import TimeoutPenaltyComponent
from wrappers.rewards.track_completion import FinishAheadBonusComponent


def test_terminal_vehicle_controller_is_deterministic() -> None:
    simulator = SimpleNamespace(
        agents=[SimpleNamespace(state=np.array([0, 0, 0, 2, 0, 1, 0.2], dtype=float))]
    )
    controller = TerminalVehicleController(
        ["car_0"], TerminalAgentConfig(finish_clearance_steps=4)
    )
    controller.capture(
        "car_0",
        status=AgentRaceStatus.FINISHED,
        terminal_step=10,
        action=np.array([0.2, 2.0], dtype=np.float32),
        vehicle_state=simulator.agents[0].state,
    )
    assert controller.states["car_0"].last_vehicle_state[3] == 2.0
    actions = np.zeros((1, 2), dtype=np.float32)
    controller.apply(actions, agent_index={"car_0": 0}, simulator=simulator, step=12)
    assert actions[0] == pytest.approx([0.1, 1.0])

    controller.apply(actions, agent_index={"car_0": 0}, simulator=simulator, step=14)
    assert actions[0] == pytest.approx([0.0, 0.0])
    assert simulator.agents[0].state[[3, 5, 6]].tolist() == [0.0, 0.0, 0.0]


def test_cause_based_reward_attribution_is_agent_specific() -> None:
    intermediate = {"info": {"lap_crossed": True, "race_completed": False}}
    completed = {
        "terminated": True,
        "info": {
            "lap_crossed": True,
            "race_completed": True,
            "terminal_reason": "race_complete",
            "finish_position": 1,
            "target_finish_position": None,
            "collision": False,
        },
    }
    assert PerLapBonusComponent({"bonus": 5}).compute(intermediate) == {
        "per_lap/bonus": 5.0
    }
    assert LapCompletionComponent({"bonus": 20}).compute(intermediate) == {}
    assert LapCompletionComponent({"bonus": 20}).compute(completed) == {
        "lap_completion/bonus": 20.0
    }
    assert FinishAheadBonusComponent({"bonus": 7}).compute(completed) == {
        "finish_ahead/bonus": 7.0
    }
    assert CollisionRewardComponent({"penalty": -9}).compute(completed) == {}
    assert TimeoutPenaltyComponent({"penalty": -11}).compute(completed) == {}


def test_target_finish_penalty_is_emitted_once() -> None:
    component = TargetFinishComponent({"penalty": -10})
    step = {"info": {"target_race_completed": True, "collision": False}}
    assert component.compute(step) == {"target_finish/penalty": -10.0}
    assert component.compute(step) == {}
    component.reset()
    assert component.compute(step) == {"target_finish/penalty": -10.0}


def test_dataset_v2_round_trip_and_old_schema_detection(tmp_path) -> None:
    output = tmp_path / "v2"
    writer = DatasetWriter(output)
    finished = TransitionRecord(
        obs=np.array([1], dtype=np.float32),
        action_norm=np.array([0, 1], dtype=np.float32),
        action_phys=np.array([0, 2], dtype=np.float32),
        reward=3.0,
        reward_components={},
        next_obs=np.array([2], dtype=np.float32),
        terminated=True,
        truncated=False,
        info={},
        global_state=np.array([4], dtype=np.float32),
        map_id="map",
        spawn_id="spawn",
        episode_id="ep",
        step_idx=5,
        agent_id="car_0",
        lap_crossed=True,
        lap_count=3,
        target_laps=3,
        race_completed=True,
        terminal_reason="race_complete",
        lifecycle_status="finished",
        finish_position=1,
        lifecycle_masks={
            "active_mask": np.array([False, True]),
            "finished_mask": np.array([True, False]),
        },
    )
    writer.add(finished)
    writer.add(
        replace(
            finished,
            agent_id="car_1",
            lap_crossed=False,
            lap_count=2,
            race_completed=False,
            terminal_reason="collision",
            lifecycle_status="crashed",
            finish_position=None,
        )
    )
    writer.add(
        replace(
            finished,
            agent_id="car_2",
            lap_crossed=False,
            lap_count=1,
            race_completed=False,
            terminal_reason="time_limit",
            lifecycle_status="truncated",
            finish_position=None,
        )
    )
    writer.close()

    assert detect_dataset_schema(output) == "2.0"
    chunk = np.load(output / "transitions_000000.npz", allow_pickle=True)
    assert chunk["lap_count"].tolist() == [3, 2, 1]
    assert chunk["terminal_reason"].tolist() == [
        "race_complete",
        "collision",
        "time_limit",
    ]
    assert chunk["lifecycle_masks"].shape == (3, 4, 2)

    old = tmp_path / "old"
    old.mkdir()
    (old / "metadata.json").write_text(json.dumps({"schema_version": "1.0"}))
    assert detect_dataset_schema(old) == "1.0"


def test_2v2_scenario_expands_explicit_race_contract() -> None:
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_vs_hybrid_pp_ftg.yaml")
    env = apply_map_split(scenario["environment"], scenario["experiment"], "train")
    kwargs = build_env_kwargs(env, scenario["agents"], seed=42)

    assert env["target_laps"] == 3
    assert env["episode_termination"]["mode"] == "all_agents"
    assert env["map_bundles_train"]
    assert env["map_bundles_eval"]
    assert kwargs["target_laps"] == 3
    assert kwargs["terminal_agents"]["remain_collidable"] is True
    assert all(
        agent["reward"].endswith("race_team_2v2_completion.yaml")
        for agent in (scenario["agents"]["car_0"], scenario["agents"]["car_1"])
    )


@pytest.mark.parametrize(
    ("scenario_path", "reward_mode", "critic_mode"),
    [
        ("scenarios/complete_4_individual.yaml", "individual", "agent_conditioned"),
        ("scenarios/complete_4_team_shared.yaml", "team_shared", "shared_team"),
        ("scenarios/mappo_2v2_individual.yaml", "individual", "agent_conditioned"),
        ("scenarios/mappo_2v2_team_shared.yaml", "team_shared", "shared_team"),
    ],
)
def test_mappo_comparison_scenarios_have_explicit_contracts(
    scenario_path: str,
    reward_mode: str,
    critic_mode: str,
) -> None:
    scenario = load_and_expand_scenario(scenario_path)

    assert scenario["mappo"] == {
        "reward_mode": reward_mode,
        "critic_mode": critic_mode,
        "team_reward_reduction": "mean",
    }


def test_individual_rewards_reject_shared_team_critic() -> None:
    scenario = load_and_expand_scenario("scenarios/complete_4_individual.yaml")
    scenario["mappo"]["critic_mode"] = "shared_team"

    with pytest.raises(ScenarioError, match="individual rewards require"):
        validate_scenario(scenario)


def test_global_state_exposes_distinct_lifecycle_masks() -> None:
    lifecycle = RaceLifecycle(["car_0", "car_1", "car_2"], 1)
    lifecycle.record_lap_crossing("car_0", step=1)
    lifecycle.record_collision("car_1", step=2)
    state = build_global_state(
        possible_agents=lifecycle.agent_ids,
        active_agents=lifecycle.active_agents,
        central_vector=np.zeros(3, dtype=np.float32),
        lifecycle_records=lifecycle.records,
    )
    assert state.masks["active_mask"].tolist() == [False, False, True]
    assert state.masks["finished_mask"].tolist() == [True, False, False]
    assert state.masks["crashed_mask"].tolist() == [False, True, False]
    assert state.masks["truncated_mask"].tolist() == [False, False, False]


def test_four_car_standings_remain_immutable() -> None:
    episode = create_episode_facts(
        episode=0,
        agent_ids=["car_0", "car_1", "car_2", "car_3"],
        trainable_ids=["car_0", "car_1"],
        opponent_ids=["car_2", "car_3"],
    )
    steps = [
        {
            "car_1": {
                "race_completed": True,
                "terminal_reason": "race_complete",
                "terminal_step": 1,
                "finish_position": 1,
                "lap_count": 3,
            }
        },
        {
            "car_1": {
                "race_completed": True,
                "terminal_reason": "race_complete",
                "collision": True,
                "finish_position": 1,
            },
            "car_2": {"terminal_reason": "collision", "terminal_step": 2},
        },
        {
            "car_0": {
                "race_completed": True,
                "terminal_reason": "race_complete",
                "terminal_step": 3,
                "finish_position": 2,
                "lap_count": 3,
            }
        },
        {"car_3": {"terminal_reason": "time_limit", "time_limit": True}},
    ]
    terminal_maps = [
        {"car_1": True},
        {"car_1": True, "car_2": True},
        {"car_0": True, "car_1": True, "car_2": True},
        {"car_0": True, "car_1": True, "car_2": True},
    ]
    trunc_maps = [{}, {}, {}, {"car_3": True}]
    for idx, info in enumerate(steps, start=1):
        update_agent_step_facts(
            episode,
            step_idx=idx,
            infos=info,
            terminations=terminal_maps[idx - 1],
            truncations=trunc_maps[idx - 1],
        )
    finalize_episode_facts(episode)

    assert episode.agents["car_1"].outcome == "finished"
    assert episode.agents["car_1"].finish_position == 1
    assert episode.agents["car_1"].collision_step is None
    assert episode.agents["car_2"].outcome == "crashed"
    assert episode.agents["car_3"].outcome == "truncated"
    summary = aggregate_eval_episodes([episode])
    assert summary["team_both_finished_rate"] == 1.0
    assert summary["team_mean_finish_position"] == 1.5
    assert summary["team_best_finish_position"] == 1.0


def test_cooperative_team_collision_rate_does_not_require_opponents() -> None:
    episode = create_episode_facts(
        episode=0,
        agent_ids=["car_0", "car_1"],
        trainable_ids=["car_0", "car_1"],
        opponent_ids=[],
    )
    update_agent_step_facts(
        episode,
        step_idx=1,
        infos={
            "car_0": {"terminal_reason": "collision"},
            "car_1": {"terminal_reason": "collision"},
        },
        terminations={"car_0": True, "car_1": True},
    )
    finalize_episode_facts(episode)

    summary = aggregate_eval_episodes([episode])
    assert summary["team_collision_rate"] == 1.0
    assert summary["team_both_finished_rate"] == 0.0


def test_finished_vehicle_remains_physical_and_can_crash_active_vehicle() -> None:
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_vs_hybrid_pp_ftg.yaml")
    env_cfg = apply_map_split(scenario["environment"], scenario["experiment"], "train")
    env_cfg = dict(env_cfg)
    env_cfg["map_cycle"] = ""
    env = create_environment(
        env_cfg,
        {"car_0": scenario["agents"]["car_0"], "car_1": scenario["agents"]["car_1"]},
        seed=42,
    )
    try:
        env.reset(seed=42)
        winner = env.lifecycle.records["car_0"]
        winner.lap_count = winner.target_laps - 1
        env.lifecycle.record_lap_crossing("car_0", step=0)
        env.agents = list(env.lifecycle.active_agents)
        env._terminal_controller.capture(
            "car_0",
            status=winner.status,
            terminal_step=0,
            action=np.zeros(2, dtype=np.float32),
            vehicle_state=env.sim.agents[0].state,
        )

        # The simulator has no collision impulses: overlap produces collision
        # facts but cannot move or rewrite the parked winner's result.
        env.sim.agents[1].state[:] = env.sim.agents[0].state
        _, _, terminations, _, _ = env.step(
            {"car_1": np.zeros(2, dtype=np.float32)}
        )

        assert env.lifecycle.records["car_0"].status == AgentRaceStatus.FINISHED
        assert env.lifecycle.records["car_0"].finish_position == 1
        assert env.lifecycle.records["car_1"].status == AgentRaceStatus.CRASHED
        assert terminations == {"car_0": True, "car_1": True}
        assert env.physical_agents == ("car_0", "car_1")
        assert env.get_global_state().agent_ids == ("car_0", "car_1")
    finally:
        env.close()
