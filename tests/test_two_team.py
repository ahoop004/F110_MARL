from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.mappo import MAPPOAgent
from core.scenario import load_and_expand_scenario, validate_scenario, ScenarioError
from training.two_team import TeamEvents, TwoTeamTrainer, evaluate_pair, resolve_teams
from wrappers.rewards.composer import RewardComposer


TEAMS = {"team_a": ["car_0", "car_1"], "team_b": ["car_2", "car_3"]}


def serial_selfplay_scenario():
    """Exercise serial semantics independently of the user's environment count."""
    scenario = load_and_expand_scenario("scenarios/mappo_2v2_selfplay.yaml", validate=False)
    scenario["experiment"]["num_envs"] = 1
    validate_scenario(scenario)
    return scenario


def test_selfplay_matches_continuous_step_budget_and_cli_can_override(monkeypatch):
    import run

    scenario = serial_selfplay_scenario()
    continuous = load_and_expand_scenario("scenarios/mappo_2v2_continuous.yaml")
    assert scenario["experiment"]["total_steps"] == continuous["experiment"]["total_steps"] == 120_000_000
    assert scenario["evaluation"]["every_steps"] == continuous["evaluation"]["every_steps"] == 1_024_000
    assert scenario["training_defaults"]["checkpoint_every_steps"] == 1_024_000
    monkeypatch.setattr("sys.argv", ["run.py", "--scenario", "scenarios/mappo_2v2_selfplay.yaml", "--episodes", "2"])
    overridden = run.apply_cli_overrides(deepcopy(scenario), run.parse_args())
    assert "total_steps" not in overridden["experiment"]
    assert overridden["experiment"]["episodes"] == 2
    monkeypatch.setattr("sys.argv", ["run.py", "--scenario", "scenarios/mappo_2v2_selfplay.yaml", "--total-steps", "7"])
    assert run.apply_cli_overrides(deepcopy(scenario), run.parse_args())["experiment"]["total_steps"] == 7


@pytest.mark.parametrize("budget", [6, 7])
def test_step_budget_runs_across_resets_and_schedules_at_updates(tmp_path, monkeypatch, budget):
    import run

    scenario = serial_selfplay_scenario()
    scenario["experiment"].update(total_steps=budget)
    scenario["environment"]["max_steps"] = 3
    scenario["training_defaults"].update(device="cpu", pi_hidden_dims=[4], vf_hidden_dims=[4],
        n_steps=2, n_epochs=1, batch_size=16, checkpoint_every_steps=2)
    scenario["evaluation"].update(max_steps=3, episodes=1, every_steps=2)
    scenario["wandb"]["enabled"] = False
    monkeypatch.setattr("sys.argv", ["run.py", "--scenario", "scenarios/mappo_2v2_selfplay.yaml",
                                    "--output-dir", str(tmp_path), "--no-wandb"])
    args = run.parse_args()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run._run_two_team(scenario, args, run.ConsoleLogger(verbose=False), Path("scenarios").resolve())
    finally:
        torch.set_num_threads(threads)

    def read_rows(name):
        return [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]

    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["environment_steps"] == summary["total_steps"] == budget
    assert summary["agent_steps"] == 4 * budget
    episodes = read_rows("team_metrics.jsonl")
    assert [row["selfplay/steps"] for row in episodes] == ([3, 3] if budget == 6 else [3, 3, 1])
    assert episodes[-1]["selfplay/budget_cut"] == (budget == 7)
    assert episodes[-1]["selfplay/completed"] == (budget == 6)
    # Threshold 4 lands inside a rollout, so the checkpoint/evaluation waits
    # until update 5. Step 2 is mid-race. Final step 6 must not evaluate twice.
    assert [row["selfplay_eval/environment_steps"] for row in read_rows("evaluation_metrics.jsonl")] == (
        [2, 5, 6] if budget == 6 else [2, 5, 6, 7])
    for step in (2, 5, 6):
        checkpoint = json.loads((tmp_path / f"pair_step{step:012d}" / "pair.json").read_text())
        assert checkpoint["environment_steps"] == step
    final = json.loads((tmp_path / "final_pair" / "pair.json").read_text())
    assert final["environment_steps"] == budget
    updates = read_rows("updates.jsonl")
    assert updates[-1]["train/budget_fraction"] == 1.0
    assert updates[-1]["train/total_steps"] == budget


def test_scenario_requires_separate_two_car_teams_and_finite_race():
    scenario = serial_selfplay_scenario()
    assert resolve_teams(scenario) == TEAMS
    for change in ({"max_steps": 0}, {"episode_termination": {"mode": "any_agent"}},
                   {"agent_teams": {aid: "one_team" for aid in scenario["agents"]}}):
        invalid = deepcopy(scenario)
        invalid["environment"].update(change)
        with pytest.raises(ScenarioError):
            validate_scenario(invalid)


def test_events_count_both_opponents_once_and_never_reward_timeout():
    events = TeamEvents(TEAMS)
    infos = {aid: {"status": "active"} for ids in TEAMS.values() for aid in ids}
    for aid in TEAMS["team_b"]:
        infos[aid] = {"status": "crashed", "terminal_reason": "collision"}
    assert events.step(infos)["team_a"]["opponent_crash"] == 2
    assert events.step(infos)["team_a"]["opponent_crash"] == 0
    for aid in TEAMS["team_a"]:
        infos[aid] = {"status": "finished"}
    assert events.step(infos)["team_a"]["both_finished"] == 2
    assert events.step(infos)["team_a"]["both_finished"] == 0
    events.reset()
    assert events.step({aid: {"terminal_reason": "time_limit"} for aid in infos}) == {
        team: {"opponent_crash": 0, "both_finished": 0} for team in TEAMS}


class ScriptedEnv:
    max_steps = 3
    target_laps = 3
    timestep = 0.05
    centerline_track_length = 100

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.agents = [aid for ids in TEAMS.values() for aid in ids]
        self.infos = {aid: {"status": "active", "lap_count": 0} for aid in self.agents}
        self.episode_done = False
        return {aid: {} for aid in self.infos}, deepcopy(self.infos)

    def get_global_state(self):
        return SimpleNamespace(vector=np.array([self.steps], dtype=np.float32))

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.steps += 1
        changes = {1: {"car_0": "collision", "car_2": "collision"},
                   2: {"car_1": "race_complete"}, 3: {"car_3": "collision"}}
        for aid, reason in changes[self.steps].items():
            self.agents.remove(aid)
            self.infos[aid] = dict(status="finished" if reason == "race_complete" else "crashed",
                terminal_reason=reason, race_completed=reason == "race_complete",
                lap_crossed=reason == "race_complete",
                lap_count=3 if reason == "race_complete" else 0)
        self.episode_done = not self.agents
        return ({aid: {} for aid in self.infos}, {},
                {aid: aid not in self.agents for aid in self.infos},
                {aid: False for aid in self.infos}, deepcopy(self.infos))


class Observation:
    def reset(self): pass
    def update_prev_action(self, action): pass
    def wrap(self, obs, info): return np.zeros(1, dtype=np.float32)


class Action:
    def reset(self): pass
    def process(self, action): return action


def make_trainer():
    agents = {team: MAPPOAgent(3, 2, -np.ones(2), np.ones(2), ids,
        dict(hidden_dims=[4], device="cpu", n_steps=8, n_epochs=1, batch_size=8,
             gamma=1., gae_lambda=1., team_return_mode="joint", reward_mode="team_shared",
             critic_mode="shared_team", team_reward_reduction="sum")) for team, ids in TEAMS.items()}
    for agent in agents.values():
        with torch.no_grad():
            for parameter in agent.critic.parameters():
                parameter.zero_()
    ids = [aid for members in TEAMS.values() for aid in members]
    return TwoTeamTrainer(env=ScriptedEnv(), teams=TEAMS, agents=agents,
        observations={aid: Observation() for aid in ids},
        actions={aid: Action() for aid in ids},
        rewards={aid: RewardComposer.from_file("configs/reward/tasks/race_two_trainable_teams.yaml") for aid in ids})


def test_independent_rewards_survivor_finish_and_later_opponent_crash_credit(monkeypatch):
    trainer = make_trainer()
    captured = {}
    for team, agent in trainer.agents.items():
        def capture(state, team=team, agent=agent):
            captured[team] = (agent.compute_team_gae(999)[1].tolist(),
                              deepcopy(agent._team_step_indices))
            return {}
        monkeypatch.setattr(agent, "update", capture)
    metrics = trainer.episode()
    assert metrics["team_a/reward"] == 3  # -3 own +2 opponent +4 finish
    assert metrics["team_b/reward"] == -5  # -6 own +1 opponent
    assert metrics["team_a/finish_count"] == 1
    assert metrics["team_b/eliminated"] == 1
    assert metrics["steps"] == 3  # continues after all A cars become inactive
    assert captured["team_a"] == ([3., 5., 1.], {"car_0": [0], "car_1": [0, 1]})
    assert captured["team_b"] == ([-5., -3., -3.], {"car_2": [0], "car_3": [0, 1, 2]})
    assert trainer.agent_steps == 7


def test_budget_cut_bootstraps_but_does_not_claim_completed_race(monkeypatch):
    trainer = make_trainer()
    observed = {}
    for team, agent in trainer.agents.items():
        def capture(state, team=team, agent=agent):
            observed[team] = agent.compute_team_gae(10)[1].tolist()
            return {}
        monkeypatch.setattr(agent, "update", capture)
    metrics = trainer.episode(step_budget=1)
    assert metrics["budget_cut"] == 1 and metrics["completed"] == 0
    assert observed == {"team_a": [8.], "team_b": [8.]}


def test_race_deadline_is_terminal_for_team_value_even_with_env_truncations(monkeypatch):
    trainer = make_trainer()
    def deadline(actions):
        trainer.env.steps += 1
        trainer.env.agents = []
        trainer.env.episode_done = True
        infos = {aid: {"status": "truncated", "terminal_reason": "time_limit"}
                 for ids in TEAMS.values() for aid in ids}
        return {}, {}, {aid: False for aid in infos}, {aid: True for aid in infos}, infos
    monkeypatch.setattr(trainer.env, "step", deadline)
    observed = {}
    for team, agent in trainer.agents.items():
        def capture(state, team=team, agent=agent):
            observed[team] = agent.compute_team_gae(100)[1].tolist()
            return {}
        monkeypatch.setattr(agent, "update", capture)
    metrics = trainer.episode()
    assert metrics["completed"] == 1 and metrics["budget_cut"] == 0
    assert observed == {"team_a": [0.], "team_b": [0.]}


def test_evaluation_preserves_policies_buffers_and_random_streams():
    trainer = make_trainer()
    before = {team: deepcopy(agent.actor.state_dict()) for team, agent in trainer.agents.items()}
    rng = torch.random.get_rng_state().clone()
    summary, rows = evaluate_pair(trainer, {"episodes": 2, "seed": 10})
    assert summary["team_a/reward"] == 3 and len(rows) == 2
    assert trainer.environment_steps == 0 and trainer.updates == 0
    assert torch.equal(rng, torch.random.get_rng_state())
    for team, agent in trainer.agents.items():
        assert not agent._team_rollout
        assert not any(b.size() for b in agent.buffers.values())
        assert all(torch.equal(value, before[team][key]) for key, value in agent.actor.state_dict().items())


def test_real_optimizers_update_both_teams_without_shared_parameters():
    trainer = make_trainer()
    a, b = trainer.agents.values()
    assert not {p.data_ptr() for p in a.actor.parameters()} & {p.data_ptr() for p in b.actor.parameters()}
    before = {team: deepcopy(agent.actor.state_dict()) for team, agent in trainer.agents.items()}
    trainer.episode()
    for team, agent in trainer.agents.items():
        assert any(not torch.equal(value, before[team][key]) for key, value in agent.actor.state_dict().items())
        assert agent.optimizer.state


def test_inactive_team_fragments_train_only_critics():
    trainer = make_trainer()
    before = {team: deepcopy(agent.actor.state_dict()) for team, agent in trainer.agents.items()}
    metrics = []
    trainer.on_update = metrics.append
    trainer._global_rollout = [np.array([0., 1.]), np.array([1., .5])]
    for agent in trainer.agents.values():
        agent.store_team_step([], reward=0., value=0., terminal=False)
        agent.store_team_step([], reward=1., value=0., terminal=True)
    trainer.update(np.array([2., 0.]))
    for team, agent in trainer.agents.items():
        assert all(torch.equal(value, before[team][key]) for key, value in agent.actor.state_dict().items())
        assert any(torch.count_nonzero(p) for p in agent.critic.parameters())
        assert metrics[0][f"train/{team}/inactive_value_loss"] > 0


def test_cleared_car_no_longer_collides_or_appears_in_neighbors_and_reset_restores():
    from core.setup import create_training_setup
    from env.types import AgentRaceStatus

    scenario = serial_selfplay_scenario()
    for key in ("map_bundles", "map_bundles_train", "map_bundles_eval"):
        scenario["environment"][key] = ["circle_map"]
    env, _, _ = create_training_setup(scenario, scenario_dir=Path("scenarios"))
    try:
        env.reset(seed=42)
        # Overlapping hulls must collide while visible, but not once cleared.
        poses = env.sim.agent_poses.copy()
        poses[1] = poses[0]
        env.sim.reset(poses)
        assert env.sim.step(np.zeros((4, 2)))["collisions"][:2].all()
        env.sim.reset(poses)
        controller = env._terminal_controller
        controller.capture("car_1", status=AgentRaceStatus.CRASHED, terminal_step=0,
                           action=np.zeros(2), vehicle_state=env.sim.agents[1].physics_state)
        controller.apply(np.zeros((4, 2)), agent_index={aid: i for i, aid in enumerate(env.possible_agents)},
                         simulator=env.sim, step=39)
        assert env.sim.collidable_mask[1]
        controller.apply(np.zeros((4, 2)), agent_index={aid: i for i, aid in enumerate(env.possible_agents)},
                         simulator=env.sim, step=40)
        assert not env.sim.collidable_mask[1]
        collision = env.sim.step(np.zeros((4, 2)))["collisions"]
        assert not collision[0] and not collision[1]
        infos = {}
        env._inject_frenet_neighbors(infos)
        assert all(neighbor["agent_id"] != "car_1" for neighbor in infos["car_0"]["frenet_neighbors"])
        env.reset(seed=42)
        assert env.sim.collidable_mask.all()
    finally:
        env.close()
