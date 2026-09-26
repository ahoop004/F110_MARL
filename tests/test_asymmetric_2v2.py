"""Role rewards, stable identities, and an executable asymmetric 2v2 race."""
from pathlib import Path

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario
from env.centerline_state import build_relative_frenet_facts
from wrappers.observations.neighbors import FrenetNeighborsComponent
from wrappers.rewards.events import OpponentCrashBonusComponent


IDS = ["car_0", "car_1", "car_2", "car_3"]
SCENARIO = "scenarios/mappo_2v2_asymmetric.yaml"


def test_vehicle_and_ego_identity_survive_reordering_and_missing_neighbors():
    component = FrenetNeighborsComponent(max_neighbors=3, include_team=True, agent_ids=IDS)
    teams = dict(zip(IDS, ["learners", "learners", "opponents", "opponents"]))
    states = {aid: {"s": i * 2.} for i, aid in enumerate(IDS)}
    for teammate_s in [2., 5.]:
        states["car_1"]["s"] = teammate_s
        facts = build_relative_frenet_facts(states, track_length=100, closed=True, agent_teams=teams)
        for ego_id in IDS[:2]:
            out = component.compute({}, {"agent_id": ego_id, "frenet_neighbors": facts[ego_id]})
            assert out.shape == (34,)
            np.testing.assert_array_equal(out[-4:], np.eye(4)[IDS.index(ego_id)])
            for slot, neighbor in zip(out[:30].reshape(3, 10), facts[ego_id]):
                np.testing.assert_array_equal(slot[6:], np.eye(4)[IDS.index(neighbor["agent_id"])])
                assert slot[5] == (teams[ego_id] == teams[neighbor["agent_id"]])
    remaining = [facts["car_0"][0]]
    out = component.compute({}, {"agent_id": "car_0", "frenet_neighbors": remaining})
    assert np.count_nonzero(out[10:30]) == 0
    np.testing.assert_array_equal(out[-4:], [1, 0, 0, 0])
    out = component.compute({}, {"agent_id": "car_1"})
    assert np.count_nonzero(out[:30]) == 0
    np.testing.assert_array_equal(out[-4:], [0, 1, 0, 0])


@pytest.mark.parametrize("ids", [[], ["car_0", "car_0"], "car_0", [0]])
def test_invalid_identity_schema_is_rejected(ids):
    with pytest.raises(ValueError, match="agent_ids"):
        FrenetNeighborsComponent(max_neighbors=3, agent_ids=ids)


def test_missing_or_unknown_identity_is_rejected():
    component = FrenetNeighborsComponent(max_neighbors=3, agent_ids=IDS)
    with pytest.raises(ValueError, match="ego agent_id"):
        component.compute({}, {})
    for bad_id in ["unknown", "car_0", None]:
        with pytest.raises(ValueError, match="neighbor agent_id"):
            component.compute({}, {"agent_id": "car_0", "frenet_neighbors": [{"agent_id": bad_id}]})


def test_crash_bonus_counts_each_opponent_once_and_resets():
    reward = OpponentCrashBonusComponent({"bonus": 1.0})
    context = {"opponent_agent_ids": IDS[2:], "all_infos": {
        "car_0": {"terminal_reason": "collision"},
        "car_1": {"terminal_reason": "collision"},
        "car_2": {"terminal_reason": "finished"},
        "car_3": {"terminal_reason": "time_limit"},
    }}
    assert reward.compute(context) == {}
    context["all_infos"]["car_2"]["terminal_reason"] = "collision"
    assert reward.compute(context) == {"opponent_crash/bonus": 1.}
    assert reward.compute(context) == {}
    context["all_infos"]["car_3"]["terminal_reason"] = "collision"
    assert reward.compute(context) == {"opponent_crash/bonus": 1.}
    assert reward.compute(context) == {}
    reward.reset()
    # Both crash on the same step, including an ego collision.
    context["info"] = {"terminal_reason": "collision"}
    assert reward.compute(context) == {"opponent_crash/bonus": 2.}
    assert reward.compute(context) == {}


@pytest.mark.parametrize("bonus", [-1, float("nan"), float("inf")])
def test_invalid_crash_bonus_is_rejected(bonus):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        OpponentCrashBonusComponent({"bonus": bonus})


def test_progress_car_keeps_pretraining_reward_and_support_car_has_distinct_objective():
    from run import build_reward_composers
    from wrappers.rewards.composer import RewardComposer
    scenario = load_and_expand_scenario(SCENARIO)
    assert scenario["mappo"]["actor_mode"] == "independent"
    rewards = build_reward_composers(scenario["agents"], IDS[:2], Path("scenarios"))
    baseline = RewardComposer.from_file("configs/reward/tasks/lap_completion_pretraining.yaml")
    context = {"track_length": 100., "opponent_agent_ids": IDS[2:],
               "info": {"centerline": {"progress_delta": .001}, "track_limits": {"exceeded": False}},
               "all_infos": {"car_0": {"centerline": {"progress_delta": .002}}}}
    assert rewards["car_0"].compute(context)[0] == baseline.compute(context)[0]
    value, parts = rewards["car_1"].compute(context)
    assert value == pytest.approx(.02 + .2)
    assert "opponent_crash/bonus" not in parts
    context["info"]["terminal_reason"] = "collision"
    _, parts = rewards["car_1"].compute(context)
    assert parts['collision/penalty'] == -5
    assert 'team_support/progress' not in parts


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("arm", ["full", "lora"])
def test_scenario_trains_and_writes_192_input_checkpoint(tmp_path, monkeypatch, num_envs, arm):
    import sys
    import run
    from utils.torch_io import safe_load

    from copy import deepcopy
    from agents.ppo import PPOAgent
    from env.spaces_builder import build_action_spaces
    path = Path(SCENARIO if arm == "full" else "scenarios/mappo_2v2_asymmetric_lora.yaml").resolve()
    scenario = load_and_expand_scenario(str(path))
    # Exercise a partial rollout budget with matched pretrained initialization.
    scenario["experiment"].update(total_steps=9, num_envs=num_envs, num_workers=1, torch_threads=1)
    scenario["environment"].update(info_level="minimal")
    scenario["training_defaults"].update(n_steps=4, rollout_steps_per_env=4, n_epochs=1,
                                          batch_size=4, device="cpu", checkpoint_every_steps=8)
    scenario["evaluation"].update(every_steps=8, episodes=1, max_steps=4)
    config = scenario['agents']['car_0']
    params = run.resolve_training_params(config, scenario)
    source_config = {**config, 'observation': '../configs/observations/rl_racer_simulated_wheel.yaml'}
    obs = run.build_obs_composer(source_config, scenario['environment'], path.parent)
    params['_observation_contract'] = obs.contract
    space, _ = build_action_spaces(['car_0'], scenario['environment']['vehicle_params'])
    source = PPOAgent(obs.obs_dim, space.low, space.high, params)
    source_path = tmp_path / 'source.pt'
    source.save(str(source_path))
    scenario['training_defaults']['pretrained_actor_checkpoint'] = str(source_path)
    monkeypatch.setattr(run, "load_and_expand_scenario", lambda *_args, **_kwargs: deepcopy(scenario))
    monkeypatch.setattr(sys, "argv", ["run.py", "--scenario", SCENARIO, "--no-wandb",
                                     "--quiet", "--output-dir", str(tmp_path)])
    run.main()
    checkpoint = safe_load(str(tmp_path / "final_model.pt"), map_location="cpu")
    assert checkpoint["environment_steps"] == 9
    assert (tmp_path / "checkpoint_step000000008.pt").exists()
    assert checkpoint["obs_dim"] == 192
    assert checkpoint['actor_mode'] == ('independent' if arm == 'full' else 'shared')
    if arm == 'full':
        assert set(checkpoint['actors']) == set(IDS[:2])
    else:
        assert checkpoint['lora_contract']['per_agent_log_std'] is True
    assert checkpoint["reward_mode"] == "individual"
    assert checkpoint["critic_mode"] == "agent_conditioned"
    assert checkpoint["observation_contract"]["observation"]["frenet_neighbors"]["agent_ids"] == IDS
    assert (tmp_path / "best_model.pt").exists()
    source_path.unlink()  # Playback must be self-contained.
    scenario['environment']['max_steps'] = 4
    monkeypatch.setattr(sys, "argv", ["run.py", "--scenario", str(path), "--eval", "--checkpoint",
        str(tmp_path / "best_model.pt"), "--eval-episodes", "1", "--allow-provenance-mismatch",
        "--no-wandb", "--quiet", "--output-dir", str(tmp_path / "playback")])
    run.main()
    assert (tmp_path / 'playback' / 'evaluation_report.json').exists()
    if num_envs > 1:
        import csv
        with (tmp_path / "collector_progress.csv").open() as stream:
            progress = list(csv.DictReader(stream))
        assert progress[0]["collector/phase"] == "startup"
        assert int(progress[-1]["collector/updated_environment_steps"]) == 9


def test_asymmetric_training_resets_after_learners_and_evaluation_runs_full_race():
    from core.setup import create_training_setup
    from env.collision_state import apply_episode_termination_policy
    scenario = load_and_expand_scenario(SCENARIO)
    envs = []
    try:
        for phase, expected in [('train', 'all_trainable'), ('eval', 'all_agents')]:
            env, _, _ = create_training_setup(scenario, mode=phase, scenario_dir=Path('scenarios'))
            envs.append(env)
            assert env.episode_termination_mode == expected
            assert env.max_steps == 16000
            assert env.lifecycle.finish_on_laps
            env.reset(seed=42)
            for lap in range(env.target_laps):
                env.lifecycle.record_lap_crossing("car_0", step=lap + 1)
            assert not env.lifecycle.records["car_0"].is_active
            terms = dict(zip(IDS, [True, True, False, False]))
            _, done = apply_episode_termination_policy(terms, {}, active_agents=IDS,
                possible_agents=IDS, trainable_agents=IDS[:2], mode=env.episode_termination_mode)
            assert done == (phase == 'train')
    finally:
        for env in envs:
            env.close()


def test_evaluation_duration_is_written_to_history(tmp_path):
    import json
    from types import SimpleNamespace
    from training.hooks import EvaluationCheckpointHook
    evaluator = SimpleNamespace(evaluate=lambda: {'completion_rate': 1., 'collision_rate': 0.,
                                                   'mean_progress': 1., 'mean_clean_finish_time_s': 10.})
    hook = EvaluationCheckpointHook(None, str(tmp_path), evaluator, evaluate_every=1)
    hook._save = lambda *args, **kwargs: None
    hook.on_episode_end(0, 0., {}, {})
    record = json.loads((tmp_path / 'evaluation_history.jsonl').read_text())
    assert record['evaluation_seconds'] >= 0
    assert record['evaluation_seconds_total'] == hook.evaluation_seconds


def test_asymmetric_budget_and_cadence_match_pretraining():
    from core.scenario import validate_scenario

    scenario = load_and_expand_scenario(SCENARIO)
    pretrain = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    validate_scenario(scenario)
    assert scenario["experiment"]["total_steps"] == pretrain["experiment"]["total_steps"] == 120000000
    assert scenario["training_defaults"]["progress_unit"] == "environment_steps"
    assert scenario["training_defaults"]["checkpoint_every_steps"] == pretrain["evaluation"]["every_steps"]
    assert scenario["evaluation"]["every_steps"] == pretrain["evaluation"]["every_steps"]
