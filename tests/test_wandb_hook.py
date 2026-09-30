"""Episode logging reuses trainer facts; it never requires transition capture."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from training.hooks import WandbHook, transition_record_hooks
from training.on_policy_trainer import _WorkerHook


def race_metrics(rewards, components=None):
    return {
        "episode_steps": 300,
        "agent_rewards": rewards,
        "agent_individual_rewards": dict(rewards),
        "agent_outcomes": {aid: "finished" for aid in rewards},
        "agent_terminal_reasons": {aid: "race_complete" for aid in rewards},
        "race_record": {"map_id": "circle_map", "mean_net_progress_laps": .5,
                        "agents": {aid: {"team": "trainable", "reward_components":
                                         (components or {}).get(aid, {})} for aid in rewards}},
    }


def test_wandb_reuses_episode_components_and_preserves_team_breakdown():
    logs = []
    hook = WandbHook(SimpleNamespace(log_metrics=logs.append))
    assert transition_record_hooks([hook]) == []
    hook.on_update({"train/policy_loss": .25, "train/updates": 4})
    metrics = race_metrics({"car_0": 10., "car_1": -190.},
                          {"car_0": {"progress": 2.}, "car_1": {"progress": -1., "crash": -200.}})
    metrics["agent_outcomes"]["car_1"] = "self_crash"
    metrics["agent_terminal_reasons"]["car_1"] = "collision"
    metrics["race_record"]["team_reward_components"] = {"team_result/sweep": 3.}
    original = deepcopy(metrics)
    hook.on_episode_end(7, 10., {}, metrics)
    assert logs[0] == {"train/updates": 4, "train/policy_loss": .25}
    row = logs[1]
    assert row["episode/team/completion_rate"] == .5
    assert row["episode/team/failure_rate"] == .5
    assert row["episode/reward_component/progress/car_0"] == 2.
    assert row["episode/reward_component_mean/progress"] == .5
    assert row["episode/team_reward_component/team_result/sweep"] == 3.
    assert not any(key.startswith("episode/individual_reward/") for key in row)
    assert metrics == original


def test_single_learner_deduplicates_rewards_and_reports_attack_facts():
    logs = []
    hook = WandbHook(SimpleNamespace(log_metrics=logs.append))
    metrics = race_metrics({"car_0": 10.}, {"car_0": {"progress": 2.}})
    metrics["race_record"]["agents"]["car_0"].update(
        attack_successes=2, attack_target_crashes=4, attack_eligible_crashes=3,
        attack_ego_failed=True)
    hook.on_episode_end(0, 10., {}, metrics)
    row = logs[-1]
    assert row["episode/attack_successes"] == 2
    assert row["episode/attack_target_crashes"] == 4
    assert row["episode/attack_eligible_crashes"] == 3
    assert row["episode/attack_ego_failed"] == 1.
    assert row["episode/net_progress_laps"] == .5
    assert "episode/reward/car_0" not in row
    assert not any(key.startswith("episode/reward_component_mean/") for key in row)
    hook.on_episode_end(1, 0., {}, race_metrics({"car_0": 0.}))
    assert not any("reward_component" in key for key in logs[-1])
    assert "episode/attack_successes" not in logs[-1]


def test_disabled_component_group_does_not_traverse_components():
    class Unreadable(dict):
        def items(self):
            raise AssertionError("Disabled reward components were collected")
    metrics = race_metrics({"car_0": 1.})
    metrics["race_record"]["agents"]["car_0"]["reward_components"] = Unreadable()
    hook = WandbHook(SimpleNamespace(log_metrics=lambda row: None,
                                    should_log=lambda group: group != "reward_components"))
    hook.on_episode_end(0, 1., {}, metrics)


@pytest.mark.parametrize("record_transitions", [False, True])
def test_worker_forwards_episode_facts_without_wandb_accumulator(record_transitions):
    import numpy as np
    from env.types import TransitionRecord

    events, logs = [], []
    workers = [_WorkerHook(SimpleNamespace(send=events.append), i, 42+i, record_transitions)
               for i in range(2)]
    hook = WandbHook(SimpleNamespace(log_metrics=logs.append))
    for i, worker in enumerate(workers):
        assert worker.requires_transition_record is record_transitions
        record = TransitionRecord(
            obs=np.zeros(3), action_norm=np.zeros(2), action_phys=np.zeros(2),
            reward=1., reward_components={"progress": i+1.}, next_obs=np.ones(3),
            terminated=False, truncated=True, info={}, global_state=np.zeros(4),
            map_id="circle_map", spawn_id=None, episode_id=f"worker{i}", step_idx=0,
            agent_id="car_0")
        worker.on_step(record)
        metrics = race_metrics({"car_0": i+1.}, {"car_0": {"progress": i+1.}})
        worker.on_episode_end(0, i+1., {}, metrics)
        _, (reward, info, forwarded) = events[-1]
        assert forwarded["race_record"] is metrics["race_record"]
        assert "_wandb_episode_state" not in info
        hook.on_episode_end(i, reward, info, forwarded)
    assert sum(kind == "transition" for kind, _ in events) == (2 if record_transitions else 0)
    assert [r["episode/reward_component/progress/car_0"] for r in logs] == [1., 2.]
    assert [r["episode/worker_id"] for r in logs] == [0, 1]


def test_curriculum_uses_supported_logger_api():
    from training.hooks import CurriculumHook
    logs = []
    manager = SimpleNamespace(on_episode_end=lambda outcome: False,
        summary=lambda: {'curriculum/phase_index': 2, 'curriculum/success_rate': .5})
    hook = CurriculumHook(manager, SimpleNamespace(log_metrics=logs.append))
    hook.on_episode_end(10, 1., {'outcome': 'finished'}, {})
    assert logs == [manager.summary()]


def test_attack_console_uses_attack_facts_without_empty_race_fields():
    from training.hooks import ConsoleHook, MAPPOConsoleHook
    lines = []
    metrics = race_metrics({'car_0': 3.})
    race = metrics['race_record']
    race.update(training_return=3., race_mode='finite')
    race['agents']['car_0'].update(attack_successes=2, attack_target_crashes=3, attack_ego_failed=False)
    console = SimpleNamespace(print_info=lines.append)
    ConsoleHook(console).on_episode_end(1, 3., {'outcome': 'finished'}, metrics)
    monitor = MAPPOConsoleHook(console, every_updates=1)
    monitor.on_episode_end(1, 3., {}, metrics)
    monitor.on_update({'train/updates': 1, 'train/environment_steps': 10})
    assert 'attacks=2' in lines[0]
    assert any('attacks/ep=2.00' in line for line in lines)
    assert all('first_place=' not in line and 'lap_time=n/a' not in line for line in lines)
