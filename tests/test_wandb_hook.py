from types import SimpleNamespace

import pytest

from training.hooks import WandbHook


class _RecordingWandbLogger:
    def __init__(self) -> None:
        self.payloads = []

    def log_metrics(self, metrics, step=None) -> None:
        self.payloads.append((dict(metrics), step))


def test_wandb_hook_logs_mappo_updates_and_episode_research_metrics() -> None:
    logger = _RecordingWandbLogger()
    hook = WandbHook(logger)

    hook.on_update({"train/policy_loss": 0.25})
    hook.on_step(
        SimpleNamespace(
            agent_id="car_0",
            map_id="circle_map",
            reward_components={"progress/bonus": 2.0},
        )
    )
    hook.on_step(
        SimpleNamespace(
            agent_id="car_1",
            map_id="circle_map",
            reward_components={
                "progress/bonus": -1.0,
                "collision/penalty": -200.0,
            },
        )
    )
    hook.on_episode_end(
        7,
        10.0,
        {"outcome": "finished"},
        {
            "episode_steps": 300,
            "agent_rewards": {"car_0": 10.0, "car_1": -190.0},
            "agent_individual_rewards": {"car_0": 10.0, "car_1": -190.0},
            "agent_outcomes": {"car_0": "finished", "car_1": "self_crash"},
            "agent_terminal_reasons": {
                "car_0": "race_complete",
                "car_1": "collision",
            },
            "agent_finish_positions": {"car_0": 1, "car_1": None},
            "agent_lap_counts": {"car_0": 1, "car_1": 0},
        },
    )

    update, update_step = logger.payloads[0]
    assert update_step is None
    assert update == {"train/update": 1, "train/policy_loss": 0.25}

    episode, episode_step = logger.payloads[1]
    assert episode_step is None
    assert episode["episode/number"] == 7
    assert episode["episode/map_bundle"] == "circle_map"
    assert episode["episode/steps"] == 300
    assert episode["episode/team/completion_rate"] == pytest.approx(0.5)
    assert episode["episode/team/all_finished"] == 0.0
    assert episode["episode/team/collision_rate"] == pytest.approx(0.5)
    assert episode["episode/team/timeout_rate"] == 0.0
    assert episode["episode/reward_component/progress/bonus/car_0"] == 2.0
    assert episode["episode/reward_component/collision/penalty/car_1"] == -200.0
    assert episode["episode/reward_component_mean/progress/bonus"] == pytest.approx(0.5)


def test_wandb_hook_clears_reward_components_between_episodes() -> None:
    logger = _RecordingWandbLogger()
    hook = WandbHook(logger)
    hook.on_step(
        SimpleNamespace(
            agent_id="car_0",
            map_id="circle_map",
            reward_components={"progress/bonus": 2.0},
        )
    )
    episode_metrics = {
        "agent_outcomes": {"car_0": "finished"},
        "agent_terminal_reasons": {"car_0": "race_complete"},
    }
    hook.on_episode_end(0, 2.0, {}, episode_metrics)
    hook.on_episode_end(1, 0.0, {}, episode_metrics)

    assert "episode/reward_component_mean/progress/bonus" in logger.payloads[0][0]
    assert "episode/reward_component_mean/progress/bonus" not in logger.payloads[1][0]


def test_wandb_hook_keeps_parallel_worker_episodes_separate():
    logger = _RecordingWandbLogger()
    hook = WandbHook(logger)
    for worker_id, reward in [(0, 2.0), (1, 7.0)]:
        hook.on_step(SimpleNamespace(
            agent_id="car_0", map_id=f"map-{worker_id}",
            info={"worker_id": worker_id}, reward_components={"progress": reward},
        ))
    for worker_id in (0, 1):
        hook.on_episode_end(worker_id, 0.0, {"worker_id": worker_id, "worker_seed": 42 + worker_id}, {})
    for worker_id, reward in [(0, 2.0), (1, 7.0)]:
        log, _ = logger.payloads[worker_id]
        assert log["episode/reward_component/progress/car_0"] == reward
        assert log["episode/map_bundle"] == f"map-{worker_id}"
        assert log["episode/worker_seed"] == 42 + worker_id


@pytest.mark.parametrize("record_transitions", [False, True])
def test_worker_wandb_summary_matches_transition_stream(record_transitions):
    from dataclasses import replace
    import pickle
    import numpy as np
    from env.types import TransitionRecord
    from training.on_policy_trainer import _WorkerHook

    messages = []
    reference_logger, summary_logger = _RecordingWandbLogger(), _RecordingWandbLogger()
    reference, summarized = WandbHook(reference_logger), WandbHook(summary_logger)
    workers = [
        _WorkerHook(SimpleNamespace(send=messages.append), i, 42 + i,
                    record_transitions, aggregate_wandb=True)
        for i in range(2)
    ]
    reference_records = []
    for episode in range(2):
        for step in range(2):
            for worker_id, worker in enumerate(workers):
                record = TransitionRecord(
                    obs=np.zeros(115, dtype=np.float32), action_norm=np.zeros(2),
                    action_phys=np.ones(2), reward=1.0,
                    reward_components={"progress": float(worker_id + step)},
                    next_obs=np.ones(115, dtype=np.float32), terminated=False,
                    truncated=step == 1, info={}, global_state=np.zeros(16),
                    map_id=f"map-{worker_id}", spawn_id=None,
                    episode_id=f"worker-{worker_id}-ep-{episode}", step_idx=step,
                    agent_id="car_0",
                )
                reference_record = replace(record, info={"worker_id": worker_id,
                                                         "worker_seed": 42 + worker_id})
                reference_records.append(reference_record)
                reference.on_step(reference_record)
                worker.on_step(record)
        for worker_id, worker in enumerate(workers):
            worker.on_episode_end(episode, 2.0, {"outcome": "timeout"}, {"episode_steps": 2})
            _, (reward, info, metrics) = messages[-1]
            global_episode = episode * 2 + worker_id
            reference.on_episode_end(global_episode, reward,
                                     {k: v for k, v in info.items() if k != "_wandb_episode_state"}, metrics)
            summarized.on_episode_end(global_episode, reward, info, metrics)

    assert summary_logger.payloads == reference_logger.payloads
    sent_records = [payload for kind, payload in messages if kind == "transition"]
    assert len(sent_records) == (8 if record_transitions else 0)
    if record_transitions:
        assert pickle.dumps(sent_records) == pickle.dumps(reference_records)
    assert sum(kind == "episode" for kind, _ in messages) == 4
