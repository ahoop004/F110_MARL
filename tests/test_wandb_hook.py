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
