from threading import Event
from types import SimpleNamespace

import pytest

from training.collector_progress import CollectorProgress
from training.hooks import WandbHook


@pytest.mark.parametrize('interval', [0, -1, float('inf'), float('nan')])
def test_invalid_progress_interval(interval):
    with pytest.raises(ValueError, match='finite and positive'):
        CollectorProgress(None, workers=2, environments=4, horizon=256, interval=interval)


def test_heartbeat_runs_while_parent_is_blocked_and_stops():
    observed = Event()
    lines = []

    def log(line):
        lines.append(line)
        observed.set()

    progress = CollectorProgress(SimpleNamespace(print_info=log), workers=2,
                                 environments=4, horizon=256, interval=.01)
    progress.set(phase='updating', waiting_workers=2, actions_dispatched=1024)
    progress.received()
    progress.start()
    try:
        # Parent makes no calls: the console heartbeat must still appear.
        assert observed.wait(2.)
    finally:
        progress.close()
    assert not progress.thread.is_alive()
    assert 'phase=updating' in lines[0]
    assert 'episodes=0' in lines[0]
    assert 'return=pending' in lines[0]
    assert 'messages=' not in lines[0]
    assert progress.snapshot()['collector/actions_dispatched'] == 1024


def episode(progress, reward, *, laps=0, reason='time_limit', successes=None):
    learner = dict(team='trainable', collision_dnf=reason == 'collision',
                   boundary_dnf=reason == 'track_boundary', timeout=reason == 'time_limit')
    if successes is not None:
        learner.update(attack_successes=successes, attack_target_crashes=successes + 1)
    # Opponent failures must not count as learner failures.
    opponent = dict(team='opponent', collision_dnf=True, boundary_dnf=False, timeout=False)
    race = dict(mean_learner_laps=laps, both_finished=reason == 'race_complete',
                agents={'car_0': learner, 'car_1': opponent})
    progress.episode_completed(reward, {}, dict(episode_steps=100, race_record=race))


def test_recent_episode_window_reports_rewards_outcomes_and_attacks():
    progress = CollectorProgress(None, workers=2, environments=4, horizon=256, window=2)
    progress.set(phase='collecting')
    episode(progress, -100., reason='collision', successes=0)
    episode(progress, 20., laps=5, reason='race_complete', successes=3)
    episode(progress, -10., laps=2, reason='track_boundary', successes=1)
    row = progress.snapshot()
    assert row['collector/completed_episodes'] == 3
    assert row['collector/recent_episodes'] == 2
    assert row['collector/recent_reward_mean'] == 5.
    assert row['collector/recent_laps_mean'] == 3.5
    text = progress.console_message()
    for expected in ('episodes=3', 'recent=2', 'return_mean=+5.00', 'return_last=-10.00',
                     'attacks/ep=2.00', 'target_crashes/ep=3.00', 'finished=50.0%',
                     'crash_or_exit=50.0%', 'timeout=0.0%'):
        assert expected in text


def test_regular_and_continuous_episodes_do_not_invent_attack_or_finish_metrics():
    progress = CollectorProgress(None, workers=1, environments=2, horizon=256, window=1)
    progress.set(phase='collecting')
    episode(progress, 2.)
    assert 'timeout=100.0%' in progress.console_message()
    assert 'attacks/ep' not in progress.console_message()
    progress.episode_completed(3., {}, {'race_record': {'both_finished': None}})
    assert 'return_mean=+3.00' in progress.console_message()
    assert 'finished=' not in progress.console_message()


def test_evaluation_replaces_idle_worker_status_and_clears_afterwards(monkeypatch):
    import training.collector_progress as module
    now = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    lines = []
    progress = CollectorProgress(SimpleNamespace(print_info=lines.append),
                                 workers=2, environments=4, horizon=256)
    episode(progress, 12., reason='race_complete')
    progress.set(phase='evaluation_checkpoint_logging')
    row = dict(episode=2, episodes=8, map='Spa_map', status='starting', steps=0,
               max_steps=40000, sim_seconds=0., laps='car_0:0/5', outcome='car_0:active')
    progress.evaluation_progress(row)
    assert 'MAPPO eval episode=2/8 map=Spa_map' in lines[-1]
    assert 'train_return_mean=+12.00' in lines[-1]
    row.update(status='running', steps=1000, sim_seconds=50., laps='car_0:1/5',
               attack_successes=2, target_crashes=3)
    progress.evaluation_progress(row)
    assert len(lines) == 1  # Running messages come from the watchdog, not each callback.
    now[0] = 10.
    text = progress.console_message()
    for expected in ('steps=1000/40000', 'laps=car_0:1/5', 'attacks=2',
                     'target_crashes=3', 'progress_age_s=10'):
        assert expected in text
    progress.evaluation_progress({**row, 'status': 'complete', 'outcome': 'car_0:race_complete'})
    assert 'outcome=car_0:race_complete' in lines[-1]
    progress.evaluation_progress(None)
    assert progress.console_message().startswith('MAPPO train')
    assert not any('evaluation_' in key for key in progress.snapshot())


def test_progress_throttles_without_advancing_optimizer_or_worker_activity(monkeypatch):
    import training.collector_progress as module
    now = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    rows = []
    hook = WandbHook(SimpleNamespace(log_metrics=lambda m: rows.append(dict(m))))
    progress = CollectorProgress(None, workers=2, environments=4, horizon=256, interval=15)
    progress.publish([hook])
    now[0] = 10.
    progress.received()
    progress.publish([hook])
    assert len(rows) == 1
    now[0] = 15.
    progress.set(phase='collecting', actions_dispatched=4)
    progress.publish([hook])
    assert len(rows) == 2
    assert rows[-1]['collector/seconds_since_worker_message'] == 5.
    assert rows[-1]['collector/phase_seconds'] == 0.
    assert rows[-1]['collector/worker_messages'] == 1
    assert rows[-1]['collector/updates'] == 0
    hook.on_update({'train/policy_loss': .5})
    assert rows[-1]['train/update'] == 1
