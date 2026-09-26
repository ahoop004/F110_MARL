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
    assert 'messages=1' in lines[0]
    assert 'actions_dispatched=1024' in lines[0]


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
