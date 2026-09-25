import multiprocessing as mp

import pytest

from training.collector_scheduling import CollectorScheduler


def test_ready_dispatch_does_not_wait_for_slow_worker():
    parent_a, child_a = mp.Pipe()
    parent_b, child_b = mp.Pipe()
    try:
        scheduler = CollectorScheduler('ready', 1)
        child_b.send(('requests', {}))
        assert scheduler.workers({0: parent_a, 1: parent_b}, {}) == [1]
        assert parent_b.recv() == ('requests', {})
        # A worker at the update barrier must not prevent another from running.
        child_a.send(('rollout', {}))
        assert scheduler.workers({0: parent_a, 1: parent_b}, {1: True}) == [0]
    finally:
        for connection in (parent_a, child_a, parent_b, child_b):
            connection.close()


def test_ready_dispatch_deadline_and_update_pause(monkeypatch):
    import training.collector_scheduling as module
    monkeypatch.setattr(module.time, 'monotonic', lambda: 20.)
    scheduler = CollectorScheduler('ready', 5)
    scheduler.last_response = {0: 10.}
    with pytest.raises(RuntimeError, match='worker 0 timed out'):
        scheduler.workers({0: object()}, {})
    scheduler.reset()
    assert scheduler.last_response == {}
    assert scheduler.workers({0: object()}, {0: True}) == []


def test_synchronous_dispatch_preserves_worker_order():
    scheduler = CollectorScheduler('synchronous', 1)
    assert scheduler.workers({2: None, 0: None, 1: None}, {0: True}) == [2, 1]
