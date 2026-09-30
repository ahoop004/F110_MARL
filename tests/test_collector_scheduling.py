import multiprocessing as mp

import pytest

from training.collector_scheduling import CollectorScheduler


@pytest.mark.parametrize('allowed,expected', [({0, 1, 96, 97}, 2), ({0, 97}, 2), ({0, 96}, 1)])
def test_affinity_counts_accessible_cores_without_double_counting_siblings(tmp_path, monkeypatch, allowed, expected):
    import training.collector_scheduling as module
    monkeypatch.setattr(module.os, 'sched_getaffinity', lambda pid: allowed, raising=False)
    for cpu, siblings in {0: '0,96', 1: '1,97', 96: '0,96', 97: '1,97'}.items():
        topology = tmp_path / f'cpu{cpu}' / 'topology'
        topology.mkdir(parents=True)
        (topology / 'thread_siblings_list').write_text(siblings + '\n')
    assert module.cpu_affinity_count() == len(allowed)
    assert module.cpu_affinity_core_count(sysfs_root=tmp_path) == expected


def test_unavailable_topology_does_not_guess_cores_from_node_cpu_count(tmp_path, monkeypatch):
    import training.collector_scheduling as module
    monkeypatch.setattr(module.os, 'sched_getaffinity', lambda pid: {0, 96}, raising=False)
    monkeypatch.setattr(module.os, 'cpu_count', lambda: 192)
    assert module.cpu_affinity_count() == 2
    assert module.cpu_affinity_core_count(sysfs_root=tmp_path) is None
    monkeypatch.delattr(module.os, 'sched_getaffinity')
    assert module.cpu_affinity_count() is None
    assert module.cpu_affinity_core_count(sysfs_root=tmp_path) is None


def test_unreadable_cpu_affinity_is_reported_as_unknown(monkeypatch):
    import training.collector_scheduling as module
    def unavailable(pid):
        raise OSError('affinity unavailable')
    monkeypatch.setattr(module.os, 'sched_getaffinity', unavailable, raising=False)
    assert module.cpu_affinity_count() is None
    assert module.cpu_affinity_core_count() is None


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
