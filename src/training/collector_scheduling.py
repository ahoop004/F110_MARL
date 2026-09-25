"""Optional readiness-based inference dispatch with per-worker deadlines."""
import time
from multiprocessing.connection import wait


class CollectorScheduler:
    def __init__(self, mode, timeout):
        if mode not in {'synchronous', 'ready'}:
            raise ValueError('collector_scheduling must be synchronous or ready')
        self.mode = mode
        self.timeout = timeout
        self.last_response = {}

    def reset(self):
        # Policy updates and evaluation deliberately pause all collectors.
        self.last_response.clear()

    def workers(self, connections, paused):
        active = [worker for worker in connections if worker not in paused]
        if self.mode == 'synchronous' or not active:
            return active
        now = time.monotonic()
        for worker in active:
            self.last_response.setdefault(worker, now)
        remaining = min(self.timeout - (now - self.last_response[w]) for w in active)
        if remaining <= 0:
            worker = min(active, key=self.last_response.__getitem__)
            raise RuntimeError(f'Collector worker {worker} timed out after {self.timeout}s')
        ready = wait([connections[w] for w in active], timeout=remaining)
        if not ready:
            worker = min(active, key=self.last_response.__getitem__)
            raise RuntimeError(f'Collector worker {worker} timed out after {self.timeout}s')
        return [w for w in active if connections[w] in ready]

    def exclude_parent_time(self, seconds):
        # Logging/evaluation can block the parent while a response is already
        # queued; that delay must not be blamed on a healthy collector.
        for worker in self.last_response:
            self.last_response[worker] += seconds

    def received(self, worker):
        self.last_response[worker] = time.monotonic()
