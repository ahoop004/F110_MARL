"""Collector scheduling, event buffering, and CPU worker lifecycle helpers."""
import time
from multiprocessing.connection import wait


_WORKER_TIMEOUT_SECONDS = 120


def _worker_startup_settings(scenario):
    defaults = {"worker_startup_batch_size": 16, "worker_startup_timeout_s": 600,
                "worker_response_timeout_s": _WORKER_TIMEOUT_SECONDS}
    settings = {key: scenario.get("experiment", {}).get(key, default)
                for key, default in defaults.items()}
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"experiment.{key} must be a positive integer")
    return settings


def _close_collectors(connections, processes, *, failed):
    # Signal the entire group before joining: per-process waits multiplied by
    # hundreds of workers otherwise turn error handling into a long shutdown.
    if failed:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for connection in connections:
        connection.close()
    deadline = time.monotonic() + 5.0
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    remaining = [process for process in processes if process.is_alive()]
    for process in remaining:
        process.kill()
    for process in remaining:
        process.join()


def _report_worker_error(connection):
    import sys
    import traceback
    error = traceback.format_exc()
    try:
        connection.send(("error", error))
    except (BrokenPipeError, EOFError, ConnectionResetError):
        # Retain the original exception if its parent can no longer receive it.
        print(error, file=sys.stderr, flush=True)


class CollectorEventSink:
    """Buffer worker events until the next collector message."""

    def __init__(self):
        self.events = []

    def send(self, event):
        self.events.append(event)

    def take(self):
        events, self.events = self.events, []
        return events


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
