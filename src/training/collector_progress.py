"""Collector liveness independent of episode completion and optimizer updates."""
import math
import threading
import time


class CollectorProgress:
    def __init__(self, logger, *, workers, environments, horizon, interval=15.):
        self.interval = float(interval)
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError('collector_progress_interval_s must be finite and positive')
        self.logger = logger
        self.started = self.phase_started = self.last_message = time.monotonic()
        self.next_publish = 0.
        self.state = dict(phase='startup', workers=workers, environments=environments,
                          horizon=horizon, ready_workers=0, waiting_workers=0,
                          worker_messages=0, actions_dispatched=0,
                          updated_environment_steps=0, updates=0)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        if self.logger is not None:
            self.thread = threading.Thread(target=self._watch, name='collector-progress', daemon=True)
            self.thread.start()

    def set(self, **values):
        with self.lock:
            if values.get('phase', self.state['phase']) != self.state['phase']:
                self.phase_started = time.monotonic()
            self.state.update(values)

    def received(self):
        with self.lock:
            self.last_message = time.monotonic()
            self.state['worker_messages'] += 1

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            return {f'collector/{key}': value for key, value in {
                **self.state, 'elapsed_seconds': now - self.started,
                'phase_seconds': now - self.phase_started,
                'seconds_since_worker_message': now - self.last_message,
            }.items()}

    def publish(self, hooks, *, force=False):
        # Only the trainer thread calls hooks: never log fake optimizer updates
        # or touch CSV/W&B from the console watchdog.
        now = time.monotonic()
        if not force and now < self.next_publish:
            return
        self.next_publish = now + self.interval
        metrics = self.snapshot()
        for hook in hooks:
            callback = getattr(hook, 'on_collector_progress', None)
            if callback is not None:
                callback(metrics)

    def _watch(self):
        while not self.stop.wait(self.interval):
            m = {k.removeprefix('collector/'): v for k, v in self.snapshot().items()}
            self.logger.print_info(
                f"MAPPO status phase={m['phase']} phase_s={m['phase_seconds']:.0f} "
                f"ready={m['ready_workers']}/{m['workers']} "
                f"at_barrier={m['waiting_workers']} messages={m['worker_messages']} "
                f"last_message_s={m['seconds_since_worker_message']:.0f} "
                f"actions_dispatched={m['actions_dispatched']} "
                f"updated_env_steps={m['updated_environment_steps']} updates={m['updates']}")

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
