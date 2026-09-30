"""Episode-focused console progress, with separate collector diagnostics."""
from collections import deque
import math
import threading
import time


class CollectorProgress:
    def __init__(self, logger, *, workers, environments, horizon, interval=15., window=100):
        self.interval = float(interval)
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError('collector_progress_interval_s must be finite and positive')
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError('terminal_recent_episodes must be a positive integer')
        self.logger = logger
        self.started = self.phase_started = self.last_message = time.monotonic()
        self.next_publish = 0.
        self.state = dict(phase='startup', workers=workers, environments=environments,
                          horizon=horizon, ready_workers=0, waiting_workers=0,
                          worker_messages=0, actions_dispatched=0,
                          updated_environment_steps=0, updates=0, completed_episodes=0)
        self.episodes = deque(maxlen=window)
        self.evaluation = None
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

    def episode_completed(self, reward, info, metrics):
        """Consume worker episode events immediately, before the update barrier."""
        race = metrics.get('race_record', {})
        learners = [a for a in race.get('agents', {}).values() if a['team'] == 'trainable']
        row = dict(reward=float(reward), steps=metrics.get('episode_steps'),
                   laps=race.get('mean_learner_laps'), finished=race.get('both_finished'))
        if learners:
            row.update(failed=any(a['collision_dnf'] or a['boundary_dnf'] for a in learners),
                       timeout=any(a['timeout'] for a in learners))
        attacks = [a for a in learners if 'attack_successes' in a]
        if attacks:
            row.update(attack_successes=sum(a['attack_successes'] for a in attacks),
                       target_crashes=sum(a['attack_target_crashes'] for a in attacks))
        with self.lock:
            self.episodes.append(row)
            self.state['completed_episodes'] += 1

    def evaluation_progress(self, row):
        """Called on the evaluator thread; the watchdog only reads snapshots."""
        with self.lock:
            self.evaluation = None if row is None else {**row, 'reported_at': time.monotonic()}
        # Always show episode boundaries, even for evaluations shorter than a heartbeat.
        if row is not None and row['status'] in {'starting', 'complete'} and self.logger is not None:
            self.logger.print_info(self.console_message())

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            episodes = list(self.episodes)
            recent = {'recent_episodes': len(episodes)}
            if episodes:
                recent['last_reward'] = episodes[-1]['reward']
                for key in ('reward', 'steps', 'laps', 'finished', 'failed', 'timeout',
                            'attack_successes', 'target_crashes'):
                    values = [r[key] for r in episodes if r.get(key) is not None]
                    if values:
                        recent[f'recent_{key}_mean'] = sum(values) / len(values)
            evaluation = {}
            if self.evaluation is not None:
                evaluation = {f'evaluation_{k}': v for k, v in self.evaluation.items() if k != 'reported_at'}
                evaluation['evaluation_progress_age_s'] = now - self.evaluation['reported_at']
            return {f'collector/{key}': value for key, value in {
                **self.state, **recent, **evaluation, 'elapsed_seconds': now - self.started,
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

    def console_message(self):
        m = {k.removeprefix('collector/'): v for k, v in self.snapshot().items()}
        if 'evaluation_episode' in m:
            text = (f"MAPPO eval episode={m['evaluation_episode']}/{m['evaluation_episodes']} "
                    f"map={m['evaluation_map']} status={m['evaluation_status']} "
                    f"steps={m['evaluation_steps']}/{m['evaluation_max_steps'] or 'unlimited'} "
                    f"sim_s={m['evaluation_sim_seconds']:.1f} laps={m['evaluation_laps']} "
                    f"outcome={m['evaluation_outcome']}")
            if 'evaluation_workers' in m:
                text += (f" workers={m['evaluation_workers']} "
                         f"completed={m['evaluation_completed_episodes']}/{m['evaluation_episodes']}")
            if 'evaluation_attack_successes' in m:
                text += (f" attacks={m['evaluation_attack_successes']} "
                         f"target_crashes={m['evaluation_target_crashes']}")
            if 'recent_reward_mean' in m:
                text += f" train_return_mean={m['recent_reward_mean']:+.2f}"
            return text + f" progress_age_s={m['evaluation_progress_age_s']:.0f}"
        if m['phase'] == 'startup':
            return (f"MAPPO startup ready={m['ready_workers']}/{m['workers']} workers "
                    f"envs={m['environments']} elapsed_s={m['phase_seconds']:.0f}")
        text = (f"MAPPO train phase={m['phase']} env_steps={m['updated_environment_steps']:,} "
                f"updates={m['updates']} episodes={m['completed_episodes']} "
                f"recent={m['recent_episodes']}")
        if not m['recent_episodes']:
            return text + " return=pending (no completed episodes yet)"
        text += f" return_mean={m['recent_reward_mean']:+.2f} return_last={m['last_reward']:+.2f}"
        for label, key in (('ep_steps_mean', 'steps'), ('laps_mean', 'laps'),
                           ('attacks/ep', 'attack_successes'), ('target_crashes/ep', 'target_crashes')):
            if f'recent_{key}_mean' in m:
                text += f" {label}={m[f'recent_{key}_mean']:.2f}"
        for label, key in (('finished', 'finished'), ('crash_or_exit', 'failed'), ('timeout', 'timeout')):
            if f'recent_{key}_mean' in m:
                text += f" {label}={m[f'recent_{key}_mean']:.1%}"
        return text

    def _watch(self):
        while not self.stop.wait(self.interval):
            self.logger.print_info(self.console_message())

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
