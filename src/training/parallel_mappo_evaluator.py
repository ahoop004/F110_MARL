"""Concurrent checkpoint races with CPU simulators and parent-owned inference."""
from copy import deepcopy
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import time
import warnings

import torch

from training.collector_scheduling import _close_collectors, _report_worker_error, _worker_startup_settings
from training.mappo_evaluator import DeterministicMAPPOEvaluator


def evaluation_workers(scenario, episodes):
    value = scenario.get('evaluation', {}).get('num_workers', 1)
    if value == 'auto':
        exp = scenario.get('experiment', {})
        value = min(8, exp.get('num_envs', 1), exp.get('num_workers', exp.get('num_envs', 1)))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('evaluation.num_workers must be a positive integer or auto')
    return min(value, episodes)


class _RemotePolicy:
    def __init__(self, connection):
        self.connection = connection

    def act_batch(self, ids, observations, deterministic=False):
        if not deterministic:
            raise ValueError('Evaluation workers require deterministic actions')
        self.connection.send(('act', (ids, observations)))
        return self.connection.recv(), {}


def _evaluation_worker(connection, scenario, scenario_dir, kwargs):
    # No policy networks or CUDA contexts live in these processes.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYGLET_HEADLESS'] = 'true'
    torch.set_num_threads(1)
    from core.setup import create_training_setup, build_obs_composers

    env = None
    try:
        env, controllers, _ = create_training_setup(scenario, mode='eval', scenario_dir=Path(scenario_dir))
        for controller in controllers.values():
            if hasattr(controller, 'set_env'):
                controller.set_env(env)
        observations = build_obs_composers(scenario['agents'], kwargs['trainable_ids'],
                                            scenario['environment'], Path(scenario_dir))
        evaluator = DeterministicMAPPOEvaluator(env=env, other_agents=controllers,
            obs_composers=observations, **kwargs).bind_agent(_RemotePolicy(connection))
        evaluator.set_progress_callback(lambda row: connection.send(('progress', row)))
        connection.send(('ready', None))
        while True:
            kind, payload = connection.recv()
            if kind == 'close':
                return
            if kind != 'evaluate':
                raise RuntimeError(f'Unexpected evaluation command: {kind}')
            episodes, protocol = payload
            for episode in episodes:
                connection.send(('result', evaluator._evaluate_episode(episode, protocol)))
            connection.send(('done', None))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except BaseException:
        _report_worker_error(connection)
    finally:
        if env is not None:
            env.close()
        connection.close()


@torch.no_grad()
def _infer_actions(agent, requests):
    """Keep serial per-race batch shapes, independent of worker arrival order.

    Cross-race batching can change float32 GEMM rounding and hence trajectories
    through the MPC optimizer. Simulators run concurrently; inference retains
    the serial evaluator's numerics and never computes the critic.
    """
    return {worker: agent.act_batch(ids, rows, deterministic=True)[0]
            for worker, (ids, rows) in requests.items()}


class ParallelMAPPOEvaluator(DeterministicMAPPOEvaluator):
    """Reuse workers between evaluations; only frozen-policy actions cross pipes.

    Each episode retains its global index for seed, map selection, and output
    ordering. Rendering and trajectory recording use the serial implementation
    because their adapters are bound to the parent's environment.
    """

    def __init__(self, *, scenario, scenario_dir, num_workers, **kwargs):
        super().__init__(**kwargs)
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError('Evaluation workers must be a positive integer')
        self.num_workers = min(num_workers, self.episodes)
        self.scenario = deepcopy(scenario)
        self.scenario['environment']['render'] = False
        self.scenario_dir = str(scenario_dir)
        self.settings = _worker_startup_settings(scenario)
        self._connections = {}
        self._processes = []
        self._warned_serial = False

    def _receive(self, worker):
        try:
            kind, payload = self._connections[worker].recv()
        except (EOFError, ConnectionResetError) as exc:
            raise RuntimeError(f'Evaluation worker {worker} disconnected') from exc
        if kind == 'error':
            raise RuntimeError(f'Evaluation worker {worker} failed:\n{payload}')
        return kind, payload

    def _ready(self, active, last_seen, timeout):
        now = time.monotonic()
        remaining = min(timeout - (now - last_seen[w]) for w in active)
        if remaining <= 0:
            worker = min(active, key=last_seen.__getitem__)
            raise RuntimeError(f'Evaluation worker {worker} timed out after {timeout}s')
        ready = wait([self._connections[w] for w in active], timeout=remaining)
        if not ready:
            worker = min(active, key=last_seen.__getitem__)
            raise RuntimeError(f'Evaluation worker {worker} timed out after {timeout}s')
        return [w for w in active if self._connections[w] in ready]

    def _start_workers(self):
        if self._connections:
            return
        context = mp.get_context('spawn')
        thread_vars = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                       'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
        previous = {key: os.environ.get(key) for key in thread_vars}
        kwargs = dict(trainable_ids=self.trainable_ids, action_composer=self.actions[self.trainable_ids[0]],
            episodes=self.episodes, base_seed=self.base_seed, action_repeat=self.action_repeat,
            focal_agent_id=self.focal_agent_id, protocol_name=self.protocol_name)
        try:
            for key in thread_vars:
                os.environ[key] = '1'
            batch_size = self.settings['worker_startup_batch_size']
            for start in range(0, self.num_workers, batch_size):
                active, last_seen = set(), {}
                for worker in range(start, min(start + batch_size, self.num_workers)):
                    parent, child = context.Pipe()
                    process = context.Process(target=_evaluation_worker,
                        args=(child, self.scenario, self.scenario_dir, kwargs),
                        name=f'mappo-evaluation-{worker}')
                    self._connections[worker] = parent
                    try:
                        process.start()
                    finally:
                        child.close()
                    self._processes.append(process)
                    active.add(worker)
                    last_seen[worker] = time.monotonic()
                while active:
                    for worker in self._ready(active, last_seen, self.settings['worker_startup_timeout_s']):
                        kind, _ = self._receive(worker)
                        if kind != 'ready':
                            raise RuntimeError(f'Unexpected evaluation startup message: {kind}')
                        active.remove(worker)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _collect_episodes(self, protocol):
        if self.num_workers == 1 or self.recording is not None or getattr(self, 'render', False):
            if self.num_workers > 1 and not self._warned_serial:
                warnings.warn('MAPPO evaluation uses one environment when recording or rendering is enabled.',
                              RuntimeWarning, stacklevel=2)
                self._warned_serial = True
            return super()._collect_episodes(protocol)
        try:
            self._start_workers()
            active = set(self._connections)
            last_seen = {worker: time.monotonic() for worker in active}
            results = {}
            completed = set()
            for worker, connection in self._connections.items():
                connection.send(('evaluate', (list(range(worker, self.episodes, self.num_workers)), protocol)))
            while active:
                requests = {}
                ready = self._ready(active, last_seen, self.settings['worker_response_timeout_s'])
                parent_started = time.monotonic()
                for worker in ready:
                    kind, payload = self._receive(worker)
                    last_seen[worker] = time.monotonic()
                    if kind == 'act':
                        requests[worker] = payload
                    elif kind == 'progress':
                        if payload['status'] == 'complete':
                            completed.add(payload['episode'])
                        if self.progress_callback is not None:
                            self.progress_callback({**payload, 'workers': self.num_workers,
                                                    'completed_episodes': len(completed)})
                    elif kind == 'result':
                        episode = payload[0].episode
                        if episode in results:
                            raise RuntimeError(f'Duplicate evaluation episode {episode}')
                        results[episode] = payload
                    elif kind == 'done':
                        active.remove(worker)
                    else:
                        raise RuntimeError(f'Unexpected evaluation worker message: {kind}')
                if requests:
                    for worker, response in _infer_actions(self.agent, requests).items():
                        self._connections[worker].send(response)
                # Inference and progress logging are parent work, not worker stalls.
                elapsed = time.monotonic() - parent_started
                for worker in active:
                    last_seen[worker] += elapsed
            if set(results) != set(range(self.episodes)):
                raise RuntimeError('Evaluation workers did not return all scheduled episodes')
            return [results[episode] for episode in range(self.episodes)]
        except BaseException:
            self._stop_workers(failed=True)
            raise

    def _stop_workers(self, *, failed=False):
        if not failed:
            for connection in self._connections.values():
                try:
                    connection.send(('close', None))
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
        _close_collectors(list(self._connections.values()), self._processes, failed=failed)
        self._connections.clear()
        self._processes.clear()

    def close(self):
        try:
            self._stop_workers()
        finally:
            super().close()
