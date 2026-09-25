"""Grouped PPO collectors: several environments per CPU process, one learner.

Each environment runs the serial trainer's generator and owns its own buffer,
bootstrap boundaries, seed and budget. The parent batches inference and freezes
weights until every live environment has delivered its next fragment.
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from training.collector_scheduling import CollectorScheduler
from training.hooks import WandbHook
from training.on_policy_trainer import (
    OnPolicyTrainer, _RemotePolicy, _WorkerHook, _close_collectors,
    _report_worker_error, _worker_startup_settings,
)


class _EventSink:
    def __init__(self):
        self.events = []

    def send(self, event):
        self.events.append(event)

    def take(self):
        events, self.events = self.events, []
        return events


def _make_collector(scenario, directory, agent_id, env_id, quota, horizon,
                    run_id, gamma, gae_lambda, sink, record, aggregate, step_budget):
    from core.setup import create_training_setup
    from run import build_obs_composer, build_reward_composer
    from wrappers.actions.composer import ActionComposer

    scenario = copy.deepcopy(scenario)
    base_seed = int(scenario['experiment']['seed'])
    seed = (base_seed + env_id) % (2 ** 32)
    scenario['experiment']['seed'] = seed
    env_cfg = scenario['environment']
    env_seed = env_cfg.get('seed')
    env_cfg['seed'] = ((base_seed if env_seed is None else int(env_seed)) + env_id) % (2 ** 32)
    env_cfg['render'] = False
    env, opponents, _ = create_training_setup(scenario, scenario_dir=Path(directory))
    try:
        for opponent in opponents.values():
            if hasattr(opponent, 'set_env'):
                opponent.set_env(env)
        cfg = scenario['agents'][agent_id]
        space = env.action_spaces[agent_id]
        obs = build_obs_composer(cfg, env_cfg, Path(directory), space.n)
        rewards = build_reward_composer(cfg, Path(directory))
        policy = _RemotePolicy(None, horizon, obs.obs_dim, space.n, gamma, gae_lambda,
                               map_scheduler=env._map_scheduler, worker_id=env_id)
        trainer = OnPolicyTrainer(
            env, agent_id, policy, opponents, obs, rewards,
            ActionComposer.from_config(space.low, space.high, cfg.get('action_constraints', {}),
                decision_dt=float(env_cfg.get('timestep', .01)) * int(env_cfg.get('action_repeat', 1))),
            action_repeat=int(env_cfg.get('action_repeat', 1)),
            hooks=[_WorkerHook(sink, env_id, seed, record, aggregate)],
            run_id=f'{run_id}_worker{env_id:03d}',
        )
        generator = trainer.iter_train(0 if step_budget else quota, parallel=True,
                                      total_steps=quota if step_budget else None)
        return env, generator, (obs.obs_dim, space.low, space.high)
    except BaseException:
        env.close()
        raise


def _collect_worker(connection, scenario, directory, agent_id, assignments,
                    horizon, run_id, gamma, gae_lambda, record, aggregate, step_budget):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYGLET_HEADLESS'] = 'true'
    torch.set_num_threads(1)
    envs, generators, pending = {}, {}, {}
    sink = _EventSink()
    try:
        contracts = []
        for env_id, quota in assignments:
            envs[env_id], generators[env_id], contract = _make_collector(
                scenario, directory, agent_id, env_id, quota, horizon, run_id,
                gamma, gae_lambda, sink, record, aggregate, step_budget)
            contracts.append(contract)
        connection.send(('ready', contracts))
        if connection.recv() != ('start', None):
            raise RuntimeError('PPO collector expected start after readiness')

        def advance(env_id, response=None):
            try:
                pending[env_id] = generators[env_id].send(response)
            except StopIteration:
                pending.pop(env_id, None)

        for env_id in generators:
            advance(env_id)
        while pending:
            paused = {}
            while set(pending) - paused.keys():
                requests = {}
                for env_id, (kind, payload) in pending.items():
                    if env_id in paused:
                        continue
                    if kind == 'rollout':
                        paused[env_id] = payload
                    else:
                        requests[env_id] = (kind, payload)
                if requests:
                    connection.send(('requests', (requests, sink.take())))
                    for env_id, response in connection.recv().items():
                        advance(env_id, response)
            connection.send(('rollout', (paused, sink.take())))
            reply = connection.recv()
            for env_id in sorted(paused):
                advance(env_id, reply)
        connection.send(('done', sink.take()))
    except (BrokenPipeError, EOFError, ConnectionResetError):
        pass
    except BaseException:
        _report_worker_error(connection)
    finally:
        for env in envs.values():
            env.close()
        connection.close()


def train_parallel(trainer, scenario, directory, num_envs, n_episodes, *, total_steps=None):
    experiment = scenario.get('experiment', {})
    workers = min(num_envs, int(experiment.get('num_workers', num_envs)))
    agent = trainer.agent
    horizon = agent.n_steps // num_envs
    budget = total_steps if total_steps is not None else n_episodes
    if workers < 1 or horizon < 1 or agent.n_steps % num_envs or budget < num_envs:
        raise ValueError('Grouped PPO requires positive workers, n_steps divisible by num_envs, and budget >= num_envs')
    startup = _worker_startup_settings(scenario)
    scheduler = CollectorScheduler(experiment.get("collector_scheduling", "synchronous"),
                                   startup["worker_response_timeout_s"])
    record_hooks = [h for h in trainer._transition_hooks if type(h) is not WandbHook]
    aggregate = any(type(h) is WandbHook for h in trainer._transition_hooks)
    context = mp.get_context('spawn')
    connections, processes, process_by_worker = {}, [], {}
    waiting = {}
    completed = collected = updates = 0
    stopped = success = False
    started = time.perf_counter()
    trainer._set_training_progress(0, budget)
    thread_vars = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                   'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
    previous = {key: os.environ.get(key) for key in thread_vars}

    def receive(worker_id, starting=False):
        timeout = startup['worker_startup_timeout_s' if starting else 'worker_response_timeout_s']
        connection, process = connections[worker_id], process_by_worker[worker_id]
        if not connection.poll(timeout):
            raise RuntimeError(f'PPO worker {worker_id} timed out after {timeout}s '
                               f'(pid={process.pid}, exitcode={process.exitcode})')
        try:
            kind, payload = connection.recv()
        except (EOFError, ConnectionResetError) as exc:
            raise RuntimeError(f'PPO worker {worker_id} disconnected (exitcode={process.exitcode})') from exc
        if kind == 'error':
            raise RuntimeError(f'PPO worker {worker_id} failed:\n{payload}')
        return kind, payload

    def process_events(items):
        nonlocal completed
        for kind, payload in items:
            if kind == 'transition':
                for hook in record_hooks:
                    hook.on_step(payload)
            elif kind == 'episode_start':
                for hook in trainer.hooks:
                    hook.on_episode_start(payload)
            elif kind == 'episode':
                if total_steps is None:
                    trainer._set_training_progress(completed + 1, budget)
                for hook in trainer.hooks:
                    hook.on_episode_end(completed, *payload)
                completed += 1
            else:
                raise RuntimeError(f'Unexpected PPO event: {kind}')

    def events(items):
        event_start = time.monotonic()
        try:
            process_events(items)
        finally:
            scheduler.exclude_parent_time(time.monotonic() - event_start)

    try:
        for key in thread_vars:
            os.environ[key] = '1'
        batch_size = startup['worker_startup_batch_size']
        for start in range(0, workers, batch_size):
            batch = range(start, min(start + batch_size, workers))
            for worker_id in batch:
                assignments = [(i, budget // num_envs + (i < budget % num_envs))
                               for i in range(worker_id, num_envs, workers)]
                parent, child = context.Pipe()
                process = context.Process(target=_collect_worker, name=f'ppo-collector-{worker_id}', args=(
                    child, scenario, str(directory), trainer.rl_agent_id, assignments,
                    horizon, trainer.run_id, agent.gamma, agent.gae_lambda,
                    bool(record_hooks), aggregate, total_steps is not None))
                connections[worker_id] = parent
                try:
                    process.start()
                finally:
                    child.close()
                processes.append(process)
                process_by_worker[worker_id] = process
            for worker_id in batch:
                kind, contracts = receive(worker_id, starting=True)
                if kind != 'ready' or any(
                    dim != agent.obs_dim or not np.array_equal(low, agent.action_low)
                    or not np.array_equal(high, agent.action_high) for dim, low, high in contracts
                ):
                    raise ValueError(f'PPO worker {worker_id} observation/action contract mismatch')
            print(f'[PPO] Initialized {min(start + batch_size, workers)}/{workers} workers '
                  f'for {num_envs} environments', flush=True)
        startup_s = time.perf_counter() - started
        for connection in connections.values():
            connection.send(('start', None))
        round_start = time.perf_counter()
        while connections:
            requests = {}
            for worker_id in scheduler.workers(connections, waiting):
                if worker_id in waiting:
                    continue
                kind, payload = receive(worker_id)
                scheduler.received(worker_id)
                if kind == 'requests':
                    batch, items = payload
                    events(items)
                    requests.update({(worker_id, env_id): request for env_id, request in batch.items()})
                elif kind == 'rollout':
                    rollouts, items = payload
                    events(items)
                    waiting[worker_id] = rollouts
                elif kind == 'done':
                    events(payload)
                    connections.pop(worker_id).close()
                else:
                    raise RuntimeError(f'Unexpected PPO worker message: {kind}')
            if requests:
                responses = {}
                for kind in ('value', 'act'):
                    keys = sorted(key for key, request in requests.items() if request[0] == kind)
                    if not keys:
                        continue
                    observations = np.stack([requests[key][1] for key in keys])
                    if kind == 'value':
                        responses.update(zip(keys, map(float, agent.value_batch(observations))))
                    else:
                        actions, log_probs, values = agent.act_batch(observations)
                        for row, key in enumerate(keys):
                            responses[key] = (actions[row], float(log_probs[row]), float(values[row]),
                                              agent.last_raw_actions[row])
                if len(responses) != len(requests):
                    raise RuntimeError('Unknown PPO inference request')
                for worker_id in sorted({key[0] for key in requests}):
                    connections[worker_id].send({env_id: response for (worker, env_id), response
                                                 in responses.items() if worker == worker_id})
            if waiting and len(waiting) == len(connections):
                collection_s = time.perf_counter() - round_start
                rollouts = [rollout for worker in sorted(waiting)
                            for _, rollout in sorted(waiting[worker].items())]
                steps = sum(len(rollout[0]) for rollout in rollouts)
                collected += steps
                trainer.collected_steps = collected
                if total_steps is not None:
                    trainer._set_training_progress(collected, budget)
                update_start = time.perf_counter()
                metrics = agent.update_rollouts(rollouts)
                update_s = time.perf_counter() - update_start
                did_update = bool(metrics)
                updates += did_update
                metrics.update({
                    'train/environment_steps': collected, 'train/updates': updates,
                    'perf/startup_seconds': startup_s, 'perf/collection_seconds': collection_s,
                    'perf/update_seconds': update_s,
                    'perf/collection_env_steps_per_second': steps / max(collection_s, 1e-9),
                    'perf/round_env_steps_per_second': steps / max(collection_s + update_s, 1e-9),
                    'perf/elapsed_seconds': time.perf_counter() - started,
                    'perf/end_to_end_env_steps_per_second': collected / max(time.perf_counter() - started, 1e-9),
                    'perf/num_workers': workers, 'perf/num_envs': num_envs,
                })
                if did_update:
                    for hook in trainer.hooks:
                        hook.on_update(metrics)
                stopped = trainer._should_stop()
                reply = metrics
                if scenario.get('map_curriculum') is not None or stopped:
                    control = {'stop': stopped}
                    if scenario.get('map_curriculum') is not None:
                        control['training_bundles'] = trainer.env._map_scheduler.training_bundles
                    reply = {'metrics': metrics, 'collector_control': control}
                for worker_id in waiting:
                    connections[worker_id].send(reply)
                waiting.clear()
                scheduler.reset()
                round_start = time.perf_counter()
        if not stopped and total_steps is not None and collected != total_steps:
            raise RuntimeError(f'PPO collected {collected} of {total_steps} transitions')
        if not stopped and total_steps is None and completed != n_episodes:
            raise RuntimeError(f'PPO completed {completed} of {n_episodes} episodes')
        if not stopped:
            trainer._flush_pending_update()
        for hook in trainer.hooks:
            hook.on_training_end()
        success = True
    finally:
        _close_collectors(list(connections.values()), processes, failed=not success)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
