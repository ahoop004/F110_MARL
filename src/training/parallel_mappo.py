"""Synchronous MAPPO collection: multiple races per CPU process, one learner.

Collectors run the same race generator as serial training. They retain only
CPU rollout buffers; all actor/critic inference and optimization happen in the
parent. Episode fragments are bootstrapped independently before pooling, and
no worker resumes collection until the pooled policy update has finished.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import torch

from agents.mappo import MAPPOAgent, MAPPORolloutBuffer
from training.collector_progress import CollectorProgress
from training.collector_scheduling import CollectorScheduler
from training.hooks import WandbHook
from training.on_policy_trainer import (
    _WorkerHook, _close_collectors, _report_worker_error, _worker_startup_settings,
)


class CollectorAgent:
    """MAPPO rollout state without networks, optimizer, or a CUDA context."""

    store_team_step = MAPPOAgent.store_team_step
    compute_team_gae = MAPPOAgent.compute_team_gae
    clear_buffers = MAPPOAgent.clear_buffers
    any_buffer_full = MAPPOAgent.any_buffer_full

    def __init__(self, contract, horizon):
        for name, value in contract.items():
            setattr(self, name, value)
        self.n_steps = horizon
        self.device = torch.device("cpu")
        self.buffers = {
            aid: MAPPORolloutBuffer(horizon, self.obs_dim, self.global_state_dim,
                                   self.action_dim, self.device)
            for aid in self.agent_ids
        }
        self._team_rollout = []
        self._team_step_indices = {aid: [] for aid in self.agent_ids}
        self.fragments = []
        self.last_raw_actions = {}
        self.policy_version = 0
        self.physics_steps_collected = 0

    def store_batch(self, agent_ids, *, observations, global_state, actions,
                    rewards, log_probs, values, terminated, truncated, raw_actions=None):
        # Environment snapshots are read-only; own one writable CPU copy before
        # handing the same state to each teammate's tensor buffer.
        global_state = np.array(global_state, dtype=np.float32, copy=True)
        for aid in agent_ids:
            self.buffers[aid].add(
                observations[aid], global_state, actions[aid], rewards[aid],
                log_probs[aid], values[aid], terminated[aid], truncated[aid],
                None if raw_actions is None else raw_actions[aid],
            )

    def finish_fragment(self, next_values):
        ids = [aid for aid in self.agent_ids if self.buffers[aid].size()]
        if not ids:
            return {}
        team_gae = (self.compute_team_gae(next_values[ids[0]])
                    if self.team_return_mode == "joint" else None)
        for aid in ids:
            buf = self.buffers[aid]
            n = buf.size()
            if team_gae is None:
                adv, ret = buf.compute_gae(next_values[aid], self.gamma, self.gae_lambda)
            else:
                indices = self._team_step_indices[aid]
                if len(indices) != n:
                    raise ValueError("Joint team indices must match actual decisions")
                adv, ret = (values[indices] for values in team_gae)
            critic = buf.global_states[:n]
            if self.critic_mode == "agent_conditioned":
                identity = torch.zeros(n, len(self.agent_ids))
                identity[:, self.agent_ids.index(aid)] = 1
                critic = torch.cat((critic, identity), dim=1)
            packed = torch.cat((buf.obs[:n], critic, buf.actions[:n],
                                buf.log_probs[:n, None], adv[:, None], ret[:, None]), dim=1)
            self.fragments.append((packed.numpy(), buf.raw_actions[:n].numpy().copy(),
                                   np.full(n, self.agent_ids.index(aid), dtype=np.int64)))
        self.clear_buffers()
        return {}

    def take_fragments(self):
        result, self.fragments = self.fragments, []
        return result


@torch.no_grad()
def infer_requests(agent, requests):
    """Batch repeated agent IDs across independent races, preserving row order."""
    observations, actor_keys, critic_inputs, critic_keys = [], [], [], []
    result = {}
    for key, (kind, payload) in requests.items():
        if kind == "act":
            ids, obs, state = payload
            observations.extend(obs)
            actor_keys.extend((key, aid) for aid in ids)
        else:
            ids, state = agent.agent_ids, payload
        result[key] = ({}, {}, {}, {}) if kind == "act" else {}
        if not ids:
            continue
        if agent.critic_mode == "shared_team":
            critic_inputs.append(state)
            critic_keys.append((key, list(ids)))
        else:
            for aid in ids:
                critic_inputs.append(agent._critic_input(state, aid))
                critic_keys.append((key, [aid]))
    if observations:
        obs_t = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=agent.device)
        actions, log_probs, raw = agent.actor_actions(
            obs_t, [aid for _, aid in actor_keys], return_raw=True)
        rows = torch.cat((actions, log_probs[:, None], raw), dim=1).cpu().numpy()
        for (key, aid), row in zip(actor_keys, rows):
            result[key][0][aid] = row[:agent.action_dim].copy()
            result[key][1][aid] = float(row[agent.action_dim])
            result[key][3][aid] = row[agent.action_dim + 1:].copy()
    if critic_inputs:
        inputs = torch.as_tensor(np.asarray(critic_inputs), dtype=torch.float32, device=agent.device)
        values = agent.critic(inputs).cpu().numpy().reshape(-1)
        for (key, ids), value in zip(critic_keys, values):
            target = result[key][2] if requests[key][0] == "act" else result[key]
            target.update({aid: float(value) for aid in ids})
    return result


class _EventSink:
    def __init__(self):
        self.events = []

    def send(self, event):
        self.events.append(event)

    def take(self):
        events, self.events = self.events, []
        return events


def _make_collector(scenario, scenario_dir, env_id, episodes, horizon, contract,
                    run_id, sink, record_transitions, aggregate_wandb, total_steps=None):
    from core.setup import create_training_setup
    from run import build_obs_composers, build_reward_composers
    from training.marl_trainer import MARLTrainer
    from wrappers.actions.composer import ActionComposer

    scenario = copy.deepcopy(scenario)
    base_seed = int(scenario["experiment"]["seed"])
    seed = (base_seed + env_id) % (2 ** 32)
    scenario["experiment"]["seed"] = seed
    env_cfg = scenario["environment"]
    env_seed = env_cfg.get("seed")
    env_cfg["seed"] = ((base_seed if env_seed is None else int(env_seed)) + env_id) % (2 ** 32)
    env_cfg["render"] = False
    env, opponents, _ = create_training_setup(scenario, scenario_dir=Path(scenario_dir))
    try:
        ids = contract["agent_ids"]
        for opponent in opponents.values():
            if hasattr(opponent, "set_env"):
                opponent.set_env(env)
        obs = build_obs_composers(scenario["agents"], ids, env_cfg, Path(scenario_dir))
        rewards = build_reward_composers(scenario["agents"], ids, Path(scenario_dir))
        snapshot = env.get_global_state()
        if (len(snapshot.vector) != contract["global_state_dim"] or
                snapshot.metadata.get("vector_contract_version", "legacy_unspecified") !=
                contract["global_state_contract_version"]):
            raise ValueError("MAPPO collector global-state contract mismatch")
        for aid in ids:
            space = env.action_spaces[aid]
            if (obs[aid].obs_dim != contract["obs_dim"] or
                    obs[aid].contract != contract["observation_contract"] or
                    not np.array_equal(space.low, contract["action_low"]) or
                    not np.array_equal(space.high, contract["action_high"])):
                raise ValueError("MAPPO collector observation/action contract mismatch")
        space = env.action_spaces[ids[0]]
        repeat = int(env_cfg.get("action_repeat", 1))
        actions = ActionComposer.from_config(
            space.low, space.high, scenario["agents"][ids[0]].get("action_constraints", {}),
            decision_dt=float(env_cfg.get("timestep", .01)) * repeat,
        )
        agent = CollectorAgent(contract, horizon)
        recorder = None
        recording = scenario.get('recording', {})
        if recording.get('enabled', False):
            from src.replay.race_recorder import RaceRecorder
            recorder = RaceRecorder(recording, lambda event: sink.send(('race_record', event)),
                                    run_id=run_id, environment_id=env_id,
                                    num_envs=int(scenario['experiment'].get('num_envs', 1)))
        trainer = MARLTrainer(
            env, agent, ids, opponents, obs, rewards, actions, action_repeat=repeat,
            hooks=[_WorkerHook(sink, env_id, seed, record_transitions, aggregate_wandb)],
            run_id=f"{run_id}_env{env_id:04d}", reward_mode=agent.reward_mode,
            team_reward_reduction=agent.team_reward_reduction,
            race_recorder=recorder,
        )
        return env, agent, trainer.iter_train(episodes, parallel=True, total_steps=total_steps)
    except BaseException:
        env.close()
        raise


def _collect_worker(connection, scenario, scenario_dir, assignments, horizon,
                    contract, run_id, record_transitions, aggregate_wandb, step_budget=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYGLET_HEADLESS"] = "true"
    torch.set_num_threads(1)
    envs, agents, generators, pending = {}, {}, {}, {}
    sink = _EventSink()
    try:
        for env_id, quota in assignments:
            envs[env_id], agents[env_id], generators[env_id] = _make_collector(
                scenario, scenario_dir, env_id, 0 if step_budget else quota, horizon, contract, run_id,
                sink, record_transitions, aggregate_wandb,
                total_steps=quota if step_budget else None,
            )
        connection.send(("ready", len(envs)))
        if connection.recv() != ("start", None):
            raise RuntimeError("MAPPO collector expected start after readiness")

        def advance(env_id, response=None):
            try:
                pending[env_id] = generators[env_id].send(response)
            except StopIteration:
                pending.pop(env_id, None)

        for env_id in generators:
            advance(env_id)
        while pending:
            physics_start = sum(a.physics_steps_collected for a in agents.values())
            counts = {env_id: 0 for env_id in pending}
            paused = set()
            while set(pending) - paused:
                requests = {}
                for env_id in list(pending):
                    if env_id in paused:
                        continue
                    kind, payload = pending[env_id]
                    if kind == "step":
                        counts[env_id] += 1
                        if counts[env_id] >= horizon:
                            requests[env_id] = ("cut", payload)
                            continue
                        advance(env_id)
                        if env_id not in pending:
                            continue
                    requests[env_id] = pending[env_id]
                if not requests:
                    break
                connection.send(("requests", (requests, sink.take())))
                responses = connection.recv()
                for env_id, response in responses.items():
                    if requests[env_id][0] == "cut":
                        agents[env_id].finish_fragment(response)
                        paused.add(env_id)
                    else:
                        advance(env_id, response)
            fragments = [fragment for agent in agents.values() for fragment in agent.take_fragments()]
            pooled = None if not fragments else tuple(
                np.concatenate([fragment[i] for fragment in fragments]) for i in range(3)
            )
            physics = sum(a.physics_steps_collected for a in agents.values()) - physics_start
            connection.send(("rollout", (pooled, sum(counts.values()), physics, sink.take())))
            metrics = connection.recv()  # Policy update barrier, including an empty final rollout.
            for agent in agents.values():
                agent.policy_version = metrics["train/updates"]
                agent.recording_stop = metrics.get('recording/storage_full', False)
                agent.recording_progress = metrics['train/environment_steps']
                agent.recording_exhausted_windows = metrics.get('recording/exhausted_windows', [])
            for env_id in sorted(paused):
                advance(env_id)
        connection.send(("done", sink.take()))
    except (BrokenPipeError, EOFError, ConnectionResetError):
        pass
    except BaseException:
        _report_worker_error(connection)
    finally:
        for env in envs.values():
            env.close()
        connection.close()


def train_parallel(trainer, scenario, scenario_dir, num_envs, n_episodes=0, *, total_steps=None):
    import multiprocessing as mp

    experiment = scenario["experiment"]
    workers = min(num_envs, int(experiment.get("num_workers", num_envs)))
    horizon = int(scenario.get("training_defaults", {}).get("rollout_steps_per_env", 256))
    if min(workers, horizon) < 1:
        raise ValueError("Parallel MAPPO needs positive workers and horizon")
    if total_steps is not None and (isinstance(total_steps, bool)
            or not isinstance(total_steps, int) or total_steps < num_envs):
        raise ValueError("Parallel MAPPO total_steps must be an integer >= num_envs")
    if total_steps is None and n_episodes < num_envs:
        raise ValueError("Parallel MAPPO needs at least num_envs episodes")
    budget = total_steps if total_steps is not None else n_episodes
    startup = _worker_startup_settings(scenario)
    scheduler = CollectorScheduler(experiment.get("collector_scheduling", "synchronous"),
                                   startup["worker_response_timeout_s"])
    agent = trainer.agent
    contract = {name: getattr(agent, name) for name in (
        "agent_ids", "obs_dim", "global_state_dim", "global_state_contract_version",
        "action_dim", "action_low", "action_high", "observation_contract", "gamma",
        "gae_lambda", "critic_mode", "reward_mode", "team_return_mode", "team_reward_reduction",
    )}
    record_hooks = [h for h in trainer._transition_hooks if type(h) is not WandbHook]
    aggregate_wandb = any(type(h) is WandbHook for h in trainer._transition_hooks)
    race_hooks = [h for h in trainer.hooks if hasattr(h, 'on_race_record')]
    scenario = copy.deepcopy(scenario)
    # Enable worker capture only when the parent has a corresponding writer.
    scenario['recording'] = race_hooks[0].recording_config if race_hooks else {'enabled': False}
    connections, processes, process_by_worker = {}, [], {}
    completed, collected, actor_samples, updates = 0, 0, 0, 0
    physics_collected = 0
    pending_episodes = []
    success = False
    started_training = time.perf_counter()
    context = mp.get_context("spawn")
    # Set before spawning, so NumPy/BLAS/Numba imports inherit single-thread limits.
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in thread_vars}
    progress = CollectorProgress(getattr(trainer, 'console', None), workers=workers,
        environments=num_envs, horizon=horizon,
        interval=experiment.get('collector_progress_interval_s', 15.))
    dispatched = 0

    def receive(worker_id, starting=False):
        timeout = startup["worker_startup_timeout_s" if starting else "worker_response_timeout_s"]
        process = process_by_worker[worker_id]
        if not connections[worker_id].poll(timeout):
            raise RuntimeError(f"MAPPO worker {worker_id} timed out after {timeout}s "
                               f"(pid={process.pid}, exitcode={process.exitcode})")
        try:
            kind, payload = connections[worker_id].recv()
        except (EOFError, ConnectionResetError) as exc:
            raise RuntimeError(f"MAPPO worker {worker_id} disconnected (exitcode={process.exitcode})") from exc
        progress.received()
        if kind == "error":
            raise RuntimeError(f"MAPPO worker {worker_id} failed:\n{payload}")
        return kind, payload

    def process_events(items):
        for kind, payload in items:
            if kind == "transition":
                for hook in record_hooks:
                    hook.on_step(payload)
            elif kind == "episode":
                pending_episodes.append(payload)
            elif kind == 'race_record':
                for hook in race_hooks:
                    hook.on_race_record(payload)
            elif kind == 'episode_start':
                for hook in trainer.hooks:
                    callback = getattr(hook, 'on_episode_start', None)
                    if callback is not None:
                        callback(payload)
            else:
                raise RuntimeError(f"Unexpected MAPPO event: {kind}")

    def flush_episodes():
        nonlocal completed
        for reward, info, metrics in pending_episodes:
            race = metrics.get("race_record")
            if race is not None:
                race.update(run_id=trainer.run_id, environment_id=info["worker_id"],
                            reported_at_environment_steps=collected)
            metrics["train/environment_steps"] = collected
            metrics["train/physics_steps"] = physics_collected
            metrics["train/agent_steps"] = actor_samples
            metrics["train/updates"] = updates
            for hook in trainer.hooks:
                hook.on_episode_end(completed, reward, info, metrics)
            completed += 1
        pending_episodes.clear()

    def events(items):
        event_start = time.monotonic()
        try:
            process_events(items)
        finally:
            scheduler.exclude_parent_time(time.monotonic() - event_start)

    def publish_progress(*, force=False):
        # Telemetry I/O is parent work, not an unresponsive worker.
        started = time.monotonic()
        try:
            progress.publish(trainer.hooks, force=force)
        finally:
            scheduler.exclude_parent_time(time.monotonic() - started)

    try:
        progress.start()
        publish_progress(force=True)
        console = getattr(trainer, 'console', None)
        if console is not None:
            console.print_info(f"MAPPO starting {workers} workers / {num_envs} environments; "
                               f"horizon={horizon}, up to {num_envs * horizon:,} joint decisions per update")
        for key in thread_vars:
            os.environ[key] = "1"
        batch_size = startup["worker_startup_batch_size"]
        for start in range(0, workers, batch_size):
            batch = range(start, min(start + batch_size, workers))
            for worker_id in batch:
                assignments = [(i, budget // num_envs + (i < budget % num_envs))
                               for i in range(worker_id, num_envs, workers)]
                parent, child = context.Pipe()
                process = context.Process(
                    target=_collect_worker,
                    args=(child, scenario, str(scenario_dir), assignments, horizon,
                          contract, trainer.run_id, bool(record_hooks), aggregate_wandb, total_steps is not None),
                    name=f"mappo-collector-{worker_id}",
                )
                connections[worker_id] = parent
                try:
                    process.start()
                finally:
                    child.close()
                processes.append(process)
                process_by_worker[worker_id] = process
            for worker_id in batch:
                if receive(worker_id, starting=True)[0] != "ready":
                    raise RuntimeError("MAPPO worker did not report readiness")
                progress.set(ready_workers=worker_id + 1)
                publish_progress()
            if getattr(trainer, "console", None) is not None:
                trainer.console.print_info(f"MAPPO initialized {min(start + batch_size, workers)}/{workers} workers "
                                           f"for {num_envs} environments")
        for connection in connections.values():
            connection.send(("start", None))
        waiting = {}
        startup_s = time.perf_counter() - started_training
        round_start = time.perf_counter()
        progress.set(phase="collecting")
        publish_progress(force=True)
        while connections:
            requests = {}
            for worker_id in scheduler.workers(connections, waiting):
                if worker_id in waiting:
                    continue
                kind, payload = receive(worker_id)
                scheduler.received(worker_id)
                if kind == "requests":
                    batch, items = payload
                    events(items)
                    requests.update({(worker_id, env_id): request for env_id, request in batch.items()})
                elif kind == "rollout":
                    rollout, steps, physics, items = payload
                    events(items)
                    waiting[worker_id] = (rollout, steps, physics)
                    progress.set(waiting_workers=len(waiting))
                elif kind == "done":
                    events(payload)
                    connections.pop(worker_id).close()
                else:
                    raise RuntimeError(f"Unexpected MAPPO worker message: {kind}")
            if requests:
                progress.set(phase="inference")
                responses = infer_requests(agent, requests)
                for worker_id in sorted({key[0] for key in requests}):
                    connections[worker_id].send({env_id: response for (worker, env_id), response
                                                 in responses.items() if worker == worker_id})
                dispatched += sum(kind == "act" for kind, _ in requests.values())
                progress.set(phase="collecting", actions_dispatched=dispatched)
            publish_progress()
            if waiting and len(waiting) == len(connections):
                collection_s = time.perf_counter() - round_start
                steps = sum(item[1] for item in waiting.values())
                rollouts = [waiting[i][0] for i in sorted(waiting) if waiting[i][0] is not None]
                samples = sum(len(item[0]) for item in rollouts)
                collected += steps
                physics_collected += sum(item[2] for item in waiting.values())
                actor_samples += samples
                trainer._environment_steps = collected
                progress.set(phase="episode_logging")
                publish_progress(force=True)
                flush_episodes()
                progress.set(phase="updating")
                publish_progress(force=True)
                started = time.perf_counter()
                metrics = agent.update_rollouts(rollouts) if samples else {}
                update_s = time.perf_counter() - started
                updates += bool(samples)
                metrics.update({
                    "train/environment_steps": collected, "train/agent_steps": actor_samples,
                    "train/physics_steps": physics_collected,
                    "train/updates": updates, "perf/collection_seconds": collection_s,
                    "perf/update_seconds": update_s, "perf/collection_env_steps_per_second": steps / max(collection_s, 1e-9),
                    "perf/round_env_steps_per_second": steps / max(collection_s + update_s, 1e-9),
                    "train/rollout_agent_samples": samples,
                    "perf/startup_seconds": startup_s,
                    "perf/elapsed_seconds": time.perf_counter() - started_training,
                    "perf/end_to_end_env_steps_per_second": collected / max(time.perf_counter() - started_training, 1e-9),
                    "perf/num_workers": workers, "perf/num_envs": num_envs,
                })
                if race_hooks:
                    metrics['recording/storage_full'] = any(h.storage_full for h in race_hooks)
                    metrics['recording/exhausted_windows'] = sorted(set().union(*(h.exhausted_windows for h in race_hooks)))
                progress.set(phase="evaluation_checkpoint_logging",
                             updated_environment_steps=collected, updates=updates)
                publish_progress(force=True)
                if steps:
                    for hook in trainer.hooks:
                        hook.on_update(metrics)
                for worker_id in waiting:
                    connections[worker_id].send(metrics)
                waiting.clear()
                scheduler.reset()
                round_start = time.perf_counter()
                progress.set(phase="collecting", waiting_workers=0)
                publish_progress(force=True)
        flush_episodes()
        if total_steps is not None and collected != total_steps:
            raise RuntimeError(f"MAPPO collectors collected {collected} of {total_steps} environment decisions")
        if total_steps is None and completed != n_episodes:
            raise RuntimeError(f"MAPPO collectors completed {completed} of {n_episodes} episodes")
        progress.set(phase="final_checkpoint")
        publish_progress(force=True)
        for hook in trainer.hooks:
            hook.on_training_end()
        progress.set(phase="finished")
        success = True
    finally:
        progress.close()
        _close_collectors(list(connections.values()), processes, failed=not success)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
