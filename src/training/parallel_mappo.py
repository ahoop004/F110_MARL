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
            self.fragments.append((packed.numpy(), buf.raw_actions[:n].numpy().copy()))
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
        actions, log_probs, raw = agent.actor.get_action(obs_t, return_raw=True)
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
                    run_id, sink, record_transitions, aggregate_wandb):
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
        trainer = MARLTrainer(
            env, agent, ids, opponents, obs, rewards, actions, action_repeat=repeat,
            hooks=[_WorkerHook(sink, env_id, seed, record_transitions, aggregate_wandb)],
            run_id=f"{run_id}_env{env_id:04d}", reward_mode=agent.reward_mode,
            team_reward_reduction=agent.team_reward_reduction,
        )
        return env, agent, trainer.iter_train(episodes, parallel=True)
    except BaseException:
        env.close()
        raise


def _collect_worker(connection, scenario, scenario_dir, assignments, horizon,
                    contract, run_id, record_transitions, aggregate_wandb):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYGLET_HEADLESS"] = "true"
    torch.set_num_threads(1)
    envs, agents, generators, pending = {}, {}, {}, {}
    sink = _EventSink()
    try:
        for env_id, episodes in assignments:
            envs[env_id], agents[env_id], generators[env_id] = _make_collector(
                scenario, scenario_dir, env_id, episodes, horizon, contract, run_id,
                sink, record_transitions, aggregate_wandb,
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
                np.concatenate([fragment[i] for fragment in fragments]) for i in range(2)
            )
            connection.send(("rollout", (pooled, sum(counts.values()), sink.take())))
            connection.recv()  # Policy update barrier, including an empty final rollout.
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


def train_parallel(trainer, scenario, scenario_dir, num_envs, n_episodes):
    import multiprocessing as mp

    experiment = scenario["experiment"]
    workers = min(num_envs, int(experiment.get("num_workers", num_envs)))
    horizon = int(scenario.get("training_defaults", {}).get("rollout_steps_per_env", 256))
    if min(workers, horizon) < 1 or n_episodes < num_envs:
        raise ValueError("Parallel MAPPO needs positive workers/horizon and at least num_envs episodes")
    startup = _worker_startup_settings(scenario)
    agent = trainer.agent
    contract = {name: getattr(agent, name) for name in (
        "agent_ids", "obs_dim", "global_state_dim", "global_state_contract_version",
        "action_dim", "action_low", "action_high", "observation_contract", "gamma",
        "gae_lambda", "critic_mode", "reward_mode", "team_return_mode", "team_reward_reduction",
    )}
    record_hooks = [h for h in trainer._transition_hooks if type(h) is not WandbHook]
    aggregate_wandb = any(type(h) is WandbHook for h in trainer._transition_hooks)
    connections, processes, process_by_worker = {}, [], {}
    completed, collected, actor_samples, updates = 0, 0, 0, 0
    success = False
    context = mp.get_context("spawn")
    # Set before spawning, so NumPy/BLAS/Numba imports inherit single-thread limits.
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in thread_vars}

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
        if kind == "error":
            raise RuntimeError(f"MAPPO worker {worker_id} failed:\n{payload}")
        return kind, payload

    def events(items):
        nonlocal completed
        for kind, payload in items:
            if kind == "transition":
                for hook in record_hooks:
                    hook.on_step(payload)
            elif kind == "episode":
                for hook in trainer.hooks:
                    hook.on_episode_end(completed, *payload)
                completed += 1
            else:
                raise RuntimeError(f"Unexpected MAPPO event: {kind}")

    try:
        for key in thread_vars:
            os.environ[key] = "1"
        batch_size = startup["worker_startup_batch_size"]
        for start in range(0, workers, batch_size):
            batch = range(start, min(start + batch_size, workers))
            for worker_id in batch:
                assignments = [(i, n_episodes // num_envs + (i < n_episodes % num_envs))
                               for i in range(worker_id, num_envs, workers)]
                parent, child = context.Pipe()
                process = context.Process(
                    target=_collect_worker,
                    args=(child, scenario, str(scenario_dir), assignments, horizon,
                          contract, trainer.run_id, bool(record_hooks), aggregate_wandb),
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
            print(f"[MAPPO] Initialized {min(start + batch_size, workers)}/{workers} workers "
                  f"for {num_envs} environments", flush=True)
        for connection in connections.values():
            connection.send(("start", None))
        waiting = {}
        round_start = time.perf_counter()
        while connections:
            requests = {}
            for worker_id in list(connections):
                if worker_id in waiting:
                    continue
                kind, payload = receive(worker_id)
                if kind == "requests":
                    batch, items = payload
                    events(items)
                    requests.update({(worker_id, env_id): request for env_id, request in batch.items()})
                elif kind == "rollout":
                    rollout, steps, items = payload
                    events(items)
                    waiting[worker_id] = (rollout, steps)
                elif kind == "done":
                    events(payload)
                    connections.pop(worker_id).close()
                else:
                    raise RuntimeError(f"Unexpected MAPPO worker message: {kind}")
            if requests:
                responses = infer_requests(agent, requests)
                for worker_id in sorted({key[0] for key in requests}):
                    connections[worker_id].send({env_id: response for (worker, env_id), response
                                                 in responses.items() if worker == worker_id})
            if waiting and len(waiting) == len(connections):
                collection_s = time.perf_counter() - round_start
                steps = sum(item[1] for item in waiting.values())
                rollouts = [waiting[i][0] for i in sorted(waiting) if waiting[i][0] is not None]
                samples = sum(len(item[0]) for item in rollouts)
                started = time.perf_counter()
                metrics = agent.update_rollouts(rollouts)
                update_s = time.perf_counter() - started
                collected += steps
                actor_samples += samples
                updates += bool(samples)
                trainer._environment_steps = collected
                metrics.update({
                    "train/environment_steps": collected, "train/agent_steps": actor_samples,
                    "train/updates": updates, "perf/collection_seconds": collection_s,
                    "perf/update_seconds": update_s, "perf/collection_env_steps_per_second": steps / max(collection_s, 1e-9),
                    "perf/round_env_steps_per_second": steps / max(collection_s + update_s, 1e-9),
                    "train/rollout_agent_samples": samples,
                })
                for hook in trainer.hooks:
                    hook.on_update(metrics)
                print(f"[MAPPO] update={updates} env_steps={collected} samples={samples} "
                      f"collect={collection_s:.1f}s update={update_s:.1f}s "
                      f"env_steps/s={metrics['perf/round_env_steps_per_second']:.1f}", flush=True)
                for worker_id in waiting:
                    connections[worker_id].send(metrics)
                waiting.clear()
                round_start = time.perf_counter()
        if completed != n_episodes:
            raise RuntimeError(f"MAPPO collectors completed {completed} of {n_episodes} episodes")
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
