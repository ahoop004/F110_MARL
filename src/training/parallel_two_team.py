"""Grouped CPU self-play collectors with two centrally updated policies.

Workers own race state and rollout buffers only. The parent batches inference
per team and waits for every collector before updating either policy. Each
race/team fragment is bootstrapped before pooling, including inactive-team
critic targets. Evaluation and W&B remain exclusively in the parent.
"""
from __future__ import annotations

from copy import deepcopy
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from training.parallel_mappo import CollectorAgent, infer_requests
from training.on_policy_trainer import _close_collectors, _report_worker_error, _worker_startup_settings
from training.two_team import TwoTeamTrainer, update_inactive_values


CONTRACT_FIELDS = (
    "agent_ids", "obs_dim", "global_state_dim", "global_state_contract_version",
    "action_dim", "action_low", "action_high", "observation_contract", "gamma",
    "gae_lambda", "critic_mode", "reward_mode", "team_return_mode", "team_reward_reduction",
)


def _peak_rss_mib():
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


class TeamCollector(CollectorAgent):
    def __init__(self, contract, horizon):
        super().__init__(contract, horizon)
        self.value_fragments = []

    def finish_team_fragment(self, next_value, states):
        if len(states) != len(self._team_rollout):
            raise ValueError("Team rollout states and rewards must align")
        if not states:
            return
        active_steps = {index for indices in self._team_step_indices.values() for index in indices}
        inactive = [i for i in range(len(states)) if i not in active_steps]
        if inactive:
            returns = self.compute_team_gae(next_value)[1].numpy()
            self.value_fragments.append(np.column_stack((np.asarray(states)[inactive], returns[inactive])))
        super().finish_fragment({aid: next_value for aid in self.agent_ids})
        # The base collector has no actor fragment to clear when both cars were
        # inactive for the entire fragment. Its critic targets still count.
        self.clear_buffers()

    def take_value_fragments(self):
        result, self.value_fragments = self.value_fragments, []
        return result


def infer_team_requests(agents, requests):
    """Route repeated car IDs in different races through the correct policy."""
    result = {key: ({}, {}, {}, {}, {}) if kind == "act" else {}
              for key, (kind, _) in requests.items()}
    for team, agent in agents.items():
        routed = {}
        for key, (kind, payload) in requests.items():
            if kind == "act":
                observations, state = payload
                ids = [aid for aid in agent.agent_ids if aid in observations]
                routed[key] = (("act", (ids, np.stack([observations[aid] for aid in ids]), state))
                               if ids else ("value", state))
            else:
                routed[key] = ("value", payload)
        responses = infer_requests(agent, routed)
        for key, response in responses.items():
            if requests[key][0] != "act":
                result[key][team] = response[agent.agent_ids[0]]
            elif routed[key][0] == "value":
                # No dummy actions for an eliminated team, but V_team(s) is
                # still needed for delayed opponent-crash rewards.
                result[key][4][team] = response[agent.agent_ids[0]]
            else:
                for index in range(4):
                    result[key][index].update(response[index])
                result[key][4][team] = next(iter(response[2].values()))
    return result


def _make_collector(scenario, scenario_dir, env_id, horizon, contracts, teams):
    from core.setup import create_training_setup
    from run import build_obs_composers, build_reward_composers
    from wrappers.actions.composer import ActionComposer

    scenario = deepcopy(scenario)
    seed = int(scenario["experiment"]["seed"])
    scenario["experiment"]["seed"] = (seed + env_id) % (2 ** 32)
    cfg = scenario["environment"]
    cfg["seed"] = ((seed if cfg.get("seed") is None else int(cfg["seed"])) + env_id) % (2 ** 32)
    cfg["render"] = False
    env, _, _ = create_training_setup(scenario, scenario_dir=Path(scenario_dir))
    try:
        ids = [aid for members in teams.values() for aid in members]
        observations = build_obs_composers(scenario["agents"], ids, cfg, Path(scenario_dir))
        rewards = build_reward_composers(scenario["agents"], ids, Path(scenario_dir))
        actions = {aid: ActionComposer.from_config(env.action_spaces[aid].low, env.action_spaces[aid].high,
            scenario["agents"][aid].get("action_constraints", {}), decision_dt=env.timestep) for aid in ids}
        snapshot = env.get_global_state()
        for team, members in teams.items():
            contract = contracts[team]
            if (len(snapshot.vector) + 1 != contract["global_state_dim"] or
                    str(snapshot.metadata.get("vector_contract_version")) + "+race_clock_v1" !=
                    contract["global_state_contract_version"]):
                raise ValueError("Two-team collector global-state contract mismatch")
            for aid in members:
                space = env.action_spaces[aid]
                if (observations[aid].obs_dim + 2 != contract["obs_dim"] or
                        observations[aid].contract != contract["observation_contract"]["base"] or
                        not np.array_equal(space.low, contract["action_low"]) or
                        not np.array_equal(space.high, contract["action_high"])):
                    raise ValueError("Two-team collector observation/action contract mismatch")
        agents = {team: TeamCollector(contract, horizon) for team, contract in contracts.items()}
        return TwoTeamTrainer(env=env, teams=teams, agents=agents, observations=observations,
            rewards=rewards, actions=actions, event_config=scenario["two_team"].get("events"))
    except BaseException:
        env.close()
        raise


def _pool(fragments):
    if not fragments:
        return None
    return tuple(np.concatenate([fragment[i] for fragment in fragments]) for i in range(3))


def _collect_worker(connection, scenario, scenario_dir, assignments, horizon, contracts, teams, step_budget,
                    recording=None, run_id='run'):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYGLET_HEADLESS"] = "true"
    torch.set_num_threads(1)
    trainers, generators, pending, episodes = {}, {}, {}, {}
    events = []
    race_events = []
    quotas = dict(assignments)
    try:
        for env_id, _ in assignments:
            trainers[env_id] = _make_collector(scenario, scenario_dir, env_id, horizon, contracts, teams)
            trainers[env_id].run_id, trainers[env_id].environment_id = run_id, env_id
            if recording is not None:
                from src.replay.race_recorder import RaceRecorder
                trainers[env_id].race_recorder = RaceRecorder(recording, race_events.append,
                    run_id=run_id, environment_id=env_id)
            episodes[env_id] = 0
        connection.send(("ready", len(trainers)))
        if connection.recv() != ("start", None):
            raise RuntimeError("Two-team collector expected start after readiness")

        def advance(env_id, response=None):
            trainer = trainers[env_id]
            while True:
                if env_id not in generators:
                    remaining = quotas[env_id] - trainer.environment_steps if step_budget else None
                    if (step_budget and remaining <= 0) or (not step_budget and episodes[env_id] >= quotas[env_id]):
                        pending.pop(env_id, None)
                        return
                    generators[env_id] = trainer.iter_episode(parallel=True, step_budget=remaining)
                try:
                    pending[env_id] = generators[env_id].send(response)
                    return
                except StopIteration as finished:
                    episodes[env_id] += 1
                    events.append({**finished.value, "environment_id": env_id,
                                   "environment_steps": trainer.environment_steps,
                                   "seed": trainers[env_id].env.seed})
                    generators.pop(env_id)
                    response = None

        for env_id in trainers:
            advance(env_id)
        while pending:
            counts = {env_id: 0 for env_id in pending}
            actor_start = sum(t.agent_steps for t in trainers.values())
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
                if race_events:
                    connection.send(('race_records', race_events))
                    race_events.clear()
                connection.send(("requests", requests))
                responses = connection.recv()
                for env_id, response in responses.items():
                    if requests[env_id][0] == "cut":
                        trainers[env_id].finish_fragment(response)
                        paused.add(env_id)
                    else:
                        advance(env_id, response)
            rollouts = {}
            for team in teams:
                actor = [fragment for trainer in trainers.values()
                         for fragment in trainer.agents[team].take_fragments()]
                value = [fragment for trainer in trainers.values()
                         for fragment in trainer.agents[team].take_value_fragments()]
                rollouts[team] = (_pool(actor), np.concatenate(value) if value else None)
            actor_steps = sum(t.agent_steps for t in trainers.values()) - actor_start
            if race_events:
                connection.send(('race_records', race_events))
                race_events.clear()
            connection.send(("rollout", (rollouts, sum(counts.values()), actor_steps, events, _peak_rss_mib())))
            events = []
            command, state = connection.recv()
            if command != 'continue':
                raise RuntimeError("Two-team collector expected policy update barrier")
            for collector in trainers.values():
                collector.updates = state['policy_version']
                collector.recording_stop = state['recording_stop']
            for env_id in sorted(paused):
                advance(env_id)
        if race_events:
            connection.send(('race_records', race_events))
        connection.send(("done", events))
    except (BrokenPipeError, EOFError, ConnectionResetError):
        pass
    except BaseException:
        _report_worker_error(connection)
    finally:
        for trainer in trainers.values():
            trainer.env.close()
        connection.close()


def train_parallel(trainer, scenario, scenario_dir, *, on_episode, console=None):
    """Collect exact aggregate steps/episodes; update both policies at a barrier."""
    experiment = scenario["experiment"]
    race_writer = getattr(trainer, 'race_writer', None)
    num_envs = int(experiment["num_envs"])
    workers = min(num_envs, int(experiment.get("num_workers", num_envs)))
    horizon = int(scenario.get("training_defaults", {}).get("rollout_steps_per_env", 256))
    total_steps = experiment.get("total_steps")
    budget = int(total_steps if total_steps is not None else experiment.get("episodes", 5000))
    if min(num_envs, workers, horizon) < 1 or budget < num_envs:
        raise ValueError("Parallel self-play requires positive workers/horizon and a budget >= num_envs")
    startup = _worker_startup_settings(scenario)
    contracts = {team: {name: getattr(agent, name) for name in CONTRACT_FIELDS}
                 for team, agent in trainer.agents.items()}
    context = mp.get_context("spawn")
    connections, processes, process_by_worker, waiting = {}, [], {}, {}
    completed = 0
    worker_memory = {}
    success = False
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in thread_vars}

    def receive(worker_id, starting=False):
        timeout = startup["worker_startup_timeout_s" if starting else "worker_response_timeout_s"]
        process = process_by_worker[worker_id]
        if not connections[worker_id].poll(timeout):
            raise RuntimeError(f"Two-team worker {worker_id} timed out after {timeout}s "
                               f"(pid={process.pid}, exitcode={process.exitcode})")
        try:
            kind, payload = connections[worker_id].recv()
            while kind == 'race_records':
                if race_writer is None:
                    raise RuntimeError('Worker sent recording data without a writer')
                for event in payload:
                    race_writer.add_event(event)
                if not connections[worker_id].poll(timeout):
                    raise RuntimeError(f'Two-team worker {worker_id} timed out after recording transfer')
                kind, payload = connections[worker_id].recv()
        except (EOFError, ConnectionResetError) as exc:
            raise RuntimeError(f"Two-team worker {worker_id} disconnected (exitcode={process.exitcode})") from exc
        if kind == "error":
            raise RuntimeError(f"Two-team worker {worker_id} failed:\n{payload}")
        return kind, payload

    def report(rows):
        nonlocal completed
        for row in rows:
            completed += bool(row["completed"])
            on_episode(row)

    try:
        for key in thread_vars:
            os.environ[key] = "1"
        batch_size = startup["worker_startup_batch_size"]
        for start in range(0, workers, batch_size):
            batch = range(start, min(start + batch_size, workers))
            for worker_id in batch:
                assignments = [(i, budget // num_envs + (i < budget % num_envs))
                               for i in range(worker_id, num_envs, workers)]
                parent, child = context.Pipe()
                process = context.Process(target=_collect_worker,
                    args=(child, scenario, str(scenario_dir), assignments, horizon, contracts,
                          trainer.teams, total_steps is not None,
                          race_writer.config if race_writer is not None else None,
                          getattr(trainer, 'run_id', 'run')), name=f"two-team-collector-{worker_id}")
                connections[worker_id] = parent
                try:
                    process.start()
                finally:
                    child.close()
                processes.append(process)
                process_by_worker[worker_id] = process
            for worker_id in batch:
                if receive(worker_id, starting=True)[0] != "ready":
                    raise RuntimeError("Two-team worker did not report readiness")
            if console:
                console.print_info(f"Self-play initialized {min(start + batch_size, workers)}/{workers} workers for {num_envs} environments")
        for connection in connections.values():
            connection.send(("start", None))
        round_start = time.perf_counter()
        while connections:
            requests = {}
            for worker_id in list(connections):
                if worker_id in waiting:
                    continue
                kind, payload = receive(worker_id)
                if kind == "requests":
                    requests.update({(worker_id, env_id): request for env_id, request in payload.items()})
                elif kind == "rollout":
                    waiting[worker_id] = payload
                    worker_memory[worker_id] = payload[4]
                elif kind == "done":
                    report(payload)
                    connections.pop(worker_id).close()
                else:
                    raise RuntimeError(f"Unexpected two-team worker message: {kind}")
            if requests:
                responses = infer_team_requests(trainer.agents, requests)
                for worker_id in sorted({key[0] for key in requests}):
                    connections[worker_id].send({env_id: response for (worker, env_id), response
                                                in responses.items() if worker == worker_id})
            if waiting and len(waiting) == len(connections):
                collection_s = time.perf_counter() - round_start
                steps = sum(item[1] for item in waiting.values())
                trainer.environment_steps += steps
                trainer.agent_steps += sum(item[2] for item in waiting.values())
                for worker_id in sorted(waiting):
                    report(waiting[worker_id][3])
                started = time.perf_counter()
                metrics = {}
                samples = 0
                for team, agent in trainer.agents.items():
                    actor = [waiting[i][0][team][0] for i in sorted(waiting)
                             if waiting[i][0][team][0] is not None]
                    value = [waiting[i][0][team][1] for i in sorted(waiting)
                             if waiting[i][0][team][1] is not None]
                    team_metrics = {key.removeprefix("train/"): v for key, v in agent.update_rollouts(actor).items()}
                    if value:
                        value = np.concatenate(value)
                        team_metrics.update(update_inactive_values(agent, value[:, :-1], value[:, -1]))
                    team_samples = sum(len(item[0]) for item in actor)
                    samples += team_samples
                    metrics.update({f"train/{team}/{key}": v for key, v in team_metrics.items()})
                    metrics[f"train/{team}/rollout_agent_samples"] = team_samples
                update_s = time.perf_counter() - started
                if steps:
                    trainer.updates += 1
                    metrics.update({"train/environment_steps": trainer.environment_steps,
                        "train/agent_steps": trainer.agent_steps, "train/update": trainer.updates,
                        "train/rollout_agent_samples": samples,
                        "perf/collection_seconds": collection_s, "perf/update_seconds": update_s,
                        "perf/collector_peak_rss_mib": sum(worker_memory.values()),
                        "perf/parent_peak_rss_mib": _peak_rss_mib(),
                        "perf/collection_env_steps_per_second": steps / max(collection_s, 1e-9),
                        "perf/round_env_steps_per_second": steps / max(collection_s + update_s, 1e-9)})
                    if race_writer is not None:
                        metrics['recording/storage_full'] = int(race_writer.storage_full)
                    if trainer.on_update:
                        trainer.on_update(metrics)
                    every = int(experiment.get("terminal_every_updates", 10))
                    if console and (trainer.updates == 1 or trainer.updates % max(every, 1) == 0):
                        console.print_info(f"Self-play update={trainer.updates} env_steps={trainer.environment_steps:,} "
                            f"samples={samples:,} collect={collection_s:.2f}s update={update_s:.2f}s "
                            f"env_steps/s={metrics['perf/round_env_steps_per_second']:.1f}")
                for worker_id in waiting:
                    connections[worker_id].send(('continue', dict(policy_version=trainer.updates,
                        recording_stop=race_writer.storage_full if race_writer is not None else False)))
                waiting.clear()
                round_start = time.perf_counter()
        actual = trainer.environment_steps if total_steps is not None else completed
        if actual != budget:
            raise RuntimeError(f"Two-team collectors completed {actual} of {budget} requested units")
        success = True
    finally:
        _close_collectors(list(connections.values()), processes, failed=not success)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
