"""On-policy training loop for PPO."""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agents.ppo import PPOAgent
from env.types import GlobalState, TransitionRecord
from metrics.outcomes import determine_outcome
from training.hooks import TrainingHook, WandbHook, transition_record_hooks
from training.reward_context import build_reward_context, transition_lifecycle_fields
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


class OnPolicyTrainer:
    """Episode-based training loop for on-policy RL agents.

    Handles:
    - Multi-agent env with one RL agent + N fixed-policy opponents
    - Action repeat (step env K times per decision)
    - Obs composition via ObservationComposer
    - Reward computation via RewardComposer
    - Action processing via ActionComposer (denormalize + constraints)
    - Rollout buffer fill → PPO update
    - Hook callbacks at episode/update boundaries
    """

    def __init__(
        self,
        env: Any,
        rl_agent_id: str,
        agent: PPOAgent,
        other_agents: Dict[str, Any],
        obs_composer: ObservationComposer,
        reward_composer: RewardComposer,
        action_composer: ActionComposer,
        action_repeat: int = 1,
        hooks: Optional[List[TrainingHook]] = None,
        render: bool = False,
        run_id: str = "run",
        spawn_plan_fn: Optional[Callable] = None,
    ) -> None:
        self.env = env
        self.rl_agent_id = rl_agent_id
        self.agent = agent
        self.other_agents = other_agents
        self.obs_composer = obs_composer
        self.reward_composer = reward_composer
        if getattr(reward_composer, "team_contract", []):
            raise ValueError("Team race rewards require MAPPO joint team returns")
        self.action_composer = action_composer
        self.action_repeat = max(1, int(action_repeat))
        self.hooks = hooks or []
        self._transition_hooks = transition_record_hooks(self.hooks)
        self.render = render
        self.run_id = run_id
        self.spawn_plan_fn = spawn_plan_fn
        self.collected_steps = 0

    def _set_training_progress(self, completed: int, total: int) -> None:
        # Remote collectors own no optimizer; the parent uses global progress.
        setter = getattr(self.agent, "set_training_progress", None)
        if setter is not None:
            setter(completed / max(total, 1))

    def train_parallel(self, scenario: Dict, scenario_dir, num_envs: int, n_episodes: int, *, total_steps: Optional[int] = None) -> None:
        """Batch CPU collector requests; update only when all live workers pause.

        n_steps is the pooled collection-round size, divided evenly across workers.
        With a transition budget, resets do not flush buffers; only a full rollout
        or the final budget remainder triggers an update.
        """
        workers = min(num_envs, int(scenario.get("experiment", {}).get("num_workers", num_envs)))
        if workers < num_envs or scenario.get("experiment", {}).get("collector_scheduling") == "ready":
            from training.parallel_ppo import train_parallel
            return train_parallel(self, scenario, scenario_dir, num_envs, n_episodes,
                                  total_steps=total_steps)
        import multiprocessing as mp

        context = mp.get_context("spawn")
        connections, processes = {}, []
        process_by_worker = {}
        startup = _worker_startup_settings(scenario)
        completed_normally = False
        started = time.perf_counter()
        thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
        previous = {key: os.environ.get(key) for key in thread_vars}
        waiting = {}
        completed = 0
        collected = 0
        stopped = False
        if total_steps is not None and total_steps < num_envs:
            raise ValueError("total_steps must be at least num_envs")
        self._set_training_progress(0, total_steps or n_episodes)
        # Standard W&B needs only episode totals. Dataset/custom hooks retain
        # the full transition stream, including custom WandbHook subclasses.
        record_hooks = [h for h in self._transition_hooks if type(h) is not WandbHook]
        aggregate_wandb = any(type(h) is WandbHook for h in self._transition_hooks)

        def receive(worker_id, *, starting=False):
            connection = connections[worker_id]
            timeout = startup["worker_startup_timeout_s"] if starting else startup["worker_response_timeout_s"]
            phase = "startup" if starting else "collection"
            if not connection.poll(timeout):
                process = process_by_worker[worker_id]
                raise RuntimeError(
                    f"PPO worker {worker_id} timed out during {phase} after {timeout}s "
                    f"(pid={getattr(process, 'pid', None)}, exitcode={getattr(process, 'exitcode', None)}). "
                    "Check system/container or scheduler logs for resource-limit kills and available CPU/RAM. "
                    + ("Startup batch size and timeout are configured in experiment.worker_startup_*."
                       if starting else "Response timeout: experiment.worker_response_timeout_s.")
                )
            try:
                kind, payload = connection.recv()
            except (EOFError, ConnectionResetError) as exc:
                process = process_by_worker[worker_id]
                raise RuntimeError(
                    f"PPO worker {worker_id} disconnected during {phase} "
                    f"(exitcode={getattr(process, 'exitcode', None)}). Check system/container or scheduler logs."
                ) from exc
            if kind == "error":
                raise RuntimeError(f"PPO worker {worker_id} failed:\n{payload}")
            return kind, payload

        try:
            for key in thread_vars:
                os.environ[key] = "1"
            batch_size = startup["worker_startup_batch_size"]
            for start in range(0, num_envs, batch_size):
                batch = range(start, min(start + batch_size, num_envs))
                for worker_id in batch:
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=_collect_ppo_worker,
                        args=(child, scenario, str(scenario_dir), self.rl_agent_id, worker_id,
                              n_episodes // num_envs + (worker_id < n_episodes % num_envs),
                              self.agent.n_steps // num_envs, self.run_id, self.agent.gamma,
                              self.agent.gae_lambda, bool(record_hooks), aggregate_wandb,
                              None if total_steps is None else
                              total_steps // num_envs + (worker_id < total_steps % num_envs)),
                        name=f"ppo-collector-{worker_id}",
                    )
                    connections[worker_id] = parent
                    try:
                        process.start()
                    finally:
                        child.close()
                    processes.append(process)
                    process_by_worker[worker_id] = process
                for worker_id in batch:
                    kind, contract = receive(worker_id, starting=True)
                    if (kind != "ready" or contract[0] != self.agent.obs_dim
                            or not np.array_equal(contract[1], self.agent.action_low)
                            or not np.array_equal(contract[2], self.agent.action_high)):
                        raise ValueError(f"PPO worker {worker_id} observation/action contract mismatch.")
                print(f"[PPO] Initialized {min(start + batch_size, num_envs)}/{num_envs} workers", flush=True)
            # Workers remain idle after readiness until every startup batch is ready.
            for connection in connections.values():
                connection.send(("start", None))

            startup_s = time.perf_counter() - started
            round_start = time.perf_counter()
            while connections:
                requests = {}
                value_requests = {}
                # Fixed worker order makes sampling and episode event order
                # independent of OS scheduling and response arrival order.
                for worker_id in list(connections):
                    if worker_id in waiting:
                        continue
                    while True:
                        kind, payload = receive(worker_id)
                        if kind == "transition":
                            for hook in record_hooks:
                                hook.on_step(payload)
                        elif kind == "episode_start":
                            for hook in self.hooks:
                                hook.on_episode_start(payload)
                        elif kind == "episode":
                            if total_steps is None:
                                self._set_training_progress(completed + 1, n_episodes)
                            for hook in self.hooks:
                                hook.on_episode_end(completed, *payload)
                            completed += 1
                        elif kind == "value":
                            value_requests[worker_id] = payload
                            break
                        elif kind == "act":
                            requests[worker_id] = payload
                            break
                        elif kind == "rollout":
                            waiting[worker_id] = payload
                            collected += len(payload[0])
                            self.collected_steps = collected
                            break
                        elif kind == "done":
                            connections.pop(worker_id).close()
                            break
                        else:
                            raise RuntimeError(f"Unexpected PPO worker message: {kind}")
                if value_requests:
                    values = self.agent.value_batch(np.stack(list(value_requests.values())))
                    for row, worker_id in enumerate(value_requests):
                        connections[worker_id].send(float(values[row]))
                if requests:
                    actions, log_probs, values = self.agent.act_batch(np.stack(list(requests.values())))
                    for row, worker_id in enumerate(requests):
                        connections[worker_id].send((actions[row], float(log_probs[row]), float(values[row]),
                                                     self.agent.last_raw_actions[row]))
                if waiting and len(waiting) == len(connections):
                    if total_steps is not None:
                        self._set_training_progress(collected, total_steps)
                    collection_s = time.perf_counter() - round_start
                    steps = sum(len(item[0]) for item in waiting.values())
                    update_start = time.perf_counter()
                    metrics = self.agent.update_rollouts([waiting[i] for i in sorted(waiting)])
                    update_s = time.perf_counter() - update_start
                    if metrics:
                        metrics.update({
                            "perf/startup_seconds": startup_s,
                            "perf/collection_seconds": collection_s, "perf/update_seconds": update_s,
                            "perf/collection_env_steps_per_second": steps / max(collection_s, 1e-9),
                            "perf/round_env_steps_per_second": steps / max(collection_s + update_s, 1e-9),
                            "perf/elapsed_seconds": time.perf_counter() - started,
                            "perf/end_to_end_env_steps_per_second": collected / max(time.perf_counter() - started, 1e-9),
                            "perf/num_workers": num_envs, "perf/num_envs": num_envs,
                        })
                        metrics["train/environment_steps"] = collected
                        for hook in self.hooks:
                            hook.on_update(metrics)
                    stopped = self._should_stop()
                    reply = metrics
                    if scenario.get("map_curriculum") is not None or stopped:
                        control = {"stop": stopped}
                        if scenario.get("map_curriculum") is not None:
                            control["training_bundles"] = self.env._map_scheduler.training_bundles
                        reply = {"metrics": metrics, "collector_control": control}
                    for worker_id in waiting:
                        connections[worker_id].send(reply)
                    waiting.clear()
                    round_start = time.perf_counter()
            if not stopped and total_steps is not None and collected != total_steps:
                raise RuntimeError(f"PPO collected {collected} of {total_steps} transitions")
            if not stopped and total_steps is None and completed != n_episodes:
                raise RuntimeError(f"PPO workers completed {completed} of {n_episodes} episodes.")
            if not stopped:
                self._flush_pending_update()
            for hook in self.hooks:
                hook.on_training_end()
            completed_normally = True
        finally:
            _close_collectors(connections.values(), processes, failed=not completed_normally)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _build_actions(
        self,
        rl_action_phys: np.ndarray,
        obs_dict: Dict,
    ) -> Dict[str, np.ndarray]:
        active = set(getattr(self.env, "agents", obs_dict))
        actions: Dict[str, np.ndarray] = {}
        if self.rl_agent_id in active:
            actions[self.rl_agent_id] = rl_action_phys
        for aid, other_agent in self.other_agents.items():
            if aid in active:
                try:
                    act = other_agent.act(obs_dict[aid])
                except Exception:
                    act = np.zeros(2, dtype=np.float32)
                actions[aid] = np.asarray(act, dtype=np.float32)
        return actions

    def _episode_id(self, episode: int) -> str:
        return f"{self.run_id}_ep{episode:06d}"

    def _map_id(self) -> Optional[str]:
        return getattr(self.env, "_map_bundle_active", None) or getattr(
            self.env, "map_name", None
        )

    def _spawn_id(self) -> Optional[str]:
        sm = getattr(self.env, "_spawn_manager", None)
        if sm is None:
            return None
        meta = getattr(sm, "last_spawn_metadata", {}) or {}
        spawn_ids = meta.get("spawn_ids", {})
        return spawn_ids.get(self.rl_agent_id) or meta.get("spawn_id")

    def _reset_env(self) -> tuple:
        """Reset env, injecting a curriculum spawn plan when one is available."""
        spawn_plan = self.spawn_plan_fn() if self.spawn_plan_fn is not None else None
        options = {"spawn_plan": spawn_plan} if spawn_plan is not None else None
        return self.env.reset(options=options)

    def _reward_context(
        self,
        *,
        agent_id: str,
        info_dict: Dict,
        obs_dict: Dict,
        actions: Dict[str, np.ndarray],
        global_state: Optional[GlobalState] = None,
    ) -> Dict[str, Any]:
        return build_reward_context(
            env=self.env,
            agent_id=agent_id,
            info_dict=info_dict,
            obs_dict=obs_dict,
            actions=actions,
            global_state=global_state,
        )

    def train(self, n_episodes: int = 0, *, total_steps: Optional[int] = None) -> None:
        for _ in self.iter_train(n_episodes, total_steps=total_steps):
            pass

    def _policy_request(self, kind, observation, parallel):
        if parallel:
            response = yield kind, observation
            if kind == "act":
                action, log_prob, value, raw = response
                self.agent.last_raw_actions = np.asarray(raw)[None]
                return action, log_prob, value
            return response
        return getattr(self.agent, kind)(observation)

    def iter_train(self, n_episodes: int = 0, *, total_steps: Optional[int] = None,
                   parallel: bool = False):
        """The serial rollout loop, also scheduled by grouped CPU collectors."""
        if total_steps is not None and total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self._set_training_progress(0, total_steps or n_episodes)
        collected = 0
        episode = 0
        self.agent.buffer.clear()
        while (collected < total_steps if total_steps is not None else episode < n_episodes):
            obs_dict, info_dict = self._reset_env()
            for controller in self.other_agents.values():
                if hasattr(controller, "reset"):
                    controller.reset()
            reset_actions = getattr(self.action_composer, "reset", None)
            if reset_actions is not None:
                reset_actions()
            self.obs_composer.reset()
            self.reward_composer.reset()
            if total_steps is None:
                self.agent.buffer.clear()

            obs = self.obs_composer.wrap(obs_dict.get(self.rl_agent_id, {}), info_dict.get(self.rl_agent_id, {}))
            done = False
            episode_reward = 0.0
            episode_truncated = False
            update_metrics: Dict = {}
            last_info: Dict = {}
            step_idx = 0
            episode_id = self._episode_id(episode)
            map_id = self._map_id()
            spawn_id = self._spawn_id()

            while not done:
                # Copy before stepping: dataset state and observation describe
                # the same decision, even if an environment reuses its arrays.
                global_state = (
                    np.asarray(self.env.get_global_state().vector, dtype=np.float32).copy()
                    if self._transition_hooks else None
                )
                action_norm, log_prob, value = yield from self._policy_request("act", obs, parallel)
                raw_batch = getattr(self.agent, "last_raw_actions", None)
                raw_action = raw_batch[0].copy() if raw_batch is not None else None
                action_phys = self.action_composer.process(action_norm)
                actions = self._build_actions(action_phys, obs_dict)
                acted_agents = set(actions)

                # Accumulate reward across action_repeat sub-steps so
                # progress-based reward components aren't missed.
                reward = 0.0
                rl_term = False
                rl_trunc = False
                post_step_global_snapshot: Optional[GlobalState] = None
                for _ in range(self.action_repeat):
                    obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(actions)
                    step_facts = getattr(self.env, "last_step_facts", None)
                    post_step_global_snapshot = getattr(
                        step_facts, "global_state", None
                    )
                    if post_step_global_snapshot is None:
                        post_step_global_snapshot = self.env.get_global_state()
                    if self.render:
                        try:
                            self.env.render()
                        except Exception:
                            pass
                    rl_term = bool(term_dict.get(self.rl_agent_id, False))
                    rl_trunc = bool(trunc_dict.get(self.rl_agent_id, False))
                    sub_done = rl_term or rl_trunc
                    sub_info = info_dict.get(self.rl_agent_id, {})
                    sub_step_info = {
                        "obs": obs,
                        "next_obs": obs_dict.get(self.rl_agent_id, {}),
                        "info": sub_info,
                        "done": sub_done,
                        "terminated": rl_term,
                        "truncated": rl_trunc,
                        "action": action_norm,
                        "timestep": float(getattr(self.env, "timestep", 0.01)),
                    }
                    sub_step_info.update(
                        self._reward_context(
                            agent_id=self.rl_agent_id,
                            info_dict=info_dict,
                            obs_dict=obs_dict,
                            actions=actions,
                            global_state=post_step_global_snapshot,
                        )
                    )
                    sub_reward, _ = self.reward_composer.compute(sub_step_info)
                    reward += sub_reward
                    membership_changed = not acted_agents.issubset(
                        set(getattr(self.env, "agents", []))
                    )
                    if sub_done or membership_changed:
                        done = True
                        episode_truncated = bool(rl_trunc)
                        if sub_done:
                            break
                        done = False
                        break

                last_info = info_dict.get(self.rl_agent_id, {})
                episode_reward += reward

                # The next observation corresponds to the state produced by
                # action_norm, so publish that action before composing it.
                self.obs_composer.update_prev_action(action_norm)
                next_obs = self.obs_composer.wrap(
                    obs_dict.get(self.rl_agent_id, {}),
                    last_info,
                )

                # --- Emit transition record for dataset hooks ---
                if self._transition_hooks:
                    record = TransitionRecord(
                        obs=obs,
                        action_norm=action_norm,
                        action_phys=action_phys,
                        reward=reward,
                        reward_components={},
                        next_obs=next_obs,
                        terminated=rl_term if done else False,
                        truncated=rl_trunc if done else False,
                        info=last_info,
                        global_state=global_state,
                        map_id=map_id,
                        spawn_id=spawn_id,
                        episode_id=episode_id,
                        step_idx=step_idx,
                        agent_id=self.rl_agent_id,
                        **transition_lifecycle_fields(
                            self.env,
                            last_info,
                            global_state=post_step_global_snapshot,
                        ),
                    )
                    for hook in self._transition_hooks:
                        hook.on_step(record)
                step_idx += 1

                collected += 1
                self.collected_steps = collected
                budget_done = total_steps is not None and collected >= total_steps
                final_value = None
                if total_steps is not None and rl_trunc and not rl_term:
                    final_value = yield from self._policy_request("value", next_obs, parallel)
                self.agent.buffer.add(
                    obs,
                    action_norm,
                    reward,
                    log_prob,
                    value,
                    terminated=rl_term,
                    truncated=rl_trunc,
                    **({"raw_action": raw_action} if raw_action is not None else {}),
                    **({"final_value": final_value} if final_value is not None else {}),
                )

                if self.agent.buffer.is_full() or budget_done or (done and total_steps is None):
                    # Time-limit truncations have a valid final observation and
                    # should bootstrap. Only true terminal states force V=0.
                    if not rl_term:
                        if total_steps is not None:
                            next_value = final_value
                            if next_value is None:
                                next_value = yield from self._policy_request("value", next_obs, parallel)
                        else:
                            _, _, next_value = yield from self._policy_request("act", next_obs, parallel)
                    else:
                        next_value = 0.0
                    if total_steps is not None:
                        self._set_training_progress(collected, total_steps)
                    if parallel:
                        reply = yield "rollout", self.agent.pack_rollout(next_value)
                        update_metrics = self.agent.apply_reply(reply)
                    else:
                        update_metrics = self.agent.update(next_value)
                    self.agent.buffer.clear()
                    if update_metrics:
                        update_metrics["train/environment_steps"] = collected
                        for hook in self.hooks:
                            hook.on_update(update_metrics)

                obs = next_obs
                if budget_done or self._should_stop():
                    break

            # Exhausting the training budget is not an environment terminal.
            if not done:
                break
            outcome = determine_outcome(last_info, truncated=episode_truncated)
            last_info["outcome"] = outcome.value
            update_metrics = dict(update_metrics)
            update_metrics["episode_steps"] = step_idx
            # Lifecycle timing persists after the crossing and counts physics
            # steps, not policy decisions (so do not multiply by action_repeat).
            lap_time_steps = last_info.get("lap_time_steps")
            timestep = getattr(self.env, "timestep", None)
            update_metrics["lap_time_s"] = (
                float(lap_time_steps) * float(timestep)
                if lap_time_steps is not None and timestep is not None else None
            )

            if total_steps is None:
                self._set_training_progress(episode + 1, n_episodes)
            for hook in self.hooks:
                hook.on_episode_end(episode, episode_reward, last_info, update_metrics)
            episode += 1
            if self._should_stop():
                break

        # A passed evaluation gate freezes the evaluated weights exactly.
        if not self._should_stop():
            self._flush_pending_update()
        for hook in self.hooks:
            hook.on_training_end()

    def _should_stop(self) -> bool:
        return bool(getattr(self.agent, "should_stop", False)) or any(
            getattr(hook, "should_stop", False) for hook in self.hooks)

    def _flush_pending_update(self) -> None:
        flush = getattr(self.agent, "flush_pending_update", None)
        metrics = flush() if flush is not None else {}
        if metrics:
            metrics["train/environment_steps"] = self.collected_steps
            for hook in self.hooks:
                hook.on_update(metrics)


# Spawned CPU collectors reuse the exact single-environment trainer above.
# Only the parent owns a policy, optimizer, logging hooks, and CUDA context.
_WORKER_TIMEOUT_SECONDS = 120
_WORKER_THREADS = 1


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


class _RemotePolicy:
    def __init__(self, connection, n_steps, obs_dim, action_dim, gamma, gae_lambda,
                 *, map_scheduler=None, worker_id=0):
        import torch
        from agents.ppo import RolloutBuffer

        self.connection = connection
        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, torch.device("cpu"))
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.map_scheduler = map_scheduler
        self.worker_id = worker_id
        self.should_stop = False
        self._training_bundles = None

    def act(self, obs):
        self.connection.send(("act", obs))
        action, log_prob, value, raw_action = self.connection.recv()
        self.last_raw_actions = np.asarray(raw_action)[None]
        return action, log_prob, value

    def value(self, obs):
        self.connection.send(("value", obs))
        return float(self.connection.recv())

    def update(self, next_value):
        self.connection.send(("rollout", self.pack_rollout(next_value)))
        return self.apply_reply(self.connection.recv())

    def pack_rollout(self, next_value):
        buffer = self.buffer
        n = buffer.size()
        advantages, returns = buffer.compute_gae(next_value, self.gamma, self.gae_lambda)
        return tuple(t.numpy() for t in (
            buffer.obs[:n], buffer.actions[:n], buffer.log_probs[:n], advantages, returns,
            buffer.raw_actions[:n],
        ))

    def apply_reply(self, reply):
        if "collector_control" not in reply:
            return reply
        control = reply["collector_control"]
        self.should_stop = bool(control["stop"])
        bundles = tuple(control.get("training_bundles", ()))
        if bundles and bundles != self._training_bundles:
            if self.map_scheduler is None:
                raise RuntimeError("Collector received map curriculum without a scheduler")
            # Stagger equal-length workers across the same weighted schedule.
            offset = self.worker_id % len(bundles)
            self.map_scheduler.set_training_bundles(list(bundles[offset:] + bundles[:offset]))
            self._training_bundles = bundles
        return reply["metrics"]


class _WorkerHook(TrainingHook):
    def __init__(self, connection, worker_id, seed, record_transitions, aggregate_wandb=False):
        self.connection = connection
        self.worker_id = worker_id
        self.seed = seed
        self._record_transitions = record_transitions
        self._wandb = WandbHook(None) if aggregate_wandb else None
        self.requires_transition_record = record_transitions or aggregate_wandb

    def on_step(self, record):
        if self._wandb is not None:
            self._wandb.on_step(record)
        if not self._record_transitions:
            return
        from dataclasses import replace
        record = replace(record, info={**record.info, "worker_id": self.worker_id,
                                       "worker_seed": self.seed})
        self.connection.send(("transition", record))

    def on_episode_start(self, metadata):
        self.connection.send(('episode_start', metadata))

    def on_episode_end(self, episode, reward, info, metrics):
        info = {**info, "worker_id": self.worker_id, "worker_seed": self.seed,
                "worker_episode": episode}
        if self._wandb is not None:
            info["_wandb_episode_state"] = self._wandb.take_episode_state()
        metrics = {**metrics, "worker_id": self.worker_id, "worker_seed": self.seed,
                   "worker_episode": episode}
        self.connection.send(("episode", (reward, info, metrics)))


def _collect_ppo_worker(connection, scenario, scenario_dir, agent_id, worker_id,
                        n_episodes, n_steps, run_id, gamma, gae_lambda, record_transitions,
                        aggregate_wandb, total_steps=None):
    import copy
    from pathlib import Path

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYGLET_HEADLESS"] = "true"
    env = None
    try:
        import torch
        torch.set_num_threads(_WORKER_THREADS)
        from core.setup import create_training_setup
        from run import build_obs_composer, build_reward_composer

        scenario = copy.deepcopy(scenario)
        base_seed = int(scenario["experiment"]["seed"])
        seed = (base_seed + worker_id) % (2 ** 32)
        scenario["experiment"]["seed"] = seed
        env_cfg = scenario["environment"]
        env_seed = env_cfg.get("seed")
        env_cfg["seed"] = ((base_seed if env_seed is None else int(env_seed)) + worker_id) % (2 ** 32)
        env_cfg["render"] = False
        env, opponents, _ = create_training_setup(scenario, scenario_dir=Path(scenario_dir))
        for opponent in opponents.values():
            if hasattr(opponent, "set_env"):
                opponent.set_env(env)
        cfg = scenario["agents"][agent_id]
        space = env.action_spaces[agent_id]
        observations = build_obs_composer(cfg, env_cfg, Path(scenario_dir), space.n)
        rewards = build_reward_composer(cfg, Path(scenario_dir))
        policy = _RemotePolicy(connection, n_steps, observations.obs_dim, space.n, gamma, gae_lambda,
            **({"map_scheduler": env._map_scheduler, "worker_id": worker_id}
               if scenario.get("map_curriculum") is not None else {}))
        trainer = OnPolicyTrainer(
            env, agent_id, policy, opponents, observations, rewards,
            ActionComposer.from_config(
                space.low, space.high, cfg.get("action_constraints", {}),
                decision_dt=float(env_cfg.get("timestep", 0.01)) * int(env_cfg.get("action_repeat", 1)),
            ),
            action_repeat=int(env_cfg.get("action_repeat", 1)),
            hooks=[_WorkerHook(connection, worker_id, seed, record_transitions, aggregate_wandb)],
            run_id=f"{run_id}_worker{worker_id:03d}",
        )
        connection.send(("ready", (observations.obs_dim, space.low, space.high)))
        if connection.recv() != ("start", None):
            raise RuntimeError("PPO collector expected a start message after readiness")
        trainer.train(n_episodes, total_steps=total_steps)
        connection.send(("done", None))
    except (BrokenPipeError, EOFError, ConnectionResetError):
        # The parent already stopped or failed; do not report through a dead pipe.
        pass
    except BaseException:
        _report_worker_error(connection)
    finally:
        if env is not None:
            env.close()
        connection.close()
