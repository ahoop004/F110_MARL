"""Multi-agent training loop for MAPPO (and future MARL algorithms).

``MARLTrainer`` drives an episode-based training loop for scenarios with
multiple trainable agents sharing one :class:`~agents.mappo.MAPPOAgent`.

Key differences from :class:`~training.on_policy_trainer.OnPolicyTrainer`
--------------------------------------------------------------------------
- Actions are collected for **all** trainable agents each step via the
  shared actor.  Fixed-policy opponents are still polled via ``other_agents``.
- The centralized critic uses ``env.get_global_state().vector`` — a flat
  concatenation of all agents' state — rather than local observations.
- Per-agent factual rewards are computed independently, then either retained
  or reduced to one shared team learning reward according to the MAPPO config.
- The buffer-full / update trigger is checked across all agents: when **any**
  agent's buffer is full, an update is triggered for all.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from agents.mappo import MAPPOAgent
from env.types import GlobalState, TransitionRecord
from metrics.outcomes import determine_outcome
from training.hooks import TrainingHook, transition_record_hooks
from training.reward_context import build_reward_context, transition_lifecycle_fields, validate_team_reward_composers
from metrics.racing_eval import (team_finish_result, create_episode_facts,
                                update_agent_step_facts, finalize_episode_facts,
                                episode_race_record, capture_spawn_context)
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


def map_mappo_learning_rewards(
    individual_rewards: Dict[str, float],
    *,
    trainable_ids: List[str],
    reward_mode: str,
    team_reward_reduction: str,
) -> Dict[str, float]:
    """Convert per-agent factual rewards into MAPPO learning rewards."""
    if reward_mode == "individual":
        return dict(individual_rewards)
    team_reward = float(sum(individual_rewards.values()))
    if team_reward_reduction == "mean":
        team_reward /= max(len(trainable_ids), 1)
    return {aid: team_reward for aid in individual_rewards}


class MARLTrainer:
    """Episode-based MAPPO training loop.

    Parameters
    ----------
    env:
        The F110 parallel environment.
    agent:
        A :class:`~agents.mappo.MAPPOAgent` with shared actor and centralized critic.
    trainable_ids:
        Ordered list of agent IDs that are being trained.
    other_agents:
        Dict of ``{agent_id: heuristic_policy}`` for fixed-policy opponents.
    obs_composers:
        ``{agent_id: ObservationComposer}`` — one per trainable agent.
    reward_composers:
        ``{agent_id: RewardComposer}`` — one per trainable agent.
    action_composer:
        Template :class:`~wrappers.actions.composer.ActionComposer`, cloned per agent for
        denormalizing and constraining actions.  Applied to every trainable
        agent's output.
    action_repeat:
        Number of env steps per agent decision.
    hooks:
        Optional list of :class:`~training.hooks.TrainingHook` callbacks.
    render:
        Whether to call ``env.render()`` each step.
    focal_agent_id:
        Agent ID used for episode-level logging (reward, outcome).  Defaults
        to the first element of *trainable_ids*.
    run_id:
        Stable run identifier used in per-agent dataset transition records.
    """

    def __init__(
        self,
        env: Any,
        agent: MAPPOAgent,
        trainable_ids: List[str],
        other_agents: Dict[str, Any],
        obs_composers: Dict[str, ObservationComposer],
        reward_composers: Dict[str, RewardComposer],
        action_composer: ActionComposer,
        action_repeat: int = 1,
        hooks: Optional[List[TrainingHook]] = None,
        render: bool = False,
        focal_agent_id: Optional[str] = None,
        run_id: str = "run",
        reward_mode: str = "individual",
        team_reward_reduction: str = "mean",
        race_recorder=None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.trainable_ids = list(trainable_ids)
        self.other_agents = other_agents
        self.obs_composers = obs_composers
        self.reward_composers = reward_composers
        self.action_composer = action_composer
        self.action_composers = {aid: copy.deepcopy(action_composer) for aid in self.trainable_ids}
        self.action_repeat = max(1, int(action_repeat))
        self.hooks = hooks or []
        self._environment_steps = 0
        self._physics_steps = 0
        self._agent_steps = 0
        self._updates = 0
        self._transition_hooks = transition_record_hooks(self.hooks)
        self.render = render
        self.focal_id = focal_agent_id or (trainable_ids[0] if trainable_ids else "")
        self.run_id = run_id
        self.race_recorder = race_recorder
        self.reward_mode = str(reward_mode).strip().lower()
        self.team_reward_reduction = str(team_reward_reduction).strip().lower()
        self.team_return_mode = getattr(agent, "team_return_mode", "per_agent")
        self._has_team_rewards = validate_team_reward_composers(
            reward_composers, trainable_ids=self.trainable_ids,
            opponent_ids=list(other_agents), reward_mode=self.reward_mode,
            critic_mode=getattr(agent, "critic_mode", "agent_conditioned"),
            team_return_mode=self.team_return_mode, action_repeat=self.action_repeat,
        )
        if self.team_return_mode == "joint" and self.action_repeat != 1:
            raise ValueError("Joint team returns currently require action_repeat=1")
        if self.reward_mode not in {"individual", "team_shared"}:
            raise ValueError(
                "MAPPO reward_mode must be 'individual' or 'team_shared', "
                f"got {self.reward_mode!r}."
            )
        if self.team_reward_reduction not in {"mean", "sum"}:
            raise ValueError(
                "MAPPO team_reward_reduction must be 'mean' or 'sum', "
                f"got {self.team_reward_reduction!r}."
            )
        agent_ids = getattr(agent, "agent_ids", None)
        if agent_ids is not None and list(agent_ids) != self.trainable_ids:
            raise ValueError(
                "MARLTrainer trainable_ids must exactly match the MAPPO agent ID order."
            )
        for field in ("reward_mode", "team_reward_reduction"):
            agent_value = getattr(agent, field, None)
            trainer_value = getattr(self, field)
            if agent_value is not None and agent_value != trainer_value:
                raise ValueError(
                    f"MARLTrainer {field}={trainer_value!r} does not match "
                    f"MAPPOAgent {field}={agent_value!r}."
                )

    # ------------------------------------------------------------------
    # Action assembly
    # ------------------------------------------------------------------

    def _build_actions(
        self,
        trainable_actions: Dict[str, np.ndarray],
        obs_dict: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Combine trainable and fixed-policy actions into one dict."""
        actions: Dict[str, np.ndarray] = dict(trainable_actions)
        self._opponent_action_errors = set()
        active_agents = set(getattr(self.env, "agents", obs_dict))
        for aid, other_agent in self.other_agents.items():
            if aid in active_agents:
                try:
                    act = other_agent.act(obs_dict[aid])
                except Exception:
                    act = np.zeros(2, dtype=np.float32)
                    self._opponent_action_errors.add(aid)
                actions[aid] = np.asarray(act, dtype=np.float32)
        return actions

    def _episode_id(self, episode: int) -> str:
        return f"{self.run_id}_ep{episode:06d}"

    def _map_id(self) -> Optional[str]:
        return getattr(self.env, "_map_bundle_active", None) or getattr(
            self.env, "map_name", None
        )

    def _spawn_id(self, agent_id: str) -> Optional[str]:
        spawn_manager = getattr(self.env, "_spawn_manager", None)
        if spawn_manager is None:
            return None
        metadata = getattr(spawn_manager, "last_spawn_metadata", {}) or {}
        spawn_ids = metadata.get("spawn_ids", {})
        return (spawn_ids.get(agent_id) or metadata.get("spawn_id") or
                getattr(spawn_manager, "last_spawn_mapping", {}).get(agent_id))

    def _reward_context(
        self,
        *,
        agent_id: str,
        info_dict: Dict[str, Any],
        obs_dict: Dict[str, Any],
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

    def _learning_rewards(
        self,
        individual_rewards: Dict[str, float],
    ) -> Dict[str, float]:
        """Map factual per-agent rewards to the configured learning signal.

        Team means always use the configured team size as denominator.  A
        teammate that already terminated therefore contributes zero instead
        of silently changing reward scale as the active set shrinks.
        """
        return map_mappo_learning_rewards(
            individual_rewards,
            trainable_ids=self.trainable_ids,
            reward_mode=self.reward_mode,
            team_reward_reduction=self.team_reward_reduction,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 0, *, total_steps: Optional[int] = None) -> None:
        """Train to an episode count or exact joint environment-decision budget."""
        for _ in self.iter_train(n_episodes, total_steps=total_steps):
            pass

    def train_parallel(self, scenario, scenario_dir, num_envs, n_episodes=0, *, total_steps=None):
        from training.parallel_mappo import train_parallel
        train_parallel(self, scenario, scenario_dir, num_envs, n_episodes, total_steps=total_steps)

    def iter_train(self, n_episodes: int, *, parallel: bool = False,
                   total_steps: Optional[int] = None):
        """Shared race loop; a budget cut bootstraps without ending the race."""
        if total_steps is not None and (isinstance(total_steps, bool)
                or not isinstance(total_steps, int) or total_steps <= 0):
            raise ValueError("total_steps must be a positive integer")
        collected, episode = 0, 0
        while (collected < total_steps if total_steps is not None else episode < n_episodes):
            obs_dict, info_dict = self.env.reset()
            for controller in self.other_agents.values():
                if hasattr(controller, "reset"):
                    controller.reset()
            global_snapshot = self.env.get_global_state()
            global_state = global_snapshot.vector

            # Reset per-agent composers and buffers
            for aid in self.trainable_ids:
                self.obs_composers[aid].reset()
                self.reward_composers[aid].reset()
                reset_actions = getattr(self.action_composers[aid], "reset", None)
                if reset_actions is not None:
                    reset_actions()
            self.agent.clear_buffers()

            # Wrap initial observations
            wrapped_obs: Dict[str, np.ndarray] = {
                aid: self.obs_composers[aid].wrap(
                    obs_dict.get(aid, {}), info_dict.get(aid, {})
                )
                for aid in self.trainable_ids
            }

            episode_done = False
            episode_reward = 0.0
            episode_truncated = False
            update_metrics: Dict = {}
            last_info: Dict = {}

            # Per-agent tracking — every trainable agent gets its own episode
            # reward total, last info payload, and truncation flag so each
            # agent's outcome can be reported independently of focal_id.
            agent_episode_rewards: Dict[str, float] = {aid: 0.0 for aid in self.trainable_ids}
            agent_individual_rewards: Dict[str, float] = {
                aid: 0.0 for aid in self.trainable_ids
            }
            agent_last_info: Dict[str, Dict] = {aid: {} for aid in self.trainable_ids}
            agent_truncated: Dict[str, bool] = {aid: False for aid in self.trainable_ids}
            step_idx = 0
            episode_id = self._episode_id(episode)
            map_id = self._map_id()
            for hook in self.hooks:
                callback = getattr(hook, 'on_episode_start', None)
                if callback is not None:
                    callback(dict(episode_id=episode_id, map_id=map_id,
                                  physics=info_dict.get(self.focal_id, {}).get('physics')))
            facts = create_episode_facts(episode=episode,
                agent_ids=[*self.trainable_ids, *self.other_agents],
                trainable_ids=self.trainable_ids, opponent_ids=list(self.other_agents))
            spawn_context = capture_spawn_context(self.env, facts.agents)
            physics_steps = 0
            start_policy = getattr(self.agent, "policy_version", self._updates)
            finite_race = getattr(getattr(self.env, "lifecycle", None), "finish_on_laps", True)
            team_components = {}
            recorder = self.race_recorder
            if recorder is not None:
                recorder.start(episode_id=episode_id, episode=episode, map_id=map_id,
                    seed=getattr(self.env, 'seed', None), timestep=float(self.env.timestep),
                    action_repeat=self.action_repeat, trainable_ids=self.trainable_ids,
                    agent_ids=list(facts.agents), track_length=float(getattr(self.env, 'centerline_track_length', 0.) or 0.),
                    policy_version=start_policy, spawn=spawn_context,
                    physics=info_dict.get(self.focal_id, {}).get('physics'),
                    termination=dict(mode=getattr(self.env, 'episode_termination_mode', None), finish_on_laps=finite_race))

            while not episode_done:
                if recorder is not None and (getattr(self.agent, 'recording_stop', False) or
                        any(getattr(h, 'storage_full', False) for h in self.hooks)):
                    if not recorder.stopped:
                        recorder.stop()
                decision_policy = getattr(self.agent, "policy_version", self._updates)
                # --- Act: all trainable agents via shared actor ---
                active_before = set(getattr(self.env, "agents", obs_dict))
                active_trainable_ids = [
                    aid for aid in self.trainable_ids if aid in active_before
                ]
                stacked_observations = np.stack(
                    [wrapped_obs[aid] for aid in active_trainable_ids], axis=0
                ) if active_trainable_ids else np.empty((0, getattr(self.agent, "obs_dim", 0)), dtype=np.float32)
                if parallel:
                    actions_norm, log_probs, values, raw_actions = yield (
                        "act", (active_trainable_ids, stacked_observations, global_state)
                    )
                    self.agent.last_raw_actions = raw_actions
                else:
                    if active_trainable_ids:
                        actions_norm, log_probs = self.agent.act_batch(
                            active_trainable_ids, stacked_observations,
                        )
                    else:
                        actions_norm, log_probs = {}, {}
                    values = self.agent.evaluate_states(global_state, active_trainable_ids)
                actions_phys: Dict[str, np.ndarray] = {}
                for aid, action in actions_norm.items():
                    actions_phys[aid] = self.action_composers[aid].process(action)

                all_actions = self._build_actions(actions_phys, obs_dict)

                # --- Step env (action_repeat times) ---
                # Accumulate per-agent rewards across sub-steps so progress-based
                # reward components (centerline delta, etc.) are not missed.
                term_dict: Dict[str, bool] = {}
                trunc_dict: Dict[str, bool] = {}
                accumulated_individual_rewards: Dict[str, float] = {
                    aid: 0.0 for aid in actions_norm
                }
                accumulated_learning_rewards: Dict[str, float] = {
                    aid: 0.0 for aid in actions_norm
                }
                reward_breakdowns: Dict[str, Dict[str, float]] = {
                    aid: {} for aid in actions_norm
                }
                decision_terminated = {aid: False for aid in actions_norm}
                decision_truncated = {aid: False for aid in actions_norm}
                post_step_global_snapshot: Optional[GlobalState] = None
                team_step_reward = 0.0
                team_breakdowns: Dict[str, float] = {}

                for substep in range(self.action_repeat):
                    if recorder is not None:
                        exhausted = set(getattr(self.agent, 'recording_exhausted_windows', ()))
                        for hook in self.hooks:
                            exhausted.update(getattr(hook, 'exhausted_windows', ()))
                        recorder.prepare(getattr(self.agent, 'recording_progress', 0) if parallel else self._environment_steps,
                            physics_steps, decision_policy, exhausted_windows=exhausted,
                            clock='last_completed_collection_barrier' if parallel else 'joint_environment_decisions')
                    recording_active = recorder is not None and recorder.capturing
                    self.env.record_applied_commands = recording_active
                    if recording_active:
                        from replay.race_recorder import capture_state, plain
                        recorded_pre = capture_state(self.env, info_dict, obs_dict, facts.agents)
                    obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(
                        all_actions
                    )
                    physics_steps += 1
                    self._physics_steps += 1
                    if parallel:
                        self.agent.physics_steps_collected = self._physics_steps
                    update_agent_step_facts(facts, step_idx=physics_steps, infos=info_dict,
                        terminations=term_dict, truncations=trunc_dict, collect_speed=False)
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
                    # Compute factual sub-step rewards independently first.
                    substep_individual_rewards: Dict[str, float] = {}
                    substep_breakdowns = {}
                    for aid in actions_norm:
                        agent_term = bool(term_dict.get(aid, False))
                        agent_trunc = bool(trunc_dict.get(aid, False))
                        agent_done = agent_term or agent_trunc
                        decision_terminated[aid] = decision_terminated[aid] or agent_term
                        decision_truncated[aid] = decision_truncated[aid] or agent_trunc
                        sub_step_info = {
                            "obs": wrapped_obs.get(aid, {}),
                            "next_obs": obs_dict.get(aid, {}),
                            "info": info_dict.get(aid, {}),
                            "done": agent_done,
                            "terminated": agent_term,
                            "truncated": agent_trunc,
                            "action": actions_norm[aid],
                            "timestep": float(getattr(self.env, "timestep", 0.01)),
                        }
                        sub_step_info.update(
                            self._reward_context(
                                agent_id=aid,
                                info_dict=info_dict,
                                obs_dict=obs_dict,
                                actions=all_actions,
                                global_state=post_step_global_snapshot,
                            )
                        )
                        sub_reward, breakdown = self.reward_composers[aid].compute(sub_step_info)
                        if recording_active:
                            substep_breakdowns[aid] = dict(breakdown)
                        substep_individual_rewards[aid] = float(sub_reward)
                        accumulated_individual_rewards[aid] += float(sub_reward)
                        for name, component_reward in breakdown.items():
                            reward_breakdowns[aid][name] = (
                                reward_breakdowns[aid].get(name, 0.0)
                                + float(component_reward)
                            )

                    # Convert the factual signals into the configured learning
                    # signal exactly once per joint environment sub-step.
                    substep_learning_rewards = self._learning_rewards(
                        substep_individual_rewards
                    )
                    if self._has_team_rewards:
                        # Shared components have one stateful composer and are
                        # added after the local mean, even with one survivor.
                        team_context = self._reward_context(
                            agent_id=self.focal_id, info_dict=info_dict,
                            obs_dict=obs_dict, actions=all_actions,
                            global_state=post_step_global_snapshot,
                        )
                        bonus, team_breakdowns = self.reward_composers[self.focal_id].compute(team_context, team=True)
                        for name, value in team_breakdowns.items():
                            team_components[name] = team_components.get(name, 0.0) + float(value)
                        for aid in substep_learning_rewards:
                            substep_learning_rewards[aid] += bonus
                    if self.team_return_mode == "joint" and substep_learning_rewards:
                        team_step_reward += next(iter(substep_learning_rewards.values()))
                    for aid, learning_reward in substep_learning_rewards.items():
                        accumulated_learning_rewards[aid] += learning_reward

                    if recording_active:
                        applied = getattr(self.env, '_recorded_applied_commands', None)
                        agent_order = list(getattr(self.env, 'possible_agents', facts.agents))
                        recorder.step(plain(dict(
                            physics_index=physics_steps - 1, physics_index_end=physics_steps,
                            decision_index=step_idx, substep_index=substep,
                            policy_version=decision_policy, timestep_s=float(self.env.timestep),
                            action_repeat=self.action_repeat,
                            simulation_time_s=(physics_steps - 1) * float(self.env.timestep),
                            simulation_time_end_s=physics_steps * float(self.env.timestep),
                            pre_state=recorded_pre,
                            post_state=capture_state(self.env, info_dict, obs_dict, facts.agents),
                            commands={aid: dict(
                                requested=all_actions.get(aid),
                                applied=applied[index] if applied is not None else None,
                                source=('learner' if aid in actions_norm else 'controller_error_fallback'
                                        if aid in self._opponent_action_errors else 'opponent'
                                        if aid in all_actions else 'terminal_controller'))
                                for index, aid in enumerate(agent_order)},
                            learners={aid: dict(observation=wrapped_obs[aid], action_normalized=actions_norm[aid],
                                reward=substep_learning_rewards[aid], individual_reward=substep_individual_rewards[aid],
                                reward_components=substep_breakdowns[aid]) for aid in actions_norm},
                            team_reward_components=team_breakdowns,
                            terminated=term_dict, truncated=trunc_dict,
                        )))

                    active_after_substep = set(getattr(self.env, "agents", []))
                    repeat_boundary = (
                        bool(getattr(self.env, "episode_done", False))
                        or not set(all_actions).issubset(active_after_substep)
                    )
                    if repeat_boundary:
                        break

                episode_done = bool(getattr(self.env, "episode_done", False)) or not bool(
                    getattr(self.env, "agents", [])
                )
                episode_truncated = episode_truncated or bool(
                    decision_truncated.get(self.focal_id, False)
                )
                if post_step_global_snapshot is None:
                    post_step_global_snapshot = self.env.get_global_state()
                next_global_state = post_step_global_snapshot.vector

                # --- Store transitions with accumulated rewards ---
                step_reward = 0.0
                next_wrapped_obs: Dict[str, np.ndarray] = {}
                agent_infos: Dict[str, Dict[str, Any]] = {}
                for aid in actions_norm:
                    reward = accumulated_learning_rewards.get(aid, 0.0)
                    individual_reward = accumulated_individual_rewards.get(aid, 0.0)
                    if aid == self.focal_id:
                        step_reward = reward

                    agent_info = dict(info_dict.get(aid, {}))
                    agent_info.update(
                        {
                            "individual_reward": individual_reward,
                            "learning_reward": reward,
                            "reward_mode": self.reward_mode,
                            "team_reward_reduction": self.team_reward_reduction,
                            "team_return_mode": self.team_return_mode,
                        }
                    )
                    # The next observation must expose the action that produced
                    # the next environment state. Updating after composition
                    # leaves PrevActionComponent one decision behind.
                    self.obs_composers[aid].update_prev_action(actions_norm[aid])
                    next_obs = self.obs_composers[aid].wrap(
                        obs_dict.get(aid, {}), agent_info
                    )
                    next_wrapped_obs[aid] = next_obs
                    agent_infos[aid] = agent_info

                    agent_episode_rewards[aid] += reward
                    agent_individual_rewards[aid] += individual_reward
                    facts.agents[aid].reward_total += reward
                    facts.agents[aid].individual_reward_total += individual_reward
                    for name, value in reward_breakdowns[aid].items():
                        components = facts.agents[aid].reward_components
                        components[name] = components.get(name, 0.0) + value
                    agent_last_info[aid] = agent_info
                    agent_truncated[aid] = (
                        agent_truncated[aid] or decision_truncated[aid]
                    )

                ordered_ids = list(actions_norm)
                self.agent.store_batch(
                    ordered_ids,
                    observations=wrapped_obs,
                    global_state=global_state,
                    actions=actions_norm,
                    rewards=accumulated_learning_rewards,
                    log_probs=log_probs,
                    values=values,
                    terminated=decision_terminated,
                    truncated=decision_truncated,
                    **({"raw_actions": self.agent.last_raw_actions}
                       if hasattr(self.agent, "last_raw_actions") else {}),
                )
                if self.team_return_mode == "joint" and ordered_ids:
                    self.agent.store_team_step(
                        ordered_ids, reward=team_step_reward,
                        value=values[ordered_ids[0]],
                        terminal=not any(aid in getattr(self.env, "agents", []) for aid in self.trainable_ids),
                    )

                if self._transition_hooks:
                    for aid in ordered_ids:
                        record = TransitionRecord(
                            obs=wrapped_obs[aid],
                            action_norm=actions_norm[aid],
                            action_phys=actions_phys[aid],
                            reward=accumulated_learning_rewards[aid],
                            reward_components={**reward_breakdowns[aid], **team_breakdowns},
                            next_obs=next_wrapped_obs[aid],
                            terminated=decision_terminated[aid],
                            truncated=decision_truncated[aid],
                            info=agent_infos[aid],
                            global_state=np.asarray(
                                global_state, dtype=np.float32
                            ).copy(),
                            map_id=map_id,
                            spawn_id=self._spawn_id(aid),
                            episode_id=episode_id,
                            step_idx=step_idx,
                            agent_id=aid,
                            **transition_lifecycle_fields(
                                self.env,
                                agent_infos[aid],
                                global_state=post_step_global_snapshot,
                            ),
                        )
                        for hook in self._transition_hooks:
                            hook.on_step(record)

                episode_reward += team_step_reward if self.team_return_mode == "joint" else step_reward

                # --- Update observation wrappers ---
                for aid in self.trainable_ids:
                    if aid in getattr(self.env, "agents", []):
                        wrapped_obs[aid] = next_wrapped_obs[aid]

                global_state = next_global_state
                step_idx += 1
                self._environment_steps += 1
                self._agent_steps += len(ordered_ids)
                collected += 1
                budget_done = total_steps is not None and collected >= total_steps

                if parallel:
                    yield "step", next_global_state

                # --- Trigger update when any buffer is full or episode ends ---
                if self.agent.any_buffer_full() or episode_done or budget_done:
                    if parallel:
                        next_values = yield "value", next_global_state
                        update_metrics = self.agent.finish_fragment(next_values)
                    else:
                        update_metrics = self.agent.update(
                            next_global_state=next_global_state,
                        )
                        self._updates += bool(update_metrics)
                    self.agent.clear_buffers()
                    update_metrics["train/environment_steps"] = self._environment_steps
                    update_metrics["train/physics_steps"] = self._physics_steps
                    update_metrics["train/agent_steps"] = self._agent_steps
                    update_metrics["train/updates"] = self._updates
                    for hook in self.hooks:
                        hook.on_update(update_metrics)

                if budget_done:
                    break

            # A budget cut is neither a crash nor a completed episode. The last
            # fragment was bootstrapped above; do not fabricate episode metrics.
            if recorder is not None:
                recorder.end(episode_done)
            if not episode_done:
                break
            last_info = agent_last_info.get(self.focal_id, {})
            outcome = determine_outcome(last_info, truncated=episode_truncated)
            last_info["outcome"] = outcome.value

            # Per-agent outcome/reward breakdown — every trainable agent's
            # info carries its own target_id-relative collision/finish flags,
            # so each agent's outcome is determined independently.
            agent_outcomes = {
                aid: determine_outcome(
                    agent_last_info.get(aid, {}), truncated=agent_truncated.get(aid, False)
                ).value
                for aid in self.trainable_ids
            }
            episode_metrics = dict(update_metrics)
            episode_metrics["episode_steps"] = step_idx
            episode_metrics["agent_rewards"] = dict(agent_episode_rewards)
            episode_metrics["agent_individual_rewards"] = dict(
                agent_individual_rewards
            )
            episode_metrics["reward_mode"] = self.reward_mode
            episode_metrics["team_reward_reduction"] = self.team_reward_reduction
            if self.team_return_mode == "joint":
                episode_metrics["team_episode_reward"] = episode_reward
            if self._has_team_rewards:
                episode_metrics.update({
                    f"team_result/{key}": value for key, value in team_finish_result(
                        info_dict, self.trainable_ids, list(self.other_agents)
                    ).items()
                })
            episode_metrics["agent_outcomes"] = agent_outcomes
            episode_metrics["agent_terminal_reasons"] = {
                aid: agent_last_info.get(aid, {}).get("terminal_reason")
                for aid in self.trainable_ids
            }
            episode_metrics["agent_finish_positions"] = {
                aid: agent_last_info.get(aid, {}).get("finish_position")
                for aid in self.trainable_ids
            }
            episode_metrics["agent_lap_counts"] = {
                aid: int(agent_last_info.get(aid, {}).get("lap_count", 0))
                for aid in self.trainable_ids
            }
            episode_metrics["race_record"] = {
                **episode_race_record(finalize_episode_facts(facts),
                    timestep=float(getattr(self.env, "timestep", .01)), finite_race=finite_race),
                "run_id": self.run_id, "environment_id": 0, "episode_id": episode_id,
                "environment_episode": episode, "map_id": map_id,
                "environment_seed": getattr(self.env, "seed", None),
                "spawn_ids": {aid: self._spawn_id(aid) for aid in facts.agents},
                "spawn_configuration": spawn_context,
                "environment_decisions": step_idx,
                "environment_decisions_total": self._environment_steps,
                "action_repeat": self.action_repeat,
                "policy_version_start": start_policy,
                "policy_version_end": decision_policy,
                "reported_at_environment_steps": self._environment_steps,
                "team_reward_components": team_components,
                "training_return": episode_reward,
            }

            for hook in self.hooks:
                hook.on_episode_end(episode, episode_reward, last_info, episode_metrics)
            episode += 1

        for hook in self.hooks:
            hook.on_training_end()
