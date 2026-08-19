"""Multi-agent training loop for MAPPO (and future MARL algorithms).

``MARLTrainer`` drives an episode-based training loop for scenarios with
multiple trainable agents sharing one :class:`~agents.mappo.MAPPOAgent`.

Key differences from :class:`~training.on_policy_trainer.OnPolicyTrainer`
--------------------------------------------------------------------------
- Actions are collected for **all** trainable agents each step via the
  shared actor.  Fixed-policy opponents are still polled via ``other_agents``.
- The centralized critic uses ``env.get_global_state().vector`` — a flat
  concatenation of all agents' state — rather than local observations.
- Per-agent rewards are computed independently by each agent's
  :class:`~wrappers.rewards.composer.RewardComposer`.
- The buffer-full / update trigger is checked across all agents: when **any**
  agent's buffer is full, an update is triggered for all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from agents.mappo import MAPPOAgent
from env.types import TransitionRecord
from metrics.outcomes import determine_outcome
from training.hooks import TrainingHook
from training.reward_context import build_reward_context
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


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
        Shared :class:`~wrappers.actions.composer.ActionComposer` for
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
    ) -> None:
        self.env = env
        self.agent = agent
        self.trainable_ids = list(trainable_ids)
        self.other_agents = other_agents
        self.obs_composers = obs_composers
        self.reward_composers = reward_composers
        self.action_composer = action_composer
        self.action_repeat = max(1, int(action_repeat))
        self.hooks = hooks or []
        self.render = render
        self.focal_id = focal_agent_id or (trainable_ids[0] if trainable_ids else "")
        self.run_id = run_id

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
        active_agents = set(getattr(self.env, "agents", obs_dict))
        for aid, other_agent in self.other_agents.items():
            if aid in active_agents:
                try:
                    act = other_agent.act(obs_dict[aid])
                except Exception:
                    act = np.zeros(2, dtype=np.float32)
                actions[aid] = np.asarray(act, dtype=np.float32)
        return actions

    def _episode_id(self, episode: int) -> str:
        return f"{self.run_id}_ep{episode:06d}"

    def _map_id(self) -> Optional[str]:
        return getattr(self.env, "map_name", None)

    def _spawn_id(self, agent_id: str) -> Optional[str]:
        spawn_manager = getattr(self.env, "_spawn_manager", None)
        if spawn_manager is None:
            return None
        metadata = getattr(spawn_manager, "last_spawn_metadata", {}) or {}
        spawn_ids = metadata.get("spawn_ids", {})
        return spawn_ids.get(agent_id) or metadata.get("spawn_id")

    def _reward_context(
        self,
        *,
        agent_id: str,
        info_dict: Dict[str, Any],
        obs_dict: Dict[str, Any],
        actions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        return build_reward_context(
            env=self.env,
            agent_id=agent_id,
            info_dict=info_dict,
            obs_dict=obs_dict,
            actions=actions,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: int) -> None:
        """Run *n_episodes* of MAPPO training."""
        for episode in range(n_episodes):
            obs_dict, info_dict = self.env.reset()
            global_state = self.env.get_global_state().vector

            # Reset per-agent composers and buffers
            for aid in self.trainable_ids:
                self.obs_composers[aid].reset()
                self.reward_composers[aid].reset()
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
            agent_last_info: Dict[str, Dict] = {aid: {} for aid in self.trainable_ids}
            agent_truncated: Dict[str, bool] = {aid: False for aid in self.trainable_ids}
            step_idx = 0
            episode_id = self._episode_id(episode)
            map_id = self._map_id()

            while not episode_done:
                # --- Act: all trainable agents via shared actor ---
                active_before = set(getattr(self.env, "agents", obs_dict))
                actions_norm: Dict[str, np.ndarray] = {}
                actions_phys: Dict[str, np.ndarray] = {}
                log_probs: Dict[str, float] = {}

                for aid in self.trainable_ids:
                    if aid in active_before:
                        a_norm, lp = self.agent.act(wrapped_obs[aid])
                        a_phys = self.action_composer.process(a_norm)
                        actions_norm[aid] = a_norm
                        actions_phys[aid] = a_phys
                        log_probs[aid] = lp

                # Centralized value estimate from current global state
                value = self.agent.evaluate_state(global_state)

                all_actions = self._build_actions(actions_phys, obs_dict)

                # --- Step env (action_repeat times) ---
                # Accumulate per-agent rewards across sub-steps so progress-based
                # reward components (centerline delta, etc.) are not missed.
                term_dict: Dict[str, bool] = {}
                trunc_dict: Dict[str, bool] = {}
                accumulated_rewards: Dict[str, float] = {aid: 0.0 for aid in self.trainable_ids}
                reward_breakdowns: Dict[str, Dict[str, float]] = {
                    aid: {} for aid in actions_norm
                }
                decision_terminated = {aid: False for aid in actions_norm}
                decision_truncated = {aid: False for aid in actions_norm}

                for _ in range(self.action_repeat):
                    obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(
                        all_actions
                    )
                    if self.render:
                        try:
                            self.env.render()
                        except Exception:
                            pass
                    # Compute sub-step reward for each trainable agent
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
                            "timestep": 0.01,
                        }
                        sub_step_info.update(
                            self._reward_context(
                                agent_id=aid,
                                info_dict=info_dict,
                                obs_dict=obs_dict,
                                actions=all_actions,
                            )
                        )
                        sub_reward, breakdown = self.reward_composers[aid].compute(sub_step_info)
                        accumulated_rewards[aid] = accumulated_rewards.get(aid, 0.0) + sub_reward
                        for name, component_reward in breakdown.items():
                            reward_breakdowns[aid][name] = (
                                reward_breakdowns[aid].get(name, 0.0)
                                + float(component_reward)
                            )

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
                next_global_state = self.env.get_global_state().vector

                # --- Store transitions with accumulated rewards ---
                step_reward = 0.0
                next_wrapped_obs: Dict[str, np.ndarray] = {}
                for aid in actions_norm:
                    reward = accumulated_rewards.get(aid, 0.0)
                    if aid == self.focal_id:
                        step_reward = reward

                    agent_info = info_dict.get(aid, {})
                    next_obs = self.obs_composers[aid].wrap(
                        obs_dict.get(aid, {}), agent_info
                    )
                    next_wrapped_obs[aid] = next_obs

                    self.agent.store(
                        agent_id=aid,
                        obs=wrapped_obs[aid],
                        global_state=global_state,
                        action=actions_norm[aid],
                        reward=reward,
                        log_prob=log_probs[aid],
                        value=value,
                        terminated=decision_terminated[aid],
                        truncated=decision_truncated[aid],
                    )

                    record = TransitionRecord(
                        obs=wrapped_obs[aid],
                        action_norm=actions_norm[aid],
                        action_phys=actions_phys[aid],
                        reward=reward,
                        reward_components=reward_breakdowns[aid],
                        next_obs=next_obs,
                        terminated=decision_terminated[aid],
                        truncated=decision_truncated[aid],
                        info=agent_info,
                        global_state=np.asarray(global_state, dtype=np.float32).copy(),
                        map_id=map_id,
                        spawn_id=self._spawn_id(aid),
                        episode_id=episode_id,
                        step_idx=step_idx,
                        agent_id=aid,
                    )
                    for hook in self.hooks:
                        hook.on_step(record)

                    agent_episode_rewards[aid] += reward
                    agent_last_info[aid] = agent_info
                    agent_truncated[aid] = (
                        agent_truncated[aid] or decision_truncated[aid]
                    )

                episode_reward += step_reward

                # --- Update observation wrappers ---
                for aid in self.trainable_ids:
                    if aid in getattr(self.env, "agents", []):
                        wrapped_obs[aid] = next_wrapped_obs[aid]
                        self.obs_composers[aid].update_prev_action(
                            actions_norm.get(aid, np.zeros(2, dtype=np.float32))
                        )

                global_state = next_global_state
                step_idx += 1

                # --- Trigger update when any buffer is full or episode ends ---
                if self.agent.any_buffer_full() or episode_done:
                    update_metrics = self.agent.update(
                        next_global_state=next_global_state,
                    )
                    self.agent.clear_buffers()
                    for hook in self.hooks:
                        hook.on_update(update_metrics)

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
            episode_metrics["agent_rewards"] = dict(agent_episode_rewards)
            episode_metrics["agent_outcomes"] = agent_outcomes

            for hook in self.hooks:
                hook.on_episode_end(episode, episode_reward, last_info, episode_metrics)

        for hook in self.hooks:
            hook.on_training_end()
