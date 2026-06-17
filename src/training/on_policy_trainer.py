"""On-policy training loop for PPO (and future A2C)."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agents.ppo import PPOAgent
from env.types import TransitionRecord
from metrics.outcomes import determine_outcome
from training.hooks import TrainingHook
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
        self.action_composer = action_composer
        self.action_repeat = max(1, int(action_repeat))
        self.hooks = hooks or []
        self.render = render
        self.run_id = run_id
        self.spawn_plan_fn = spawn_plan_fn

    def _build_actions(
        self,
        rl_action_phys: np.ndarray,
        obs_dict: Dict,
    ) -> Dict[str, np.ndarray]:
        actions: Dict[str, np.ndarray] = {self.rl_agent_id: rl_action_phys}
        for aid, other_agent in self.other_agents.items():
            if aid in obs_dict:
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
    ) -> Dict[str, Any]:
        try:
            global_state = self.env.get_global_state().vector
        except Exception:
            global_state = np.zeros(0, dtype=np.float32)
        return {
            "agent_id": agent_id,
            "all_infos": info_dict or {},
            "all_obs": obs_dict or {},
            "all_actions": actions or {},
            "global_state": global_state,
            "last_step_facts": getattr(self.env, "last_step_facts", None),
        }

    def train(self, n_episodes: int) -> None:
        for episode in range(n_episodes):
            obs_dict, info_dict = self._reset_env()
            self.obs_composer.reset()
            self.reward_composer.reset()
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
                action_norm, log_prob, value = self.agent.act(obs)
                action_phys = self.action_composer.process(action_norm)
                actions = self._build_actions(action_phys, obs_dict)

                # Accumulate reward across action_repeat sub-steps so
                # progress-based reward components aren't missed.
                reward = 0.0
                rl_term = False
                rl_trunc = False
                for _ in range(self.action_repeat):
                    obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(actions)
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
                        "timestep": 0.01,
                    }
                    sub_step_info.update(
                        self._reward_context(
                            agent_id=self.rl_agent_id,
                            info_dict=info_dict,
                            obs_dict=obs_dict,
                            actions=actions,
                        )
                    )
                    sub_reward, _ = self.reward_composer.compute(sub_step_info)
                    reward += sub_reward
                    if sub_done:
                        done = True
                        episode_truncated = bool(rl_trunc)
                        break

                last_info = info_dict.get(self.rl_agent_id, {})
                episode_reward += reward

                next_obs = self.obs_composer.wrap(
                    obs_dict.get(self.rl_agent_id, {}),
                    last_info,
                )
                self.obs_composer.update_prev_action(action_norm)

                # --- Emit transition record for dataset hooks ---
                try:
                    global_state = self.env.get_global_state().vector
                except Exception:
                    global_state = np.zeros(0, dtype=np.float32)
                record = TransitionRecord(
                    obs=obs,
                    action_norm=action_norm,
                    action_phys=action_phys,
                    reward=reward,
                    reward_components={},  # breakdown available via reward_composer if needed
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
                )
                for hook in self.hooks:
                    hook.on_step(record)
                step_idx += 1

                self.agent.buffer.add(obs, action_norm, reward, log_prob, value, done)

                if self.agent.buffer.is_full() or done:
                    if not done:
                        _, _, next_value = self.agent.act(next_obs)
                    else:
                        next_value = 0.0
                    update_metrics = self.agent.update(next_value, done)
                    self.agent.buffer.clear()
                    for hook in self.hooks:
                        hook.on_update(update_metrics)

                obs = next_obs

            outcome = determine_outcome(last_info, truncated=episode_truncated)
            last_info["outcome"] = outcome.value

            for hook in self.hooks:
                hook.on_episode_end(episode, episode_reward, last_info, update_metrics)

        for hook in self.hooks:
            hook.on_training_end()
