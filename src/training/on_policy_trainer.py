"""On-policy training loop for PPO (and future A2C)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from agents.ppo import PPOAgent
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

    def train(self, n_episodes: int) -> None:
        for episode in range(n_episodes):
            obs_dict, info_dict = self.env.reset()
            self.obs_composer.reset()
            self.reward_composer.reset()
            self.agent.buffer.clear()

            obs = self.obs_composer.wrap(obs_dict.get(self.rl_agent_id, {}), info_dict.get(self.rl_agent_id, {}))
            done = False
            episode_reward = 0.0
            episode_truncated = False
            update_metrics: Dict = {}
            last_info: Dict = {}

            while not done:
                action_norm, log_prob, value = self.agent.act(obs)
                action_phys = self.action_composer.process(action_norm)
                actions = self._build_actions(action_phys, obs_dict)

                for _ in range(self.action_repeat):
                    obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(actions)
                    if self.render:
                        try:
                            self.env.render()
                        except Exception:
                            pass
                    rl_term = term_dict.get(self.rl_agent_id, False)
                    rl_trunc = trunc_dict.get(self.rl_agent_id, False)
                    if rl_term or rl_trunc:
                        done = True
                        episode_truncated = bool(rl_trunc)
                        break

                last_info = info_dict.get(self.rl_agent_id, {})
                step_info = {
                    "obs": obs,
                    "next_obs": obs_dict.get(self.rl_agent_id, {}),
                    "info": last_info,
                    "done": done,
                    "terminated": rl_term if done else False,
                    "truncated": rl_trunc if done else False,
                    "action": action_norm,
                    "timestep": 0.01,
                }
                reward, breakdown = self.reward_composer.compute(step_info)
                episode_reward += reward

                next_obs = self.obs_composer.wrap(
                    obs_dict.get(self.rl_agent_id, {}),
                    last_info,
                )
                self.obs_composer.update_prev_action(action_norm)

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
