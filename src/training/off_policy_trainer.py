"""Step-based off-policy training loop for SAC, TD3, DQN."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from metrics.outcomes import determine_outcome
from replay.replay_buffer import ReplayBuffer
from training.hooks import TrainingHook
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


class OffPolicyTrainer:
    """Step-based training loop for off-policy RL agents.

    Flow per step:
      1. act (random if step < learning_starts, else agent.act)
      2. step env (with action_repeat)
      3. compute reward via RewardComposer
      4. add transition to ReplayBuffer
      5. every train_freq steps after learning_starts: sample + agent.update
      6. on episode end: call hooks.on_episode_end, reset env
    """

    def __init__(
        self,
        env: Any,
        rl_agent_id: str,
        agent: Any,
        other_agents: Dict[str, Any],
        obs_composer: ObservationComposer,
        reward_composer: RewardComposer,
        action_composer: ActionComposer,
        replay_buffer: ReplayBuffer,
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
        self.replay_buffer = replay_buffer
        self.action_repeat = max(1, int(action_repeat))
        self.hooks = hooks or []
        self.render = render

    def _build_actions(self, rl_action_phys: np.ndarray, obs_dict: Dict) -> Dict[str, np.ndarray]:
        actions: Dict[str, np.ndarray] = {self.rl_agent_id: rl_action_phys}
        for aid, other_agent in self.other_agents.items():
            if aid in obs_dict:
                try:
                    act = other_agent.act(obs_dict[aid])
                except Exception:
                    act = np.zeros(2, dtype=np.float32)
                actions[aid] = np.asarray(act, dtype=np.float32)
        return actions

    def train(
        self,
        total_steps: int,
        learning_starts: int,
        train_freq: int,
        gradient_steps: int,
        batch_size: int,
    ) -> None:
        obs_dict, info_dict = self.env.reset()
        self.obs_composer.reset()
        self.reward_composer.reset()
        obs = self.obs_composer.wrap(obs_dict.get(self.rl_agent_id, {}), info_dict.get(self.rl_agent_id, {}))

        episode = 0
        episode_reward = 0.0
        episode_truncated = False
        last_info: Dict = {}
        update_metrics: Dict = {}

        for step in range(total_steps):
            # Choose action
            if step < learning_starts:
                action = self.agent.explore()
            else:
                action = self.agent.act(obs)

            action_phys = self.action_composer.process(action)
            actions = self._build_actions(action_phys, obs_dict)

            # Step env (with action_repeat, accumulate reward)
            done = False
            rl_term = False
            rl_trunc = False
            for _ in range(self.action_repeat):
                obs_dict, _, term_dict, trunc_dict, info_dict = self.env.step(actions)
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
                "action": action,
                "timestep": 0.01,
            }
            reward, _ = self.reward_composer.compute(step_info)
            episode_reward += reward

            next_obs = self.obs_composer.wrap(obs_dict.get(self.rl_agent_id, {}), last_info)

            self.replay_buffer.add(obs, action, reward, next_obs, float(done))

            # Update
            if step >= learning_starts and (step - learning_starts) % train_freq == 0:
                for _ in range(gradient_steps):
                    if len(self.replay_buffer) >= batch_size:
                        batch = self.replay_buffer.sample(batch_size)
                        update_metrics = self.agent.update(batch)
                        for hook in self.hooks:
                            hook.on_update(update_metrics)

            if done:
                outcome = determine_outcome(last_info, truncated=episode_truncated)
                last_info["outcome"] = outcome.value
                for hook in self.hooks:
                    hook.on_episode_end(episode, episode_reward, last_info, update_metrics)

                obs_dict, info_dict = self.env.reset()
                self.obs_composer.reset()
                self.reward_composer.reset()
                obs = self.obs_composer.wrap(obs_dict.get(self.rl_agent_id, {}), info_dict.get(self.rl_agent_id, {}))
                episode_reward = 0.0
                episode_truncated = False
                episode += 1
            else:
                obs = next_obs

        for hook in self.hooks:
            hook.on_training_end()
