"""Simple training loop - replaces the 2,011-line train_runner.py."""
from typing import Any, Dict, List, Optional, Callable
import numpy as np
from pettingzoo import ParallelEnv

from core.protocol import Agent, is_on_policy_agent, is_off_policy_agent


class TrainingLoop:
    """Simple training loop for RL agents.

    This replaces the complex train_runner.py with a clean, focused implementation.
    No wrapper layers, no factory abstractions - just the core training logic.
    """

    def __init__(
        self,
        env: ParallelEnv,
        agents: Dict[str, Agent],
        max_episodes: int,
        max_steps_per_episode: int = 1000,
        update_frequency: int = 1,
        log_callback: Optional[Callable] = None,
        checkpoint_callback: Optional[Callable] = None,
    ):
        """Initialize training loop.

        Args:
            env: PettingZoo ParallelEnv
            agents: Dictionary mapping agent_id -> Agent
            max_episodes: Number of episodes to train
            max_steps_per_episode: Max steps per episode
            update_frequency: How often to call agent.update() (in episodes for on-policy, steps for off-policy)
            log_callback: Optional callback(episode, stats) for logging
            checkpoint_callback: Optional callback(episode, agents) for checkpointing
        """
        self.env = env
        self.agents = agents
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_frequency = update_frequency
        self.log_callback = log_callback
        self.checkpoint_callback = checkpoint_callback

        # Determine agent types
        self.agent_types = {
            agent_id: "on_policy" if is_on_policy_agent(agent) else "off_policy"
            for agent_id, agent in agents.items()
        }

    def run(self) -> Dict[str, List[Dict[str, float]]]:
        """Run the training loop.

        Returns:
            training_history: Dictionary mapping agent_id -> list of episode stats
        """
        training_history = {agent_id: [] for agent_id in self.agents.keys()}

        for episode in range(self.max_episodes):
            episode_stats = self._run_episode(episode)

            # Store stats
            for agent_id, stats in episode_stats.items():
                training_history[agent_id].append(stats)

            # Callbacks
            if self.log_callback:
                self.log_callback(episode, episode_stats)

            if self.checkpoint_callback:
                self.checkpoint_callback(episode, self.agents)

        return training_history

    def _run_episode(self, episode_num: int) -> Dict[str, Dict[str, float]]:
        """Run a single episode.

        Args:
            episode_num: Current episode number

        Returns:
            episode_stats: Dictionary mapping agent_id -> episode statistics
        """
        obs, info = self.env.reset()
        episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        episode_lengths = {agent_id: 0 for agent_id in self.agents.keys()}
        training_stats = {agent_id: {} for agent_id in self.agents.keys()}

        step = 0
        done = {agent_id: False for agent_id in self.agents.keys()}

        while not all(done.values()) and step < self.max_steps_per_episode:
            # Select actions for all agents
            actions = {}
            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    actions[agent_id] = agent.act(obs[agent_id], deterministic=False)

            # Step environment
            next_obs, rewards, terminations, truncations, info = self.env.step(actions)

            # Store transitions and update done flags
            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    reward = rewards.get(agent_id, 0.0)
                    terminated = terminations.get(agent_id, False)
                    truncated = truncations.get(agent_id, False)
                    episode_done = terminated or truncated

                    # Store transition (method differs by agent type)
                    if is_on_policy_agent(agent):
                        agent.store(obs[agent_id], actions[agent_id], reward, episode_done, terminated=terminated)
                    else:
                        agent.store_transition(obs[agent_id], actions[agent_id], reward, next_obs[agent_id], episode_done)

                    episode_rewards[agent_id] += reward
                    episode_lengths[agent_id] += 1

                    # Update done flag
                    if episode_done:
                        done[agent_id] = True

                        # For on-policy agents, finish the path
                        if is_on_policy_agent(agent):
                            agent.finish_path()

            # Update observation
            obs = next_obs
            step += 1

            # Off-policy updates (every step)
            for agent_id, agent in self.agents.items():
                if self.agent_types.get(agent_id) != "off_policy":
                    continue
                if step % self.update_frequency == 0:
                    stats = agent.update()
                    if stats:
                        training_stats[agent_id] = stats

        # On-policy updates (end of episode)
        for agent_id, agent in self.agents.items():
            if self.agent_types[agent_id] == "on_policy":
                if episode_num % self.update_frequency == 0:
                    stats = agent.update()
                    if stats:
                        training_stats[agent_id] = stats

        # Compile episode statistics
        episode_stats = {}
        for agent_id in self.agents.keys():
            episode_stats[agent_id] = {
                "episode_reward": episode_rewards[agent_id],
                "episode_length": episode_lengths[agent_id],
                **training_stats.get(agent_id, {}),
            }

        return episode_stats


class EvaluationLoop:
    """Simple evaluation loop for trained agents."""

    def __init__(
        self,
        env: ParallelEnv,
        agents: Dict[str, Agent],
        num_episodes: int,
        max_steps_per_episode: int = 1000,
    ):
        """Initialize evaluation loop.

        Args:
            env: PettingZoo ParallelEnv
            agents: Dictionary mapping agent_id -> Agent
            num_episodes: Number of episodes to evaluate
            max_steps_per_episode: Max steps per episode
        """
        self.env = env
        self.agents = agents
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode

    def run(self) -> Dict[str, Dict[str, float]]:
        """Run evaluation.

        Returns:
            results: Dictionary with mean/std episode rewards and lengths per agent
        """
        episode_rewards = {agent_id: [] for agent_id in self.agents.keys()}
        episode_lengths = {agent_id: [] for agent_id in self.agents.keys()}

        for episode in range(self.num_episodes):
            obs, info = self.env.reset()
            done = {agent_id: False for agent_id in self.agents.keys()}
            ep_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
            ep_lengths = {agent_id: 0 for agent_id in self.agents.keys()}

            step = 0
            while not all(done.values()) and step < self.max_steps_per_episode:
                actions = {}
                for agent_id, agent in self.agents.items():
                    if not done[agent_id]:
                        actions[agent_id] = agent.act(obs[agent_id], deterministic=True)

                next_obs, rewards, terminations, truncations, info = self.env.step(actions)

                for agent_id in self.agents.keys():
                    if not done[agent_id]:
                        ep_rewards[agent_id] += rewards.get(agent_id, 0.0)
                        ep_lengths[agent_id] += 1

                        if terminations.get(agent_id, False) or truncations.get(agent_id, False):
                            done[agent_id] = True

                obs = next_obs
                step += 1

            for agent_id in self.agents.keys():
                episode_rewards[agent_id].append(ep_rewards[agent_id])
                episode_lengths[agent_id].append(ep_lengths[agent_id])

        # Compute statistics
        results = {}
        for agent_id in self.agents.keys():
            results[agent_id] = {
                "mean_reward": np.mean(episode_rewards[agent_id]),
                "std_reward": np.std(episode_rewards[agent_id]),
                "mean_length": np.mean(episode_lengths[agent_id]),
                "std_length": np.std(episode_lengths[agent_id]),
            }

        return results
