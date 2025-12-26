"""Enhanced training loop with integrated metrics, rewards, and logging.

Extends the basic TrainingLoop with:
- Metrics tracking (outcomes, rolling stats)
- Custom reward computation
- W&B and console logging integration
- Multi-agent outcome determination
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from pettingzoo import ParallelEnv

from v2.core.protocol import Agent, is_on_policy_agent
from v2.core.obs_flatten import flatten_observation
from v2.metrics import MetricsTracker, determine_outcome, EpisodeOutcome
from v2.loggers import WandbLogger, ConsoleLogger
from v2.rewards.base import RewardStrategy


class EnhancedTrainingLoop:
    """Enhanced training loop with integrated v2 components.

    Integrates metrics tracking, custom rewards, and logging into a unified
    training pipeline. Designed to work seamlessly with scenario configuration.

    Example:
        >>> # Create from scenario
        >>> loop = EnhancedTrainingLoop(
        ...     env=env,
        ...     agents=agents,
        ...     agent_rewards=agent_rewards,  # Dict of RewardStrategy per agent
        ...     wandb_logger=wandb_logger,
        ...     console_logger=console_logger,
        ... )
        >>> loop.run(episodes=1500)
    """

    def __init__(
        self,
        env: ParallelEnv,
        agents: Dict[str, Agent],
        agent_rewards: Optional[Dict[str, RewardStrategy]] = None,
        observation_presets: Optional[Dict[str, str]] = None,
        target_ids: Optional[Dict[str, Optional[str]]] = None,
        wandb_logger: Optional[WandbLogger] = None,
        console_logger: Optional[ConsoleLogger] = None,
        max_steps_per_episode: int = 5000,
        rolling_window: int = 100,
    ):
        """Initialize enhanced training loop.

        Args:
            env: PettingZoo ParallelEnv
            agents: Dict mapping agent_id -> Agent
            agent_rewards: Optional dict mapping agent_id -> RewardStrategy
                If provided, these custom rewards override env rewards
            observation_presets: Optional dict mapping agent_id -> preset name
                Used to flatten Dict observations. If None, observations are passed as-is.
            target_ids: Optional dict mapping agent_id -> target_id
                For adversarial tasks where agent observes target state
            wandb_logger: Optional W&B logger
            console_logger: Optional console logger
            max_steps_per_episode: Max steps per episode
            rolling_window: Window size for rolling statistics
        """
        self.env = env
        self.agents = agents
        self.agent_rewards = agent_rewards or {}
        self.observation_presets = observation_presets or {}
        self.target_ids = target_ids or {}
        self.wandb_logger = wandb_logger
        self.console_logger = console_logger
        self.max_steps_per_episode = max_steps_per_episode
        self.rolling_window = rolling_window

        # Initialize metrics tracker for each agent
        self.metrics_trackers = {
            agent_id: MetricsTracker()
            for agent_id in agents.keys()
        }

    def _flatten_obs(self, agent_id: str, obs: Any, all_obs: Optional[Dict[str, Any]] = None) -> Any:
        """Flatten observation for agent if preset is configured.

        Args:
            agent_id: Agent ID
            obs: Raw observation from environment for this agent
            all_obs: Optional dict of all agent observations (for extracting target state)

        Returns:
            Flattened observation if preset configured, otherwise original obs
        """
        if agent_id not in self.observation_presets:
            return obs

        preset = self.observation_presets[agent_id]
        target_id = self.target_ids.get(agent_id, None)

        # If target_id specified and we have all observations, add target state to obs
        if target_id and all_obs and target_id in all_obs:
            # Create combined observation dict with central_state
            combined_obs = dict(obs)  # Copy agent's own observation
            combined_obs['central_state'] = all_obs[target_id]  # Add target as central_state
            return flatten_observation(combined_obs, preset=preset, target_id=target_id)
        else:
            return flatten_observation(obs, preset=preset, target_id=target_id)

    def run(self, episodes: int) -> Dict[str, Any]:
        """Run training for specified number of episodes.

        Args:
            episodes: Number of episodes to train

        Returns:
            Training statistics dict
        """
        if self.console_logger:
            progress = self.console_logger.create_progress(
                total=episodes,
                description="Training"
            )
        else:
            progress = None

        # Run training with progress tracking
        if progress:
            with progress:
                task = progress.add_task("[cyan]Episodes", total=episodes)
                for episode in range(episodes):
                    self._run_episode(episode)
                    progress.update(task, advance=1)
        else:
            for episode in range(episodes):
                self._run_episode(episode)

        # Print final summary
        if self.console_logger:
            self._print_final_summary()

        return self._get_training_stats()

    def _run_episode(self, episode_num: int):
        """Run a single episode with integrated metrics and logging.

        Args:
            episode_num: Current episode number
        """
        # Reset environment and reward strategies
        obs, info = self.env.reset()

        for agent_id, reward_strategy in self.agent_rewards.items():
            reward_strategy.reset()

        # Episode tracking
        episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        episode_reward_components = {agent_id: {} for agent_id in self.agents.keys()}
        episode_steps = 0
        done = {agent_id: False for agent_id in self.agents.keys()}

        # Run episode
        while not all(done.values()) and episode_steps < self.max_steps_per_episode:
            # Select actions
            actions = {}
            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    # Flatten observation if preset configured
                    flat_obs = self._flatten_obs(agent_id, obs[agent_id], all_obs=obs)
                    actions[agent_id] = agent.act(flat_obs, deterministic=False)

            # Step environment
            next_obs, env_rewards, terminations, truncations, step_info = self.env.step(actions)

            # Compute custom rewards if provided
            rewards = {}
            for agent_id in self.agents.keys():
                if agent_id in self.agent_rewards:
                    # Use custom reward
                    reward_info = self._build_reward_info(
                        agent_id, obs, next_obs, step_info, episode_steps
                    )
                    total_reward, components = self.agent_rewards[agent_id].compute(reward_info)
                    rewards[agent_id] = total_reward

                    # Accumulate components
                    for comp_name, comp_value in components.items():
                        if comp_name not in episode_reward_components[agent_id]:
                            episode_reward_components[agent_id][comp_name] = 0.0
                        episode_reward_components[agent_id][comp_name] += comp_value
                else:
                    # Use environment reward
                    rewards[agent_id] = env_rewards.get(agent_id, 0.0)

            # Update agents
            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    episode_rewards[agent_id] += rewards[agent_id]

                    # Store experience (flatten observations if configured)
                    flat_obs = self._flatten_obs(agent_id, obs[agent_id], all_obs=obs)
                    flat_next_obs = self._flatten_obs(agent_id, next_obs[agent_id], all_obs=next_obs)

                    # Different storage methods for on-policy vs off-policy
                    terminated = terminations.get(agent_id, False)
                    truncated = truncations.get(agent_id, False)
                    done_flag = terminated or truncated

                    if is_on_policy_agent(agent):
                        # On-policy agents (PPO): store(obs, action, reward, done, terminated)
                        agent.store(flat_obs, actions[agent_id], rewards[agent_id], done_flag, terminated)
                    else:
                        # Off-policy agents: store_transition(obs, action, reward, next_obs, done)
                        agent.store_transition(flat_obs, actions[agent_id], rewards[agent_id], flat_next_obs, done_flag)

            # Update observations and done flags
            obs = next_obs
            for agent_id in self.agents.keys():
                done[agent_id] = terminations.get(agent_id, False) or truncations.get(agent_id, False)

            episode_steps += 1

        # Finish paths for on-policy agents
        for agent_id, agent in self.agents.items():
            if is_on_policy_agent(agent):
                agent.finish_path()

        # Determine episode outcome for each agent
        final_info = step_info if episode_steps > 0 else info

        for agent_id in self.agents.keys():
            # Determine outcome
            outcome = self._determine_agent_outcome(
                agent_id, final_info, any(truncations.values())
            )

            # Create episode metrics
            metrics = self.metrics_trackers[agent_id].add_episode(
                episode=episode_num,
                outcome=outcome,
                total_reward=episode_rewards[agent_id],
                steps=episode_steps,
                reward_components=episode_reward_components.get(agent_id, {}),
            )

            # Get rolling stats
            rolling_stats = self.metrics_trackers[agent_id].get_rolling_stats(
                window=self.rolling_window
            )

            # Log to W&B
            if self.wandb_logger:
                self.wandb_logger.log_episode(
                    episode=episode_num,
                    metrics=metrics,
                    rolling_stats=rolling_stats,
                    extra={'agent_id': agent_id},
                )

            # Log to console (only for first agent to avoid clutter)
            if self.console_logger and agent_id == list(self.agents.keys())[0]:
                self.console_logger.log_episode(
                    episode=episode_num,
                    outcome=outcome.value,
                    reward=episode_rewards[agent_id],
                    steps=episode_steps,
                    success_rate=rolling_stats.get('success_rate'),
                    avg_reward=rolling_stats.get('avg_reward'),
                )

    def _build_reward_info(
        self,
        agent_id: str,
        obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """Build reward info dict for custom reward computation.

        Args:
            agent_id: ID of agent to build info for
            obs: Current observations
            next_obs: Next observations
            info: Step info from environment
            step: Current step number

        Returns:
            Reward info dict for RewardStrategy.compute()
        """
        # For adversarial tasks, we need target agent info
        # Assume target_id is stored in agent config or derived from roles
        target_id = self._get_target_id(agent_id, info)

        reward_info = {
            'obs': obs.get(agent_id, {}),
            'next_obs': next_obs.get(agent_id, {}),
            'info': info,
            'step': step,
            'done': False,  # Will be updated on final step
            'truncated': False,
            'timestep': 0.01,  # TODO: Get from env config
        }

        # Add target obs if available (for adversarial tasks)
        if target_id and target_id in next_obs:
            reward_info['target_obs'] = next_obs[target_id]

        return reward_info

    def _get_target_id(self, agent_id: str, info: Dict[str, Any]) -> Optional[str]:
        """Get target agent ID for adversarial reward computation.

        Args:
            agent_id: Current agent ID
            info: Environment info

        Returns:
            Target agent ID or None
        """
        # TODO: Get from agent config or scenario
        # For now, simple heuristic: if 2 agents, target is the other one
        agent_ids = list(self.agents.keys())
        if len(agent_ids) == 2:
            return agent_ids[1] if agent_id == agent_ids[0] else agent_ids[0]
        return None

    def _determine_agent_outcome(
        self,
        agent_id: str,
        info: Dict[str, Any],
        truncated: bool,
    ) -> EpisodeOutcome:
        """Determine episode outcome for an agent.

        Args:
            agent_id: Agent ID
            info: Final episode info
            truncated: Whether episode was truncated

        Returns:
            EpisodeOutcome enum
        """
        # Use standard outcome determination
        # TODO: Handle multi-agent case properly
        return determine_outcome(info, truncated)

    def _print_final_summary(self):
        """Print final training summary to console."""
        if not self.console_logger:
            return

        # Get stats for first agent (main trainable agent)
        main_agent_id = list(self.agents.keys())[0]
        tracker = self.metrics_trackers[main_agent_id]

        # Get overall stats
        stats = tracker.get_rolling_stats()
        outcome_counts = tracker.get_outcome_counts()

        # Print summary
        self.console_logger.print_summary({
            'Total Episodes': stats['total_episodes'],
            'Success Rate': stats['success_rate'],
            'Average Reward': stats['avg_reward'],
            'Average Steps': stats['avg_steps'],
        })

        # Print outcome distribution
        self.console_logger.print_outcome_distribution(outcome_counts)

    def _get_training_stats(self) -> Dict[str, Any]:
        """Get final training statistics.

        Returns:
            Dict of training statistics
        """
        stats = {}
        for agent_id, tracker in self.metrics_trackers.items():
            stats[agent_id] = {
                'episodes': tracker.episodes,
                'rolling_stats': tracker.get_rolling_stats(),
                'outcome_counts': tracker.get_outcome_counts(),
            }
        return stats


__all__ = ['EnhancedTrainingLoop']
