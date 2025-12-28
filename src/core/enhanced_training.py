"""Enhanced training loop with integrated metrics, rewards, and logging.

Extends the basic TrainingLoop with:
- Metrics tracking (outcomes, rolling stats)
- Custom reward computation
- W&B and console logging integration
- Multi-agent outcome determination
- Checkpoint management and best model tracking
"""

from typing import Dict, Any, Optional, Tuple
import time
import numpy as np
from pettingzoo import ParallelEnv

from core.protocol import Agent, is_on_policy_agent
from core.obs_flatten import flatten_observation
from core.spawn_curriculum import SpawnCurriculumManager
from core.checkpoint_manager import CheckpointManager
from core.best_model_tracker import BestModelTracker
from metrics import MetricsTracker, determine_outcome, EpisodeOutcome
from loggers import WandbLogger, ConsoleLogger, CSVLogger, RichConsole
from rewards.base import RewardStrategy
from wrappers.normalize import ObservationNormalizer


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
        agent_algorithms: Optional[Dict[str, str]] = None,
        spawn_curriculum: Optional[SpawnCurriculumManager] = None,
        wandb_logger: Optional[WandbLogger] = None,
        console_logger: Optional[ConsoleLogger] = None,
        csv_logger: Optional[CSVLogger] = None,
        rich_console: Optional[RichConsole] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        best_model_tracker: Optional[BestModelTracker] = None,
        max_steps_per_episode: int = 5000,
        rolling_window: int = 100,
        save_every_n_episodes: Optional[int] = None,
        normalize_observations: bool = True,
        obs_clip: float = 10.0,
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
            agent_algorithms: Optional dict mapping agent_id -> algorithm name (e.g., 'ppo', 'sac')
                Used for namespacing algorithm-specific metrics in wandb
            spawn_curriculum: Optional SpawnCurriculumManager for progressive difficulty
            wandb_logger: Optional W&B logger
            console_logger: Optional console logger
            csv_logger: Optional CSV/JSON file logger
            rich_console: Optional Rich live dashboard for real-time metrics
            checkpoint_manager: Optional CheckpointManager for saving/loading models
            best_model_tracker: Optional BestModelTracker for tracking best models
            max_steps_per_episode: Max steps per episode
            rolling_window: Window size for rolling statistics
            save_every_n_episodes: Save checkpoint every N episodes (None = disabled)
            normalize_observations: Whether to normalize observations with running mean/std
            obs_clip: Clip normalized observations to [-clip, clip]
        """
        self.env = env
        self.agents = agents
        self.agent_rewards = agent_rewards or {}
        self.observation_presets = observation_presets or {}
        self.target_ids = target_ids or {}
        self.agent_algorithms = agent_algorithms or {}
        self.spawn_curriculum = spawn_curriculum
        self.wandb_logger = wandb_logger
        self.console_logger = console_logger
        self.csv_logger = csv_logger
        self.rich_console = rich_console
        self.checkpoint_manager = checkpoint_manager
        self.best_model_tracker = best_model_tracker
        self.max_steps_per_episode = max_steps_per_episode
        self.rolling_window = rolling_window
        self.save_every_n_episodes = save_every_n_episodes

        # Observation normalization
        self.normalize_observations = normalize_observations
        self.obs_normalizer: Optional[ObservationNormalizer] = None
        if self.normalize_observations:
            # Get observation shape from first agent
            first_agent_id = list(agents.keys())[0]
            obs_space = env.observation_spaces[first_agent_id]
            obs_shape = obs_space.shape
            self.obs_normalizer = ObservationNormalizer(obs_shape, clip=obs_clip)

        # Initialize metrics tracker for each agent
        self.metrics_trackers = {
            agent_id: MetricsTracker()
            for agent_id in agents.keys()
        }

        # Training state tracking
        self.training_start_time: Optional[float] = None
        self.episodes_trained: int = 0

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

    def run(self, episodes: int, start_episode: int = 0) -> Dict[str, Any]:
        """Run training for specified number of episodes.

        Args:
            episodes: Number of episodes to train
            start_episode: Starting episode number (for resuming training)

        Returns:
            Training statistics dict
        """
        # Track training time
        self.training_start_time = time.time()

        # Update metadata if checkpoint manager enabled
        if self.checkpoint_manager:
            self.checkpoint_manager.run_metadata.mark_running()
            self.checkpoint_manager.run_metadata.total_episodes = episodes
            self.checkpoint_manager.save_metadata()

        # Start Rich console dashboard
        if self.rich_console:
            self.rich_console.start()

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
                for episode in range(start_episode, episodes):
                    self._run_episode(episode)
                    progress.update(task, advance=1)
        else:
            for episode in range(start_episode, episodes):
                self._run_episode(episode)

        # Mark training complete
        if self.checkpoint_manager:
            training_time = time.time() - self.training_start_time
            self.checkpoint_manager.run_metadata.update_progress(
                episodes_completed=episodes,
                training_time_seconds=training_time
            )
            self.checkpoint_manager.run_metadata.mark_completed()
            self.checkpoint_manager.save_metadata()

        # Stop Rich console dashboard
        if self.rich_console:
            self.rich_console.stop()

        # Print final summary
        if self.console_logger:
            self._print_final_summary()

        # Save CSV summary
        if self.csv_logger:
            final_stats = self._get_training_stats()
            self.csv_logger.save_summary(final_stats)
            self.csv_logger.close()

        return self._get_training_stats()

    def _run_episode(self, episode_num: int):
        """Run a single episode with integrated metrics and logging.

        Args:
            episode_num: Current episode number
        """
        # Sample spawn configuration if curriculum enabled
        if self.spawn_curriculum:
            spawn_info = self.spawn_curriculum.sample_spawn()
            # Reset with spawn configuration
            obs, info = self.env.reset(options={
                'poses': spawn_info['poses'],
                'velocities': spawn_info['velocities'],
                'lock_speed_steps': spawn_info['lock_speed_steps']
            })
            # Store spawn info for logging
            self._current_spawn_stage = spawn_info['stage']
        else:
            # Standard reset
            obs, info = self.env.reset()
            self._current_spawn_stage = None

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
            reward_components_this_step = {}  # Track components for this step
            for agent_id in self.agents.keys():
                if agent_id in self.agent_rewards:
                    # Use custom reward
                    reward_info = self._build_reward_info(
                        agent_id, obs, next_obs, step_info, episode_steps,
                        terminations=terminations, truncations=truncations
                    )
                    total_reward, components = self.agent_rewards[agent_id].compute(reward_info)
                    rewards[agent_id] = total_reward
                    reward_components_this_step[agent_id] = components

                    # Accumulate components
                    for comp_name, comp_value in components.items():
                        if comp_name not in episode_reward_components[agent_id]:
                            episode_reward_components[agent_id][comp_name] = 0.0
                        episode_reward_components[agent_id][comp_name] += comp_value
                else:
                    # Use environment reward
                    rewards[agent_id] = env_rewards.get(agent_id, 0.0)

            # Render if enabled
            if hasattr(self.env, 'render_mode') and self.env.render_mode is not None:
                # Update telemetry extensions before rendering
                if hasattr(self.env, 'renderer') and self.env.renderer is not None:
                    for ext in self.env.renderer._extensions:
                        if ext.__class__.__name__ == 'TelemetryHUD':
                            # Update episode info
                            ext.update_episode_info(episode_num, episode_steps)

                            # Update rewards for each agent
                            for agent_id in rewards.keys():
                                components = reward_components_this_step.get(agent_id, {})
                                ext.update_rewards(
                                    agent_id=agent_id,
                                    reward=rewards[agent_id],
                                    components=components,
                                    reset=(episode_steps == 1)
                                )

                                # Update collision status
                                collision = step_info.get(agent_id, {}).get('collision', False)
                                ext.update_collision_status(agent_id, collision)

                self.env.render()

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

        # Update agents and collect trainer stats
        trainer_stats = {}
        for agent_id, agent in self.agents.items():
            update_stats = agent.update()
            if update_stats:
                # Namespace stats by agent_id
                trainer_stats[agent_id] = update_stats

                # Log trainer stats to W&B
                if self.wandb_logger:
                    log_dict = {}
                    # Get algorithm name for this agent (if available)
                    algo_name = self.agent_algorithms.get(agent_id, None)

                    for stat_name, stat_value in update_stats.items():
                        # Namespace by agent_id/algorithm/metric
                        if algo_name:
                            log_dict[f'{agent_id}/{algo_name}/{stat_name}'] = stat_value
                        else:
                            # Fallback to old style if algorithm not specified
                            log_dict[f'trainer/{agent_id}/{stat_name}'] = stat_value
                    self.wandb_logger.log_metrics(log_dict, step=episode_num)

        # Determine episode outcome for each agent
        final_info = step_info if episode_steps > 0 else info

        for agent_id in self.agents.keys():
            # Determine outcome using agent-specific info
            agent_info = final_info.get(agent_id, {}) if isinstance(final_info, dict) else {}
            agent_truncated = truncations.get(agent_id, False)
            outcome = self._determine_agent_outcome(
                agent_id, agent_info, agent_truncated
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
                    agent_id=agent_id,
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

                # Log trainer stats if available
                if trainer_stats:
                    for aid, stats in trainer_stats.items():
                        stats_str = ", ".join([f"{k}={v:.4f}" for k, v in stats.items()])
                        self.console_logger.print_info(f"[{aid}] Trainer: {stats_str}")

            # Log to CSV (only for first agent for episode metrics, all agents for agent metrics)
            if self.csv_logger and agent_id == list(self.agents.keys())[0]:
                # Collect per-agent metrics
                agent_metrics_dict = {}
                for aid in self.agents.keys():
                    agent_tracker = self.metrics_trackers[aid]
                    if agent_tracker.episodes:
                        latest_metrics = agent_tracker.get_latest(1)[0]
                        agent_metrics_dict[aid] = latest_metrics.to_dict()

                self.csv_logger.log_episode(
                    episode=episode_num,
                    metrics=metrics,
                    agent_metrics=agent_metrics_dict,
                    rolling_stats=rolling_stats,
                )

            # Update Rich console (only for first agent to avoid duplicate updates)
            if self.rich_console and agent_id == list(self.agents.keys())[0]:
                self.rich_console.update_episode(
                    episode=episode_num,
                    outcome=outcome.value,
                    reward=episode_rewards[agent_id],
                    steps=episode_steps,
                    outcome_stats=rolling_stats,
                )

        # Update spawn curriculum if enabled (only for first/primary agent)
        if self.spawn_curriculum:
            # Determine success for curriculum (use first agent's outcome)
            primary_agent_id = list(self.agents.keys())[0]
            primary_outcome = determine_outcome(
                final_info.get(primary_agent_id, {}),
                truncations.get(primary_agent_id, False)
            )
            success = primary_outcome.is_success()

            # Observe episode outcome
            curriculum_state = self.spawn_curriculum.observe(episode_num, success)

            # Log curriculum state transition
            if curriculum_state['changed']:
                msg = (
                    f"Spawn curriculum: {curriculum_state['stage']} "
                    f"(success rate: {curriculum_state['success_rate']:.2%})"
                )
                if self.console_logger:
                    self.console_logger.print_info(msg)

            # Log curriculum metrics to W&B
            if self.wandb_logger:
                self.wandb_logger.log_metrics({
                    'curriculum/stage': curriculum_state['stage'],
                    'curriculum/stage_index': curriculum_state['stage_index'],
                    'curriculum/success_rate': curriculum_state['success_rate'] or 0.0,
                    'curriculum/stage_success_rate': curriculum_state['stage_success_rate'] or 0.0,
                }, step=episode_num)

        # Handle checkpointing
        if self.checkpoint_manager:
            self._handle_checkpointing(episode_num)

    def _build_reward_info(
        self,
        agent_id: str,
        obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
        step: int,
        terminations: Optional[Dict[str, bool]] = None,
        truncations: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Build reward info dict for custom reward computation.

        Args:
            agent_id: ID of agent to build info for
            obs: Current observations
            next_obs: Next observations
            info: Step info from environment
            step: Current step number
            terminations: Termination flags from environment (done flags)
            truncations: Truncation flags from environment (timeout flags)

        Returns:
            Reward info dict for RewardStrategy.compute()
        """
        # For adversarial tasks, we need target agent info
        # Assume target_id is stored in agent config or derived from roles
        target_id = self._get_target_id(agent_id, info)

        # Get done and truncated flags for this agent
        terminated = terminations.get(agent_id, False) if terminations else False
        truncated = truncations.get(agent_id, False) if truncations else False
        done = terminated or truncated

        reward_info = {
            'obs': obs.get(agent_id, {}),
            'next_obs': next_obs.get(agent_id, {}),
            'info': info,
            'step': step,
            'done': done,  # Actual done flag from environment
            'truncated': truncated,  # Actual truncated flag from environment
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

    def _handle_checkpointing(self, episode_num: int):
        """Handle checkpoint saving logic.

        Args:
            episode_num: Current episode number
        """
        # Get rolling stats for primary agent (first trainable agent)
        primary_agent_id = self._get_primary_agent_id()
        if not primary_agent_id:
            return

        tracker = self.metrics_trackers[primary_agent_id]
        rolling_stats = tracker.get_rolling_stats(window=self.rolling_window)
        success_rate = rolling_stats.get('success_rate', 0.0)

        # Update metadata with progress
        if self.training_start_time:
            training_time = time.time() - self.training_start_time
            self.checkpoint_manager.run_metadata.update_progress(
                episodes_completed=episode_num + 1,
                training_time_seconds=training_time,
                latest_metric_value=success_rate
            )

        # Check if this is a new best model
        is_new_best = False
        if self.best_model_tracker:
            is_new_best = self.best_model_tracker.is_new_best(success_rate, episode_num)

        # Save checkpoint if needed
        should_save_periodic = (
            self.save_every_n_episodes and
            (episode_num + 1) % self.save_every_n_episodes == 0
        )

        if is_new_best or should_save_periodic:
            # Collect agent states
            agent_states = {}
            optimizer_states = {}

            for agent_id, agent in self.agents.items():
                # Skip non-trainable agents (e.g., FTG)
                if hasattr(agent, 'get_state'):
                    agent_states[agent_id] = agent.get_state()
                if hasattr(agent, 'get_optimizer_state'):
                    optimizer_states[agent_id] = agent.get_optimizer_state()

            # Build training state
            training_state = {
                'episode': episode_num,
                'rolling_stats': rolling_stats,
            }

            if self.spawn_curriculum:
                training_state['curriculum_stage'] = self.spawn_curriculum.current_stage

            # Save checkpoint
            checkpoint_type = "best" if is_new_best else "periodic"
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                episode=episode_num,
                agent_states=agent_states,
                optimizer_states=optimizer_states if optimizer_states else None,
                training_state=training_state,
                checkpoint_type=checkpoint_type,
                metric_value=success_rate,
            )

            # Update best model tracker
            if is_new_best:
                self.best_model_tracker.update_best(
                    value=success_rate,
                    episode=episode_num,
                    checkpoint_path=checkpoint_path
                )

                self.checkpoint_manager.run_metadata.update_best(
                    metric_value=success_rate,
                    episode=episode_num,
                    checkpoint_path=checkpoint_path
                )

                if self.console_logger:
                    smoothed = self.best_model_tracker.get_smoothed_value()
                    self.console_logger.print_info(
                        f"New best model! Success rate: {smoothed:.2%} @ episode {episode_num}"
                    )

            # Cleanup old checkpoints
            best_checkpoints = None
            if self.best_model_tracker:
                best_checkpoints = self.best_model_tracker.get_best_checkpoints(
                    n=self.checkpoint_manager.keep_best_n
                )

            self.checkpoint_manager.cleanup_checkpoints(best_checkpoints=best_checkpoints)

        # Save metadata every episode
        self.checkpoint_manager.save_metadata()

    def _get_primary_agent_id(self) -> Optional[str]:
        """Get primary trainable agent ID.

        Returns:
            Agent ID of primary trainable agent, or None if none found
        """
        for agent_id, agent in self.agents.items():
            # Skip non-trainable agents
            if hasattr(agent, 'get_state'):
                return agent_id
        return None

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint and restore agent states.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dict containing training state

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not self.checkpoint_manager:
            raise ValueError("CheckpointManager not configured")

        # Load checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        # Restore agent states
        agent_states = checkpoint.get('agent_states', {})
        for agent_id, state in agent_states.items():
            if agent_id in self.agents and hasattr(self.agents[agent_id], 'load_state'):
                self.agents[agent_id].load_state(state)

        # Restore optimizer states
        optimizer_states = checkpoint.get('optimizer_states', {})
        if optimizer_states:
            for agent_id, state in optimizer_states.items():
                if agent_id in self.agents and hasattr(self.agents[agent_id], 'load_optimizer_state'):
                    self.agents[agent_id].load_optimizer_state(state)

        # Return training state for caller to handle
        return checkpoint.get('training_state', {})


__all__ = ['EnhancedTrainingLoop']
