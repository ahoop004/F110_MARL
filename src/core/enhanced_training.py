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
import logging
import numpy as np
from pettingzoo import ParallelEnv

from core.protocol import Agent, is_on_policy_agent
from core.obs_flatten import flatten_observation
from core.spawn_curriculum import SpawnCurriculumManager
from core.checkpoint_manager import CheckpointManager
from core.best_model_tracker import BestModelTracker
from core.evaluator import Evaluator, EvaluationConfig
from metrics import MetricsTracker, determine_outcome, EpisodeOutcome
from loggers import WandbLogger, ConsoleLogger, CSVLogger, RichConsole
from rewards.base import RewardStrategy
from wrappers.normalize import ObservationNormalizer

# Set up logger for error handling
logger = logging.getLogger(__name__)


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
        ftg_schedules: Optional[Dict[str, Dict[str, Any]]] = None,
        wandb_logger: Optional[WandbLogger] = None,
        console_logger: Optional[ConsoleLogger] = None,
        csv_logger: Optional[CSVLogger] = None,
        rich_console: Optional[RichConsole] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        best_model_tracker: Optional[BestModelTracker] = None,
        best_eval_model_tracker: Optional[BestModelTracker] = None,
        evaluation_config: Optional[EvaluationConfig] = None,
        eval_every_n_episodes: Optional[int] = None,
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
            best_model_tracker: Optional BestModelTracker for tracking best training models
            best_eval_model_tracker: Optional BestModelTracker for tracking best eval models
            evaluation_config: Optional EvaluationConfig for periodic evaluation
            eval_every_n_episodes: Run evaluation every N episodes (None = disabled)
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
        self.ftg_schedules = ftg_schedules or {}
        self.wandb_logger = wandb_logger
        self.console_logger = console_logger
        self.csv_logger = csv_logger
        self.rich_console = rich_console
        self.checkpoint_manager = checkpoint_manager
        self.best_model_tracker = best_model_tracker
        self.best_eval_model_tracker = best_eval_model_tracker
        self.evaluation_config = evaluation_config
        self.eval_every_n_episodes = eval_every_n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.rolling_window = rolling_window
        self.save_every_n_episodes = save_every_n_episodes
        self.obs_scales = self._resolve_obs_scales()

        # Create evaluator if evaluation is enabled
        self.evaluator: Optional[Evaluator] = None
        if self.evaluation_config and self.eval_every_n_episodes:
            # Get spawn configs from environment (if available)
            spawn_configs = getattr(env, 'spawn_configs', {})
            self.evaluator = Evaluator(
                env=env,
                agents=agents,
                config=evaluation_config,
                observation_presets=observation_presets,
                target_ids=target_ids,
                obs_normalizer=None,  # Don't use normalization for eval
                obs_scales=self.obs_scales,
                spawn_configs=spawn_configs,
            )

        # Observation normalization
        self.normalize_observations = normalize_observations
        self.obs_clip = obs_clip
        self.obs_normalizer: Optional[ObservationNormalizer] = None
        if self.normalize_observations:
            # Get observation shape from first agent
            first_agent_id = list(agents.keys())[0]
            obs_space = env.observation_spaces[first_agent_id]
            obs_shape = getattr(obs_space, "shape", None)
            if obs_shape is not None:
                self.obs_normalizer = ObservationNormalizer(obs_shape, clip=obs_clip)

        # Initialize metrics tracker for each agent
        self.metrics_trackers = {
            agent_id: MetricsTracker()
            for agent_id in agents.keys()
        }

        # Primary agent selection for unified logging
        self._non_trainable_algorithms = {"ftg", "pp", "pure_pursuit"}
        self.primary_agent_id = self._get_primary_agent_id()
        self.primary_target_id = (
            self.target_ids.get(self.primary_agent_id)
            if self.primary_agent_id
            else None
        )

        # Training state tracking
        self.training_start_time: Optional[float] = None
        self.episodes_trained: int = 0
        self._phase_curriculum_state: Optional[Dict[str, Any]] = None
        self._current_spawn_stage: Optional[str] = None
        self._current_spawn_mapping: Dict[str, str] = {}

        # Evaluation tracking (separate counter for continuous eval plots)
        self.total_eval_episodes: int = 0
        self.last_eval_training_episode: int = -1

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
            return flatten_observation(
                combined_obs,
                preset=preset,
                target_id=target_id,
                scales=self.obs_scales,
            )
        else:
            return flatten_observation(
                obs,
                preset=preset,
                target_id=target_id,
                scales=self.obs_scales,
            )

    def _resolve_obs_scales(self) -> Dict[str, float]:
        """Resolve fixed observation scales from the environment."""
        scales: Dict[str, float] = {}

        lidar_range = getattr(self.env, "lidar_range", None)
        if lidar_range is not None:
            try:
                scales["lidar_range"] = float(lidar_range)
            except (TypeError, ValueError):
                pass

        params = getattr(self.env, "params", None)
        if isinstance(params, dict):
            candidates = []
            for value in (params.get("v_max"), params.get("v_min")):
                if value is None:
                    continue
                try:
                    candidates.append(abs(float(value)))
                except (TypeError, ValueError):
                    continue
            if candidates:
                speed_scale = max(candidates)
                if speed_scale > 0.0:
                    scales["speed"] = speed_scale

        primary_id = next(iter(self.agents), None)
        if primary_id:
            obs_space = getattr(self.env, "observation_spaces", {}).get(primary_id)
            pose_space = None
            if hasattr(obs_space, "spaces"):
                pose_space = obs_space.spaces.get("pose")
            if pose_space is not None and hasattr(pose_space, "low") and hasattr(pose_space, "high"):
                low = np.asarray(pose_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(pose_space.high, dtype=np.float32).reshape(-1)
                if low.size >= 2 and high.size >= 2:
                    span_x = float(high[0] - low[0])
                    span_y = float(high[1] - low[1])
                    position_scale = max(span_x, span_y)
                    if position_scale > 0.0:
                        scales["position"] = position_scale

        return scales

    def _ensure_obs_normalizer(self, obs: np.ndarray) -> None:
        """Initialize or refresh observation normalizer based on actual obs shape."""
        if not self.normalize_observations:
            return
        if self.obs_normalizer is None:
            self.obs_normalizer = ObservationNormalizer(obs.shape, clip=self.obs_clip)
            return
        if getattr(self.obs_normalizer, "obs_shape", None) != obs.shape:
            logger.warning(
                "Observation shape changed from %s to %s; resetting normalizer.",
                self.obs_normalizer.obs_shape,
                obs.shape,
            )
            self.obs_normalizer = ObservationNormalizer(obs.shape, clip=self.obs_clip)

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

        # Apply initial FTG schedule (if configured)
        if self.spawn_curriculum and self.ftg_schedules:
            self._apply_ftg_schedule(
                stage_name=self.spawn_curriculum.current_stage.name,
                stage_index=self.spawn_curriculum.current_stage_idx,
            )

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
            self._current_spawn_mapping = dict(spawn_info.get('spawn_points', {}))
        else:
            # Standard reset
            obs, info = self.env.reset()
            self._current_spawn_stage = None
            self._current_spawn_mapping = {}

        # Capture spawn points from env info (random spawn mode)
        if isinstance(info, dict):
            for agent_id, agent_info in info.items():
                if not isinstance(agent_info, dict):
                    continue
                spawn_point = agent_info.get("spawn_point")
                if spawn_point and agent_id not in self._current_spawn_mapping:
                    self._current_spawn_mapping[agent_id] = spawn_point

        for agent_id, reward_strategy in self.agent_rewards.items():
            reward_strategy.reset()

        # Episode tracking
        episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        episode_reward_components = {agent_id: {} for agent_id in self.agents.keys()}
        episode_steps = 0
        done = {agent_id: False for agent_id in self.agents.keys()}
        success_transitions = {
            agent_id: []
            for agent_id, agent in self.agents.items()
            if callable(getattr(agent, "store_success_transition", None))
        }

        # Run episode
        while not all(done.values()) and episode_steps < self.max_steps_per_episode:
            # Select actions and cache processed observations
            actions = {}
            flat_obs_cache = {}  # Cache to avoid duplicate processing
            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    # Flatten observation if preset configured
                    flat_obs = self._flatten_obs(agent_id, obs[agent_id], all_obs=obs)

                    # Normalize observation if enabled (only for numpy arrays, not dicts)
                    if isinstance(flat_obs, np.ndarray):
                        self._ensure_obs_normalizer(flat_obs)
                    if self.obs_normalizer and isinstance(flat_obs, np.ndarray):
                        flat_obs = self.obs_normalizer.normalize(flat_obs, agent_id, update_stats=True)

                    # Cache for reuse in storage phase
                    flat_obs_cache[agent_id] = flat_obs

                    # Pass info to agent for curriculum-based velocity control (if supported)
                    agent_info = info.get(agent_id, {}) if info else {}
                    try:
                        actions[agent_id] = agent.act(flat_obs, deterministic=False, info=agent_info)
                    except TypeError:
                        # Agent doesn't support info parameter (backward compatibility)
                        actions[agent_id] = agent.act(flat_obs, deterministic=False)

            # Step environment with error handling
            try:
                next_obs, env_rewards, terminations, truncations, step_info = self.env.step(actions)
            except Exception as e:
                logger.error(f"Environment step failed at episode {episode_num}, step {episode_steps}: {e}")
                logger.error(f"Actions: {actions}")
                # Mark all agents as terminated to end episode gracefully
                terminations = {agent_id: True for agent_id in self.agents.keys()}
                truncations = {agent_id: False for agent_id in self.agents.keys()}
                next_obs = obs  # Use current obs as next_obs
                env_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
                step_info = {}

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

                    # Reuse cached observation (already flattened and normalized)
                    flat_obs = flat_obs_cache[agent_id]

                    # Process next observation (only needs to be done once)
                    flat_next_obs = self._flatten_obs(agent_id, next_obs[agent_id], all_obs=next_obs)
                    if isinstance(flat_next_obs, np.ndarray):
                        self._ensure_obs_normalizer(flat_next_obs)
                    if self.obs_normalizer and isinstance(flat_next_obs, np.ndarray):
                        flat_next_obs = self.obs_normalizer.normalize(flat_next_obs, agent_id, update_stats=False)

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

                        # Hindsight Experience Replay (HER) for off-policy agents
                        if hasattr(agent, 'store_hindsight_transition') and callable(agent.store_hindsight_transition):
                            # Calculate distance to target for HER
                            target_id = self.target_ids.get(agent_id, None)
                            if target_id and target_id in next_obs:
                                distance = self._calculate_distance_to_target(
                                    next_obs[agent_id], next_obs[target_id]
                                )
                                agent.store_hindsight_transition(
                                    flat_obs, actions[agent_id], rewards[agent_id],
                                    flat_next_obs, done_flag, distance,
                                    info=step_info.get(agent_id, {})
                                )

                        # Update off-policy agents every step (they internally check if buffer is ready)
                        try:
                            step_update_stats = agent.update()

                            # Log per-step training metrics to wandb
                            # Per-step update stats are intentionally not logged to keep W&B clean.
                        except Exception as e:
                            logger.error(f"Agent {agent_id} update failed at episode {episode_num}, step {episode_steps}: {e}")
                            # Continue training - this update will be skipped but training continues

                    if (
                        agent_id in success_transitions
                        and isinstance(flat_obs, np.ndarray)
                        and isinstance(flat_next_obs, np.ndarray)
                    ):
                        success_transitions[agent_id].append((
                            np.asarray(flat_obs, dtype=np.float32).copy(),
                            np.asarray(actions[agent_id]).copy() if not np.isscalar(actions[agent_id]) else actions[agent_id],
                            float(rewards[agent_id]),
                            np.asarray(flat_next_obs, dtype=np.float32).copy(),
                            bool(done_flag),
                        ))

            # Update observations and done flags
            obs = next_obs
            info = step_info  # Update info for next iteration (for curriculum velocity control)
            for agent_id in self.agents.keys():
                done[agent_id] = terminations.get(agent_id, False) or truncations.get(agent_id, False)

            episode_steps += 1

        # Update agents and collect trainer stats
        # Note: On-policy agents will call finish_path() internally in their update() method
        trainer_stats = {}
        for agent_id, agent in self.agents.items():
            update_stats = None
            try:
                update_stats = agent.update()
                if update_stats:
                    # Namespace stats by agent_id
                    trainer_stats[agent_id] = update_stats
                else:
                    # Debug: Log when update returns None for on-policy agents
                    if is_on_policy_agent(agent):
                        logger.debug(f"On-policy agent {agent_id} update() returned None at episode {episode_num}")
            except Exception as e:
                logger.error(f"Agent {agent_id} update failed after episode {episode_num}: {e}")
                # Continue with next agent
                continue

            # Log trainer stats to W&B
            # Per-episode update stats are intentionally not logged to keep W&B clean.

        # Determine episode outcome for each agent
        final_info = step_info if episode_steps > 0 else info

        episode_metrics: Dict[str, Any] = {}
        rolling_stats_by_agent: Dict[str, Dict[str, float]] = {}
        outcomes_by_agent: Dict[str, EpisodeOutcome] = {}

        for agent_id in self.agents.keys():
            # Determine outcome using agent-specific info
            agent_info = final_info.get(agent_id, {}) if isinstance(final_info, dict) else {}
            target_id = self.target_ids.get(agent_id)
            if target_id and isinstance(final_info, dict):
                target_info = final_info.get(target_id, {})
                if isinstance(target_info, dict):
                    agent_info = dict(agent_info)
                    agent_info["target_finished"] = bool(
                        target_info.get("finish_line", False)
                        or target_info.get("target_finished", False)
                    )
            agent_truncated = truncations.get(agent_id, False)
            outcome = self._determine_agent_outcome(
                agent_id, agent_info, agent_truncated
            )
            outcomes_by_agent[agent_id] = outcome

            # Create episode metrics
            metrics = self.metrics_trackers[agent_id].add_episode(
                episode=episode_num,
                outcome=outcome,
                total_reward=episode_rewards[agent_id],
                steps=episode_steps,
                reward_components=episode_reward_components.get(agent_id, {}),
            )
            episode_metrics[agent_id] = metrics

            # Get rolling stats
            rolling_stats = self.metrics_trackers[agent_id].get_rolling_stats(
                window=self.rolling_window
            )
            rolling_stats_by_agent[agent_id] = rolling_stats

            # Populate success replay buffer for off-policy agents
            if outcome.is_success() and agent_id in success_transitions:
                transitions = success_transitions.get(agent_id, [])
                if transitions:
                    store_success = getattr(self.agents[agent_id], "store_success_transition", None)
                    if callable(store_success):
                        for obs_t, act_t, rew_t, next_obs_t, done_t in transitions:
                            store_success(obs_t, act_t, rew_t, next_obs_t, done_t)

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

                spawn_point = self._current_spawn_mapping.get(agent_id)
                spawn_stage = self._current_spawn_stage
                csv_extra = {
                    "spawn_point": spawn_point or "",
                    "spawn_stage": spawn_stage or "",
                }

                self.csv_logger.log_episode(
                    episode=episode_num,
                    metrics=metrics,
                    agent_metrics=agent_metrics_dict,
                    rolling_stats=rolling_stats,
                    extra=csv_extra,
                )

        if self.wandb_logger and self.primary_agent_id in episode_metrics:
            primary_id = self.primary_agent_id
            metrics = episode_metrics[primary_id]
            rolling_stats = rolling_stats_by_agent.get(primary_id, {})
            primary_info = (
                final_info.get(primary_id, {}) if isinstance(final_info, dict) else {}
            )

            target_finished = False
            target_collision = False
            if self.primary_target_id and isinstance(final_info, dict):
                target_info = final_info.get(self.primary_target_id, {})
                if isinstance(target_info, dict):
                    target_finished = bool(
                        target_info.get("finish_line", False)
                        or target_info.get("target_finished", False)
                    )
                    target_collision = bool(target_info.get("collision", False))

            if primary_info.get("target_collision") is not None:
                target_collision = bool(primary_info.get("target_collision", False))

            log_dict = {
                "train/outcome": metrics.outcome.value,
                "train/success": int(metrics.success),
                "train/episode_reward": float(metrics.total_reward),
                "train/episode_steps": int(metrics.steps),
                "train/success_rate": float(rolling_stats.get("success_rate", 0.0)),
                "train/reward_mean": float(rolling_stats.get("avg_reward", 0.0)),
                "train/steps_mean": float(rolling_stats.get("avg_steps", 0.0)),
            }

            spawn_point = self._current_spawn_mapping.get(primary_id)
            if spawn_point:
                log_dict["train/spawn_point"] = spawn_point
            if self._current_spawn_stage:
                log_dict["train/spawn_stage"] = self._current_spawn_stage

            if self.primary_target_id:
                log_dict.update({
                    "target/success": int(target_finished),
                    "target/crash": int(target_collision),
                })

            self.wandb_logger.log_metrics(log_dict, step=episode_num)

        # Update spawn curriculum if enabled (only for first/primary agent)
        spawn_curriculum_state = None
        if self.spawn_curriculum:
            # Determine success for curriculum (use first agent's outcome)
            primary_agent_id = self.primary_agent_id or list(self.agents.keys())[0]
            primary_outcome = determine_outcome(
                final_info.get(primary_agent_id, {}),
                truncations.get(primary_agent_id, False)
            )
            success = primary_outcome.is_success()

            # Observe episode outcome
            curriculum_state = self.spawn_curriculum.observe(episode_num, success)
            spawn_curriculum_state = curriculum_state

            # Log curriculum state transition
            if curriculum_state['changed']:
                msg = (
                    f"Spawn curriculum: {curriculum_state['stage']} "
                    f"(success rate: {curriculum_state['success_rate']:.2%})"
                )
                if self.console_logger:
                    self.console_logger.print_info(msg)
                if self.ftg_schedules:
                    self._apply_ftg_schedule(
                        stage_name=curriculum_state['stage'],
                        stage_index=curriculum_state['stage_index'],
                    )

            # Log curriculum metrics to W&B
            if self.wandb_logger:
                self.wandb_logger.log_metrics({
                    'curriculum/stage': curriculum_state['stage'],
                    'curriculum/stage_index': curriculum_state['stage_index'],
                    'curriculum/success_rate': curriculum_state['success_rate'] or 0.0,
                    'curriculum/stage_success_rate': curriculum_state['stage_success_rate'] or 0.0,
                }, step=episode_num)

        # Update Rich console once per episode (use primary agent metrics)
        if self.rich_console:
            primary_id = self.primary_agent_id or list(self.agents.keys())[0]
            if primary_id in episode_metrics:
                metrics = episode_metrics[primary_id]
                rolling_stats = rolling_stats_by_agent.get(primary_id, {})
                algo_name = None
                if self.agent_algorithms:
                    algo_name = self.agent_algorithms.get(primary_id)
                curriculum_state = {}
                if spawn_curriculum_state:
                    curriculum_state.update({
                        "stage": spawn_curriculum_state.get("stage"),
                        "stage_index": spawn_curriculum_state.get("stage_index"),
                        "stage_success_rate": spawn_curriculum_state.get("stage_success_rate"),
                    })
                if self._phase_curriculum_state:
                    curriculum_state.update(self._phase_curriculum_state)

                self.rich_console.update_episode(
                    episode=episode_num,
                    outcome=metrics.outcome.value,
                    reward=metrics.total_reward,
                    steps=metrics.steps,
                    outcome_stats=rolling_stats,
                    curriculum_state=curriculum_state or None,
                    algo_name=algo_name,
                )

        # Handle periodic evaluation
        if self.evaluator and self.eval_every_n_episodes:
            if (episode_num + 1) % self.eval_every_n_episodes == 0:
                self._run_periodic_evaluation(episode_num)

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

        # Get timestep from environment if available, otherwise use default
        timestep = getattr(self.env, 'timestep', 0.01)

        info_for_agent = info
        if isinstance(info, dict):
            info_for_agent = info.get(agent_id, info)

        reward_info = {
            'obs': obs.get(agent_id, {}),
            'next_obs': next_obs.get(agent_id, {}),
            'info': info_for_agent,
            'step': step,
            'done': done,  # Actual done flag from environment
            'truncated': truncated,  # Actual truncated flag from environment
            'timestep': timestep,
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
        # Check if target_id is stored in target_ids mapping (set during init)
        if agent_id in self.target_ids:
            return self.target_ids[agent_id]

        # Fallback: simple heuristic for 2-agent scenarios
        agent_ids = list(self.agents.keys())
        if len(agent_ids) == 2:
            return agent_ids[1] if agent_id == agent_ids[0] else agent_ids[0]
        return None

    def _calculate_distance_to_target(
        self,
        agent_obs: Dict[str, Any],
        target_obs: Dict[str, Any],
    ) -> float:
        """Calculate Euclidean distance from agent to target.

        Args:
            agent_obs: Agent's observation dict (must contain 'pose' key)
            target_obs: Target's observation dict (must contain 'pose' key)

        Returns:
            Distance in meters, or float('inf') if positions unavailable
        """
        try:
            agent_pose = agent_obs.get('pose', None)
            target_pose = target_obs.get('pose', None)

            if agent_pose is None or target_pose is None:
                return float('inf')

            # Extract x, y positions (first 2 elements of pose)
            agent_x, agent_y = agent_pose[0], agent_pose[1]
            target_x, target_y = target_pose[0], target_pose[1]

            # Euclidean distance
            distance = np.sqrt((agent_x - target_x)**2 + (agent_y - target_y)**2)
            return float(distance)

        except (IndexError, KeyError, TypeError):
            return float('inf')

    def _apply_ftg_schedule(self, stage_name: Optional[str], stage_index: Optional[int]) -> None:
        """Apply FTG parameter schedule for the current curriculum stage."""
        if not self.ftg_schedules:
            return

        for agent_id, schedule in self.ftg_schedules.items():
            if not schedule or not schedule.get("enabled", True):
                continue

            agent = self.agents.get(agent_id)
            if agent is None:
                continue

            apply_config = getattr(agent, "apply_config", None)
            if not callable(apply_config):
                continue

            params: Dict[str, Any] = {}
            by_stage = schedule.get("by_stage", {})
            if stage_name and stage_name in by_stage:
                params.update(by_stage[stage_name])

            by_index = schedule.get("by_stage_index", {})
            if stage_index is not None:
                if stage_index in by_index:
                    params.update(by_index[stage_index])
                elif str(stage_index) in by_index:
                    params.update(by_index[str(stage_index)])

            if not params:
                continue

            apply_config(params)
            if self.console_logger:
                stage_label = stage_name if stage_name is not None else f"stage_{stage_index}"
                self.console_logger.print_info(
                    f"FTG schedule applied for {agent_id} at {stage_label}"
                )

    def _determine_agent_outcome(
        self,
        agent_id: str,
        info: Dict[str, Any],
        truncated: bool,
    ) -> EpisodeOutcome:
        """Determine episode outcome for an agent.

        Args:
            agent_id: Agent ID
            info: Final episode info (can be nested by agent or flat)
            truncated: Whether episode was truncated

        Returns:
            EpisodeOutcome enum
        """
        # Handle multi-agent case: extract agent-specific info if available
        agent_info = info.get(agent_id, info) if agent_id in info else info

        # Use standard outcome determination with agent-specific or flat info
        return determine_outcome(agent_info, truncated)

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
                'episodes': [ep.to_dict() for ep in tracker.episodes],
                'rolling_stats': tracker.get_rolling_stats(),
                'outcome_counts': tracker.get_outcome_counts(),
            }
        return stats

    def _run_periodic_evaluation(self, episode_num: int):
        """Run periodic evaluation and track best eval model.

        Args:
            episode_num: Current episode number (training episodes)
        """
        if not self.evaluator:
            return

        # Track which training episode triggered this eval
        self.last_eval_training_episode = episode_num

        if self.console_logger:
            self.console_logger.print_info(
                f"Running evaluation at training episode {episode_num + 1} "
                f"(eval episodes {self.total_eval_episodes + 1}-{self.total_eval_episodes + self.evaluation_config.num_episodes})..."
            )

        # Run evaluation
        eval_result = self.evaluator.evaluate(verbose=False)

        # Log each eval episode individually for continuous plots
        if self.wandb_logger:
            for ep_data in eval_result.episodes:
                self.total_eval_episodes += 1
                self.wandb_logger.log_metrics({
                    'eval/episode_reward': ep_data['reward'],
                    'eval/episode_steps': ep_data['steps'],
                    'eval/episode_success': int(ep_data['success']),
                    'eval/spawn_point': ep_data['spawn_point'],
                    'eval/training_episode': episode_num,  # Track which training ep this eval came from
                }, step=self.total_eval_episodes)

        # Log aggregate results (using last eval episode as step for aggregate metrics)
        if self.wandb_logger:
            self.wandb_logger.log_metrics({
                'eval_agg/success_rate': eval_result.success_rate,
                'eval_agg/avg_reward': eval_result.avg_reward,
                'eval_agg/avg_episode_length': eval_result.avg_episode_length,
                'eval_agg/std_reward': eval_result.std_reward,
                'eval_agg/std_episode_length': eval_result.std_episode_length,
                'eval_agg/training_episode': episode_num,
            }, step=self.total_eval_episodes)

            # Log outcome distribution (aggregate)
            for outcome, count in eval_result.outcome_counts.items():
                pct = (count / eval_result.num_episodes) * 100
                self.wandb_logger.log_metrics({
                    f'eval_agg/outcome_{outcome}': pct,
                }, step=self.total_eval_episodes)

        # Log to console (aggregate summary)
        if self.console_logger:
            self.console_logger.print_info(
                f"Eval complete: Success rate={eval_result.success_rate:.2%}, "
                f"Avg reward={eval_result.avg_reward:.2f}, "
                f"Avg steps={eval_result.avg_episode_length:.1f} "
                f"(eval episodes {self.total_eval_episodes - self.evaluation_config.num_episodes + 1}-{self.total_eval_episodes})"
            )

        # Check if this is a new best eval model
        if self.best_eval_model_tracker and self.checkpoint_manager:
            is_new_best_eval = self.best_eval_model_tracker.is_new_best(
                eval_result.success_rate,
                episode_num
            )

            if is_new_best_eval:
                # Save checkpoint for best eval model
                agent_states = {}
                optimizer_states = {}

                for agent_id, agent in self.agents.items():
                    if hasattr(agent, 'get_state'):
                        agent_states[agent_id] = agent.get_state()
                    if hasattr(agent, 'get_optimizer_state'):
                        optimizer_states[agent_id] = agent.get_optimizer_state()

                # Save with eval_best type
                try:
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        episode=episode_num,
                        agent_states=agent_states,
                        optimizer_states=optimizer_states if optimizer_states else None,
                        training_state={'eval_result': eval_result.to_dict()},
                        checkpoint_type="best_eval",
                        metric_value=eval_result.success_rate,
                    )

                    # Update best eval model tracker
                    self.best_eval_model_tracker.update_best(
                        value=eval_result.success_rate,
                        episode=episode_num,
                        checkpoint_path=checkpoint_path
                    )

                    if self.console_logger:
                        smoothed = self.best_eval_model_tracker.get_smoothed_value()
                        self.console_logger.print_info(
                            f"New best eval model! Eval success rate: {smoothed:.2%} @ episode {episode_num}"
                        )
                except Exception as e:
                    logger.error(f"Failed to save best eval checkpoint at episode {episode_num}: {e}")

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
                'total_eval_episodes': self.total_eval_episodes,
                'last_eval_training_episode': self.last_eval_training_episode,
            }

            if self.spawn_curriculum:
                training_state['curriculum_stage'] = self.spawn_curriculum.current_stage

            # Save checkpoint with error handling
            checkpoint_type = "best" if is_new_best else "periodic"
            try:
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
            except Exception as e:
                logger.error(f"Checkpoint save failed at episode {episode_num}: {e}")
                # Continue training despite checkpoint failure

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
        if self.agent_algorithms:
            for agent_id, algo in self.agent_algorithms.items():
                if not algo:
                    continue
                if algo.lower() not in self._non_trainable_algorithms:
                    return agent_id
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'get_state'):
                return agent_id
        return next(iter(self.agents), None)

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

        # Restore eval counters from training state
        training_state = checkpoint.get('training_state', {})
        if 'total_eval_episodes' in training_state:
            self.total_eval_episodes = training_state['total_eval_episodes']
        if 'last_eval_training_episode' in training_state:
            self.last_eval_training_episode = training_state['last_eval_training_episode']

        # Return training state for caller to handle
        return training_state


__all__ = ['EnhancedTrainingLoop']
