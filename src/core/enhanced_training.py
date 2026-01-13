"""Enhanced training loop with integrated metrics, rewards, and logging.

Extends the basic TrainingLoop with:
- Metrics tracking (outcomes, rolling stats)
- Custom reward computation
- W&B and console logging integration
- Multi-agent outcome determination
- Checkpoint management and best model tracking
"""

from typing import Dict, Any, Optional, Tuple
from collections import deque
import time
import logging
import numpy as np
from pettingzoo import ParallelEnv

from src.core.protocol import Agent, is_on_policy_agent
from src.core.obs_flatten import flatten_observation
from src.core.spawn_curriculum import SpawnCurriculumManager
from src.core.checkpoint_manager import CheckpointManager
from src.core.best_model_tracker import BestModelTracker
from src.core.evaluator import Evaluator, EvaluationConfig
from src.metrics import MetricsTracker, determine_outcome, EpisodeOutcome
from src.loggers import WandbLogger, ConsoleLogger, CSVLogger, RichConsole
from src.rewards.base import RewardStrategy

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
        frame_stacks: Optional[Dict[str, int]] = None,
        action_repeat: int = 1,
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
    ):
        """Initialize enhanced training loop.

        Args:
            env: PettingZoo ParallelEnv
            agents: Dict mapping agent_id -> Agent
            agent_rewards: Optional dict mapping agent_id -> RewardStrategy
                If provided, these custom rewards override env rewards
            observation_presets: Optional dict mapping agent_id -> preset name
                Used to flatten Dict observations. Flattening includes normalization.
            frame_stacks: Optional dict mapping agent_id -> number of frames to stack
            action_repeat: Number of environment steps to repeat each action
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

        Note on Observation Normalization:
            Observations are normalized within flatten_observation() based on the preset.
            For gaplock preset:
            - LiDAR: normalized to [0, 1] using max_range
            - Velocities: clipped to [-1, 1] using speed_scale
            - Positions: clipped to [-1, 1] using position_scale
            - Angles: sin/cos encoding (inherently bounded)
            No additional running mean/std normalization is applied.
        """
        self.env = env
        self.agents = agents
        self.agent_rewards = agent_rewards or {}
        self.observation_presets = observation_presets or {}
        self.frame_stacks = {}
        if frame_stacks:
            for agent_id, stack_size in frame_stacks.items():
                try:
                    stack_size = int(stack_size)
                except (TypeError, ValueError):
                    stack_size = 1
                if stack_size < 1:
                    stack_size = 1
                self.frame_stacks[agent_id] = stack_size
        try:
            action_repeat = int(action_repeat)
        except (TypeError, ValueError):
            action_repeat = 1
        self.action_repeat = max(1, action_repeat)
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
        self._frame_buffers: Dict[str, deque] = {}

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
                frame_stacks=self.frame_stacks,
                target_ids=target_ids,
                obs_scales=self.obs_scales,
                spawn_configs=spawn_configs,
                action_repeat=self.action_repeat,
            )

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
        self.best_eval_per_phase: Dict[int, float] = {}

    def _flatten_obs(
        self,
        agent_id: str,
        obs: Any,
        all_obs: Optional[Dict[str, Any]] = None,
        update_stack: bool = True,
    ) -> Any:
        """Flatten observation for agent if preset is configured.

        Args:
            agent_id: Agent ID
            obs: Raw observation from environment for this agent
            all_obs: Optional dict of all agent observations (for extracting target state)

        Returns:
            Flattened observation if preset configured, otherwise original obs
        """
        if agent_id not in self.observation_presets:
            flat_obs = obs
        else:
            preset = self.observation_presets[agent_id]
            target_id = self.target_ids.get(agent_id, None)

            # If target_id specified and we have all observations, add target state to obs
            if target_id and all_obs and target_id in all_obs:
                # Create combined observation dict with central_state
                combined_obs = dict(obs)  # Copy agent's own observation
                combined_obs['central_state'] = all_obs[target_id]  # Add target as central_state
                flat_obs = flatten_observation(
                    combined_obs,
                    preset=preset,
                    target_id=target_id,
                    scales=self.obs_scales,
                )
            else:
                flat_obs = flatten_observation(
                    obs,
                    preset=preset,
                    target_id=target_id,
                    scales=self.obs_scales,
                )

        return self._stack_obs(agent_id, flat_obs, update=update_stack)

    def _stack_obs(self, agent_id: str, obs: Any, update: bool = True) -> Any:
        stack_size = int(self.frame_stacks.get(agent_id, 1))
        if stack_size <= 1:
            return obs
        if isinstance(obs, dict):
            raise ValueError(
                f"Frame stacking requires flattened observations for {agent_id}"
            )
        if not isinstance(obs, np.ndarray):
            try:
                obs = np.asarray(obs, dtype=np.float32)
            except Exception as exc:
                raise ValueError(
                    f"Frame stacking requires array observations for {agent_id}"
                ) from exc
        buffer = self._frame_buffers.get(agent_id)
        if buffer is None:
            buffer = deque(maxlen=stack_size)
            self._frame_buffers[agent_id] = buffer
        if len(buffer) == 0:
            frames = [obs] * stack_size
            if update:
                buffer.extend(frames)
            return np.concatenate(frames, axis=0)
        if update:
            buffer.append(obs)
            frames = list(buffer)
        else:
            frames = list(buffer) + [obs]
            if len(frames) > stack_size:
                frames = frames[-stack_size:]
        return np.concatenate(frames, axis=0)

    def _reset_frame_buffers(self) -> None:
        self._frame_buffers = {
            agent_id: deque(maxlen=stack_size)
            for agent_id, stack_size in self.frame_stacks.items()
            if stack_size > 1
        }

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


    def run(self, episodes: int, start_episode: int = 0) -> Dict[str, Any]:
        """Run training for specified number of episodes.

        Args:
            episodes: Number of episodes to train
            start_episode: Starting episode number (for resuming training)

        Returns:
            Training statistics dict
        """
        # ========================================
        # TRAINING INITIALIZATION
        # ========================================

        # Track training time for performance metrics
        self.training_start_time = time.time()

        # Apply initial FTG (Follow-the-Gap) defender schedule if spawn curriculum enabled
        # This configures defender difficulty parameters based on curriculum stage
        if self.spawn_curriculum and self.ftg_schedules:
            self._apply_ftg_schedule(
                stage_name=self.spawn_curriculum.current_stage.name,
                stage_index=self.spawn_curriculum.current_stage_idx,
            )

        # Update checkpoint manager metadata to track run status
        if self.checkpoint_manager:
            self.checkpoint_manager.run_metadata.mark_running()
            self.checkpoint_manager.run_metadata.total_episodes = episodes
            self.checkpoint_manager.save_metadata()

        # Initialize console visualization (Rich live dashboard)
        if self.rich_console:
            self.rich_console.start()

        # Create progress bar for episode tracking
        if self.console_logger:
            progress = self.console_logger.create_progress(
                total=episodes,
                description="Training"
            )
        else:
            progress = None

        # ========================================
        # MAIN TRAINING LOOP
        # ========================================

        # Run episodes with optional progress tracking
        if progress:
            with progress:
                task = progress.add_task("[cyan]Episodes", total=episodes)
                for episode in range(start_episode, episodes):
                    self._run_episode(episode)  # Execute single episode (see below)
                    progress.update(task, advance=1)
        else:
            for episode in range(start_episode, episodes):
                self._run_episode(episode)

        # ========================================
        # TRAINING FINALIZATION
        # ========================================

        # Update final training metadata
        if self.checkpoint_manager:
            training_time = time.time() - self.training_start_time
            self.checkpoint_manager.run_metadata.update_progress(
                episodes_completed=episodes,
                training_time_seconds=training_time
            )
            self.checkpoint_manager.run_metadata.mark_completed()
            self.checkpoint_manager.save_metadata()

        # Stop console visualization
        if self.rich_console:
            self.rich_console.stop()

        # Print final training summary (success rates, rewards, etc.)
        if self.console_logger:
            self._print_final_summary()

        # Save aggregated CSV summary statistics
        if self.csv_logger:
            final_stats = self._get_training_stats()
            self.csv_logger.save_summary(final_stats)
            self.csv_logger.close()

        return self._get_training_stats()

    def _run_episode(self, episode_num: int):
        """Run a single episode with integrated metrics and logging.

        Episode Structure:
        1. Environment Reset (with curriculum sampling if enabled)
        2. Episode Loop: Action Selection → Env Step → Reward Computation → Agent Update
        3. Episode Finalization: Metrics, Logging, Checkpointing

        Args:
            episode_num: Current episode number
        """
        # ========================================
        # PHASE 1: ENVIRONMENT RESET
        # ========================================

        # Sample spawn configuration from curriculum (if enabled)
        # Curriculum controls: spawn positions, initial velocities, speed lock duration
        if self.spawn_curriculum:
            spawn_info = self.spawn_curriculum.sample_spawn(episode=episode_num)
            # Reset environment with curriculum-specified configuration
            obs, info = self.env.reset(options={
                'poses': spawn_info['poses'],              # Agent starting positions [x, y, theta]
                'velocities': spawn_info['velocities'],    # Agent initial velocities
                'lock_speed_steps': spawn_info['lock_speed_steps']  # Steps to freeze defender speed
            })
            # Store spawn metadata for logging/analysis
            self._current_spawn_stage = spawn_info['stage']
            self._current_spawn_mapping = dict(spawn_info.get('spawn_points', {}))
        else:
            # Standard reset without curriculum (random spawns)
            obs, info = self.env.reset()
            self._current_spawn_stage = None
            self._current_spawn_mapping = {}

        self._reset_frame_buffers()

        # Capture spawn points from env info (for logging in random spawn mode)
        if isinstance(info, dict):
            for agent_id, agent_info in info.items():
                if not isinstance(agent_info, dict):
                    continue
                spawn_point = agent_info.get("spawn_point")
                if spawn_point and agent_id not in self._current_spawn_mapping:
                    self._current_spawn_mapping[agent_id] = spawn_point

        # Reset reward strategies (clears internal state for new episode)
        for agent_id, reward_strategy in self.agent_rewards.items():
            reward_strategy.reset()

        # ========================================
        # PHASE 2: EPISODE INITIALIZATION
        # ========================================

        # Initialize episode-level tracking variables
        episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}  # Cumulative reward per agent
        episode_reward_components = {agent_id: {} for agent_id in self.agents.keys()}  # Component breakdown
        episode_steps = 0  # Step counter
        done = {agent_id: False for agent_id in self.agents.keys()}  # Termination flags

        # Success replay buffer: stores transitions from successful episodes
        # Used for off-policy agents that support HER (Hindsight Experience Replay)
        success_transitions = {
            agent_id: []
            for agent_id, agent in self.agents.items()
            if callable(getattr(agent, "store_success_transition", None))
        }

        # ========================================
        # PHASE 3: EPISODE EXECUTION LOOP
        # ========================================

        # Run until all agents done OR max steps reached
        while not all(done.values()) and episode_steps < self.max_steps_per_episode:

            # -------------------- ACTION SELECTION --------------------

            actions = {}
            flat_obs_cache = {}  # Cache processed obs to avoid duplicate flattening

            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    # Step 1: Flatten and normalize observation from dict to vector
                    # Converts {scans: [...], pose: [...], velocity: [...]} → np.ndarray
                    # Normalization is performed within flatten_observation() based on preset
                    # For gaplock: LiDAR→[0,1], velocities→[-1,1], positions→[-1,1], angles→sin/cos
                    flat_obs = self._flatten_obs(agent_id, obs[agent_id], all_obs=obs)

                    # Cache processed observation for later reuse in storage
                    flat_obs_cache[agent_id] = flat_obs

                    # Step 2: Select action from agent's policy
                    # Pass info dict for agents that use curriculum-based velocity control
                    agent_info = info.get(agent_id, {}) if info else {}
                    try:
                        actions[agent_id] = agent.act(flat_obs, deterministic=False, info=agent_info)
                    except TypeError:
                        # Backward compatibility: some agents don't accept info parameter
                        actions[agent_id] = agent.act(flat_obs, deterministic=False)

            # -------------------- ENVIRONMENT STEP --------------------

            rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
            reward_components_this_step: Dict[str, Dict[str, float]] = {}
            terminations = {agent_id: False for agent_id in self.agents.keys()}
            truncations = {agent_id: False for agent_id in self.agents.keys()}
            step_info: Dict[str, Any] = {}
            next_obs = obs
            steps_this_action = 0
            done_snapshot = dict(done)

            while (
                steps_this_action < self.action_repeat
                and episode_steps < self.max_steps_per_episode
                and not all(done.values())
            ):
                prev_obs = obs
                # Execute physics simulation step with all agents' actions
                try:
                    next_obs, env_rewards, terminations, truncations, step_info = self.env.step(actions)
                except Exception as e:
                    # Graceful degradation: log error and mark episode as terminated
                    logger.error(f"Environment step failed at episode {episode_num}, step {episode_steps}: {e}")
                    logger.error(f"Actions: {actions}")
                    terminations = {agent_id: True for agent_id in self.agents.keys()}
                    truncations = {agent_id: False for agent_id in self.agents.keys()}
                    next_obs = prev_obs  # Reuse current observation
                    env_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
                    step_info = {}

                # -------------------- REWARD COMPUTATION --------------------

                # Compute custom rewards (if configured) or use environment rewards
                for agent_id in self.agents.keys():
                    if agent_id in self.agent_rewards:
                        # Custom reward computation (e.g., gaplock_full reward strategy)
                        # Builds reward from multiple components: terminal, distance, pressure, etc.
                        reward_info = self._build_reward_info(
                            agent_id, prev_obs, next_obs, step_info, episode_steps,
                            terminations=terminations, truncations=truncations
                        )
                        total_reward, components = self.agent_rewards[agent_id].compute(reward_info)
                        rewards[agent_id] += float(total_reward)
                        if components:
                            comp_totals = reward_components_this_step.setdefault(agent_id, {})
                            for comp_name, comp_value in components.items():
                                comp_totals[comp_name] = comp_totals.get(comp_name, 0.0) + float(comp_value)
                    else:
                        # Fall back to environment's native reward signal
                        rewards[agent_id] += float(env_rewards.get(agent_id, 0.0))

                # Update state for next iteration
                obs = next_obs
                info = step_info  # Carry forward info for curriculum velocity control
                for agent_id in self.agents.keys():
                    done_snapshot[agent_id] = (
                        done_snapshot.get(agent_id, False)
                        or terminations.get(agent_id, False)
                        or truncations.get(agent_id, False)
                    )

                episode_steps += 1
                steps_this_action += 1

                if all(done_snapshot.values()):
                    break

            # Accumulate component values for episode-level analysis
            for agent_id, components in reward_components_this_step.items():
                for comp_name, comp_value in components.items():
                    if comp_name not in episode_reward_components[agent_id]:
                        episode_reward_components[agent_id][comp_name] = 0.0
                    episode_reward_components[agent_id][comp_name] += comp_value

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

            # -------------------- AGENT UPDATES & STORAGE --------------------

            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    # Accumulate episode reward
                    episode_rewards[agent_id] += rewards[agent_id]

                    # Reuse cached observation (already flattened and normalized)
                    flat_obs = flat_obs_cache[agent_id]

                    # Process next observation for storage (flatten + normalize)
                    # Normalization is performed within flatten_observation()
                    flat_next_obs = self._flatten_obs(
                        agent_id,
                        next_obs[agent_id],
                        all_obs=next_obs,
                        update_stack=False,
                    )

                    # Extract termination flags
                    terminated = terminations.get(agent_id, False)
                    truncated = truncations.get(agent_id, False)
                    done_flag = done_snapshot.get(agent_id, False)

                    # ======== ON-POLICY vs OFF-POLICY STORAGE ========

                    if is_on_policy_agent(agent):
                        # ON-POLICY (PPO, A2C): Store rollout experience
                        # Uses internal buffer, updates at episode end
                        agent.store(flat_obs, actions[agent_id], rewards[agent_id], done_flag, terminated)
                    else:
                        # OFF-POLICY (SAC, TD3, TQC, DQN): Store in replay buffer
                        # Requires next_obs for TD learning: Q(s,a) = r + γ*Q(s',a')
                        agent.store_transition(flat_obs, actions[agent_id], rewards[agent_id], flat_next_obs, done_flag)

                        # Hindsight Experience Replay (HER) - optional advanced technique
                        # Relabels transitions with alternate goals for better sample efficiency
                        if hasattr(agent, 'store_hindsight_transition') and callable(agent.store_hindsight_transition):
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

                        # OFF-POLICY UPDATE: Perform gradient step every timestep
                        # Agent internally checks if buffer has enough samples (learning_starts)
                        try:
                            step_update_stats = agent.update()
                            # Note: Per-step stats intentionally not logged to reduce W&B overhead
                        except Exception as e:
                            logger.error(f"Agent {agent_id} update failed at episode {episode_num}, step {episode_steps}: {e}")
                            # Continue training despite failed update

                    # ======== SUCCESS REPLAY BUFFER ========
                    # Store transitions from potentially successful episodes
                    # Used for success-biased replay (experimental feature)
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

            # -------------------- STEP BOOKKEEPING --------------------
            done = done_snapshot

        # ========================================
        # PHASE 4: EPISODE FINALIZATION
        # ========================================

        # -------------------- FINAL AGENT UPDATES --------------------

        # ON-POLICY UPDATE: Perform policy gradient update using full episode rollout
        # Off-policy agents already updated every step (see above)
        trainer_stats = {}
        for agent_id, agent in self.agents.items():
            if not is_on_policy_agent(agent):
                continue
            update_stats = None
            try:
                update_stats = agent.update()  # PPO/A2C compute policy gradient here
                if update_stats:
                    trainer_stats[agent_id] = update_stats
                else:
                    # Debug: On-policy agents should return stats (policy_loss, value_loss, etc.)
                    if is_on_policy_agent(agent):
                        logger.debug(f"On-policy agent {agent_id} update() returned None at episode {episode_num}")
            except Exception as e:
                logger.error(f"Agent {agent_id} update failed after episode {episode_num}: {e}")
                continue

        # -------------------- OUTCOME DETERMINATION --------------------

        # Determine final episode outcome: target_crash (success), self_crash, timeout, etc.
        final_info = step_info if episode_steps > 0 else info

        episode_metrics: Dict[str, Any] = {}
        rolling_stats_by_agent: Dict[str, Dict[str, float]] = {}
        outcomes_by_agent: Dict[str, EpisodeOutcome] = {}

        for agent_id in self.agents.keys():
            # Extract agent-specific info from final step
            agent_info = final_info.get(agent_id, {}) if isinstance(final_info, dict) else {}

            # For adversarial tasks: check if target agent finished
            target_id = self.target_ids.get(agent_id)
            if target_id and isinstance(final_info, dict):
                target_info = final_info.get(target_id, {})
                if isinstance(target_info, dict):
                    agent_info = dict(agent_info)
                    agent_info["target_finished"] = bool(
                        target_info.get("finish_line", False)
                        or target_info.get("target_finished", False)
                    )

            # Determine outcome based on collision flags, finish line, timeout
            agent_truncated = truncations.get(agent_id, False)
            outcome = self._determine_agent_outcome(
                agent_id, agent_info, agent_truncated
            )
            outcomes_by_agent[agent_id] = outcome

            # -------------------- METRICS TRACKING --------------------

            # Record episode metrics (outcome, reward, steps, components)
            metrics = self.metrics_trackers[agent_id].add_episode(
                episode=episode_num,
                outcome=outcome,
                total_reward=episode_rewards[agent_id],
                steps=episode_steps,
                reward_components=episode_reward_components.get(agent_id, {}),
            )
            episode_metrics[agent_id] = metrics

            # Compute rolling statistics (success rate, avg reward) over recent episodes
            rolling_stats = self.metrics_trackers[agent_id].get_rolling_stats(
                window=self.rolling_window
            )
            rolling_stats_by_agent[agent_id] = rolling_stats

            # -------------------- SUCCESS REPLAY BUFFER --------------------

            # If episode was successful, store all transitions in success buffer
            # Allows success-biased sampling for improved learning (experimental)
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

            log_train = self.wandb_logger.should_log("train")
            log_spawn = self.wandb_logger.should_log("spawn")
            log_target = self.wandb_logger.should_log("target")

            log_dict = {}

            if log_train:
                log_dict.update({
                    "train/outcome": metrics.outcome.value,
                    "train/success": int(metrics.success),
                    "train/episode": int(episode_num),
                    "train/episode_reward": float(metrics.total_reward),
                    "train/episode_steps": int(metrics.steps),
                    "train/success_rate": float(rolling_stats.get("success_rate", 0.0)),
                    "train/reward_mean": float(rolling_stats.get("avg_reward", 0.0)),
                    "train/steps_mean": float(rolling_stats.get("avg_steps", 0.0)),
                })

            if log_spawn:
                spawn_point = self._current_spawn_mapping.get(primary_id)
                if spawn_point:
                    log_dict["train/spawn_point"] = spawn_point
                if self._current_spawn_stage:
                    log_dict["train/spawn_stage"] = self._current_spawn_stage

            if log_target and self.primary_target_id:
                log_dict.update({
                    "target/success": int(target_finished),
                    "target/crash": int(target_collision),
                })

            if log_dict:
                self.wandb_logger.log_metrics(log_dict, step=episode_num)

        # Update spawn curriculum if enabled (only for first/primary agent)
        spawn_curriculum_state = None
        if self.spawn_curriculum and not getattr(self, "phased_curriculum", None):
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

            # Log minimal curriculum metrics to W&B
            if self.wandb_logger and self.wandb_logger.should_log("curriculum"):
                self.wandb_logger.log_metrics({
                    'train/episode': int(episode_num),
                    'curriculum/stage': curriculum_state['stage'],
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

        # Use a single frame (post-step when available) for reward computation.
        obs_source = next_obs if isinstance(next_obs, dict) else obs

        reward_info = {
            'obs': obs_source.get(agent_id, {}) if isinstance(obs_source, dict) else {},
            'next_obs': next_obs.get(agent_id, {}) if isinstance(next_obs, dict) else {},
            'info': info_for_agent,
            'step': step,
            'done': done,  # Actual done flag from environment
            'truncated': truncated,  # Actual truncated flag from environment
            'timestep': timestep,
        }

        # Add target obs if available (for adversarial tasks)
        if target_id and isinstance(obs_source, dict) and target_id in obs_source:
            reward_info['target_obs'] = obs_source[target_id]

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

        # Enter eval mode in Rich dashboard
        if self.rich_console:
            self.rich_console.enter_eval_mode(
                num_eval_episodes=self.evaluation_config.num_episodes,
                training_episode=episode_num
            )

        # Track eval start for CSV/aggregate logging
        eval_episode_start = self.total_eval_episodes + 1

        # Run evaluation and stream per-episode logging
        successes_so_far = 0

        def _handle_eval_episode(ep_data: Dict[str, Any], eval_idx: int, total_eval: int) -> None:
            nonlocal successes_so_far
            self.total_eval_episodes += 1

            # Track success rate so far (for Rich dashboard)
            if ep_data['success']:
                successes_so_far += 1
            success_rate_so_far = successes_so_far / max(1, eval_idx)

            # Log to WandB
            if self.wandb_logger and self.wandb_logger.should_log("eval"):
                reward_value = float(ep_data['reward'])
                if not np.isfinite(reward_value):
                    reward_value = 0.0
                self.wandb_logger.log_metrics({
                    'eval/episode': int(self.total_eval_episodes),
                }, step=episode_num)
                self.wandb_logger.log_metrics({
                    'eval/episode_reward': reward_value,
                    'eval/episode_steps': int(ep_data['steps']),
                    'eval/episode_success': int(ep_data['success']),
                    'eval/spawn_point': ep_data['spawn_point'],
                    'eval/training_episode': episode_num,
                }, step=episode_num)

            # Log to CSV
            if self.csv_logger:
                self.csv_logger.log_eval_episode(
                    eval_episode=self.total_eval_episodes,
                    training_episode=episode_num,
                    outcome=ep_data['outcome'],
                    success=ep_data['success'],
                    reward=ep_data['reward'],
                    steps=ep_data['steps'],
                    spawn_point=ep_data['spawn_point'],
                    spawn_speed=ep_data['spawn_speed'],
                )

            # Update Rich dashboard
            if self.rich_console:
                self.rich_console.update_eval_episode(
                    eval_episode_num=eval_idx,
                    outcome=ep_data['outcome'],
                    reward=ep_data['reward'],
                    steps=ep_data['steps'],
                    spawn_point=ep_data['spawn_point'],
                    success_rate_so_far=success_rate_so_far,
                )

        eval_result = self.evaluator.evaluate(
            verbose=False,
            on_episode_end=_handle_eval_episode,
        )

        phase_info = self._get_phase_info()
        eval_training_state = {'eval_result': eval_result.to_dict()}
        if phase_info:
            eval_training_state.update(phase_info)

        agent_states: Optional[Dict[str, Any]] = None
        optimizer_states: Optional[Dict[str, Any]] = None

        def _collect_agent_states() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
            nonlocal agent_states, optimizer_states
            if agent_states is None:
                agent_states = {}
                optimizer_states = {}

                for agent_id, agent in self.agents.items():
                    if hasattr(agent, 'get_state'):
                        agent_states[agent_id] = agent.get_state()
                    if hasattr(agent, 'get_optimizer_state'):
                        opt_state = agent.get_optimizer_state()
                        if opt_state is not None:
                            optimizer_states[agent_id] = opt_state

                if not optimizer_states:
                    optimizer_states = None

            return agent_states, optimizer_states

        # Log aggregate results (keep step monotonic for W&B)
        if self.wandb_logger and self.wandb_logger.should_log("eval_agg"):
            agg_metrics = {
                'eval/episode': int(self.total_eval_episodes),
                'eval_agg/success_rate': eval_result.success_rate,
                'eval_agg/avg_reward': eval_result.avg_reward,
                'eval_agg/avg_episode_length': eval_result.avg_episode_length,
                'eval_agg/std_reward': eval_result.std_reward,
                'eval_agg/std_episode_length': eval_result.std_episode_length,
            }

            # Add outcome distribution to aggregate metrics
            for outcome, count in eval_result.outcome_counts.items():
                pct = (count / eval_result.num_episodes) * 100
                agg_metrics[f'eval_agg/outcome_{outcome}'] = pct

            self.wandb_logger.log_metrics(agg_metrics, step=episode_num)

        # Log aggregate results to CSV
        if self.csv_logger:
            self.csv_logger.log_eval_aggregate(
                training_episode=episode_num,
                eval_episode_start=eval_episode_start,
                eval_episode_end=self.total_eval_episodes,
                num_episodes=eval_result.num_episodes,
                success_count=eval_result.success_count,
                success_rate=eval_result.success_rate,
                avg_reward=eval_result.avg_reward,
                std_reward=eval_result.std_reward,
                avg_steps=eval_result.avg_episode_length,
                std_steps=eval_result.std_episode_length,
                outcome_counts=eval_result.outcome_counts,
            )

        # Exit eval mode in Rich dashboard
        if self.rich_console:
            self.rich_console.exit_eval_mode()

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
                agent_states, optimizer_states = _collect_agent_states()

                # Save with eval_best type
                try:
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        episode=episode_num,
                        agent_states=agent_states,
                        optimizer_states=optimizer_states if optimizer_states else None,
                        training_state=eval_training_state,
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

        # Save best eval per phase (phased curriculum only)
        if self.checkpoint_manager and phase_info.get("phase_index") is not None:
            phase_index = phase_info["phase_index"]
            best_value = self.best_eval_per_phase.get(phase_index)
            if best_value is None or eval_result.success_rate > best_value:
                agent_states, optimizer_states = _collect_agent_states()
                try:
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        episode=episode_num,
                        agent_states=agent_states,
                        optimizer_states=optimizer_states if optimizer_states else None,
                        training_state=eval_training_state,
                        checkpoint_type=f"best_eval_phase_{phase_index}",
                        metric_value=eval_result.success_rate,
                    )
                    self.best_eval_per_phase[phase_index] = eval_result.success_rate
                    if self.console_logger:
                        phase_name = phase_info.get("phase_name", phase_index)
                        self.console_logger.print_info(
                            f"New phase-best eval model ({phase_name}): "
                            f"{eval_result.success_rate:.2%} @ episode {episode_num}"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to save phase-best eval checkpoint at episode {episode_num}: {e}"
                    )

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

            phase_info = self._get_phase_info()
            if phase_info:
                training_state.update(phase_info)

            if self.spawn_curriculum:
                training_state['curriculum_stage'] = self.spawn_curriculum.current_stage
            if getattr(self, "phased_curriculum", None):
                from curriculum.curriculum_env import create_curriculum_checkpoint_data
                training_state['phased_curriculum'] = create_curriculum_checkpoint_data(
                    self.phased_curriculum
                )

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

    def _get_phase_info(self) -> Dict[str, Any]:
        """Get phased curriculum info for checkpoint tagging."""
        if not self._phase_curriculum_state:
            return {}

        phase_info: Dict[str, Any] = {}
        phase_index = self._phase_curriculum_state.get("phase_index")
        phase_name = self._phase_curriculum_state.get("phase_name")

        if phase_index is not None:
            phase_info["phase_index"] = int(phase_index)
        if phase_name is not None:
            phase_info["phase_name"] = str(phase_name)

        return phase_info

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
