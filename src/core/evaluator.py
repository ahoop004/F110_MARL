"""Evaluation system for trained agents.

Provides deterministic evaluation with fixed scenarios for consistent
performance measurement during and after training.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pettingzoo import ParallelEnv

from core.protocol import Agent
from core.obs_flatten import flatten_observation
from metrics import determine_outcome, EpisodeOutcome
from wrappers.normalize import ObservationNormalizer


class EvaluationConfig:
    """Configuration for evaluation runs."""

    def __init__(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        spawn_points: Optional[List[str]] = None,
        spawn_speeds: Optional[List[float]] = None,
        lock_speed_steps: int = 0,
        ftg_override: Optional[Dict[str, Any]] = None,
        max_steps: int = 2500,
    ):
        """Initialize evaluation configuration.

        Args:
            num_episodes: Number of episodes to run (default: 10)
            deterministic: Use deterministic actions (default: True)
            spawn_points: List of spawn point names for sequential spawning
                If None, uses environment defaults. Sequential order for consistency.
            spawn_speeds: List of speeds corresponding to spawn_points
                If None, uses default speed (0.44)
            lock_speed_steps: Number of steps to lock speed (default: 0 = no locking)
            ftg_override: Override FTG parameters for full strength evaluation
                Example: {'max_speed': 1.0, 'bubble_radius': 3.0, 'steering_gain': 0.35}
            max_steps: Maximum steps per episode (default: 2500)
        """
        self.num_episodes = num_episodes
        self.deterministic = deterministic
        self.spawn_points = spawn_points or ['spawn_pinch_left', 'spawn_pinch_right']
        self.spawn_speeds = spawn_speeds or [0.44, 0.44]
        self.lock_speed_steps = lock_speed_steps
        self.ftg_override = ftg_override or {}
        self.max_steps = max_steps


class EvaluationResult:
    """Results from an evaluation run."""

    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        agent_id: str,
        target_id: Optional[str] = None,
    ):
        """Initialize evaluation result.

        Args:
            episodes: List of episode result dicts
            agent_id: ID of evaluated agent
            target_id: ID of target agent (optional)
        """
        self.episodes = episodes
        self.agent_id = agent_id
        self.target_id = target_id

        # Compute aggregate statistics
        self.num_episodes = len(episodes)
        self.success_count = sum(1 for ep in episodes if ep['success'])
        self.success_rate = self.success_count / self.num_episodes if self.num_episodes > 0 else 0.0

        # Episode lengths
        episode_lengths = [ep['steps'] for ep in episodes]
        self.avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
        self.std_episode_length = np.std(episode_lengths) if episode_lengths else 0.0

        # Rewards
        rewards = [ep['reward'] for ep in episodes]
        self.avg_reward = np.mean(rewards) if rewards else 0.0
        self.std_reward = np.std(rewards) if rewards else 0.0

        # Outcome distribution
        self.outcome_counts = {}
        for ep in episodes:
            outcome = ep['outcome']
            self.outcome_counts[outcome] = self.outcome_counts.get(outcome, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'agent_id': self.agent_id,
            'target_id': self.target_id,
            'num_episodes': self.num_episodes,
            'success_count': self.success_count,
            'success_rate': self.success_rate,
            'avg_episode_length': self.avg_episode_length,
            'std_episode_length': self.std_episode_length,
            'avg_reward': self.avg_reward,
            'std_reward': self.std_reward,
            'outcome_counts': self.outcome_counts,
            'episodes': self.episodes,
        }

    def summary_str(self) -> str:
        """Get formatted summary string."""
        lines = [
            f"Evaluation Results ({self.num_episodes} episodes)",
            f"  Success Rate: {self.success_rate:.2%} ({self.success_count}/{self.num_episodes})",
            f"  Avg Episode Length: {self.avg_episode_length:.1f} ± {self.std_episode_length:.1f} steps",
            f"  Avg Reward: {self.avg_reward:.2f} ± {self.std_reward:.2f}",
            "",
            "Outcome Distribution:",
        ]

        for outcome, count in sorted(self.outcome_counts.items()):
            pct = (count / self.num_episodes) * 100
            lines.append(f"  {outcome}: {count} ({pct:.1f}%)")

        return "\n".join(lines)


class Evaluator:
    """Evaluates trained agents with deterministic scenarios.

    Runs agents through fixed evaluation scenarios to measure
    performance consistently during and after training.

    Example:
        >>> evaluator = Evaluator(env, agents, eval_config)
        >>> result = evaluator.evaluate()
        >>> print(result.summary_str())
    """

    def __init__(
        self,
        env: ParallelEnv,
        agents: Dict[str, Agent],
        config: EvaluationConfig,
        observation_presets: Optional[Dict[str, str]] = None,
        target_ids: Optional[Dict[str, Optional[str]]] = None,
        obs_normalizer: Optional[ObservationNormalizer] = None,
        obs_scales: Optional[Dict[str, float]] = None,
        spawn_configs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize evaluator.

        Args:
            env: PettingZoo ParallelEnv
            agents: Dict mapping agent_id -> Agent
            config: EvaluationConfig
            observation_presets: Optional dict mapping agent_id -> preset name
            target_ids: Optional dict mapping agent_id -> target_id
            obs_normalizer: Optional observation normalizer (for consistency with training)
            obs_scales: Optional observation scales for flattening
            spawn_configs: Optional spawn configurations (if None, reads from env or uses default poses)
        """
        self.env = env
        self.agents = agents
        self.config = config
        self.observation_presets = observation_presets or {}
        self.target_ids = target_ids or {}
        self.obs_normalizer = obs_normalizer
        self.obs_scales = obs_scales or {}
        self.spawn_configs = spawn_configs or getattr(env, 'spawn_configs', {})

        # Get primary agent (first trainable agent)
        self.primary_agent_id = self._get_primary_agent_id()
        self.primary_target_id = (
            self.target_ids.get(self.primary_agent_id)
            if self.primary_agent_id
            else None
        )

    def evaluate(self, verbose: bool = False) -> EvaluationResult:
        """Run evaluation and return results.

        Args:
            verbose: Print progress during evaluation (default: False)

        Returns:
            EvaluationResult with aggregate statistics and per-episode data
        """
        # Apply FTG override if configured
        if self.config.ftg_override:
            self._apply_ftg_override()

        episodes = []

        for ep_idx in range(self.config.num_episodes):
            # Sequential spawn selection (cycle through spawn points)
            spawn_idx = ep_idx % len(self.config.spawn_points)
            spawn_point = self.config.spawn_points[spawn_idx]
            spawn_speed = self.config.spawn_speeds[spawn_idx] if spawn_idx < len(self.config.spawn_speeds) else 0.44

            if verbose:
                print(f"Episode {ep_idx + 1}/{self.config.num_episodes}: {spawn_point} @ {spawn_speed:.2f} m/s")

            # Run episode
            episode_result = self._run_episode(spawn_point, spawn_speed, ep_idx)
            episodes.append(episode_result)

            if verbose:
                outcome = episode_result['outcome']
                reward = episode_result['reward']
                steps = episode_result['steps']
                print(f"  Result: {outcome}, Reward: {reward:.2f}, Steps: {steps}")

        # Create result
        result = EvaluationResult(
            episodes=episodes,
            agent_id=self.primary_agent_id,
            target_id=self.primary_target_id,
        )

        return result

    def _run_episode(
        self,
        spawn_point: str,
        spawn_speed: float,
        episode_num: int
    ) -> Dict[str, Any]:
        """Run a single evaluation episode.

        Args:
            spawn_point: Name of spawn point to use
            episode_num: Episode number (for logging)

        Returns:
            Dict with episode results
        """
        # Get spawn configuration
        if spawn_point not in self.spawn_configs:
            raise ValueError(
                f"Spawn point '{spawn_point}' not found in spawn_configs. "
                f"Available: {list(self.spawn_configs.keys())}"
            )

        spawn_config = self.spawn_configs[spawn_point]

        # Build poses array and velocities dict for reset
        poses_list = []
        velocities = {}

        for agent_id in sorted(self.agents.keys()):  # Sort for consistent ordering
            if agent_id in spawn_config:
                pose = spawn_config[agent_id]
                poses_list.append(pose)
                # Set initial velocity as scalar speed (environment expects this format)
                velocities[agent_id] = spawn_speed

        # Convert poses to numpy array (N, 3) format expected by environment
        poses_array = np.array(poses_list, dtype=np.float32)

        # Reset environment with spawn configuration
        reset_options = {
            'poses': poses_array,
            'velocities': velocities,
            'lock_speed_steps': self.config.lock_speed_steps,
        }

        obs, info = self.env.reset(options=reset_options)

        # Episode tracking
        episode_reward = 0.0
        episode_steps = 0
        done = {agent_id: False for agent_id in self.agents.keys()}

        # Run episode
        while not all(done.values()) and episode_steps < self.config.max_steps:
            # Select actions (deterministic for eval)
            actions = {}
            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    # Flatten observation if preset configured
                    flat_obs = self._flatten_obs(agent_id, obs[agent_id], all_obs=obs)

                    # Normalize observation if normalizer provided
                    if self.obs_normalizer and isinstance(flat_obs, np.ndarray):
                        # Don't update stats during eval
                        flat_obs = self.obs_normalizer.normalize(flat_obs, agent_id, update_stats=False)

                    # Get action (deterministic)
                    try:
                        actions[agent_id] = agent.act(flat_obs, deterministic=self.config.deterministic)
                    except TypeError:
                        # Agent doesn't support deterministic parameter
                        actions[agent_id] = agent.act(flat_obs)

            # Step environment
            next_obs, rewards, terminations, truncations, step_info = self.env.step(actions)

            # Track primary agent reward
            if self.primary_agent_id:
                episode_reward += rewards.get(self.primary_agent_id, 0.0)

            # Update observations and done flags
            obs = next_obs
            info = step_info
            for agent_id in self.agents.keys():
                done[agent_id] = terminations.get(agent_id, False) or truncations.get(agent_id, False)

            episode_steps += 1

        # Determine outcome
        final_info = step_info if episode_steps > 0 else info

        # Get primary agent outcome
        if self.primary_agent_id:
            agent_info = final_info.get(self.primary_agent_id, {}) if isinstance(final_info, dict) else {}

            # Add target info if available
            if self.primary_target_id and isinstance(final_info, dict):
                target_info = final_info.get(self.primary_target_id, {})
                if isinstance(target_info, dict):
                    agent_info = dict(agent_info)
                    agent_info["target_finished"] = bool(
                        target_info.get("finish_line", False)
                        or target_info.get("target_finished", False)
                    )

            truncated = truncations.get(self.primary_agent_id, False)
            outcome = determine_outcome(agent_info, truncated)
        else:
            outcome = EpisodeOutcome.UNKNOWN

        return {
            'episode': episode_num,
            'spawn_point': spawn_point,
            'spawn_speed': spawn_speed,
            'outcome': outcome.value,
            'success': outcome.is_success(),
            'reward': episode_reward,
            'steps': episode_steps,
        }

    def _flatten_obs(self, agent_id: str, obs: Any, all_obs: Optional[Dict[str, Any]] = None) -> Any:
        """Flatten observation for agent if preset is configured."""
        if agent_id not in self.observation_presets:
            return obs

        preset = self.observation_presets[agent_id]
        target_id = self.target_ids.get(agent_id, None)

        # If target_id specified and we have all observations, add target state
        if target_id and all_obs and target_id in all_obs:
            # Create combined observation dict with central_state
            combined_obs = dict(obs)
            combined_obs['central_state'] = all_obs[target_id]
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

    def _apply_ftg_override(self) -> None:
        """Apply FTG parameter overrides for full-strength evaluation."""
        for agent_id, agent in self.agents.items():
            # Check if agent supports apply_config (FTG agents)
            apply_config = getattr(agent, "apply_config", None)
            if callable(apply_config):
                apply_config(self.config.ftg_override)

    def _get_primary_agent_id(self) -> Optional[str]:
        """Get primary trainable agent ID."""
        # For now, just return first agent
        # Could be made smarter by checking for trainable agents
        return next(iter(self.agents), None)


__all__ = ['Evaluator', 'EvaluationConfig', 'EvaluationResult']
