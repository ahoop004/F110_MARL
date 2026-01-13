"""Stable-Baselines3 wrapper for F110 multi-agent environment.

Converts PettingZoo ParallelEnv to single-agent Gymnasium environment
for training with SB3 algorithms (SAC, TD3, PPO).
"""

from typing import Any, Dict, Optional, Tuple
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.core.obs_flatten import flatten_observation
from src.rewards.base import RewardStrategy
from src.metrics.outcomes import determine_outcome


class SB3SingleAgentWrapper(gym.Env):
    """Wrapper to convert multi-agent F110 env to single-agent for SB3.

    Wraps one agent (attacker) while other agents (e.g., FTG defender)
    run their own policies.

    Args:
        env: PettingZoo ParallelEnv (F110 environment)
        agent_id: ID of agent to control (default: 'car_0')
        obs_dim: Observation dimension (default: 126 for gaplock)
        action_low: Action space lower bounds (default: [-0.46, -1.0])
        action_high: Action space upper bounds (default: [0.46, 1.0])

    Example:
        >>> from core.setup import create_training_setup
        >>> env, agents, _ = create_training_setup(scenario)
        >>>
        >>> # Wrap for SB3
        >>> wrapped_env = SB3SingleAgentWrapper(
        ...     env,
        ...     agent_id='car_0',
        ...     obs_dim=126
        ... )
        >>>
        >>> # Train with SB3
        >>> from stable_baselines3 import SAC
        >>> model = SAC("MlpPolicy", wrapped_env)
        >>> model.learn(total_timesteps=1000000)
    """

    def __init__(
        self,
        env,
        agent_id: str = 'car_0',
        obs_dim: int = 126,
        action_low: np.ndarray = np.array([-0.46, -1.0]),
        action_high: np.ndarray = np.array([0.46, 1.0]),
        observation_preset: Optional[str] = None,
        target_id: Optional[str] = None,
        reward_strategy: Optional[RewardStrategy] = None,
        action_set: Optional[np.ndarray] = None,
        spawn_curriculum: Optional[Any] = None,
        frame_stack: int = 1,
        action_repeat: int = 1,
    ):
        super().__init__()

        self.env = env
        self.agent_id = agent_id
        self.observation_preset = observation_preset
        self.target_id = target_id
        self.reward_strategy = reward_strategy
        self.spawn_curriculum = spawn_curriculum
        self._episode_count = 0
        self._last_spawn_info: Optional[Dict[str, Any]] = None
        try:
            frame_stack = int(frame_stack)
        except (TypeError, ValueError):
            frame_stack = 1
        self.frame_stack = max(1, frame_stack)
        try:
            action_repeat = int(action_repeat)
        except (TypeError, ValueError):
            action_repeat = 1
        self.action_repeat = max(1, action_repeat)
        self._frame_buffer: Optional[deque] = None

        # Define observation space (Box for continuous observations)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Define action space (Discrete or Box)
        self.action_set = action_set
        if action_set is not None:
            # Discrete action space for DQN/QR-DQN
            action_set_array = np.asarray(action_set, dtype=np.float32)
            if action_set_array.ndim != 2:
                raise ValueError("action_set must be 2D array of shape (n_actions, action_dim)")
            self.action_set = action_set_array
            self.action_space = spaces.Discrete(len(action_set_array))
        else:
            # Continuous action space for SAC/TD3/PPO
            self.action_space = spaces.Box(
                low=action_low,
                high=action_high,
                dtype=np.float32
            )

        # Store agents dict for non-SB3 agents
        self.other_agents = None

        # Resolve observation scales from environment
        self.obs_scales = self._resolve_obs_scales()

        # Track episode state for reward computation
        self.current_obs_dict = None
        self.episode_steps = 0

        if self.frame_stack > 1:
            self._reset_frame_buffer()

    def _reset_frame_buffer(self) -> None:
        if self.frame_stack > 1:
            self._frame_buffer = deque(maxlen=self.frame_stack)
        else:
            self._frame_buffer = None

    def _stack_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if self.frame_stack <= 1:
            return obs
        if self._frame_buffer is None:
            self._reset_frame_buffer()
        buffer = self._frame_buffer
        if buffer is None:
            return obs
        if len(buffer) == 0:
            frames = [obs] * self.frame_stack
            if update:
                buffer.extend(frames)
            return np.concatenate(frames, axis=0)
        if update:
            buffer.append(obs)
            frames = list(buffer)
        else:
            frames = list(buffer) + [obs]
            if len(frames) > self.frame_stack:
                frames = frames[-self.frame_stack:]
        return np.concatenate(frames, axis=0)

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

        return scales

    def _flatten_obs(self, obs: Any, all_obs: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Flatten observation if it's a dict, otherwise return as-is.

        Args:
            obs: Raw observation (can be dict or numpy array)
            all_obs: Optional dict of all agent observations (for extracting target state)

        Returns:
            Flattened observation as numpy array
        """
        # If already a numpy array, just return it
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)

        # If it's a dict and we have a preset, flatten it
        if isinstance(obs, dict) and self.observation_preset:
            # If target_id specified and we have all observations, add target state to obs
            if self.target_id and all_obs and self.target_id in all_obs:
                # Create combined observation dict with central_state
                combined_obs = dict(obs)  # Copy agent's own observation
                combined_obs['central_state'] = all_obs[self.target_id]  # Add target as central_state
                return flatten_observation(
                    combined_obs,
                    preset=self.observation_preset,
                    target_id=self.target_id,
                    scales=self.obs_scales,
                )
            else:
                return flatten_observation(
                    obs,
                    preset=self.observation_preset,
                    target_id=self.target_id,
                    scales=self.obs_scales,
                )

        # Otherwise, try to convert to array
        try:
            return np.asarray(obs, dtype=np.float32)
        except Exception:
            raise ValueError(f"Cannot convert observation of type {type(obs)} to numpy array")

    def set_other_agents(self, agents: Dict[str, Any]):
        """Set other agents (e.g., FTG defender) that act in environment.

        Args:
            agents: Dict mapping agent_id to agent object
        """
        self.other_agents = {
            aid: agent for aid, agent in agents.items()
            if aid != self.agent_id
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            obs: Initial observation for controlled agent
            info: Info dict
        """
        # Apply spawn curriculum if configured and no explicit options provided
        if options is None and self.spawn_curriculum is not None:
            spawn_info = self.spawn_curriculum.sample_spawn(episode=self._episode_count)
            self._last_spawn_info = spawn_info
            options = {
                'poses': spawn_info['poses'],
                'velocities': spawn_info['velocities'],
                'lock_speed_steps': spawn_info['lock_speed_steps'],
            }
        else:
            self._last_spawn_info = None

        # Reset underlying environment
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)

        # Store current observations for reward computation
        self.current_obs_dict = obs_dict
        self.episode_steps = 0
        self._reset_frame_buffer()

        # Extract observation for controlled agent
        obs = self._flatten_obs(obs_dict[self.agent_id], all_obs=obs_dict)
        obs = self._stack_obs(obs, update=True)
        info = info_dict.get(self.agent_id, {})
        if self._last_spawn_info:
            spawn_mapping = self._last_spawn_info.get('spawn_points', {})
            spawn_point = spawn_mapping.get(self.agent_id)
            if spawn_point:
                info = dict(info)
                info['spawn_point'] = spawn_point
        self._episode_count += 1

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action for controlled agent (discrete index or continuous action)

        Returns:
            obs: Next observation
            reward: Reward
            terminated: Whether episode ended (goal reached or failed)
            truncated: Whether episode was truncated (time limit)
            info: Info dict
        """
        # Convert discrete action to continuous if using action_set
        if self.action_set is not None:
            action_idx = int(action)
            continuous_action = self.action_set[action_idx]
        else:
            continuous_action = action

        # Build action dict for all agents
        actions = {self.agent_id: continuous_action}

        # Get actions from other agents (if they exist)
        if self.other_agents:
            # Get current observations for other agents from stored obs dict
            for agent_id, agent in self.other_agents.items():
                # Other agents select their own actions
                # (This assumes they have an act() method)
                obs = self.current_obs_dict.get(agent_id) if self.current_obs_dict else None
                if obs is not None and hasattr(agent, 'act'):
                    actions[agent_id] = agent.act(obs)

        prev_obs_dict = self.current_obs_dict
        total_reward = 0.0
        reward_components: Dict[str, float] = {}
        terminated = False
        truncated = False
        obs_dict = None
        info_dict: Dict[str, Any] = {}

        for _ in range(self.action_repeat):
            # Step environment with all actions
            obs_dict, reward_dict, done_dict, truncated_dict, info_dict = self.env.step(actions)

            step_terminated = bool(done_dict[self.agent_id])
            step_truncated = bool(truncated_dict[self.agent_id])

            # Compute reward using custom strategy if provided
            if self.reward_strategy and prev_obs_dict:
                reward_info = self._build_reward_info(
                    prev_obs=prev_obs_dict,
                    next_obs=obs_dict,
                    info=info_dict,
                    terminated=step_terminated,
                    truncated=step_truncated,
                )
                step_reward, components = self.reward_strategy.compute(reward_info)
                if components:
                    for name, value in components.items():
                        reward_components[name] = reward_components.get(name, 0.0) + float(value)
                step_reward_value = float(step_reward)
            else:
                # Use environment's default reward
                step_reward_value = float(reward_dict[self.agent_id])

            total_reward += step_reward_value
            self.current_obs_dict = obs_dict
            self.episode_steps += 1
            prev_obs_dict = obs_dict
            terminated = step_terminated
            truncated = step_truncated

            if terminated or truncated:
                break

        if obs_dict is None:
            raise RuntimeError("Environment returned no observations during step.")

        # Extract results for controlled agent
        obs = self._flatten_obs(obs_dict[self.agent_id], all_obs=obs_dict)
        obs = self._stack_obs(obs, update=True)
        info = info_dict.get(self.agent_id, {})
        target_finished = False
        if self.target_id and isinstance(info_dict, dict):
            target_info = info_dict.get(self.target_id, {})
            if isinstance(target_info, dict):
                target_finished = bool(
                    target_info.get("finish_line", False)
                    or target_info.get("target_finished", False)
                )
        info["target_finished"] = target_finished

        # Determine episode outcome for curriculum tracking
        if terminated or truncated:
            outcome = determine_outcome(info, truncated=truncated)
            info["outcome"] = outcome.value
            info['is_success'] = outcome.is_success()

        if reward_components:
            info['reward_components'] = reward_components

        return obs, total_reward, terminated, truncated, info

    def _build_reward_info(
        self,
        prev_obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        """Build reward info dict for custom reward computation.

        Args:
            prev_obs: Previous observations for all agents
            next_obs: Next observations for all agents
            info: Step info from environment
            terminated: Whether episode terminated
            truncated: Whether episode was truncated

        Returns:
            Reward info dict for RewardStrategy.compute()
        """
        # Get timestep from environment if available
        timestep = getattr(self.env, 'timestep', 0.01)

        # Extract info for this agent
        info_for_agent = info.get(self.agent_id, {}) if isinstance(info, dict) else {}

        reward_info = {
            'obs': prev_obs.get(self.agent_id, {}),
            'next_obs': next_obs.get(self.agent_id, {}),
            'info': info_for_agent,
            'step': self.episode_steps,
            'done': terminated or truncated,
            'truncated': truncated,
            'timestep': timestep,
        }

        # Add target obs if target_id is specified (for adversarial tasks)
        if self.target_id and self.target_id in next_obs:
            reward_info['target_obs'] = next_obs[self.target_id]

        return reward_info

    def render(self):
        """Render environment (delegates to underlying env)."""
        if hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


__all__ = ['SB3SingleAgentWrapper']
