"""Spawn curriculum system for progressive difficulty scaling.

Manages transitions between spawn configurations based on agent success rate,
implementing a curriculum learning approach that starts with easy scenarios
and gradually increases difficulty.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SpawnStage:
    """Configuration for a curriculum stage.

    Attributes:
        name: Human-readable stage name
        spawn_points: List of spawn point IDs or "all" for all available
        speed_range: (min, max) speed range in m/s
        enable_rate: Success rate threshold to advance to this stage
        disable_rate: Success rate threshold to regress from this stage
        enable_patience: Consecutive episodes above threshold to advance
        disable_patience: Consecutive episodes below threshold to regress
    """
    name: str
    spawn_points: List[str] | str
    speed_range: Tuple[float, float]
    enable_rate: float
    disable_rate: float = 0.0
    enable_patience: Optional[int] = None
    disable_patience: Optional[int] = None


class SpawnCurriculumManager:
    """Manages progressive spawn difficulty based on success rate.

    Implements curriculum learning by transitioning between spawn configurations
    as the agent's success rate improves or degrades. Tracks rolling window of
    recent episodes and uses patience counters to avoid rapid oscillation.

    Example:
        >>> config = {
        ...     'window': 200,
        ...     'enable_patience': 5,
        ...     'stages': [
        ...         {'name': 'easy', 'spawn_points': ['pinch_right'], 'speed_range': [0.5, 0.5], 'enable_rate': 0.70},
        ...         {'name': 'hard', 'spawn_points': 'all', 'speed_range': [0.3, 1.0], 'disable_rate': 0.50}
        ...     ]
        ... }
        >>> manager = SpawnCurriculumManager(config, spawn_points_dict)
        >>>
        >>> # Sample spawn for current stage
        >>> spawn_info = manager.sample_spawn()
        >>>
        >>> # After episode completion
        >>> state = manager.observe(episode_idx=100, success=True)
        >>> if state['changed']:
        ...     print(f"Advanced to {state['stage']}")
    """

    def __init__(self, config: Dict[str, Any], available_spawn_points: Dict[str, Any]):
        """Initialize spawn curriculum manager.

        Args:
            config: Configuration dict with keys:
                - window: Rolling window size for success tracking (default: 200)
                - activation_samples: Min samples before enabling curriculum (default: window)
                - min_episode: Min episode before first transition (default: 100)
                - enable_patience: Consecutive successes to advance (default: 5)
                - disable_patience: Consecutive failures to regress (default: 3)
                - cooldown: Episodes between transitions (default: 20)
                - lock_speed_steps: Steps to lock starting speed (default: 150)
                - stages: List of stage configurations
            available_spawn_points: Dict of spawn point name -> spawn data
        """
        self.config = config
        self.available_spawn_points = available_spawn_points

        # Success tracking parameters
        self.window = max(1, int(config.get('window', 200)))
        self.activation_samples = max(1, int(config.get('activation_samples', self.window)))
        self.min_episode = max(0, int(config.get('min_episode', 100)))

        # Speed control parameters
        self.lock_speed_steps = max(0, int(config.get('lock_speed_steps', 150)))

        # Transition parameters
        self.enable_patience_default = max(1, int(config.get('enable_patience', 5)))
        self.disable_patience_default = max(1, int(config.get('disable_patience', 3)))
        self.cooldown = max(0, int(config.get('cooldown', 20)))

        # Parse stages
        self.stages = self._parse_stages(config.get('stages', []))
        if not self.stages:
            raise ValueError("Spawn curriculum requires at least one stage")

        # State tracking
        self.current_stage_idx = 0
        self.success_history: deque = deque(maxlen=self.window)
        self.stage_histories: Dict[int, deque] = {
            i: deque(maxlen=self.window) for i in range(len(self.stages))
        }

        # Transition tracking
        self.promote_streak = 0
        self.regress_streak = 0
        self.last_transition_episode = -self.cooldown  # Allow immediate first transition

    def _parse_stages(self, raw_stages: List[Dict[str, Any]]) -> List[SpawnStage]:
        """Parse stage configurations from config.

        Args:
            raw_stages: List of stage config dicts

        Returns:
            List of SpawnStage objects
        """
        stages = []
        for idx, stage_config in enumerate(raw_stages):
            # Parse spawn points
            spawn_points_raw = stage_config.get('spawn_points', [])
            if spawn_points_raw == 'all':
                spawn_points = 'all'
            elif isinstance(spawn_points_raw, (list, tuple)):
                spawn_points = list(spawn_points_raw)
            else:
                spawn_points = [str(spawn_points_raw)]

            # Parse speed range
            speed_range_raw = stage_config.get('speed_range', [0.5, 1.0])
            if isinstance(speed_range_raw, (list, tuple)) and len(speed_range_raw) == 2:
                speed_range = (float(speed_range_raw[0]), float(speed_range_raw[1]))
            else:
                speed_range = (0.5, 1.0)

            # Create stage
            stage = SpawnStage(
                name=stage_config.get('name', f'stage_{idx}'),
                spawn_points=spawn_points,
                speed_range=speed_range,
                enable_rate=float(stage_config.get('enable_rate', 0.70)),
                disable_rate=float(stage_config.get('disable_rate', 0.0)),
                enable_patience=stage_config.get('enable_patience'),
                disable_patience=stage_config.get('disable_patience'),
            )
            stages.append(stage)

        return stages

    @property
    def current_stage(self) -> SpawnStage:
        """Get current stage configuration."""
        return self.stages[self.current_stage_idx]

    @property
    def success_rate(self) -> Optional[float]:
        """Get overall success rate from history."""
        if not self.success_history:
            return None
        return float(sum(self.success_history) / len(self.success_history))

    def _stage_success_rate(self, stage_idx: int) -> Optional[float]:
        """Get success rate for specific stage."""
        history = self.stage_histories.get(stage_idx)
        if not history:
            return None
        return float(sum(history) / len(history))

    def _can_transition(self, episode: int) -> bool:
        """Check if cooldown period has elapsed since last transition."""
        return episode >= (self.last_transition_episode + self.cooldown)

    def _can_advance(self, episode: int, success_rate: float) -> bool:
        """Check if conditions are met to advance to next stage.

        Args:
            episode: Current episode number
            success_rate: Current overall success rate

        Returns:
            True if should advance to next stage
        """
        # Can't advance from final stage
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        # Check basic requirements
        if episode < self.min_episode:
            return False
        if len(self.success_history) < self.activation_samples:
            return False
        if success_rate is None:
            return False
        if not self._can_transition(episode):
            return False

        # Check success rate threshold
        next_stage = self.stages[self.current_stage_idx + 1]
        enable_rate = next_stage.enable_rate
        enable_patience = next_stage.enable_patience or self.enable_patience_default

        if success_rate >= enable_rate:
            self.promote_streak += 1
        else:
            self.promote_streak = 0

        return self.promote_streak >= enable_patience

    def _can_regress(self, episode: int, success_rate: float) -> bool:
        """Check if conditions are met to regress to previous stage.

        Args:
            episode: Current episode number
            success_rate: Current overall success rate

        Returns:
            True if should regress to previous stage
        """
        # Can't regress from first stage
        if self.current_stage_idx <= 0:
            return False

        # Check basic requirements
        if len(self.success_history) < self.activation_samples:
            return False
        if success_rate is None:
            return False
        if not self._can_transition(episode):
            return False

        # Check success rate threshold
        current_stage = self.stages[self.current_stage_idx]
        disable_rate = current_stage.disable_rate
        disable_patience = current_stage.disable_patience or self.disable_patience_default

        if success_rate < disable_rate:
            self.regress_streak += 1
        else:
            self.regress_streak = 0

        return self.regress_streak >= disable_patience

    def _transition_to_stage(self, stage_idx: int, episode: int):
        """Transition to a new stage.

        Args:
            stage_idx: Index of stage to transition to
            episode: Current episode number
        """
        self.current_stage_idx = stage_idx
        self.last_transition_episode = episode
        self.promote_streak = 0
        self.regress_streak = 0

        # Clear history for new stage to get fresh statistics
        self.stage_histories[stage_idx].clear()

    def observe(self, episode: int, success: bool) -> Dict[str, Any]:
        """Record episode outcome and update curriculum state.

        Args:
            episode: Episode number
            success: Whether episode was successful

        Returns:
            Dict with curriculum state:
                - changed: Whether stage changed this episode
                - stage: Current stage name
                - stage_index: Current stage index
                - success_rate: Overall success rate
                - stage_success_rate: Success rate for current stage
        """
        # Record outcome
        value = 1.0 if success else 0.0
        self.success_history.append(value)
        self.stage_histories[self.current_stage_idx].append(value)

        # Get current metrics
        success_rate = self.success_rate
        stage_success_rate = self._stage_success_rate(self.current_stage_idx)

        changed = False

        # Check for advancement
        if self._can_advance(episode, success_rate):
            old_idx = self.current_stage_idx
            self._transition_to_stage(self.current_stage_idx + 1, episode)
            changed = self.current_stage_idx != old_idx

        # Check for regression
        elif self._can_regress(episode, success_rate):
            old_idx = self.current_stage_idx
            self._transition_to_stage(self.current_stage_idx - 1, episode)
            changed = self.current_stage_idx != old_idx

        return {
            'changed': changed,
            'stage': self.current_stage.name,
            'stage_index': self.current_stage_idx,
            'success_rate': success_rate,
            'stage_success_rate': stage_success_rate,
        }

    def sample_spawn(self) -> Dict[str, Any]:
        """Sample spawn configuration for current stage.

        Returns:
            Dict with:
                - spawn_points: Dict mapping agent_id -> spawn_point_name
                - poses: numpy array of poses (N, 3) with [x, y, theta]
                - velocities: Dict mapping agent_id -> initial velocity (only car_0)
                - stage: Current stage name
        """
        stage = self.current_stage

        # Select spawn points
        if stage.spawn_points == 'all':
            # Sample from all available spawn points
            spawn_names = list(self.available_spawn_points.keys())
        else:
            # Use specified spawn points
            spawn_names = [name for name in stage.spawn_points
                          if name in self.available_spawn_points]

        if not spawn_names:
            raise ValueError(f"No valid spawn points for stage {stage.name}")

        # Sample random spawn point
        spawn_name = np.random.choice(spawn_names)
        spawn_data = self.available_spawn_points[spawn_name]

        # Extract poses and agent order
        poses = []
        spawn_mapping = {}
        agent_ids = []
        for agent_id, pose in spawn_data.items():
            if agent_id == 'metadata':
                continue
            poses.append(pose)
            spawn_mapping[agent_id] = spawn_name
            agent_ids.append(agent_id)

        poses_array = np.array(poses, dtype=np.float32)

        # Sample velocity ONLY for car_0 (attacker/learner)
        # car_1 (defender/FTG) uses its own speed control
        min_speed, max_speed = stage.speed_range
        velocities = {}

        # Only set velocity for car_0
        if 'car_0' in agent_ids:
            if min_speed == max_speed:
                velocities['car_0'] = float(min_speed)
            else:
                velocities['car_0'] = float(np.random.uniform(min_speed, max_speed))

        return {
            'spawn_points': spawn_mapping,
            'poses': poses_array,
            'velocities': velocities,  # Dict {agent_id: velocity}
            'lock_speed_steps': self.lock_speed_steps,  # Steps to lock speed
            'stage': stage.name,
        }


__all__ = ['SpawnCurriculumManager', 'SpawnStage']
