"""Phased curriculum learning system.

Provides a modular curriculum that progressively increases difficulty across
multiple orthogonal dimensions:
- Speed lock duration (defender speed constraints)
- Speed range variation (defender speed diversity)
- Spawn position diversity (geometric scenarios)
- FTG resistance (defender competence)

Example:
    >>> curriculum = PhaseBasedCurriculum.from_config(config)
    >>>
    >>> # After each episode
    >>> curriculum.update(episode_outcome, episode_reward, episode_num)
    >>>
    >>> # Get current phase configuration
    >>> phase_config = curriculum.get_current_config()
    >>> spawn_points = phase_config['spawn']['points']
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np


@dataclass
class AdvancementCriteria:
    """Criteria for advancing to next phase.

    Attributes:
        success_rate: Minimum success rate required (0.0-1.0)
        avg_reward: Minimum average reward required
        min_episodes: Minimum episodes before advancement allowed
        patience: Maximum episodes before forced advancement
        window_size: Number of episodes for rolling statistics
    """
    success_rate: float = 0.70
    avg_reward: Optional[float] = None
    min_episodes: int = 50
    patience: int = 200
    window_size: int = 100


@dataclass
class Phase:
    """Single curriculum phase configuration.

    Attributes:
        name: Phase identifier (e.g., "1_foundation", "2a_reduce_lock")
        description: Human-readable description
        criteria: Advancement criteria for this phase
        spawn_config: Spawn point configuration
        ftg_config: FTG defender configuration
        lock_speed_steps: Steps to lock defender speed
        mixture_weights: Optional weights for sampling from multiple difficulty levels
        keep_foundation: Fraction of episodes to use foundation-level difficulty (0.0-1.0)
        keep_previous: Fraction of episodes to use previous phase difficulty (0.0-1.0)
    """
    name: str
    description: str
    criteria: AdvancementCriteria
    spawn_config: Dict[str, Any]
    ftg_config: Dict[str, Any]
    lock_speed_steps: Optional[int] = None
    mixture_weights: Optional[Dict[str, float]] = None
    keep_foundation: float = 0.0
    keep_previous: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert phase to dictionary representation."""
        return {
            'name': self.name,
            'description': self.description,
            'spawn': self.spawn_config,
            'ftg': self.ftg_config,
            'lock_speed_steps': self.lock_speed_steps,
            'mixture_weights': self.mixture_weights,
            'keep_foundation': self.keep_foundation,
            'keep_previous': self.keep_previous,
        }


@dataclass
class CurriculumState:
    """Tracks curriculum progression state.

    Attributes:
        current_phase_idx: Index of current phase
        episodes_in_phase: Number of episodes in current phase
        success_history: Recent episode outcomes (True/False)
        reward_history: Recent episode rewards
        phase_start_episode: Global episode number when phase started
        advancement_log: History of phase advancements
    """
    current_phase_idx: int = 0
    episodes_in_phase: int = 0
    success_history: deque = field(default_factory=lambda: deque(maxlen=100))
    reward_history: deque = field(default_factory=lambda: deque(maxlen=100))
    phase_start_episode: int = 0
    advancement_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            'current_phase_idx': self.current_phase_idx,
            'episodes_in_phase': self.episodes_in_phase,
            'success_history': list(self.success_history),
            'reward_history': list(self.reward_history),
            'phase_start_episode': self.phase_start_episode,
            'advancement_log': self.advancement_log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurriculumState':
        """Deserialize state from checkpoint."""
        state = cls(
            current_phase_idx=data['current_phase_idx'],
            episodes_in_phase=data['episodes_in_phase'],
            phase_start_episode=data['phase_start_episode'],
            advancement_log=data.get('advancement_log', []),
        )
        state.success_history = deque(data.get('success_history', []), maxlen=100)
        state.reward_history = deque(data.get('reward_history', []), maxlen=100)
        return state


class PhaseBasedCurriculum:
    """Modular phased curriculum learning system.

    Manages progression through distinct training phases with automatic
    advancement based on performance metrics.

    Example:
        >>> config = {
        ...     'phases': [
        ...         {
        ...             'name': '1_foundation',
        ...             'description': 'Learn basics against weak defender',
        ...             'criteria': {'success_rate': 0.70, 'patience': 200},
        ...             'spawn': {'points': ['pinch_right', 'pinch_left'],
        ...                       'speed_range': [0.44, 0.44]},
        ...             'ftg': {'steering_gain': 0.25, 'bubble_radius': 2.0},
        ...             'lock_speed_steps': 800,
        ...         },
        ...         # ... more phases
        ...     ]
        ... }
        >>> curriculum = PhaseBasedCurriculum.from_config(config)
    """

    def __init__(self, phases: List[Phase], start_phase: int = 0):
        """Initialize curriculum.

        Args:
            phases: List of Phase objects defining curriculum
            start_phase: Index of starting phase (for ablation studies)
        """
        if not phases:
            raise ValueError("Curriculum must have at least one phase")

        self.phases = phases
        self.state = CurriculumState(current_phase_idx=start_phase)
        self._warned_performance_drop = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PhaseBasedCurriculum':
        """Create curriculum from configuration dict.

        Args:
            config: Configuration dict with 'phases' list

        Returns:
            PhaseBasedCurriculum instance
        """
        phases = []
        for phase_config in config.get('phases', []):
            criteria_config = phase_config.get('criteria', {})
            criteria = AdvancementCriteria(
                success_rate=criteria_config.get('success_rate', 0.70),
                avg_reward=criteria_config.get('avg_reward'),
                min_episodes=criteria_config.get('min_episodes', 50),
                patience=criteria_config.get('patience', 200),
                window_size=criteria_config.get('window_size', 100),
            )

            phase = Phase(
                name=phase_config['name'],
                description=phase_config.get('description', ''),
                criteria=criteria,
                spawn_config=phase_config.get('spawn', {}),
                ftg_config=phase_config.get('ftg', {}),
                lock_speed_steps=phase_config.get('lock_speed_steps'),
                mixture_weights=phase_config.get('mixture_weights'),
                keep_foundation=phase_config.get('keep_foundation', 0.0),
                keep_previous=phase_config.get('keep_previous', 0.0),
            )
            phases.append(phase)

        start_phase = config.get('start_phase', 0)
        return cls(phases, start_phase=start_phase)

    def get_current_phase(self) -> Phase:
        """Get current phase object."""
        return self.phases[self.state.current_phase_idx]

    def get_current_config(self, sample_mixture: bool = True) -> Dict[str, Any]:
        """Get current phase configuration for environment setup.

        Args:
            sample_mixture: If True and phase has mixture config, sample from mixture.
                           If False, return base phase config.

        Returns:
            Phase configuration dict
        """
        phase = self.get_current_phase()
        base_config = phase.to_dict()

        # If no mixture sampling or not requested, return base config
        if not sample_mixture or (phase.keep_foundation == 0.0 and phase.keep_previous == 0.0):
            return base_config

        # Sample difficulty level based on mixture weights
        import random
        rand = random.random()

        # Determine which difficulty level to use
        if rand < phase.keep_foundation and self.state.current_phase_idx > 0:
            # Use foundation (phase 0) difficulty
            return self._get_mixture_config(base_config, 0)
        elif rand < (phase.keep_foundation + phase.keep_previous) and self.state.current_phase_idx > 1:
            # Use previous phase difficulty
            return self._get_mixture_config(base_config, self.state.current_phase_idx - 1)
        else:
            # Use current phase difficulty
            return base_config

    def _get_mixture_config(self, base_config: Dict[str, Any], phase_idx: int) -> Dict[str, Any]:
        """Get configuration mixed with an earlier phase.

        Args:
            base_config: Current phase config
            phase_idx: Index of phase to mix in

        Returns:
            Mixed configuration
        """
        if phase_idx < 0 or phase_idx >= len(self.phases):
            return base_config

        earlier_phase = self.phases[phase_idx]
        mixed_config = dict(base_config)

        # Use earlier phase's spawn and FTG config, keep current lock_speed_steps
        mixed_config['spawn'] = earlier_phase.spawn_config
        mixed_config['ftg'] = earlier_phase.ftg_config
        if earlier_phase.lock_speed_steps is not None:
            mixed_config['lock_speed_steps'] = earlier_phase.lock_speed_steps

        return mixed_config

    def is_complete(self) -> bool:
        """Check if all phases completed."""
        return self.state.current_phase_idx >= len(self.phases) - 1

    def update(
        self,
        episode_outcome: str,
        episode_reward: float,
        episode_num: int
    ) -> Optional[Dict[str, Any]]:
        """Update curriculum state after episode.

        Args:
            episode_outcome: Episode outcome ('target_crash', 'self_crash', 'timeout', etc.)
            episode_reward: Total episode reward
            episode_num: Global episode number

        Returns:
            Advancement info dict if phase advanced, None otherwise
        """
        # Record episode metrics
        is_success = episode_outcome == 'target_crash'
        self.state.success_history.append(is_success)
        self.state.reward_history.append(episode_reward)
        self.state.episodes_in_phase += 1

        # Check if we should advance
        if not self.is_complete():
            advancement_info = self._check_advancement(episode_num)
            if advancement_info:
                return advancement_info

        # Check for performance drops (warning only)
        self._check_performance_drop()

        return None

    def _check_advancement(self, episode_num: int) -> Optional[Dict[str, Any]]:
        """Check if criteria met for advancement.

        Returns:
            Advancement info dict if advancing, None otherwise
        """
        phase = self.get_current_phase()
        criteria = phase.criteria

        # Must have minimum episodes
        if self.state.episodes_in_phase < criteria.min_episodes:
            return None

        # Calculate metrics over window
        window_size = min(criteria.window_size, len(self.state.success_history))
        if window_size == 0:
            return None

        recent_successes = list(self.state.success_history)[-window_size:]
        recent_rewards = list(self.state.reward_history)[-window_size:]

        success_rate = np.mean(recent_successes)
        avg_reward = np.mean(recent_rewards)

        # Check advancement criteria
        forced = self.state.episodes_in_phase >= criteria.patience

        criteria_met = success_rate >= criteria.success_rate
        if criteria.avg_reward is not None:
            criteria_met = criteria_met and avg_reward >= criteria.avg_reward

        should_advance = forced or criteria_met

        if should_advance:
            return self._advance_phase(
                episode_num=episode_num,
                success_rate=success_rate,
                avg_reward=avg_reward,
                forced=forced,
            )

        return None

    def _advance_phase(
        self,
        episode_num: int,
        success_rate: float,
        avg_reward: float,
        forced: bool
    ) -> Dict[str, Any]:
        """Advance to next phase.

        Returns:
            Advancement info dict
        """
        old_phase = self.get_current_phase()
        self.state.current_phase_idx += 1
        new_phase = self.get_current_phase()

        advancement_info = {
            'old_phase': old_phase.name,
            'new_phase': new_phase.name,
            'episode': episode_num,
            'episodes_in_old_phase': self.state.episodes_in_phase,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'forced': forced,
        }

        # Log advancement
        self.state.advancement_log.append(advancement_info)

        # Reset phase tracking
        self.state.episodes_in_phase = 0
        self.state.phase_start_episode = episode_num
        self._warned_performance_drop = False

        return advancement_info

    def _check_performance_drop(self):
        """Check for significant performance drop (warning only)."""
        if len(self.state.reward_history) < 50 or self._warned_performance_drop:
            return

        # Compare recent performance to phase start
        recent_rewards = list(self.state.reward_history)[-25:]
        early_rewards = list(self.state.reward_history)[:25]

        if len(early_rewards) < 25:
            return

        recent_avg = np.mean(recent_rewards)
        early_avg = np.mean(early_rewards)

        # Warn if performance dropped significantly
        if recent_avg < early_avg * 0.7:  # 30% drop
            print(f"\n⚠️  Warning: Performance drop detected in phase {self.get_current_phase().name}")
            print(f"   Early avg reward: {early_avg:.1f} → Recent: {recent_avg:.1f}")
            self._warned_performance_drop = True

    def get_state(self) -> CurriculumState:
        """Get current curriculum state."""
        return self.state

    def set_state(self, state: CurriculumState):
        """Set curriculum state (for checkpointing)."""
        self.state = state

    def get_metrics(self) -> Dict[str, Any]:
        """Get current curriculum metrics for logging.

        Returns:
            Dict with curriculum metrics
        """
        phase = self.get_current_phase()

        metrics = {
            'curriculum/phase_idx': self.state.current_phase_idx,
            'curriculum/phase_name': phase.name,
            'curriculum/episodes_in_phase': self.state.episodes_in_phase,
            'curriculum/total_phases': len(self.phases),
            'curriculum/progress': self.state.current_phase_idx / max(1, len(self.phases) - 1),
        }

        # Add rolling metrics if we have data
        if len(self.state.success_history) > 0:
            window_size = min(phase.criteria.window_size, len(self.state.success_history))
            recent_successes = list(self.state.success_history)[-window_size:]
            recent_rewards = list(self.state.reward_history)[-window_size:]

            metrics['curriculum/success_rate'] = np.mean(recent_successes)
            metrics['curriculum/avg_reward'] = np.mean(recent_rewards)
            metrics['curriculum/criteria_success_rate'] = phase.criteria.success_rate
            if phase.criteria.avg_reward is not None:
                metrics['curriculum/criteria_avg_reward'] = phase.criteria.avg_reward

        return metrics

    def __repr__(self) -> str:
        """String representation."""
        phase = self.get_current_phase()
        return (
            f"PhaseBasedCurriculum("
            f"phase={self.state.current_phase_idx + 1}/{len(self.phases)}, "
            f"name='{phase.name}', "
            f"episodes={self.state.episodes_in_phase})"
        )
