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
        regress_success_rate: Success rate threshold to trigger regression (optional)
        regress_min_episodes: Minimum episodes before regression allowed (optional)
        regress_patience: Minimum episodes in phase before regression allowed (optional)
        regress_window_size: Window size for regression success rate (optional)
        max_self_crash_rate: Maximum allowed self-crash rate (0.0-1.0)
        max_collision_rate: Maximum allowed collision rate (0.0-1.0)
        max_target_finish_rate: Maximum allowed target-finish rate (0.0-1.0)
        eval_success_rate: Required eval success rate for advancement (0.0-1.0)
        eval_required_runs: Required consecutive eval runs meeting eval_success_rate
    """
    success_rate: float = 0.70
    avg_reward: Optional[float] = None
    min_episodes: int = 50
    patience: int = 200
    window_size: int = 100
    regress_success_rate: Optional[float] = None
    regress_min_episodes: Optional[int] = None
    regress_patience: Optional[int] = None
    regress_window_size: Optional[int] = None
    max_self_crash_rate: Optional[float] = None
    max_collision_rate: Optional[float] = None
    max_target_finish_rate: Optional[float] = None
    eval_success_rate: float = 1.0
    eval_required_runs: int = 4


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
        outcome_history: Recent episode outcomes (string labels)
        reward_history: Recent episode rewards
        phase_start_episode: Global episode number when phase started
        advancement_log: History of phase advancements and regressions
        eval_success_streak: Consecutive eval runs meeting eval criteria
    """
    current_phase_idx: int = 0
    episodes_in_phase: int = 0
    success_history: deque = field(default_factory=lambda: deque(maxlen=100))
    outcome_history: deque = field(default_factory=lambda: deque(maxlen=100))
    reward_history: deque = field(default_factory=lambda: deque(maxlen=100))
    phase_start_episode: int = 0
    advancement_log: List[Dict[str, Any]] = field(default_factory=list)
    eval_success_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            'current_phase_idx': self.current_phase_idx,
            'episodes_in_phase': self.episodes_in_phase,
            'success_history': list(self.success_history),
            'outcome_history': list(self.outcome_history),
            'reward_history': list(self.reward_history),
            'phase_start_episode': self.phase_start_episode,
            'advancement_log': self.advancement_log,
            'eval_success_streak': self.eval_success_streak,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurriculumState':
        """Deserialize state from checkpoint."""
        success_history = data.get('success_history', [])
        outcome_history = data.get('outcome_history', [])
        reward_history = data.get('reward_history', [])
        history_len = max(100, len(success_history), len(outcome_history), len(reward_history), 1)
        state = cls(
            current_phase_idx=data['current_phase_idx'],
            episodes_in_phase=data['episodes_in_phase'],
            phase_start_episode=data['phase_start_episode'],
            advancement_log=data.get('advancement_log', []),
            eval_success_streak=data.get('eval_success_streak', 0),
        )
        state.success_history = deque(success_history, maxlen=history_len)
        state.outcome_history = deque(outcome_history, maxlen=history_len)
        state.reward_history = deque(reward_history, maxlen=history_len)
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
        self.eval_gate_enabled = True
        self._warned_performance_drop = False
        self._reset_phase_histories(self.get_current_phase().criteria.window_size)

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
                regress_success_rate=criteria_config.get('regress_success_rate'),
                regress_min_episodes=criteria_config.get('regress_min_episodes'),
                regress_patience=criteria_config.get('regress_patience'),
                regress_window_size=criteria_config.get('regress_window_size'),
                max_self_crash_rate=criteria_config.get('max_self_crash_rate'),
                max_collision_rate=criteria_config.get('max_collision_rate'),
                max_target_finish_rate=criteria_config.get('max_target_finish_rate'),
                eval_success_rate=criteria_config.get('eval_success_rate', 1.0),
                eval_required_runs=criteria_config.get('eval_required_runs', 4),
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

    def record_eval_result(self, success_rate: float) -> int:
        """Record evaluation success rate for eval gating.

        Returns:
            Updated eval success streak.
        """
        if not self.eval_gate_enabled:
            return 0
        criteria = self.get_current_phase().criteria
        if success_rate >= criteria.eval_success_rate:
            self.state.eval_success_streak += 1
        else:
            self.state.eval_success_streak = 0
        return self.state.eval_success_streak

    def _eval_gate_satisfied(self) -> bool:
        if not self.eval_gate_enabled:
            return True
        criteria = self.get_current_phase().criteria
        required = max(0, int(criteria.eval_required_runs))
        if required == 0:
            return True
        return self.state.eval_success_streak >= required

    def _compute_advancement_status(self) -> Optional[Dict[str, Any]]:
        """Compute training criteria stats for advancement."""
        phase = self.get_current_phase()
        criteria = phase.criteria

        if self.state.episodes_in_phase < criteria.min_episodes:
            return None

        window_size = min(criteria.window_size, len(self.state.success_history))
        if window_size == 0:
            return None

        recent_successes = list(self.state.success_history)[-window_size:]
        recent_rewards = list(self.state.reward_history)[-window_size:]

        success_rate = float(np.mean(recent_successes))
        avg_reward = float(np.mean(recent_rewards))

        criteria_met = success_rate >= criteria.success_rate
        if criteria.avg_reward is not None:
            criteria_met = criteria_met and avg_reward >= criteria.avg_reward

        caps_met = True
        recent_outcomes = list(self.state.outcome_history)[-window_size:]
        if recent_outcomes:
            total_outcomes = len(recent_outcomes)
            outcome_counts = {
                'self_crash': recent_outcomes.count('self_crash'),
                'collision': recent_outcomes.count('collision'),
                'target_finish': recent_outcomes.count('target_finish'),
            }
            if criteria.max_self_crash_rate is not None:
                caps_met = caps_met and (
                    outcome_counts['self_crash'] / total_outcomes <= criteria.max_self_crash_rate
                )
            if criteria.max_collision_rate is not None:
                caps_met = caps_met and (
                    outcome_counts['collision'] / total_outcomes <= criteria.max_collision_rate
                )
            if criteria.max_target_finish_rate is not None:
                caps_met = caps_met and (
                    outcome_counts['target_finish'] / total_outcomes <= criteria.max_target_finish_rate
                )

        forced = self.state.episodes_in_phase >= criteria.patience

        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "criteria_met": criteria_met,
            "caps_met": caps_met,
            "forced": forced,
        }

    def is_eval_ready(self) -> bool:
        """Return True when training criteria met and eval gate not satisfied."""
        status = self._compute_advancement_status()
        if not status:
            return False
        if not status["criteria_met"] or not status["caps_met"]:
            return False
        return not self._eval_gate_satisfied()

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
        outcome_value = (
            episode_outcome.value if hasattr(episode_outcome, 'value') else str(episode_outcome)
        )
        is_success = outcome_value == 'target_crash'
        self.state.success_history.append(is_success)
        self.state.outcome_history.append(outcome_value)
        self.state.reward_history.append(episode_reward)
        self.state.episodes_in_phase += 1

        # Check if we should advance
        if not self.is_complete():
            advancement_info = self._check_advancement(episode_num)
            if advancement_info:
                return advancement_info

        # Check if we should regress
        regression_info = self._check_regression(episode_num)
        if regression_info:
            return regression_info

        # Check for performance drops (warning only)
        self._check_performance_drop()

        return None

    def _check_advancement(self, episode_num: int) -> Optional[Dict[str, Any]]:
        """Check if criteria met for advancement.

        Returns:
            Advancement info dict if advancing, None otherwise
        """
        status = self._compute_advancement_status()
        if not status:
            return None

        if not status["criteria_met"] or not status["caps_met"]:
            return None

        if not self._eval_gate_satisfied():
            return None

        return self._advance_phase(
            episode_num=episode_num,
            success_rate=status["success_rate"],
            avg_reward=status["avg_reward"],
            forced=status["forced"],
        )

        return None

    def _check_regression(self, episode_num: int) -> Optional[Dict[str, Any]]:
        """Check if criteria met for regression.

        Returns:
            Regression info dict if regressing, None otherwise
        """
        if self.state.current_phase_idx <= 0:
            return None

        phase = self.get_current_phase()
        criteria = phase.criteria

        if criteria.regress_success_rate is None:
            return None

        min_episodes = criteria.regress_min_episodes
        if min_episodes is None:
            min_episodes = criteria.min_episodes

        if self.state.episodes_in_phase < min_episodes:
            return None

        patience = criteria.regress_patience
        if patience is not None and self.state.episodes_in_phase < patience:
            return None

        window_size_cfg = criteria.regress_window_size
        if window_size_cfg is None:
            window_size_cfg = criteria.window_size

        window_size = min(int(window_size_cfg), len(self.state.success_history))
        if window_size <= 0:
            return None

        recent_successes = list(self.state.success_history)[-window_size:]
        recent_rewards = list(self.state.reward_history)[-window_size:]
        success_rate = np.mean(recent_successes)
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

        if success_rate >= criteria.regress_success_rate:
            return None

        return self._regress_phase(
            episode_num=episode_num,
            success_rate=success_rate,
            avg_reward=avg_reward,
            threshold=criteria.regress_success_rate,
        )

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
            'action': 'advance',
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
        self.state.eval_success_streak = 0
        self._warned_performance_drop = False
        self._reset_phase_histories(new_phase.criteria.window_size)

        return advancement_info

    def _regress_phase(
        self,
        episode_num: int,
        success_rate: float,
        avg_reward: float,
        threshold: float,
    ) -> Dict[str, Any]:
        """Regress to previous phase.

        Returns:
            Regression info dict
        """
        old_phase = self.get_current_phase()
        self.state.current_phase_idx -= 1
        new_phase = self.get_current_phase()

        regression_info = {
            'action': 'regress',
            'old_phase': old_phase.name,
            'new_phase': new_phase.name,
            'episode': episode_num,
            'episodes_in_old_phase': self.state.episodes_in_phase,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'threshold': threshold,
            'forced': False,
        }

        # Log regression alongside advances
        self.state.advancement_log.append(regression_info)

        # Reset phase tracking
        self.state.episodes_in_phase = 0
        self.state.phase_start_episode = episode_num
        self.state.eval_success_streak = 0
        self._warned_performance_drop = False
        self._reset_phase_histories(new_phase.criteria.window_size)

        return regression_info

    def _reset_phase_histories(self, window_size: int) -> None:
        """Reset rolling histories for a new phase."""
        history_len = max(1, int(window_size))
        self.state.success_history = deque(maxlen=history_len)
        self.state.outcome_history = deque(maxlen=history_len)
        self.state.reward_history = deque(maxlen=history_len)

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
            'curriculum/eval_success_streak': self.state.eval_success_streak,
            'curriculum/criteria_eval_success_rate': phase.criteria.eval_success_rate,
            'curriculum/criteria_eval_required_runs': phase.criteria.eval_required_runs,
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
            if len(self.state.outcome_history) > 0:
                recent_outcomes = list(self.state.outcome_history)[-window_size:]
                total_outcomes = len(recent_outcomes)
                metrics['curriculum/self_crash_rate'] = (
                    recent_outcomes.count('self_crash') / total_outcomes
                )
                metrics['curriculum/collision_rate'] = (
                    recent_outcomes.count('collision') / total_outcomes
                )
                metrics['curriculum/target_finish_rate'] = (
                    recent_outcomes.count('target_finish') / total_outcomes
                )

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
