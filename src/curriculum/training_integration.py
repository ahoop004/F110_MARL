"""Integration of phased curriculum with enhanced training loop.

Provides utilities to integrate PhaseBasedCurriculum into the existing
training pipeline with minimal modifications.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def add_curriculum_to_training_loop(
    training_loop,
    curriculum,
    scenario: Optional[Dict[str, Any]] = None
):
    """Add phased curriculum to an existing EnhancedTrainingLoop instance.

    This modifies the training loop to:
    - Apply curriculum configuration before each episode
    - Update curriculum after each episode
    - Log curriculum metrics to WandB

    Args:
        training_loop: EnhancedTrainingLoop instance
        curriculum: PhaseBasedCurriculum instance
        scenario: Optional scenario configuration (for agent lookup)
    """
    # Store curriculum reference
    training_loop.phased_curriculum = curriculum
    if hasattr(training_loop, "rich_console") and training_loop.rich_console:
        metrics = curriculum.get_metrics()
        training_loop._phase_curriculum_state = {
            "phase_index": metrics.get("curriculum/phase_idx"),
            "phase_name": metrics.get("curriculum/phase_name"),
            "phase_success_rate": metrics.get("curriculum/success_rate"),
        }

    # Store original _run_episode method
    original_run_episode = training_loop._run_episode

    def _select_curriculum_agent_id() -> str:
        """Pick the attacker/primary agent to drive curriculum updates."""
        if scenario:
            for agent_id, agent_config in scenario.get('agents', {}).items():
                if str(agent_config.get('role', '')).lower() == 'attacker':
                    return agent_id
        primary_id = getattr(training_loop, "primary_agent_id", None)
        if primary_id:
            return primary_id
        return list(training_loop.agents.keys())[0]

    curriculum_agent_id = _select_curriculum_agent_id()

    def _run_episode_with_curriculum(episode_num: int):
        """Wrapped episode runner that applies curriculum config."""
        # Get current phase configuration (with mixture sampling)
        phase_config = curriculum.get_current_config(sample_mixture=True)

        # Apply curriculum to environment
        from .curriculum_env import (
            apply_curriculum_to_env,
            apply_curriculum_to_agent,
            apply_curriculum_to_spawn_curriculum,
        )

        apply_curriculum_to_env(training_loop.env, phase_config)
        if getattr(training_loop, "spawn_curriculum", None):
            apply_curriculum_to_spawn_curriculum(training_loop.spawn_curriculum, phase_config)

        # Apply curriculum to agents (e.g., FTG)
        for agent_id, agent in training_loop.agents.items():
            apply_curriculum_to_agent(agent, agent_id, phase_config)

        # Run the episode
        original_run_episode(episode_num)

        # After episode, update curriculum
        # Extract outcome from attacker/primary agent's metrics
        tracker = training_loop.metrics_trackers.get(curriculum_agent_id)
        if tracker is None:
            fallback_id = list(training_loop.agents.keys())[0]
            tracker = training_loop.metrics_trackers.get(fallback_id)

        if tracker and tracker.episodes:
            latest = tracker.get_latest(1)[0]
            outcome = latest.outcome.value if hasattr(latest.outcome, 'value') else str(latest.outcome)
            reward = latest.total_reward
            success = outcome == 'target_crash'

            # Update curriculum
            advancement_info = curriculum.update(outcome, reward, episode_num)

            # Log advancement if occurred
            if advancement_info:
                msg = (
                    f"\n{'='*60}\n"
                    f"📈 CURRICULUM ADVANCED!\n"
                    f"  Episode: {episode_num}\n"
                    f"  {advancement_info['old_phase']} → {advancement_info['new_phase']}\n"
                    f"  Success Rate: {advancement_info['success_rate']:.2%}\n"
                    f"  Avg Reward: {advancement_info['avg_reward']:.1f}\n"
                    f"  Episodes in Phase: {advancement_info['episodes_in_old_phase']}\n"
                    f"  Forced: {advancement_info['forced']}\n"
                    f"{'='*60}\n"
                )
                logger.info(msg)
                if training_loop.console_logger:
                    training_loop.console_logger.print_success(msg)

            curriculum_metrics = curriculum.get_metrics()

            # Log minimal curriculum metrics to WandB
            if training_loop.wandb_logger:
                training_loop.wandb_logger.log_metrics({
                    'train/episode': int(episode_num),
                    'train/success': int(success),
                    'curriculum/phase_idx': curriculum_metrics.get('curriculum/phase_idx'),
                    'curriculum/phase_name': curriculum_metrics.get('curriculum/phase_name'),
                    'curriculum/phase_success_rate': curriculum_metrics.get('curriculum/success_rate') or 0.0,
                    'curriculum/stage': curriculum_metrics.get('curriculum/phase_name'),
                    'curriculum/stage_success_rate': curriculum_metrics.get('curriculum/success_rate') or 0.0,
                }, step=episode_num)

            if hasattr(training_loop, "rich_console") and training_loop.rich_console:
                training_loop._phase_curriculum_state = {
                    "phase_index": curriculum_metrics.get("curriculum/phase_idx"),
                    "phase_name": curriculum_metrics.get("curriculum/phase_name"),
                    "phase_success_rate": curriculum_metrics.get("curriculum/success_rate"),
                }

    # Replace _run_episode with wrapped version
    training_loop._run_episode = _run_episode_with_curriculum

    logger.info(f"Phased curriculum integrated with training loop: {curriculum}")


def setup_curriculum_from_scenario(
    scenario: Dict[str, Any],
    training_loop
) -> Optional['PhaseBasedCurriculum']:
    """Set up phased curriculum from scenario configuration.

    Args:
        scenario: Scenario configuration dict
        training_loop: EnhancedTrainingLoop instance

    Returns:
        PhaseBasedCurriculum instance if configured, None otherwise
    """
    curriculum_config = scenario.get('curriculum')

    if not curriculum_config:
        return None

    if curriculum_config.get('type') != 'phased':
        logger.info(f"Curriculum type '{curriculum_config.get('type')}' is not 'phased', skipping")
        return None

    # Create curriculum from config
    from .phased_curriculum import PhaseBasedCurriculum

    curriculum = PhaseBasedCurriculum.from_config(curriculum_config)

    # Integrate with training loop
    add_curriculum_to_training_loop(training_loop, curriculum, scenario)

    logger.info(f"Phased curriculum initialized: {len(curriculum.phases)} phases")
    logger.info(f"Starting phase: {curriculum.get_current_phase().name}")

    return curriculum


__all__ = [
    'add_curriculum_to_training_loop',
    'setup_curriculum_from_scenario',
]
