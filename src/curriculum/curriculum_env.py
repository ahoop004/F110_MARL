"""Curriculum-aware environment configuration.

Provides utilities to apply curriculum phase configurations to the environment
and agent parameters dynamically during training.
"""

from typing import Dict, Any, Optional


def apply_curriculum_to_env(env, phase_config: Dict[str, Any]) -> None:
    """Apply curriculum phase configuration to environment.

    Args:
        env: F110 environment instance
        phase_config: Phase configuration dict with 'spawn', 'ftg', etc.
    """
    spawn_config = phase_config.get('spawn', {})
    lock_speed_steps = phase_config.get('lock_speed_steps')

    # Update spawn curriculum if it exists
    if hasattr(env, 'spawn_curriculum') and env.spawn_curriculum:
        # Update speed lock duration
        if lock_speed_steps is not None:
            env.spawn_curriculum.lock_speed_steps = lock_speed_steps

        # Update spawn points if specified
        spawn_points = spawn_config.get('points')
        if spawn_points:
            _update_spawn_points(env.spawn_curriculum, spawn_points)

        # Update speed range if specified
        speed_range = spawn_config.get('speed_range')
        if speed_range and len(speed_range) == 2:
            _update_speed_range(env.spawn_curriculum, speed_range)


def apply_curriculum_to_agent(agent, agent_id: str, phase_config: Dict[str, Any]) -> None:
    """Apply curriculum phase configuration to agent (e.g., FTG defender).

    Args:
        agent: Agent instance
        agent_id: Agent identifier
        phase_config: Phase configuration dict
    """
    ftg_config = phase_config.get('ftg', {})

    # Apply to FTG agents
    if hasattr(agent, 'steering_gain'):  # FTG agent
        for param, value in ftg_config.items():
            if hasattr(agent, param):
                setattr(agent, param, value)


def _update_spawn_points(spawn_curriculum, spawn_points: list) -> None:
    """Update active spawn points in curriculum.

    Args:
        spawn_curriculum: SpawnCurriculum instance
        spawn_points: List of spawn point names to activate
    """
    if not hasattr(spawn_curriculum, 'stages'):
        return

    # Update current stage's spawn points
    current_stage = spawn_curriculum.current_stage
    if current_stage and hasattr(current_stage, 'spawn_points'):
        # If 'all' specified, use all available points
        if spawn_points == ['all'] or spawn_points == 'all':
            # Get all available spawn points from config
            if hasattr(spawn_curriculum, 'spawn_configs'):
                current_stage.spawn_points = list(spawn_curriculum.spawn_configs.keys())
        else:
            current_stage.spawn_points = spawn_points


def _update_speed_range(spawn_curriculum, speed_range: list) -> None:
    """Update defender speed range in curriculum.

    Args:
        spawn_curriculum: SpawnCurriculum instance
        speed_range: [min_speed, max_speed]
    """
    if not hasattr(spawn_curriculum, 'stages'):
        return

    current_stage = spawn_curriculum.current_stage
    if current_stage and hasattr(current_stage, 'speed_range'):
        current_stage.speed_range = speed_range


def extract_outcome_from_info(info: Dict[str, Any]) -> str:
    """Extract episode outcome from info dict.

    Args:
        info: Episode info dict

    Returns:
        Outcome string ('target_crash', 'self_crash', 'timeout', etc.)
    """
    # Check for outcome in various possible locations
    if 'outcome' in info:
        return info['outcome']

    if 'termination_reason' in info:
        return info['termination_reason']

    # Infer from terminal flags
    if info.get('target_collision', False):
        return 'target_crash'
    if info.get('wall_collision', False) or info.get('self_collision', False):
        return 'self_crash'
    if info.get('timeout', False):
        return 'timeout'

    return 'unknown'


def create_curriculum_checkpoint_data(curriculum) -> Dict[str, Any]:
    """Create checkpoint data for curriculum state.

    Args:
        curriculum: PhaseBasedCurriculum instance

    Returns:
        Dict with curriculum state for checkpointing
    """
    return {
        'type': 'phased',
        'state': curriculum.get_state().to_dict(),
        'current_phase_name': curriculum.get_current_phase().name,
    }


def restore_curriculum_from_checkpoint(
    curriculum,
    checkpoint_data: Dict[str, Any]
) -> None:
    """Restore curriculum state from checkpoint.

    Args:
        curriculum: PhaseBasedCurriculum instance
        checkpoint_data: Checkpoint data dict
    """
    from .phased_curriculum import CurriculumState

    if checkpoint_data.get('type') != 'phased':
        print(f"Warning: Checkpoint has curriculum type '{checkpoint_data.get('type')}', expected 'phased'")
        return

    state_data = checkpoint_data.get('state')
    if state_data:
        state = CurriculumState.from_dict(state_data)
        curriculum.set_state(state)
        print(f"Restored curriculum state: {curriculum}")


__all__ = [
    'apply_curriculum_to_env',
    'apply_curriculum_to_agent',
    'extract_outcome_from_info',
    'create_curriculum_checkpoint_data',
    'restore_curriculum_from_checkpoint',
]
