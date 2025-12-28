"""Run identification and hashing utilities.

Provides consistent run ID resolution across training, checkpointing, and W&B logging.
Aligns with legacy behavior for backward compatibility.
"""

import os
import re
import time
from typing import Optional


def slugify(text: str) -> str:
    """Convert text to filesystem/URL-safe slug.

    Args:
        text: Input text to slugify

    Returns:
        Slugified text (lowercase, alphanumeric + hyphens/underscores only)

    Examples:
        >>> slugify("My Experiment!")
        'my-experiment'
        >>> slugify("test_run_123")
        'test_run_123'
        >>> slugify("Run #42 (v2)")
        'run-42-v2'
    """
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and non-alphanumeric chars with hyphens
    text = re.sub(r'[^a-z0-9_-]+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    # Collapse multiple hyphens
    text = re.sub(r'-+', '-', text)
    return text


def resolve_run_id(
    scenario_name: Optional[str] = None,
    algorithm: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """Resolve unique run identifier from environment and config.

    Priority order (first non-empty wins):
    1. F110_RUN_SUFFIX env var (legacy compatibility)
    2. WANDB_RUN_NAME env var (W&B run name)
    3. WANDB_RUN_ID env var (W&B run ID)
    4. RUN_CONFIG_HASH env var (custom hash)
    5. RUN_SEED env var (seed-based naming)
    6. Auto-generate: {scenario}_{algorithm}_s{seed}_{timestamp}_{random}

    Args:
        scenario_name: Scenario/experiment name (optional)
        algorithm: Algorithm name (optional)
        seed: Random seed (optional)

    Returns:
        Slugified run identifier (alphanumeric + hyphens/underscores only)

    Examples:
        >>> os.environ['F110_RUN_SUFFIX'] = 'my_run_001'
        >>> resolve_run_id()
        'my_run_001'

        >>> os.environ.pop('F110_RUN_SUFFIX', None)
        >>> resolve_run_id(scenario_name='gaplock_ppo', algorithm='ppo', seed=42)
        'gaplock_ppo_ppo_s42_1735315200_a1b2'
    """
    # Priority 1: F110_RUN_SUFFIX (legacy compatibility)
    run_suffix = os.environ.get('F110_RUN_SUFFIX', '').strip()
    if run_suffix:
        return slugify(run_suffix)

    # Priority 2: WANDB_RUN_NAME
    wandb_run_name = os.environ.get('WANDB_RUN_NAME', '').strip()
    if wandb_run_name:
        return slugify(wandb_run_name)

    # Priority 3: WANDB_RUN_ID
    wandb_run_id = os.environ.get('WANDB_RUN_ID', '').strip()
    if wandb_run_id:
        return slugify(wandb_run_id)

    # Priority 4: RUN_CONFIG_HASH
    config_hash = os.environ.get('RUN_CONFIG_HASH', '').strip()
    if config_hash:
        return slugify(config_hash)

    # Priority 5: RUN_SEED
    seed_str = os.environ.get('RUN_SEED', '').strip()
    if seed_str:
        return slugify(f"seed_{seed_str}")

    # Priority 6: Auto-generate
    # Format: {scenario}_{algorithm}_s{seed}_{timestamp}_{random}
    parts = []

    if scenario_name:
        parts.append(slugify(scenario_name))

    if algorithm:
        parts.append(slugify(algorithm))

    if seed is not None:
        parts.append(f"s{seed}")

    # Add timestamp (unix timestamp)
    timestamp = int(time.time())
    parts.append(str(timestamp))

    # Add random suffix (4 hex chars)
    import random
    random_suffix = f"{random.randint(0, 0xFFFF):04x}"
    parts.append(random_suffix)

    # Join with underscores
    run_id = '_'.join(parts)

    return slugify(run_id)


def set_run_id_env(run_id: str) -> None:
    """Set run ID in environment variables for child processes.

    Sets F110_RUN_SUFFIX to ensure consistent run ID across all processes.

    Args:
        run_id: Run identifier to set
    """
    os.environ['F110_RUN_SUFFIX'] = run_id


def get_checkpoint_dir(
    run_id: str,
    scenario_name: Optional[str] = None,
    base_dir: str = "outputs/checkpoints"
) -> str:
    """Get checkpoint directory path for a run.

    Directory structure:
        outputs/checkpoints/{scenario}/{run_id}/

    Args:
        run_id: Run identifier
        scenario_name: Scenario name (optional, creates flat structure if None)
        base_dir: Base checkpoint directory (default: outputs/checkpoints)

    Returns:
        Absolute path to checkpoint directory

    Examples:
        >>> get_checkpoint_dir('my_run_001', 'gaplock_ppo')
        '/home/aaron/F110_MARL/outputs/checkpoints/gaplock_ppo/my_run_001'

        >>> get_checkpoint_dir('my_run_001')
        '/home/aaron/F110_MARL/outputs/checkpoints/my_run_001'
    """
    base_path = os.path.abspath(base_dir)

    if scenario_name:
        scenario_slug = slugify(scenario_name)
        return os.path.join(base_path, scenario_slug, run_id)
    else:
        return os.path.join(base_path, run_id)


def get_output_dir(
    run_id: str,
    scenario_name: Optional[str] = None,
    base_dir: str = "outputs"
) -> str:
    """Get output directory path for a run.

    Directory structure:
        outputs/{scenario}/{run_id}/

    Args:
        run_id: Run identifier
        scenario_name: Scenario name (optional, creates flat structure if None)
        base_dir: Base output directory (default: outputs)

    Returns:
        Absolute path to output directory

    Examples:
        >>> get_output_dir('my_run_001', 'gaplock_ppo')
        '/home/aaron/F110_MARL/outputs/gaplock_ppo/my_run_001'
    """
    base_path = os.path.abspath(base_dir)

    if scenario_name:
        scenario_slug = slugify(scenario_name)
        return os.path.join(base_path, scenario_slug, run_id)
    else:
        return os.path.join(base_path, run_id)
