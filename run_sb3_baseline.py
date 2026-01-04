#!/usr/bin/env python3
"""Train SB3 baseline agents for F110 gaplock task with full curriculum support.

This script provides baselines using Stable-Baselines3 (SB3) implementations
of various RL algorithms with full support for curriculum learning from scenario files.

Features:
    - Custom reward strategies from scenario configs
    - Spawn curriculum (environment-level difficulty progression)
    - Phased curriculum (multi-stage training with advancement criteria)
    - FTG opponent scheduling (dynamic difficulty adjustment)
    - WandB integration with automatic curriculum metrics logging
    - All configuration driven by scenario YAML files

Supported algorithms:
    - Off-policy (continuous): sac, td3, ddpg, tqc
    - Off-policy (discrete): dqn, qrdqn
    - On-policy: ppo, a2c

Usage:
    # Basic training (uses all settings from scenario)
    python run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_sb3_ppo.yaml --wandb

    # Override episodes from command line
    python run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_sb3_ppo.yaml --wandb --episodes 10000

    # Custom seed
    python run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_sb3_ppo.yaml --wandb --seed 123

Note:
    The scenario file controls all training parameters including:
    - Episodes, seed, environment settings
    - Reward configuration
    - Spawn curriculum stages and criteria
    - Phased curriculum progression
    - FTG opponent schedules
    - WandB project, tags, and notes
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC, TD3, PPO, DDPG, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Optional sb3-contrib algorithms
try:
    from sb3_contrib import TQC, QRDQN
    CONTRIB_AVAILABLE = True
except ImportError:
    CONTRIB_AVAILABLE = False
    TQC = None
    QRDQN = None

# Optional wandb integration
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from src.core.scenario import load_and_expand_scenario
from src.core.setup import create_training_setup
from src.core.obs_flatten import flatten_observation
from src.baselines.sb3_wrapper import SB3SingleAgentWrapper
from src.baselines.sb3_curriculum_callback import CurriculumCallback
from src.baselines.sb3_eval_callback import SB3EvaluationCallback
from src.core.evaluator import EvaluationConfig
from src.loggers import RichConsole


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SB3 baseline for F110 gaplock'
    )
    parser.add_argument(
        '--algo',
        type=str,
        required=True,
        choices=['sac', 'td3', 'ppo', 'a2c', 'ddpg', 'tqc', 'dqn', 'qrdqn'],
        help='RL algorithm to use'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='Path to scenario YAML file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of episodes to train (default: use value from scenario)'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable wandb logging'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (default: use value from scenario)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./sb3_models',
        help='Directory to save models (default: ./sb3_models)'
    )
    parser.add_argument(
        '--override',
        type=str,
        action='append',
        default=[],
        help='Override scenario parameters (e.g., agents.car_0.params.learning_rate=0.001)'
    )
    return parser.parse_args()


def apply_overrides(scenario: dict, overrides: list[str]) -> dict:
    """Apply command-line overrides to scenario dictionary.

    Args:
        scenario: Scenario dictionary
        overrides: List of override strings in format "key.subkey.param=value"

    Returns:
        Modified scenario dictionary
    """
    for override in overrides:
        if '=' not in override:
            print(f"Warning: Invalid override format '{override}', skipping")
            continue

        path, value_str = override.split('=', 1)
        keys = path.split('.')

        # Try to parse value as number, boolean, or keep as string
        value = value_str
        try:
            # Try int
            value = int(value_str)
        except ValueError:
            try:
                # Try float
                value = float(value_str)
            except ValueError:
                # Try boolean
                if value_str.lower() in ('true', 'false'):
                    value = value_str.lower() == 'true'
                elif value_str.lower() == 'none':
                    value = None
                # Otherwise keep as string

        # Navigate to the correct nested dictionary
        current = scenario
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value
        print(f"Override: {path} = {value} ({type(value).__name__})")

    return scenario


def set_nested_value(config: dict, path: str, value) -> None:
    """Set a nested dictionary value using dot-notation path."""
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_space_dim(space) -> int:
    """Get total dimension of a gym space."""
    if isinstance(space, spaces.Dict):
        return sum(get_space_dim(s) for s in space.spaces.values())
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, spaces.Discrete):
        return 1
    if isinstance(space, spaces.MultiDiscrete):
        return len(space.nvec)
    return 1


def resolve_sb3_agent_config(scenario: dict, algo_name: str):
    """Pick the SB3-controlled agent config from the scenario."""
    agents_cfg = scenario.get('agents', {})
    target_algo = f"sb3_{algo_name}".lower()
    for agent_id, cfg in agents_cfg.items():
        if str(cfg.get('algorithm', '')).lower() == target_algo:
            return agent_id, cfg
    if 'car_0' in agents_cfg:
        return 'car_0', agents_cfg['car_0']
    if agents_cfg:
        first_id = next(iter(agents_cfg))
        return first_id, agents_cfg[first_id]
    raise ValueError("No agents configured in scenario")


def resolve_observation_preset(agent_cfg: dict) -> str | None:
    obs_cfg = agent_cfg.get('observation')
    if isinstance(obs_cfg, dict):
        if 'preset' in obs_cfg:
            return obs_cfg['preset']
        if obs_cfg:
            return 'gaplock'
    return None


def create_sb3_agent(algo_name: str, env, params: dict = None, seed: int = 42, tensorboard_log: str = None):
    """Create SB3 agent with hyperparameters from scenario or defaults.

    Args:
        algo_name: Algorithm name ('sac', 'td3', 'ppo', 'a2c', 'ddpg', 'tqc', 'dqn', 'qrdqn')
        env: Gym environment
        params: Hyperparameters dict from scenario (optional)
        seed: Random seed
        tensorboard_log: Path to tensorboard log directory

    Returns:
        SB3 model
    """
    if params is None:
        params = {}

    # Build policy_kwargs from params
    policy_kwargs = {}
    if 'hidden_dims' in params:
        policy_kwargs['net_arch'] = params['hidden_dims']
    else:
        policy_kwargs['net_arch'] = [256, 256]  # Default

    common_kwargs = {
        'policy': 'MlpPolicy',
        'env': env,
        'verbose': 1,
        'seed': seed,
        'tensorboard_log': tensorboard_log,
        'policy_kwargs': policy_kwargs,
    }

    if algo_name == 'sac':
        return SAC(
            **common_kwargs,
            learning_rate=params.get('learning_rate', 3e-4),
            buffer_size=params.get('buffer_size', 1_000_000),
            batch_size=params.get('batch_size', 256),
            tau=params.get('tau', 0.005),
            gamma=params.get('gamma', 0.995),
            ent_coef=params.get('ent_coef', 'auto'),
            target_entropy=params.get('target_entropy', 'auto'),
            learning_starts=params.get('learning_starts', 1000),
        )
    elif algo_name == 'td3':
        return TD3(
            **common_kwargs,
            learning_rate=params.get('learning_rate', 3e-4),
            buffer_size=params.get('buffer_size', 1_000_000),
            batch_size=params.get('batch_size', 256),
            tau=params.get('tau', 0.005),
            gamma=params.get('gamma', 0.995),
            policy_delay=params.get('policy_delay', 2),
            target_policy_noise=params.get('target_policy_noise', 0.2),
            target_noise_clip=params.get('target_noise_clip', 0.5),
            learning_starts=params.get('learning_starts', 1000),
        )
    elif algo_name == 'ppo':
        return PPO(
            **common_kwargs,
            learning_rate=params.get('learning_rate', 3e-4),
            n_steps=params.get('n_steps', 2048),
            batch_size=params.get('batch_size', 256),
            gamma=params.get('gamma', 0.995),
            gae_lambda=params.get('gae_lambda', 0.95),
            clip_range=params.get('clip_range', 0.2),
            ent_coef=params.get('ent_coef', 0.02),
            n_epochs=params.get('n_epochs', 10),
            vf_coef=params.get('vf_coef', 0.5),
            max_grad_norm=params.get('max_grad_norm', 0.5),
        )
    elif algo_name == 'a2c':
        return A2C(
            **common_kwargs,
            learning_rate=params.get('learning_rate', 3e-4),
            n_steps=params.get('n_steps', 5),
            gamma=params.get('gamma', 0.995),
            gae_lambda=params.get('gae_lambda', 1.0),
            ent_coef=params.get('ent_coef', 0.0),
            vf_coef=params.get('vf_coef', 0.5),
            max_grad_norm=params.get('max_grad_norm', 0.5),
        )
    elif algo_name == 'ddpg':
        from stable_baselines3.common.noise import NormalActionNoise
        n_actions = env.action_space.shape[0]
        action_noise_sigma = params.get('action_noise_sigma', 0.1)
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_sigma * np.ones(n_actions)
        )
        return DDPG(
            **common_kwargs,
            learning_rate=params.get('learning_rate', 3e-4),
            buffer_size=params.get('buffer_size', 1_000_000),
            batch_size=params.get('batch_size', 256),
            tau=params.get('tau', 0.005),
            gamma=params.get('gamma', 0.995),
            action_noise=action_noise,
            learning_starts=params.get('learning_starts', 1000),
        )
    elif algo_name == 'tqc':
        if not CONTRIB_AVAILABLE or TQC is None:
            raise ImportError("TQC requires sb3-contrib. Install with: pip install sb3-contrib")
        return TQC(
            **common_kwargs,
            learning_rate=params.get('learning_rate', 3e-4),
            buffer_size=params.get('buffer_size', 1_000_000),
            batch_size=params.get('batch_size', 256),
            tau=params.get('tau', 0.005),
            gamma=params.get('gamma', 0.995),
            ent_coef=params.get('ent_coef', 'auto'),
            target_entropy=params.get('target_entropy', 'auto'),
            top_quantiles_to_drop_per_net=params.get('top_quantiles_to_drop_per_net', 2),
            learning_starts=params.get('learning_starts', 1000),
        )
    elif algo_name == 'dqn':
        return DQN(
            **common_kwargs,
            learning_rate=params.get('learning_rate', 3e-4),
            buffer_size=params.get('buffer_size', 1_000_000),
            batch_size=params.get('batch_size', 256),
            tau=params.get('tau', 0.005),
            gamma=params.get('gamma', 0.995),
            exploration_fraction=params.get('exploration_fraction', 0.1),
            exploration_final_eps=params.get('exploration_final_eps', 0.05),
            exploration_initial_eps=params.get('exploration_initial_eps', 1.0),
            learning_starts=params.get('learning_starts', 1000),
        )
    elif algo_name == 'qrdqn':
        if not CONTRIB_AVAILABLE or QRDQN is None:
            raise ImportError("QR-DQN requires sb3-contrib. Install with: pip install sb3-contrib")
        return QRDQN(
            **common_kwargs,
            learning_rate=params.get('learning_rate', 3e-4),
            buffer_size=params.get('buffer_size', 1_000_000),
            batch_size=params.get('batch_size', 256),
            tau=params.get('tau', 0.005),
            gamma=params.get('gamma', 0.995),
            exploration_fraction=params.get('exploration_fraction', 0.1),
            exploration_final_eps=params.get('exploration_final_eps', 0.05),
            exploration_initial_eps=params.get('exploration_initial_eps', 1.0),
            learning_starts=params.get('learning_starts', 1000),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def main():
    args = parse_args()

    # Load scenario
    print(f"Loading scenario: {args.scenario}")
    scenario = load_and_expand_scenario(args.scenario)

    # Apply command-line overrides
    if args.override:
        print(f"\nApplying {len(args.override)} override(s):")
        scenario = apply_overrides(scenario, args.override)

    # Check if running in a WandB sweep
    in_sweep = 'WANDB_SWEEP_ID' in os.environ or 'WANDB_SWEEP_PARAM_PATH' in os.environ
    sweep_params_applied = {}

    # Resolve the SB3-controlled agent ID early for sweep overrides
    sb3_agent_id, _ = resolve_sb3_agent_config(scenario, args.algo)

    # Initialize wandb early for sweeps so we can apply params before env creation
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Skipping.")
        else:
            scenario_tags = scenario.get('wandb', {}).get('tags', [])
            if not scenario_tags:
                scenario_tags = ['sb3', args.algo, 'baseline']

            if in_sweep:
                print("Running in WandB sweep - using sweep configuration")
                wandb_run = wandb.init(
                    config={
                        'algorithm': args.algo,
                        'scenario': args.scenario,
                        'episodes': scenario.get('experiment', {}).get('episodes', 2500),
                        'seed': scenario.get('experiment', {}).get('seed', 42),
                    },
                    sync_tensorboard=True,
                )
                if wandb_run is not None:
                    try:
                        sweep_name = scenario.get('wandb', {}).get(
                            'name', scenario.get('experiment', {}).get('name')
                        )
                        algo_label = str(args.algo).lower()
                        wandb_run.name = f"{sweep_name}_{algo_label}_{wandb_run.id}"
                        wandb_run.save()
                    except Exception:
                        pass

                if wandb.config:
                    wandb_params = dict(wandb.config)
                    for key, value in wandb_params.items():
                        if key.startswith('_') or key in [
                            'method', 'metric', 'program', 'algorithm', 'scenario'
                        ]:
                            continue
                        if key == 'episodes':
                            scenario.setdefault('experiment', {})['episodes'] = value
                            sweep_params_applied[key] = value
                            continue
                        if key == 'seed':
                            scenario.setdefault('experiment', {})['seed'] = value
                            sweep_params_applied[key] = value
                            continue

                        override_path = key if '.' in key else f"agents.{sb3_agent_id}.params.{key}"
                        set_nested_value(scenario, override_path, value)
                        sweep_params_applied[key] = value

                    if sweep_params_applied:
                        print(f"\nApplying WandB sweep parameters ({len(sweep_params_applied)} parameter(s)):")
                        for key, value in sweep_params_applied.items():
                            print(f"  {key} = {value}")
            else:
                wandb_run = wandb.init(
                    project=scenario.get('wandb', {}).get('project', 'marl-f110'),
                    entity=scenario.get('wandb', {}).get('entity'),
                    name=scenario.get('wandb', {}).get('name', scenario.get('experiment', {}).get('name')),
                    tags=scenario_tags,
                    group=scenario.get('wandb', {}).get('group', scenario.get('experiment', {}).get('name')),
                    job_type=scenario.get('wandb', {}).get('job_type', args.algo),
                    config={
                        'algorithm': args.algo,
                        'scenario': args.scenario,
                        'episodes': args.episodes or scenario.get('experiment', {}).get('episodes', 2500),
                        'seed': args.seed if args.seed is not None else scenario.get('experiment', {}).get('seed', 42),
                    },
                    sync_tensorboard=True,
                    notes=scenario.get('wandb', {}).get('notes'),
                )

    # Use episodes from scenario unless overridden by command line
    if args.episodes is None:
        args.episodes = scenario.get('experiment', {}).get('episodes', 2500)
    else:
        scenario.setdefault('experiment', {})['episodes'] = args.episodes

    # Use seed from scenario unless overridden by command line
    if args.seed is None:
        args.seed = scenario.get('experiment', {}).get('seed', 42)
    else:
        scenario.setdefault('experiment', {})['seed'] = args.seed

    # Re-resolve SB3 agent config after sweep overrides
    sb3_agent_id, sb3_agent_cfg = resolve_sb3_agent_config(scenario, args.algo)

    # Create environment and agents (including FTG defender)
    print("Creating environment...")
    env, agents, reward_strategies = create_training_setup(scenario)
    observation_preset = resolve_observation_preset(sb3_agent_cfg)
    target_id = sb3_agent_cfg.get('target_id')

    obs_space = env.observation_spaces.get(sb3_agent_id)
    if obs_space is None:
        raise ValueError(f"Agent '{sb3_agent_id}' not found in observation spaces")

    if observation_preset:
        dummy_obs = obs_space.sample()
        if target_id:
            dummy_obs = dict(dummy_obs)
            dummy_obs['central_state'] = obs_space.sample()
        flat_dummy = flatten_observation(dummy_obs, preset=observation_preset, target_id=target_id)
        obs_dim = int(flat_dummy.shape[0])
    else:
        obs_dim = get_space_dim(obs_space)

    action_space = env.action_spaces.get(sb3_agent_id)
    if action_space is None:
        raise ValueError(f"Agent '{sb3_agent_id}' not found in action spaces")
    action_low = None
    action_high = None
    if isinstance(action_space, spaces.Box):
        action_low = action_space.low
        action_high = action_space.high

    # Get reward strategy for the SB3 agent if available
    reward_strategy = reward_strategies.get(sb3_agent_id)

    # Extract action_set if present (for DQN/QR-DQN)
    action_set = None
    if args.algo in ['dqn', 'qrdqn']:
        # Check in agent config or params for action_set
        agent_cfg = scenario.get('agents', {}).get(sb3_agent_id, {})
        params = agent_cfg.get('params', {})
        action_set = agent_cfg.get('action_set') or params.get('action_set')
        if action_set is None:
            raise ValueError(f"{args.algo.upper()} requires 'action_set' parameter for action discretization")

    # Setup curriculum if enabled in scenario
    spawn_curriculum = None
    spawn_config = scenario.get('environment', {}).get('spawn_curriculum', {})
    if spawn_config.get('enabled', False):
        from src.core.spawn_curriculum import SpawnCurriculumManager

        spawn_configs = spawn_config.get('spawn_configs', {})
        if spawn_configs:
            print("Creating spawn curriculum...")
            try:
                spawn_curriculum = SpawnCurriculumManager(
                    config=spawn_config,
                    available_spawn_points=spawn_configs
                )
                print(
                    f"  Spawn curriculum: {len(spawn_curriculum.stages)} stages, "
                    f"starting at '{spawn_curriculum.current_stage.name}'"
                )
            except Exception as e:
                print(f"Warning: Failed to create spawn curriculum: {e}")
                spawn_curriculum = None
        else:
            print("Warning: Spawn curriculum enabled but no spawn_configs provided")

    # Wrap environment for SB3
    print("Wrapping environment for SB3...")
    wrapped_env = SB3SingleAgentWrapper(
        env,
        agent_id=sb3_agent_id,
        obs_dim=obs_dim,
        action_low=action_low if action_low is not None else np.array([-0.46, -1.0]),
        action_high=action_high if action_high is not None else np.array([0.46, 1.0]),
        observation_preset=observation_preset,
        target_id=target_id,
        reward_strategy=reward_strategy,
        action_set=np.array(action_set) if action_set is not None else None,
        spawn_curriculum=spawn_curriculum,
    )

    if reward_strategy:
        print(f"Using custom reward strategy for {sb3_agent_id}")
    else:
        print(f"Using environment default rewards for {sb3_agent_id}")

    # Set other agents (FTG defender) so they can act
    wrapped_env.set_other_agents(agents)

    # Wrap with Monitor for logging
    wrapped_env = Monitor(wrapped_env)

    # Define wandb metrics once run is initialized
    if wandb_run is not None:
        try:
            wandb.define_metric("train/episode")
            wandb.define_metric("train/*", step_metric="train/episode")
            wandb.define_metric("target/*", step_metric="train/episode")
            wandb.define_metric("curriculum/*", step_metric="train/episode")
            wandb.define_metric("eval/episode")
            wandb.define_metric("eval/episode_*", step_metric="eval/episode")
            wandb.define_metric("eval/training_episode", step_metric="eval/episode")
            wandb.define_metric("eval/spawn_point", step_metric="eval/episode")
            wandb.define_metric("eval_agg/*", step_metric="eval/episode")
        except Exception:
            pass

    # Get FTG schedules from scenario
    ftg_schedules = {}
    ftg_agents_dict = {}
    for agent_id, agent_config in scenario.get('agents', {}).items():
        if agent_config.get('algorithm', '').lower() == 'ftg':
            schedule = agent_config.get('ftg_schedule')
            if isinstance(schedule, dict) and schedule.get('enabled', False):
                ftg_schedules[agent_id] = schedule
                if agent_id in agents:
                    ftg_agents_dict[agent_id] = agents[agent_id]

    if ftg_schedules:
        print(f"Loaded FTG schedules for {len(ftg_schedules)} agents")

    # Get curriculum config from scenario
    curriculum_config = scenario.get('curriculum')
    if curriculum_config and curriculum_config.get('phases'):
        print(f"Loaded phased curriculum with {len(curriculum_config['phases'])} phases")

    # Create output directory
    output_dir = Path(args.output_dir) / args.algo / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tensorboard log directory (required for wandb sync)
    tensorboard_dir = output_dir / 'tensorboard'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Rich console dashboard
    rich_console = RichConsole(enabled=True)

    # Create callbacks
    callbacks = []
    eval_wrapped_env = None

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / 'checkpoints'),
        name_prefix=f'{args.algo}_model',
        save_replay_buffer=True,
    )
    callbacks.append(checkpoint_callback)

    # Wandb callback
    if wandb_run is not None:
        wandb_callback = WandbCallback(
            model_save_freq=10000,
            model_save_path=str(output_dir / 'models'),
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # Curriculum/metrics callback
    enable_metrics_callback = bool(
        spawn_curriculum
        or curriculum_config
        or wandb_run
        or getattr(rich_console, "enabled", False)
    )
    if enable_metrics_callback:
        curriculum_callback = CurriculumCallback(
            curriculum_config=curriculum_config,
            spawn_curriculum=spawn_curriculum,
            ftg_agents=ftg_agents_dict,
            ftg_schedules=ftg_schedules,
            env_wrapper=wrapped_env,
            wandb_run=wandb_run,
            rich_console=rich_console,
            algo_name=args.algo,
            verbose=1,
        )
        callbacks.append(curriculum_callback)
        if spawn_curriculum or curriculum_config:
            print("Curriculum callback enabled")
        elif getattr(rich_console, "enabled", False):
            print("Rich console enabled")

    # Evaluation callback (SB3)
    eval_cfg = scenario.get('evaluation', {})
    if eval_cfg.get('enabled', False):
        env_config = scenario.get('environment', {})
        spawn_configs = env_config.get('spawn_configs', {})
        if not spawn_configs and 'spawn_curriculum' in env_config:
            spawn_configs = env_config['spawn_curriculum'].get('spawn_configs', {})

        if not spawn_configs:
            print("Warning: evaluation enabled but no spawn_configs provided; skipping eval")
        else:
            eval_config = EvaluationConfig(
                num_episodes=eval_cfg.get('num_episodes', 10),
                deterministic=eval_cfg.get('deterministic', True),
                spawn_points=eval_cfg.get('spawn_points', ['spawn_pinch_left', 'spawn_pinch_right']),
                spawn_speeds=eval_cfg.get('spawn_speeds', [0.44, 0.44]),
                lock_speed_steps=eval_cfg.get('lock_speed_steps', 0),
                ftg_override=eval_cfg.get('ftg_override', {}),
                max_steps=env_config.get('max_steps', 2500),
            )

            # Create evaluation environment/agents
            eval_env, eval_agents, eval_reward_strategies = create_training_setup(scenario)
            eval_reward_strategy = eval_reward_strategies.get(sb3_agent_id)
            eval_wrapped_env = SB3SingleAgentWrapper(
                eval_env,
                agent_id=sb3_agent_id,
                obs_dim=obs_dim,
                action_low=action_low if action_low is not None else np.array([-0.46, -1.0]),
                action_high=action_high if action_high is not None else np.array([0.46, 1.0]),
                observation_preset=observation_preset,
                target_id=target_id,
                reward_strategy=eval_reward_strategy,
                action_set=np.array(action_set) if action_set is not None else None,
            )
            eval_wrapped_env.set_other_agents(eval_agents)
            eval_wrapped_env = Monitor(eval_wrapped_env)

            # Apply FTG overrides for evaluation
            if eval_config.ftg_override:
                for agent in eval_agents.values():
                    apply_config = getattr(agent, "apply_config", None)
                    if callable(apply_config):
                        apply_config(eval_config.ftg_override)

            eval_every_n_episodes = eval_cfg.get('frequency', 100)
            callbacks.append(SB3EvaluationCallback(
                eval_env=eval_wrapped_env,
                evaluation_config=eval_config,
                spawn_configs=spawn_configs,
                eval_every_n_episodes=eval_every_n_episodes,
                wandb_run=wandb_run,
                verbose=1,
            ))
            print(
                f"Evaluation: {eval_config.num_episodes} episodes "
                f"every {eval_every_n_episodes} training episodes"
            )

    # Extract hyperparameters from scenario
    agent_params = sb3_agent_cfg.get('params', {})

    # Create SB3 agent
    print(f"Creating {args.algo.upper()} agent...")
    model = create_sb3_agent(
        args.algo,
        wrapped_env,
        params=agent_params,
        seed=args.seed,
        tensorboard_log=str(tensorboard_dir)
    )

    # Calculate total timesteps
    # Approximate: episodes * avg_steps_per_episode
    max_steps = scenario['environment'].get('max_steps', 2500)
    total_timesteps = args.episodes * max_steps

    print(f"\nStarting training:")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Output directory: {output_dir}")
    print(f"  TensorBoard logs: {tensorboard_dir}")
    if wandb_run is not None:
        print(f"  WandB run: {wandb_run.name}")
    print()

    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final model
        final_model_path = output_dir / f'{args.algo}_final.zip'
        model.save(final_model_path)
        print(f"\nSaved final model to: {final_model_path}")

        # Close wandb
        if wandb_run is not None:
            wandb.finish()

        # Close environment
        wrapped_env.close()
        if eval_wrapped_env is not None:
            eval_wrapped_env.close()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
