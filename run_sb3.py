#!/usr/bin/env python3
"""On-policy SB3 training runner (PPO/A2C) using the v2 scenario format."""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch.nn as nn
from gymnasium import spaces

# Allow running from repo root without installing the package.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.sb3_curriculum_callback import CurriculumCallback
from baselines.sb3_eval_callback import SB3EvaluationCallback
from baselines.sb3_wrapper import SB3SingleAgentWrapper
from core.evaluator import EvaluationConfig
from core.obs_flatten import flatten_observation
from core.run_id import resolve_run_id, set_run_id_env
from core.scenario import ScenarioError, load_and_expand_scenario
from core.setup import create_training_setup
from loggers import ConsoleLogger, WandbLogger

try:
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    print(
        "stable-baselines3 is required for on-policy runs. "
        "Install with: pip install stable-baselines3",
        file=sys.stderr,
    )
    raise


ON_POLICY_ALGOS = {"sb3_ppo", "sb3_a2c", "ppo", "a2c"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="F110 on-policy SB3 training (PPO/A2C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Path to scenario YAML file",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (overrides scenario config)",
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (overrides scenario config)",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (default: False)",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides scenario config)",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes (overrides scenario config)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console output",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID for logging alignment",
    )

    return parser.parse_args()


def resolve_cli_overrides(scenario: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI argument overrides to scenario."""
    if args.seed is not None:
        scenario.setdefault("experiment", {})["seed"] = args.seed

    if args.episodes is not None:
        scenario.setdefault("experiment", {})["episodes"] = args.episodes

    if args.wandb:
        scenario.setdefault("wandb", {})["enabled"] = True
    elif args.no_wandb:
        scenario.setdefault("wandb", {})["enabled"] = False

    if args.render:
        scenario.setdefault("environment", {})["render"] = True
    elif args.no_render:
        scenario.setdefault("environment", {})["render"] = False

    return scenario


def initialize_loggers(
    scenario: Dict[str, Any], args: argparse.Namespace, run_id: Optional[str] = None
) -> Tuple[Optional[WandbLogger], ConsoleLogger]:
    """Initialize W&B and console loggers."""
    console_logger = ConsoleLogger(verbose=not args.quiet)

    wandb_config = scenario.get("wandb", {})
    wandb_enabled = wandb_config.get("enabled", False)

    default_algo = "unknown"
    for agent_cfg in scenario.get("agents", {}).values():
        algo = agent_cfg.get("algorithm", "").lower()
        if algo and algo not in ["ftg", "pp", "pure_pursuit"]:
            default_algo = algo
            break
    default_group = scenario.get("experiment", {}).get("name")

    sweep_mode = bool(os.environ.get("WANDB_SWEEP_ID"))

    if wandb_enabled:
        console_logger.print_info("Initializing Weights & Biases...")
        wandb_logger = WandbLogger(
            project=wandb_config.get("project", "f110-marl"),
            name=wandb_config.get("name", scenario["experiment"]["name"]),
            config=None if sweep_mode else scenario,
            tags=wandb_config.get("tags", []),
            group=wandb_config.get("group", default_group),
            job_type=wandb_config.get("job_type", default_algo),
            entity=wandb_config.get("entity", None),
            notes=wandb_config.get("notes", None),
            mode=wandb_config.get("mode", "online"),
            run_id=run_id,
            logging_config=wandb_config.get("logging"),
        )
    else:
        wandb_logger = None

    return wandb_logger, console_logger


def apply_wandb_sweep_overrides(scenario: Dict[str, Any], console_logger: ConsoleLogger) -> None:
    """Apply WandB sweep parameters to the scenario if in sweep mode."""
    try:
        import wandb

        sweep_mode = bool(os.environ.get("WANDB_SWEEP_ID") or getattr(wandb.run, "sweep_id", None))
        if not sweep_mode or not wandb.config or len(dict(wandb.config)) == 0:
            return

        console_logger.print_info("Detected WandB sweep mode - applying sweep parameters...")

        def set_nested_value(d: dict, path: str, value: Any) -> None:
            keys = path.split(".")
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value

        sb3_agent_id = None
        for agent_id, agent_cfg in scenario.get("agents", {}).items():
            algo = agent_cfg.get("algorithm", "").lower()
            if algo in ON_POLICY_ALGOS:
                sb3_agent_id = agent_id
                break

        if not sb3_agent_id:
            console_logger.print_warning("Sweep mode active but no on-policy agent found to apply params.")
            return

        sweep_params_applied = {}
        wandb_params = {k: wandb.config[k] for k in wandb.config}
        agent_params = scenario.get("agents", {}).get(sb3_agent_id, {}).get("params", {})
        allowed_param_keys = set(agent_params.keys())
        skipped_keys = []

        for key, value in wandb_params.items():
            if (
                key.startswith("_")
                or "/" in key
                or key in ["method", "metric", "program", "algorithm", "scenario"]
            ):
                continue

            if key == "episodes":
                scenario.setdefault("experiment", {})["episodes"] = value
                sweep_params_applied[key] = value
                continue
            if key == "seed":
                scenario.setdefault("experiment", {})["seed"] = value
                sweep_params_applied[key] = value
                continue

            if key not in allowed_param_keys and "." not in key:
                skipped_keys.append(key)
                continue

            override_path = key if "." in key else f"agents.{sb3_agent_id}.params.{key}"
            set_nested_value(scenario, override_path, value)
            sweep_params_applied[key] = value

        if sweep_params_applied:
            console_logger.print_success(
                f"Applied {len(sweep_params_applied)} sweep parameter(s) to {sb3_agent_id}"
            )
            for key, value in sweep_params_applied.items():
                console_logger.print_info(f"  {key} = {value}")
            if skipped_keys:
                skipped_str = ", ".join(sorted(skipped_keys))
                console_logger.print_warning(
                    f"Skipped {len(skipped_keys)} sweep key(s) not in agent params: {skipped_str}"
                )
    except Exception as exc:
        console_logger.print_warning(f"Failed to apply sweep parameters: {exc}")


def select_on_policy_agent(scenario: Dict[str, Any]) -> Tuple[str, str]:
    """Select the on-policy agent from the scenario."""
    for agent_id, agent_cfg in scenario.get("agents", {}).items():
        algo = agent_cfg.get("algorithm", "").lower()
        if algo in ON_POLICY_ALGOS:
            return agent_id, algo
    raise ValueError("No on-policy agent found (expected sb3_ppo or sb3_a2c).")


def infer_observation_preset(agent_config: Dict[str, Any]) -> Optional[str]:
    """Infer observation preset name from agent config."""
    obs_config = agent_config.get("observation")
    if isinstance(obs_config, dict):
        if "preset" in obs_config:
            return obs_config["preset"]
        if len(obs_config) > 0:
            return "gaplock"
    return None


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


def parse_action_repeat(env_config: Dict[str, Any]) -> int:
    """Parse action repeat (step skip) from environment config."""
    value = None
    for key in ("action_repeat", "step_repeat", "step_skip", "frame_skip"):
        if key in env_config:
            value = env_config.get(key)
            break
    if value is None:
        return 1
    try:
        repeat = int(value)
    except (TypeError, ValueError):
        repeat = 1
    return max(1, repeat)


def compute_obs_dim(
    obs_space,
    preset: Optional[str],
    target_id: Optional[str],
    frame_stack: int,
) -> int:
    """Compute flattened observation dimension."""
    if preset:
        dummy_obs = obs_space.sample()
        if target_id:
            dummy_obs["central_state"] = obs_space.sample()
        flat_dummy = flatten_observation(dummy_obs, preset=preset, target_id=target_id)
        obs_dim = int(flat_dummy.shape[0])
    else:
        obs_dim = get_space_dim(obs_space)
    if frame_stack > 1:
        obs_dim *= frame_stack
    return obs_dim


def build_policy_kwargs(params: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Build policy_kwargs with net_arch and activation_fn."""
    hidden_dims = params.get("hidden_dims", [256, 256])
    pi_dims = params.get("pi_hidden_dims")
    qf_dims = params.get("qf_hidden_dims")
    supports_split = model_name in {"SAC", "TD3", "TQC", "DDPG"}

    if supports_split and (pi_dims is not None or qf_dims is not None):
        if pi_dims is None:
            pi_dims = hidden_dims
        if qf_dims is None:
            qf_dims = hidden_dims
        net_arch = {"pi": pi_dims, "qf": qf_dims}
    else:
        net_arch = hidden_dims

    policy_kwargs: Dict[str, Any] = {"net_arch": net_arch}

    activation = params.get("activation")
    if activation is not None:
        if isinstance(activation, str):
            activation_key = activation.strip().lower()
            activation_map = {
                "relu": nn.ReLU,
                "silu": nn.SiLU,
                "swish": nn.SiLU,
                "tanh": nn.Tanh,
            }
            if activation_key not in activation_map:
                raise ValueError(
                    f"Unsupported activation '{activation}'. Use: relu, silu/swish, tanh."
                )
            policy_kwargs["activation_fn"] = activation_map[activation_key]
        else:
            policy_kwargs["activation_fn"] = activation

    return policy_kwargs


def build_model(
    algorithm: str,
    params: Dict[str, Any],
    env,
    policy_kwargs: Dict[str, Any],
    device: str,
    seed: Optional[int],
):
    """Create SB3 on-policy model."""
    learning_rate = params.get("learning_rate", 3e-4)
    gamma = params.get("gamma", 0.995)

    if algorithm in {"sb3_ppo", "ppo"}:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=params.get("n_steps", 2048),
            batch_size=params.get("batch_size", 256),
            n_epochs=params.get("n_epochs", 10),
            gamma=gamma,
            gae_lambda=params.get("gae_lambda", 0.95),
            clip_range=params.get("clip_range", 0.2),
            ent_coef=params.get("ent_coef", 0.02),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )
    elif algorithm in {"sb3_a2c", "a2c"}:
        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=params.get("n_steps", 5),
            gamma=gamma,
            gae_lambda=params.get("gae_lambda", 1.0),
            ent_coef=params.get("ent_coef", 0.0),
            vf_coef=params.get("vf_coef", 0.5),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported on-policy algorithm: {algorithm}")

    return model


def build_spawn_curriculum(env, scenario: Dict[str, Any], console_logger: ConsoleLogger):
    """Create spawn curriculum (or sampler) if configured."""
    env_config = scenario.get("environment", {})
    spawn_curriculum = None
    phased_curriculum_enabled = scenario.get("curriculum", {}).get("type") == "phased"

    spawn_configs = env_config.get("spawn_configs", {})
    spawn_config = env_config.get("spawn_curriculum", {})
    if not spawn_configs:
        spawn_configs = spawn_config.get("spawn_configs", {})

    if spawn_config.get("enabled", False):
        from core.spawn_curriculum import SpawnCurriculumManager

        if spawn_configs:
            console_logger.print_info("Creating spawn curriculum...")
            try:
                spawn_curriculum = SpawnCurriculumManager(
                    config=spawn_config, available_spawn_points=spawn_configs
                )
                env.spawn_configs = spawn_configs
                console_logger.print_success(
                    f"Spawn curriculum: {len(spawn_curriculum.stages)} stages, "
                    f"starting at '{spawn_curriculum.current_stage.name}'"
                )
                if phased_curriculum_enabled:
                    console_logger.print_info(
                        "Phased curriculum active: spawn curriculum progression disabled"
                    )
            except Exception as exc:
                console_logger.print_warning(f"Failed to create spawn curriculum: {exc}")
                spawn_curriculum = None
        else:
            console_logger.print_warning("Spawn curriculum enabled but no spawn_configs provided")
    elif phased_curriculum_enabled and spawn_configs:
        from core.spawn_curriculum import SpawnCurriculumManager

        console_logger.print_info("Creating spawn sampler for phased curriculum...")
        try:
            spawn_curriculum = SpawnCurriculumManager(
                config={
                    "window": 1,
                    "activation_samples": 1,
                    "min_episode": 0,
                    "enable_patience": 1,
                    "disable_patience": 1,
                    "cooldown": 0,
                    "lock_speed_steps": 0,
                    "stages": [
                        {
                            "name": "phase_sampler",
                            "spawn_points": "all",
                            "speed_range": [0.0, 0.0],
                            "enable_rate": 1.0,
                        }
                    ],
                },
                available_spawn_points=spawn_configs,
            )
            env.spawn_configs = spawn_configs
            console_logger.print_success("Spawn sampler ready for phased curriculum")
        except Exception as exc:
            console_logger.print_warning(f"Failed to create spawn sampler: {exc}")
            spawn_curriculum = None

    return spawn_curriculum, spawn_configs


class StopOnEpisodeCallback(BaseCallback):
    """Stop training after a fixed number of episodes."""

    def __init__(self, max_episodes: int, console_logger: Optional[ConsoleLogger] = None):
        super().__init__()
        self.max_episodes = max_episodes
        self.console_logger = console_logger
        self.episode_count = 0

    def _count_episode_ends(self) -> int:
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return 0
            done_flags = np.logical_or(terminated, truncated)
        else:
            done_flags = dones
        if isinstance(done_flags, (list, tuple, np.ndarray)):
            return int(np.sum(done_flags))
        return int(bool(done_flags))

    def _on_step(self) -> bool:
        if self.max_episodes <= 0:
            return True
        done_count = self._count_episode_ends()
        if done_count > 0:
            self.episode_count += done_count
            if self.episode_count >= self.max_episodes:
                if self.console_logger:
                    self.console_logger.print_success(
                        f"Reached {self.episode_count} episodes; stopping training."
                    )
                return False
        return True


class EpisodeProgressCallback(BaseCallback):
    """Periodic episode progress logging."""

    def __init__(self, log_every: int, console_logger: Optional[ConsoleLogger] = None):
        super().__init__()
        self.log_every = max(1, int(log_every))
        self.console_logger = console_logger
        self.episode_count = 0

    def _count_episode_ends(self) -> int:
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return 0
            done_flags = np.logical_or(terminated, truncated)
        else:
            done_flags = dones
        if isinstance(done_flags, (list, tuple, np.ndarray)):
            return int(np.sum(done_flags))
        return int(bool(done_flags))

    def _on_step(self) -> bool:
        done_count = self._count_episode_ends()
        if done_count > 0:
            self.episode_count += done_count
            if self.console_logger and self.episode_count % self.log_every == 0:
                self.console_logger.print_info(f"Progress: episode {self.episode_count}")
        return True


def main() -> None:
    args = parse_args()

    try:
        scenario = load_and_expand_scenario(args.scenario)
    except ScenarioError as exc:
        print(f"Error loading scenario: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Scenario file not found: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    scenario = resolve_cli_overrides(scenario, args)

    train_agent_id, algorithm = select_on_policy_agent(scenario)

    run_id = args.run_id or resolve_run_id(
        scenario_name=scenario.get("experiment", {}).get("name"),
        algorithm=algorithm,
        seed=scenario.get("experiment", {}).get("seed"),
    )
    set_run_id_env(run_id)

    wandb_logger, console_logger = initialize_loggers(scenario, args, run_id=run_id)
    if wandb_logger is not None:
        apply_wandb_sweep_overrides(scenario, console_logger)

    console_logger.print_header(
        f"Training: {scenario['experiment']['name']}",
        f"Episodes: {scenario['experiment'].get('episodes', 'N/A')}",
    )

    env, agents, reward_strategies = create_training_setup(scenario)

    train_agent_cfg = scenario["agents"][train_agent_id]
    train_params = train_agent_cfg.get("params", {})

    frame_stack = int(train_agent_cfg.get("frame_stack", 1) or 1)
    if frame_stack < 1:
        frame_stack = 1

    env_config = scenario.get("environment", {})
    action_repeat = parse_action_repeat(env_config)

    target_id = train_agent_cfg.get("target_id")
    observation_preset = infer_observation_preset(train_agent_cfg)

    obs_space = env.observation_spaces.get(train_agent_id)
    action_space = env.action_spaces.get(train_agent_id)
    if obs_space is None or action_space is None:
        raise ValueError(f"Agent '{train_agent_id}' not found in environment spaces.")

    obs_dim = compute_obs_dim(obs_space, observation_preset, target_id, frame_stack)

    if not isinstance(action_space, spaces.Box):
        raise ValueError("On-policy SB3 runner expects continuous action spaces.")
    action_low = action_space.low
    action_high = action_space.high

    spawn_curriculum, spawn_configs = build_spawn_curriculum(env, scenario, console_logger)

    reward_strategy = reward_strategies.get(train_agent_id)
    sb3_env = SB3SingleAgentWrapper(
        env,
        agent_id=train_agent_id,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        observation_preset=observation_preset,
        target_id=target_id,
        reward_strategy=reward_strategy,
        spawn_curriculum=spawn_curriculum,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
    )

    other_agents = {aid: agent for aid, agent in agents.items() if aid != train_agent_id}
    sb3_env.set_other_agents(other_agents)

    monitor_env = Monitor(
        sb3_env,
        info_keywords=("is_success", "outcome", "target_finished", "target_collision", "collision"),
    )

    device = train_agent_cfg.get("device", train_params.get("device", "cuda"))
    policy_kwargs = build_policy_kwargs(train_params, "PPO" if "ppo" in algorithm else "A2C")
    model = build_model(algorithm, train_params, monitor_env, policy_kwargs, device, scenario["experiment"].get("seed"))

    episodes = int(scenario["experiment"].get("episodes", 0) or 0)
    if episodes <= 0:
        raise ValueError("Scenario must define a positive experiment.episodes for on-policy runs.")

    max_steps = int(env_config.get("max_steps", 2500))
    decision_steps = max_steps
    if action_repeat > 1:
        decision_steps = int(math.ceil(max_steps / action_repeat))
    total_timesteps = decision_steps * episodes

    callbacks = [StopOnEpisodeCallback(episodes, console_logger)]
    log_every_env = os.environ.get("F110_LOG_EVERY_EPISODES")
    if log_every_env is None:
        log_every = 25
    else:
        try:
            log_every = int(log_every_env)
        except ValueError:
            log_every = 25
    if log_every > 0:
        callbacks.append(EpisodeProgressCallback(log_every, console_logger))

    ftg_agents = {}
    ftg_schedules = {}
    for agent_id, agent_cfg in scenario.get("agents", {}).items():
        if agent_id == train_agent_id:
            continue
        if agent_cfg.get("algorithm", "").lower() == "ftg":
            if agent_id in agents:
                ftg_agents[agent_id] = agents[agent_id]
            schedule = agent_cfg.get("ftg_schedule")
            if isinstance(schedule, dict):
                ftg_schedules[agent_id] = schedule

    curriculum_config = scenario.get("curriculum")
    if curriculum_config or spawn_curriculum:
        callbacks.append(
            CurriculumCallback(
                curriculum_config=curriculum_config,
                spawn_curriculum=spawn_curriculum,
                ftg_agents=ftg_agents,
                ftg_schedules=ftg_schedules,
                env_wrapper=sb3_env,
                wandb_run=wandb_logger.run if wandb_logger else None,
                wandb_logging=scenario.get("wandb", {}).get("logging"),
                algo_name=algorithm,
            )
        )

    eval_cfg = scenario.get("evaluation", {})
    if eval_cfg.get("enabled", False):
        eval_config = EvaluationConfig(
            num_episodes=eval_cfg.get("num_episodes", 10),
            deterministic=eval_cfg.get("deterministic", True),
            spawn_points=eval_cfg.get("spawn_points", ["spawn_pinch_left", "spawn_pinch_right"]),
            spawn_speeds=eval_cfg.get("spawn_speeds", [0.44, 0.44]),
            lock_speed_steps=eval_cfg.get("lock_speed_steps", 0),
            ftg_override=eval_cfg.get("ftg_override", {}),
            max_steps=max_steps,
        )
        eval_env_raw, eval_agents, _ = create_training_setup(scenario)
        eval_other_agents = {aid: agent for aid, agent in eval_agents.items() if aid != train_agent_id}

        eval_env = SB3SingleAgentWrapper(
            eval_env_raw,
            agent_id=train_agent_id,
            obs_dim=obs_dim,
            action_low=action_low,
            action_high=action_high,
            observation_preset=observation_preset,
            target_id=target_id,
            reward_strategy=reward_strategy,
            frame_stack=frame_stack,
            action_repeat=action_repeat,
        )
        eval_env.set_other_agents(eval_other_agents)

        eval_every = eval_cfg.get("frequency", 100)
        if eval_every:
            callbacks.append(
                SB3EvaluationCallback(
                    eval_env=eval_env,
                    evaluation_config=eval_config,
                    spawn_configs=spawn_configs,
                    eval_every_n_episodes=eval_every,
                    wandb_run=wandb_logger.run if wandb_logger else None,
                    wandb_logging=scenario.get("wandb", {}).get("logging"),
                    verbose=0,
                )
            )

    callback = CallbackList(callbacks) if callbacks else None

    console_logger.print_info("Starting SB3 on-policy training...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    finally:
        if wandb_logger:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
