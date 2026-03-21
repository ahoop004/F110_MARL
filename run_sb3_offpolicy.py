#!/usr/bin/env python3
"""Off-policy SB3 training runner (SAC, TD3, DDPG, DQN, QR-DQN, TQC).

Mirrors run_sb3.py structure for on-policy algorithms. Use this script for
all off-policy algorithms; use run_sb3.py for PPO/A2C.

Usage:
    python run_sb3_offpolicy.py --scenario scenarios/comparison/sac.yaml
    python run_sb3_offpolicy.py --scenario scenarios/comparison/td3.yaml
    python run_sb3_offpolicy.py --scenario scenarios/comparison/dqn.yaml
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch.nn as nn
from gymnasium import spaces

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.sb3_curriculum_callback import CurriculumCallback
from baselines.sb3_eval_callback import SB3EvaluationCallback
from baselines.sb3_train_callback import SB3TrainLoggingCallback
from baselines.sb3_wrapper import SB3SingleAgentWrapper
from core.evaluator import EvaluationConfig
from core.obs_flatten import flatten_observation
from core.run_id import resolve_run_id, set_run_id_env
from core.scenario import ScenarioError, load_and_expand_scenario
from core.setup import create_training_setup
from loggers import ConsoleLogger, WandbLogger

try:
    from stable_baselines3 import SAC, TD3, DDPG, DQN
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
except ImportError as exc:
    print(
        "stable-baselines3 is required. Install with: pip install stable-baselines3",
        file=sys.stderr,
    )
    raise

# Optional sb3_contrib algorithms (TQC, QR-DQN)
try:
    from sb3_contrib import TQC, QRDQN
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    _SB3_CONTRIB_AVAILABLE = False

OFF_POLICY_CONTINUOUS = {"sb3_sac", "sac", "sb3_td3", "td3", "sb3_ddpg", "ddpg", "sb3_tqc", "tqc"}
OFF_POLICY_DISCRETE = {"sb3_dqn", "dqn", "sb3_qrdqn", "qrdqn"}
OFF_POLICY_ALGOS = OFF_POLICY_CONTINUOUS | OFF_POLICY_DISCRETE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F110 off-policy SB3 training (SAC/TD3/DDPG/DQN/QR-DQN/TQC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", type=str, required=True, help="Path to scenario YAML file")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--episodes", type=int, default=None, help="Episode count override")
    parser.add_argument("--quiet", action="store_true", help="Disable console output")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID")
    return parser.parse_args()


def resolve_cli_overrides(scenario: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
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

    if wandb_enabled:
        console_logger.print_info("Initializing Weights & Biases...")
        wandb_logger = WandbLogger(
            project=wandb_config.get("project", "f110-marl"),
            name=wandb_config.get("name", scenario["experiment"]["name"]),
            config=scenario,
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


def select_off_policy_agent(scenario: Dict[str, Any]) -> Tuple[str, str]:
    for agent_id, agent_cfg in scenario.get("agents", {}).items():
        algo = agent_cfg.get("algorithm", "").lower()
        if algo in OFF_POLICY_ALGOS:
            return agent_id, algo
    raise ValueError(
        f"No off-policy agent found. Expected one of: {sorted(OFF_POLICY_ALGOS)}"
    )


def infer_observation_preset(agent_config: Dict[str, Any]) -> Optional[str]:
    obs_config = agent_config.get("observation")
    if isinstance(obs_config, dict):
        if "preset" in obs_config:
            return obs_config["preset"]
        if "_preset" in obs_config:
            return obs_config["_preset"]
        if obs_config.get("speed", {}).get("enabled") or obs_config.get("prev_action", {}).get("enabled"):
            return "centerline"
        if obs_config.get("target_state", {}).get("enabled"):
            return "gaplock"
        if len(obs_config) > 0:
            return "gaplock"
    return None


def get_space_dim(space) -> int:
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


def compute_obs_dim(obs_space, preset: Optional[str], target_id: Optional[str], frame_stack: int) -> int:
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
                raise ValueError(f"Unsupported activation '{activation}'. Use: relu, silu/swish, tanh.")
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
    n_action_dims: int = 2,
):
    """Build the appropriate SB3 off-policy model."""
    learning_rate = params.get("learning_rate", 3e-4)
    gamma = params.get("gamma", 0.99)
    tau = params.get("tau", 0.005)
    buffer_size = params.get("buffer_size", 1_000_000)
    batch_size = params.get("batch_size", 256)
    learning_starts = params.get("learning_starts", 1000)
    train_freq = params.get("train_freq", 1)
    gradient_steps = params.get("gradient_steps", 1)

    if algorithm in {"sb3_sac", "sac"}:
        ent_coef = params.get("ent_coef", "auto")
        target_entropy = params.get("target_entropy", "auto")
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )

    elif algorithm in {"sb3_td3", "td3"}:
        action_noise_sigma = params.get("action_noise_sigma", 0.1)
        action_noise = (
            NormalActionNoise(
                mean=np.zeros(n_action_dims),
                sigma=action_noise_sigma * np.ones(n_action_dims),
            )
            if action_noise_sigma > 0
            else None
        )
        model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            policy_delay=params.get("policy_delay", 2),
            target_policy_noise=params.get("target_policy_noise", 0.2),
            target_noise_clip=params.get("target_noise_clip", 0.5),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )

    elif algorithm in {"sb3_ddpg", "ddpg"}:
        action_noise_sigma = params.get("action_noise_sigma", 0.1)
        action_noise = (
            NormalActionNoise(
                mean=np.zeros(n_action_dims),
                sigma=action_noise_sigma * np.ones(n_action_dims),
            )
            if action_noise_sigma > 0
            else None
        )
        model = DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )

    elif algorithm in {"sb3_dqn", "dqn"}:
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=params.get("train_freq", 4),
            gradient_steps=gradient_steps,
            exploration_fraction=params.get("exploration_fraction", 0.1),
            exploration_initial_eps=params.get("exploration_initial_eps", 1.0),
            exploration_final_eps=params.get("exploration_final_eps", 0.05),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )

    elif algorithm in {"sb3_qrdqn", "qrdqn"}:
        if not _SB3_CONTRIB_AVAILABLE:
            raise ImportError("sb3_contrib is required for QR-DQN. Install with: pip install sb3-contrib")
        model = QRDQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=params.get("train_freq", 4),
            gradient_steps=gradient_steps,
            exploration_fraction=params.get("exploration_fraction", 0.1),
            exploration_initial_eps=params.get("exploration_initial_eps", 1.0),
            exploration_final_eps=params.get("exploration_final_eps", 0.05),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )

    elif algorithm in {"sb3_tqc", "tqc"}:
        if not _SB3_CONTRIB_AVAILABLE:
            raise ImportError("sb3_contrib is required for TQC. Install with: pip install sb3-contrib")
        ent_coef = params.get("ent_coef", "auto")
        target_entropy = params.get("target_entropy", "auto")
        model = TQC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            top_quantiles_to_drop_per_net=params.get("top_quantiles_to_drop_per_net", 2),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unsupported off-policy algorithm: '{algorithm}'. "
            f"Supported: {sorted(OFF_POLICY_ALGOS)}"
        )

    return model


def build_spawn_curriculum(env, scenario: Dict[str, Any], console_logger: ConsoleLogger):
    """Create spawn curriculum (or sampler) if configured. Mirrors run_sb3.py."""
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


class RenderCallback(BaseCallback):
    def __init__(self, render_every: int = 1):
        super().__init__()
        self.render_every = max(1, int(render_every))

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every == 0:
            try:
                self.training_env.env_method("render")
            except Exception:
                return True
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

    train_agent_id, algorithm = select_off_policy_agent(scenario)

    run_id = args.run_id or resolve_run_id(
        scenario_name=scenario.get("experiment", {}).get("name"),
        algorithm=algorithm,
        seed=scenario.get("experiment", {}).get("seed"),
    )
    set_run_id_env(run_id)

    wandb_logger, console_logger = initialize_loggers(scenario, args, run_id=run_id)

    console_logger.print_header(
        f"Training: {scenario['experiment']['name']}",
        f"Algorithm: {algorithm.upper()} | Episodes: {scenario['experiment'].get('episodes', 'N/A')}",
    )

    env, agents, reward_strategies = create_training_setup(scenario, mode="train")

    train_agent_cfg = scenario["agents"][train_agent_id]
    train_params = train_agent_cfg.get("params", {})
    action_constraints = train_agent_cfg.get("action_constraints", {})

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

    # Discrete algorithms need action_set; continuous need action bounds
    is_discrete = algorithm in OFF_POLICY_DISCRETE
    action_set = None

    if is_discrete:
        action_set_raw = train_params.get("action_set")
        if action_set_raw is None:
            raise ValueError(
                f"Algorithm '{algorithm}' requires 'action_set' in agent params "
                "(list of [steer, throttle] primitives)."
            )
        action_set = np.asarray(action_set_raw, dtype=np.float32)
        # action_low/high not used for discrete, but pass env bounds for consistency
        action_low = action_space.low if isinstance(action_space, spaces.Box) else np.array([-0.46, -1.0])
        action_high = action_space.high if isinstance(action_space, spaces.Box) else np.array([0.46, 1.0])
    else:
        if not isinstance(action_space, spaces.Box):
            raise ValueError(f"Continuous algorithm '{algorithm}' expects a Box action space.")
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
        action_set=action_set,
        spawn_curriculum=spawn_curriculum,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        action_constraints=action_constraints,
    )

    other_agents = {aid: agent for aid, agent in agents.items() if aid != train_agent_id}
    sb3_env.set_other_agents(other_agents)

    monitor_env = Monitor(
        sb3_env,
        info_keywords=("is_success", "outcome", "target_finished", "target_collision", "collision"),
    )

    device = train_agent_cfg.get("device", train_params.get("device", "cuda"))

    # Map algorithm name to a display name for policy_kwargs
    algo_display_map = {
        "sb3_sac": "SAC", "sac": "SAC",
        "sb3_td3": "TD3", "td3": "TD3",
        "sb3_ddpg": "DDPG", "ddpg": "DDPG",
        "sb3_tqc": "TQC", "tqc": "TQC",
        "sb3_dqn": "DQN", "dqn": "DQN",
        "sb3_qrdqn": "QRDQN", "qrdqn": "QRDQN",
    }
    algo_display = algo_display_map.get(algorithm, algorithm.upper())
    policy_kwargs = build_policy_kwargs(train_params, algo_display)

    n_action_dims = action_set.shape[1] if action_set is not None else len(action_low)
    model = build_model(
        algorithm, train_params, monitor_env, policy_kwargs, device,
        scenario["experiment"].get("seed"), n_action_dims
    )

    episodes = int(scenario["experiment"].get("episodes", 0) or 0)
    if episodes <= 0:
        raise ValueError("Scenario must define a positive experiment.episodes.")

    max_steps = int(env_config.get("max_steps", 2500))
    decision_steps = max_steps
    if action_repeat > 1:
        decision_steps = int(math.ceil(max_steps / action_repeat))
    total_timesteps = decision_steps * episodes

    console_logger.print_info(
        f"Budget: {episodes} episodes × {decision_steps} decision steps = {total_timesteps:,} timesteps"
    )

    callbacks = [StopOnEpisodeCallback(episodes, console_logger)]

    log_every_env = os.environ.get("F110_LOG_EVERY_EPISODES")
    log_every = 25
    if log_every_env is not None:
        try:
            log_every = int(log_every_env)
        except ValueError:
            pass
    if log_every > 0:
        callbacks.append(EpisodeProgressCallback(log_every, console_logger))

    if env_config.get("render"):
        render_every = 1
        render_every_env = os.environ.get("F110_RENDER_EVERY_STEPS")
        if render_every_env is not None:
            try:
                render_every = int(render_every_env)
            except (TypeError, ValueError):
                pass
        callbacks.append(RenderCallback(render_every))

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

    eval_cfg = scenario.get("evaluation", {})
    curriculum_config = scenario.get("curriculum")
    curriculum_callback = None
    if curriculum_config or spawn_curriculum:
        eval_gate_schedule_enabled = bool(eval_cfg.get("gate_eval_schedule", True))
        eval_gate_advancement_enabled = bool(eval_cfg.get("gate_phase_advancement", True))
        curriculum_callback = CurriculumCallback(
            curriculum_config=curriculum_config,
            spawn_curriculum=spawn_curriculum,
            ftg_agents=ftg_agents,
            ftg_schedules=ftg_schedules,
            env_wrapper=sb3_env,
            wandb_run=wandb_logger.run if wandb_logger else None,
            wandb_logging=scenario.get("wandb", {}).get("logging"),
            algo_name=algorithm,
            eval_gate_enabled=bool(
                eval_cfg.get("enabled", False)
                and eval_cfg.get("frequency", 0)
                and eval_gate_advancement_enabled
            ),
            eval_gate_schedule_enabled=bool(
                eval_cfg.get("enabled", False)
                and eval_cfg.get("frequency", 0)
                and eval_gate_schedule_enabled
            ),
        )
        callbacks.append(curriculum_callback)

    if eval_cfg.get("enabled", False):
        eval_config = EvaluationConfig(
            num_episodes=eval_cfg.get("num_episodes", 10),
            deterministic=eval_cfg.get("deterministic", True),
            spawn_points=eval_cfg.get("spawn_points", ["spawn_pinch_left", "spawn_pinch_right"]),
            spawn_speeds=eval_cfg.get("spawn_speeds", [0.44, 0.44]),
            lock_speed_steps=eval_cfg.get("lock_speed_steps", 0),
            ftg_override=eval_cfg.get("ftg_override", {}),
            max_steps=max_steps,
            rolling_window=eval_cfg.get("rolling_window"),
        )
        eval_env_raw, eval_agents, _ = create_training_setup(scenario, mode="eval")
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
            action_set=action_set,
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
                    curriculum_callback=curriculum_callback,
                    verbose=1,
                )
            )

    if wandb_logger and wandb_logger.run and curriculum_callback is None:
        callbacks.append(
            SB3TrainLoggingCallback(
                wandb_run=wandb_logger.run,
                wandb_logging=scenario.get("wandb", {}).get("logging"),
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    console_logger.print_info(f"Starting {algo_display} off-policy training...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=None)
    finally:
        if wandb_logger:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
