#!/usr/bin/env python3
"""MARL role-based training runner (PPO/A2C).

Trains a shared policy for all agents of one role (e.g. all defenders or all
attackers) while the other role runs a fixed policy (FTG or a frozen checkpoint).

The focal agent cycles each episode so every role member contributes equally to
the shared policy's training data.

Usage
-----
    # Phase 1: train defenders
    python run_marl_roles.py --scenario scenarios/circle_defender.yaml

    # Phase 2: train attackers (after Phase 1 produces a defender checkpoint)
    python run_marl_roles.py --scenario scenarios/circle_attacker.yaml

    # Enable W&B logging
    python run_marl_roles.py --scenario scenarios/circle_defender.yaml --wandb
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn
from gymnasium import spaces

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.sb3_role_wrapper import SB3RoleWrapper
from baselines.sb3_train_callback import SB3TrainLoggingCallback
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
        "stable-baselines3 is required. Install with: pip install stable-baselines3",
        file=sys.stderr,
    )
    raise

ON_POLICY_ALGOS = {"sb3_ppo", "sb3_a2c", "ppo", "a2c"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F110 MARL role-based training (shared PPO/A2C per role)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", type=str, required=True,
                        help="Path to scenario YAML file")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed override")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episode count override")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable console output")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run ID")
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


# ---------------------------------------------------------------------------
# Role agent selection
# ---------------------------------------------------------------------------

def select_role_agents(scenario: Dict[str, Any]) -> Tuple[List[str], str]:
    """Return (ordered list of role agent IDs, algorithm name).

    All agents that share an on-policy algorithm form the training role.
    They must all use the *same* algorithm.
    """
    role: Dict[str, str] = {}
    for agent_id, cfg in scenario.get("agents", {}).items():
        algo = cfg.get("algorithm", "").lower()
        if algo in ON_POLICY_ALGOS:
            role[agent_id] = algo

    if not role:
        raise ValueError(
            "No on-policy agents found in scenario. "
            "Expected at least one agent with algorithm: sb3_ppo or sb3_a2c."
        )

    algos = set(role.values())
    if len(algos) > 1:
        raise ValueError(
            f"All role agents must share one algorithm, found: {algos}. "
            "Use a single algorithm for all agents of the same role."
        )

    # Preserve insertion order (Python 3.7+)
    return list(role.keys()), next(iter(algos))


# ---------------------------------------------------------------------------
# Reused helpers from run_sb3.py
# ---------------------------------------------------------------------------

def initialize_loggers(
    scenario: Dict[str, Any], args: argparse.Namespace, run_id: Optional[str] = None
) -> Tuple[Optional[WandbLogger], ConsoleLogger]:
    console_logger = ConsoleLogger(verbose=not args.quiet)
    wandb_config = scenario.get("wandb", {})
    wandb_enabled = wandb_config.get("enabled", False)

    default_algo = "unknown"
    for agent_cfg in scenario.get("agents", {}).values():
        algo = agent_cfg.get("algorithm", "").lower()
        if algo in ON_POLICY_ALGOS:
            default_algo = algo
            break

    if wandb_enabled:
        console_logger.print_info("Initializing Weights & Biases...")
        wandb_logger = WandbLogger(
            project=wandb_config.get("project", "f110-marl"),
            name=wandb_config.get("name", scenario["experiment"]["name"]),
            config=scenario,
            tags=wandb_config.get("tags", []),
            group=wandb_config.get("group"),
            job_type=wandb_config.get("job_type", default_algo),
            entity=wandb_config.get("entity"),
            notes=wandb_config.get("notes"),
            mode=wandb_config.get("mode", "online"),
            run_id=run_id,
            logging_config=wandb_config.get("logging"),
        )
    else:
        wandb_logger = None

    return wandb_logger, console_logger


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


def compute_obs_dim(obs_space, preset: Optional[str]) -> int:
    if preset:
        dummy_obs = obs_space.sample()
        flat = flatten_observation(dummy_obs, preset=preset)
        # prev_action is injected by wrapper (+2 dims); account for it here
        # by adding a dummy prev_action field to the sample
        dummy_obs_with_prev = dict(dummy_obs)
        dummy_obs_with_prev["prev_action"] = np.zeros(2, dtype=np.float32)
        flat = flatten_observation(dummy_obs_with_prev, preset=preset)
        return int(flat.shape[0])
    return get_space_dim(obs_space)


def parse_action_repeat(env_config: Dict[str, Any]) -> int:
    for key in ("action_repeat", "step_repeat", "step_skip", "frame_skip"):
        if key in env_config:
            try:
                return max(1, int(env_config[key]))
            except (TypeError, ValueError):
                return 1
    return 1


def build_policy_kwargs(params: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    hidden_dims = params.get("hidden_dims", [64, 64])
    pi_dims = params.get("pi_hidden_dims")
    vf_dims = params.get("vf_hidden_dims")

    if pi_dims is not None or vf_dims is not None:
        net_arch = {
            "pi": pi_dims if pi_dims is not None else hidden_dims,
            "vf": vf_dims if vf_dims is not None else hidden_dims,
        }
    else:
        net_arch = hidden_dims

    policy_kwargs: Dict[str, Any] = {"net_arch": net_arch}

    activation = params.get("activation")
    if isinstance(activation, str):
        activation_map = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
            "tanh": nn.Tanh,
        }
        key = activation.strip().lower()
        if key not in activation_map:
            raise ValueError(f"Unsupported activation '{activation}'. Use: relu, silu/swish, tanh.")
        policy_kwargs["activation_fn"] = activation_map[key]
    elif activation is not None:
        policy_kwargs["activation_fn"] = activation

    return policy_kwargs


def build_model(algorithm, params, env, policy_kwargs, device, seed):
    lr = params.get("learning_rate", 3e-4)
    gamma = params.get("gamma", 0.998)
    if algorithm in {"sb3_ppo", "ppo"}:
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr,
            n_steps=params.get("n_steps", 2048),
            batch_size=params.get("batch_size", 256),
            n_epochs=params.get("n_epochs", 10),
            gamma=gamma,
            gae_lambda=params.get("gae_lambda", 0.95),
            clip_range=params.get("clip_range", 0.2),
            ent_coef=params.get("ent_coef", 0.005),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            seed=seed,
        )
    if algorithm in {"sb3_a2c", "a2c"}:
        return A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr,
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
    raise ValueError(f"Unsupported on-policy algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class StopOnEpisodeCallback(BaseCallback):
    """Stop training after reaching a target episode count."""

    def __init__(self, max_episodes: int, console_logger: Optional[ConsoleLogger] = None):
        super().__init__()
        self.max_episodes = max_episodes
        self.console_logger = console_logger
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.max_episodes <= 0:
            return True
        dones = self.locals.get("dones")
        if dones is None:
            term = self.locals.get("terminateds")
            trunc = self.locals.get("truncateds")
            if term is None or trunc is None:
                return True
            dones = np.logical_or(term, trunc)
        done_count = int(np.sum(dones)) if hasattr(dones, "__len__") else int(bool(dones))
        if done_count:
            self.episode_count += done_count
            if self.episode_count >= self.max_episodes:
                if self.console_logger:
                    self.console_logger.print_success(
                        f"Reached {self.episode_count} episodes; stopping."
                    )
                return False
        return True


class EpisodeProgressCallback(BaseCallback):
    def __init__(self, log_every: int, console_logger: Optional[ConsoleLogger] = None):
        super().__init__()
        self.log_every = max(1, int(log_every))
        self.console_logger = console_logger
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            term = self.locals.get("terminateds")
            trunc = self.locals.get("truncateds")
            if term is None or trunc is None:
                return True
            dones = np.logical_or(term, trunc)
        done_count = int(np.sum(dones)) if hasattr(dones, "__len__") else int(bool(dones))
        if done_count:
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
                pass
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    role_agent_ids, algorithm = select_role_agents(scenario)
    n_role = len(role_agent_ids)
    template_id = role_agent_ids[0]

    run_id = args.run_id or resolve_run_id(
        scenario_name=scenario.get("experiment", {}).get("name"),
        algorithm=algorithm,
        seed=scenario.get("experiment", {}).get("seed"),
    )
    set_run_id_env(run_id)

    wandb_logger, console_logger = initialize_loggers(scenario, args, run_id=run_id)

    console_logger.print_header(
        f"MARL Role Training: {scenario['experiment']['name']}",
        f"Role agents ({n_role}): {role_agent_ids} | Episodes: {scenario['experiment'].get('episodes', 'N/A')}",
    )

    env, agents, reward_strategies = create_training_setup(scenario, mode="train")

    # All role agents share the same config — use template for hyperparams
    train_cfg = scenario["agents"][template_id]
    train_params = train_cfg.get("params", {})
    action_constraints = train_cfg.get("action_constraints", {})

    env_config = scenario.get("environment", {})
    action_repeat = parse_action_repeat(env_config)
    observation_preset = infer_observation_preset(train_cfg)

    obs_space = env.observation_spaces.get(template_id)
    action_space = env.action_spaces.get(template_id)
    if obs_space is None or action_space is None:
        raise ValueError(
            f"Template agent '{template_id}' not found in env spaces. "
            f"Available: {list(env.observation_spaces.keys())}"
        )

    if not isinstance(action_space, spaces.Box):
        raise ValueError("Role wrapper requires a continuous (Box) action space.")

    obs_dim = compute_obs_dim(obs_space, observation_preset)
    action_low = action_space.low
    action_high = action_space.high

    console_logger.print_info(
        f"obs_dim={obs_dim}, action_repeat={action_repeat}, preset='{observation_preset}'"
    )

    # Reward strategy — shared across all focal agents (same objective per role)
    reward_strategy = reward_strategies.get(template_id)
    if reward_strategy is None:
        console_logger.print_warning(
            f"No reward strategy found for '{template_id}'; "
            "using environment default reward."
        )

    # Fixed-policy agents (everything outside the training role)
    other_agents = {
        aid: agent for aid, agent in agents.items()
        if aid not in role_agent_ids
    }
    console_logger.print_info(
        f"Fixed-policy agents: {list(other_agents.keys())}"
    )

    # Build the role wrapper
    role_wrapper = SB3RoleWrapper(
        env,
        role_agent_ids=role_agent_ids,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        observation_preset=observation_preset,
        reward_strategy=reward_strategy,
        action_repeat=action_repeat,
        action_constraints=action_constraints,
    )
    role_wrapper.set_other_agents(other_agents)

    monitor_env = Monitor(
        role_wrapper,
        info_keywords=("outcome", "focal_agent", "role_episode"),
    )

    device = train_cfg.get("device", train_params.get("device", "cpu"))
    policy_kwargs = build_policy_kwargs(train_params, "PPO" if "ppo" in algorithm else "A2C")
    model = build_model(
        algorithm, train_params, monitor_env, policy_kwargs, device,
        scenario["experiment"].get("seed"),
    )

    # Give the wrapper access to the model so non-focal role agents can act
    role_wrapper.set_shared_model(model)

    episodes = int(scenario["experiment"].get("episodes", 0) or 0)
    if episodes <= 0:
        raise ValueError("Scenario must define a positive experiment.episodes.")

    max_steps = int(env_config.get("max_steps", 6000))
    decision_steps = max_steps if action_repeat <= 1 else int(math.ceil(max_steps / action_repeat))
    total_timesteps = decision_steps * episodes

    console_logger.print_info(
        f"Total timesteps: {total_timesteps:,}  "
        f"({episodes} episodes × {decision_steps} decision steps)"
    )

    callbacks = [StopOnEpisodeCallback(episodes, console_logger)]

    log_every_str = os.environ.get("F110_LOG_EVERY_EPISODES")
    try:
        log_every = int(log_every_str) if log_every_str else 25
    except ValueError:
        log_every = 25
    if log_every > 0:
        callbacks.append(EpisodeProgressCallback(log_every, console_logger))

    if env_config.get("render"):
        render_every = 1
        try:
            render_every = int(os.environ.get("F110_RENDER_EVERY_STEPS", "1"))
        except (TypeError, ValueError):
            render_every = 1
        callbacks.append(RenderCallback(render_every))

    if wandb_logger and wandb_logger.run:
        callbacks.append(
            SB3TrainLoggingCallback(
                wandb_run=wandb_logger.run,
                wandb_logging=scenario.get("wandb", {}).get("logging"),
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    console_logger.print_info("Starting MARL role training...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    finally:
        if wandb_logger:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
