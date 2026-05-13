#!/usr/bin/env python3
"""Unified training entry point — dispatches to the right trainer based on scenario algorithm."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.scenario import ScenarioError, load_and_expand_scenario
from core.setup import create_training_setup
from core.run_id import resolve_run_id, set_run_id_env
from loggers.console import ConsoleLogger
from loggers.wandb_logger import WandbLogger
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer
from training.hooks import CheckpointHook, ConsoleHook, WandbHook

ON_POLICY_ALGOS = {"ppo", "a2c", "sb3_ppo", "sb3_a2c"}
OFF_POLICY_ALGOS = {"sac", "td3", "ddpg", "dqn", "qrdqn", "sb3_sac", "sb3_td3", "sb3_dqn"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F110 RL training")
    p.add_argument("--scenario", required=True, help="Path to scenario YAML file")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def apply_cli_overrides(scenario: Dict, args: argparse.Namespace) -> Dict:
    if args.seed is not None:
        scenario.setdefault("experiment", {})["seed"] = args.seed
    if args.episodes is not None:
        scenario.setdefault("experiment", {})["episodes"] = args.episodes
    if args.total_steps is not None:
        scenario.setdefault("experiment", {})["total_steps"] = args.total_steps
    if args.wandb:
        scenario.setdefault("wandb", {})["enabled"] = True
    elif args.no_wandb:
        scenario.setdefault("wandb", {})["enabled"] = False
    if args.render:
        scenario.setdefault("environment", {})["render"] = True
    elif args.no_render:
        scenario.setdefault("environment", {})["render"] = False
    return scenario


def find_rl_agent(scenario: Dict) -> tuple[str, str]:
    """Return (agent_id, algorithm) for the single RL agent in the scenario."""
    for agent_id, cfg in scenario.get("agents", {}).items():
        algo = cfg.get("algorithm", "").lower()
        if algo in ON_POLICY_ALGOS | OFF_POLICY_ALGOS:
            return agent_id, algo
    raise ValueError("No RL agent found in scenario. Expected algorithm: ppo, sac, td3, etc.")


def build_obs_composer(
    agent_cfg: Dict, env_config: Dict, scenario_dir: Path, action_dim: int = 2
) -> ObservationComposer:
    obs_ref = agent_cfg.get("observation")
    if isinstance(obs_ref, str):
        obs_path = (scenario_dir / obs_ref).resolve()
        return ObservationComposer.from_file(str(obs_path), env_config, action_dim=action_dim)
    elif isinstance(obs_ref, dict):
        return ObservationComposer.from_config(obs_ref, env_config, action_dim=action_dim)
    raise ValueError("Agent 'observation' must be a file path or inline config dict.")


def build_reward_composer(agent_cfg: Dict, scenario_dir: Path) -> RewardComposer:
    reward_ref = agent_cfg.get("reward")
    if isinstance(reward_ref, str):
        reward_path = (scenario_dir / reward_ref).resolve()
        return RewardComposer.from_file(str(reward_path))
    elif isinstance(reward_ref, dict):
        return RewardComposer.from_config(reward_ref)
    raise ValueError("Agent 'reward' must be a file path or inline config dict.")


def resolve_training_params(agent_cfg: Dict, scenario: Dict) -> Dict:
    """Merge training_defaults with agent params — agent params win."""
    defaults = scenario.get("training_defaults", {})
    params = agent_cfg.get("params", {})
    return {**defaults, **params}


def main() -> None:
    args = parse_args()
    console = ConsoleLogger(verbose=not args.quiet)

    try:
        scenario = load_and_expand_scenario(args.scenario)
    except (ScenarioError, FileNotFoundError) as exc:
        console.print_error(f"Failed to load scenario: {exc}")
        sys.exit(1)

    scenario = apply_cli_overrides(scenario, args)
    scenario_dir = Path(args.scenario).resolve().parent

    rl_agent_id, algorithm = find_rl_agent(scenario)
    agent_cfg = scenario["agents"][rl_agent_id]
    exp_cfg = scenario.get("experiment", {})
    env_cfg = scenario.get("environment", {})

    run_id = args.run_id or resolve_run_id(
        scenario_name=exp_cfg.get("name"),
        algorithm=algorithm,
        seed=exp_cfg.get("seed"),
    )
    set_run_id_env(run_id)

    # Loggers
    wandb_cfg = scenario.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False)
    wandb_logger: Optional[WandbLogger] = None
    if wandb_enabled:
        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "f110"),
            name=wandb_cfg.get("name", exp_cfg.get("name")),
            config=scenario,
            tags=wandb_cfg.get("tags", []),
            group=wandb_cfg.get("group"),
            job_type=wandb_cfg.get("job_type", algorithm),
            entity=wandb_cfg.get("entity"),
            notes=wandb_cfg.get("notes"),
            mode=wandb_cfg.get("mode", "online"),
            run_id=run_id,
        )

    console.print_header(
        f"Training: {exp_cfg.get('name', 'unnamed')}",
        f"Algorithm: {algorithm}  |  RL agent: {rl_agent_id}",
    )

    # Build env + heuristic agents
    env, agents, _ = create_training_setup(scenario, mode="train")

    # Action bounds from env
    action_space = env.action_spaces.get(rl_agent_id)
    if action_space is None:
        raise ValueError(f"RL agent '{rl_agent_id}' not in env action_spaces.")
    action_low = action_space.low
    action_high = action_space.high
    action_dim = action_space.n

    # Wrappers
    obs_composer = build_obs_composer(agent_cfg, env_cfg, scenario_dir, action_dim=2)
    reward_composer = build_reward_composer(agent_cfg, scenario_dir)

    console.print_info(f"obs_dim={obs_composer.obs_dim}  action_dim={action_dim}")

    # Training params
    params = resolve_training_params(agent_cfg, scenario)

    # Other agents (fixed policy)
    other_agents = {aid: ag for aid, ag in agents.items() if aid != rl_agent_id}
    for aid, ag in other_agents.items():
        if hasattr(ag, "set_env"):
            ag.set_env(env)

    # Output directory
    output_dir = args.output_dir or os.path.join(
        "outputs", exp_cfg.get("name", "unnamed"), run_id
    )

    # Hooks
    hooks = [
        ConsoleHook(
            logger=console,
            log_every=int(os.environ.get("F110_LOG_EVERY", "1")),
            summary_every=int(os.environ.get("F110_SUMMARY_EVERY", "25")),
        ),
        CheckpointHook(agent=None, output_dir=output_dir),  # agent set below
    ]
    if wandb_logger:
        hooks.append(WandbHook(wandb_logger))

    action_constraints = agent_cfg.get("action_constraints", {})

    try:
        if algorithm in ON_POLICY_ALGOS:
            action_repeat = int(scenario.get("environment", {}).get("action_repeat", 1))
            render = bool(scenario.get("environment", {}).get("render", False))
            _run_on_policy(
                env, rl_agent_id, agent_cfg, other_agents,
                obs_composer, reward_composer, params,
                action_low, action_high, action_constraints,
                action_repeat, render, hooks, exp_cfg, output_dir, console,
            )
        elif algorithm in OFF_POLICY_ALGOS:
            console.print_error(f"Off-policy algorithm '{algorithm}' not yet implemented. Coming in Phase 2.")
            sys.exit(1)
        else:
            console.print_error(f"Unknown algorithm: '{algorithm}'")
            sys.exit(1)
    finally:
        if wandb_logger:
            wandb_logger.finish()


def _run_on_policy(
    env, rl_agent_id, agent_cfg, other_agents,
    obs_composer, reward_composer, params,
    action_low, action_high, action_constraints,
    action_repeat, render, hooks, exp_cfg, output_dir, console,
) -> None:
    from agents.ppo import PPOAgent
    from training.on_policy_trainer import OnPolicyTrainer

    n_episodes = int(exp_cfg.get("episodes", 1000))

    agent = PPOAgent(
        obs_dim=obs_composer.obs_dim,
        action_low=action_low,
        action_high=action_high,
        params=params,
    )

    # Wire checkpoint hook to the agent now that we have it
    for hook in hooks:
        if hasattr(hook, "_agent") and hook._agent is None:
            hook._agent = agent

    trainer = OnPolicyTrainer(
        env=env,
        rl_agent_id=rl_agent_id,
        agent=agent,
        other_agents=other_agents,
        obs_composer=obs_composer,
        reward_composer=reward_composer,
        action_repeat=action_repeat,
        action_constraints=action_constraints,
        hooks=hooks,
        render=render,
    )

    console.print_info(f"Starting PPO training for {n_episodes} episodes...")
    trainer.train(n_episodes=n_episodes)


if __name__ == "__main__":
    main()
