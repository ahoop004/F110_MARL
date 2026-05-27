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
from src.core.agent_builder import get_trainable_agent_ids
from loggers.console import ConsoleLogger
from loggers.wandb_logger import WandbLogger
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer
from wrappers.actions.composer import ActionComposer
from training.hooks import CheckpointHook, ConsoleHook, CurriculumHook, WandbHook

ON_POLICY_ALGOS = {"ppo", "a2c"}
OFF_POLICY_ALGOS = {"sac", "td3", "ddpg", "dqn"}
MARL_ALGOS = {"mappo"}  # multi-agent algorithms — allow multiple trainable agents


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
    p.add_argument(
        "--dataset-dir", type=str, default=None,
        help="If set, record all transitions to this directory as chunked .npz files.",
    )
    p.add_argument(
        "--dataset-chunk-size", type=int, default=10_000,
        help="Transitions per .npz chunk (default 10000).",
    )
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
    """Return (agent_id, algorithm) for the single trainable RL agent.

    Uses explicit ``trainable: true`` field when present; falls back to
    algorithm-name inference (``ppo``/``sac``/``td3``/``dqn`` → trainable).
    Raises ``ValueError`` when zero or multiple trainable agents are found,
    since current single-agent trainers require exactly one.
    """
    agent_configs = scenario.get("agents", {})
    trainable_ids = get_trainable_agent_ids(agent_configs)
    if not trainable_ids:
        raise ValueError(
            "No trainable RL agent found in scenario. "
            "Set 'trainable: true' or use a known RL algorithm (ppo, sac, td3, dqn)."
        )
    if len(trainable_ids) > 1:
        raise ValueError(
            f"Multiple trainable agents found: {trainable_ids}. "
            "Current single-agent trainers require exactly one. "
            "Use MAPPO for multi-agent training."
        )
    agent_id = trainable_ids[0]
    algo = str(agent_configs[agent_id].get("algorithm", "")).strip().lower()
    return agent_id, algo


def build_obs_composer(
    agent_cfg: Dict, env_config: Dict, scenario_dir: Path, action_dim: int = 2
) -> ObservationComposer:
    """Build one ObservationComposer for a single agent config."""
    obs_ref = agent_cfg.get("observation")
    if isinstance(obs_ref, str):
        obs_path = (scenario_dir / obs_ref).resolve()
        return ObservationComposer.from_file(str(obs_path), env_config, action_dim=action_dim)
    elif isinstance(obs_ref, dict):
        return ObservationComposer.from_config(obs_ref, env_config, action_dim=action_dim)
    raise ValueError("Agent 'observation' must be a file path or inline config dict.")


def build_reward_composer(agent_cfg: Dict, scenario_dir: Path) -> RewardComposer:
    """Build one RewardComposer for a single agent config."""
    reward_ref = agent_cfg.get("reward")
    if isinstance(reward_ref, str):
        reward_path = (scenario_dir / reward_ref).resolve()
        return RewardComposer.from_file(str(reward_path))
    elif isinstance(reward_ref, dict):
        return RewardComposer.from_config(reward_ref)
    raise ValueError("Agent 'reward' must be a file path or inline config dict.")


def build_obs_composers(
    agent_configs: Dict,
    trainable_ids: List[str],
    env_config: Dict,
    scenario_dir: Path,
    action_dim: int = 2,
) -> Dict[str, ObservationComposer]:
    """Build one ObservationComposer per trainable agent.

    Returns a dict keyed by agent_id.  Single-agent trainers index into it
    with their ``rl_agent_id``; MAPPO iterates over all entries.
    """
    return {
        aid: build_obs_composer(agent_configs[aid], env_config, scenario_dir, action_dim)
        for aid in trainable_ids
    }


def build_reward_composers(
    agent_configs: Dict,
    trainable_ids: List[str],
    scenario_dir: Path,
) -> Dict[str, RewardComposer]:
    """Build one RewardComposer per trainable agent.

    Returns a dict keyed by agent_id.
    """
    return {
        aid: build_reward_composer(agent_configs[aid], scenario_dir)
        for aid in trainable_ids
    }


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

    agent_configs = scenario.get("agents", {})
    exp_cfg = scenario.get("experiment", {})
    env_cfg = scenario.get("environment", {})

    # Detect MARL scenario (any trainable agent uses a MARL algorithm)
    trainable_ids = get_trainable_agent_ids(agent_configs)
    trainable_algos = {
        str(agent_configs[aid].get("algorithm", "")).lower() for aid in trainable_ids
    }
    if MARL_ALGOS & trainable_algos:
        algorithm = next(iter(MARL_ALGOS & trainable_algos))
        rl_agent_id = trainable_ids[0]  # focal agent for logging
    else:
        rl_agent_id, algorithm = find_rl_agent(scenario)

    agent_cfg = agent_configs[rl_agent_id]

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

    # Build env + heuristic agents (before banner so we can show real dims)
    env, agents, _ = create_training_setup(scenario, mode="train")

    # Action bounds from env
    action_space = env.action_spaces.get(rl_agent_id)
    if action_space is None:
        raise ValueError(f"RL agent '{rl_agent_id}' not in env action_spaces.")
    action_low = action_space.low
    action_high = action_space.high
    action_dim = action_space.n

    # Wrappers — build per-agent dicts, then extract the single trainable agent's composers.
    # MAPPO consumes the full dicts; single-agent trainers use rl_agent_id's entry.
    # For MARL, keep all trainable IDs; for single-agent, restrict to one.
    if algorithm not in MARL_ALGOS:
        trainable_ids = [rl_agent_id]
    # else: trainable_ids already holds the full list from get_trainable_agent_ids()

    obs_composers = build_obs_composers(agent_configs, trainable_ids, env_cfg, scenario_dir, action_dim=2)
    reward_composers = build_reward_composers(agent_configs, trainable_ids, scenario_dir)
    obs_composer = obs_composers[rl_agent_id]
    reward_composer = reward_composers[rl_agent_id]

    # Training params (needed before banner so we can show device)
    params = resolve_training_params(agent_cfg, scenario)

    # --- Startup banner ---
    maps_raw = env_cfg.get("maps", env_cfg.get("map", "?"))
    maps_str = ", ".join(maps_raw) if isinstance(maps_raw, list) else str(maps_raw)
    seed_str = str(exp_cfg.get("seed", "random"))
    from utils.torch_io import resolve_device
    device_str = str(resolve_device([params.get("device", "cpu")]))
    trainable_str = ", ".join(trainable_ids)
    fixed_str = ", ".join(k for k in agents if k not in set(trainable_ids)) or "none"

    console.print_header(
        f"Training: {exp_cfg.get('name', 'unnamed')}",
        f"algorithm={algorithm}  trainable=({trainable_str})  fixed=({fixed_str})",
    )
    console.print_info(
        f"map={maps_str}  seed={seed_str}  device={device_str}  "
        f"obs_dim={obs_composer.obs_dim}  action_dim={action_dim}"
    )

    # Other agents (fixed policy) — exclude ALL trainable agents, not just rl_agent_id
    trainable_set = set(trainable_ids)
    other_agents = {aid: ag for aid, ag in agents.items() if aid not in trainable_set}
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
        CheckpointHook(
            agent=None,
            output_dir=output_dir,
            save_every=int(params.get("checkpoint_every", os.environ.get("F110_CHECKPOINT_EVERY", 100))),
        ),  # agent set below
    ]
    if wandb_logger:
        hooks.append(WandbHook(wandb_logger))

    # Optional dataset recording
    dataset_writer = None
    if args.dataset_dir:
        from replay.dataset_writer import DatasetWriter, DatasetHook
        import hashlib, yaml as _yaml
        with open(args.scenario, "rb") as _f:
            _scenario_hash = hashlib.sha256(_f.read()).hexdigest()[:16]
        dataset_writer = DatasetWriter(
            output_dir=args.dataset_dir,
            chunk_size=args.dataset_chunk_size,
            metadata={
                "run_id": run_id,
                "algorithm": algorithm,
                "scenario": exp_cfg.get("name"),
                "scenario_hash": _scenario_hash,
                "trainable_agents": trainable_ids,
            },
        )
        hooks.append(DatasetHook(dataset_writer))
        console.print_info(f"Dataset recording → {args.dataset_dir}  (chunk_size={args.dataset_chunk_size})")

    # Optional curriculum
    spawn_plan_fn = None
    curriculum_cfg = scenario.get("curriculum")
    if curriculum_cfg:
        from training.curriculum import CurriculumManager, CurriculumPhase
        phase_cfgs = curriculum_cfg.get("phases", [])
        if phase_cfgs:
            phases = [
                CurriculumPhase(
                    name=str(p.get("name", f"phase{i}")),
                    spawn_names=list(p.get("spawn_names", [])),
                    success_threshold=float(p.get("success_threshold", 0.7)),
                    window_size=int(p.get("window_size", 50)),
                )
                for i, p in enumerate(phase_cfgs)
            ]
            curriculum = CurriculumManager(phases)
            hooks.append(CurriculumHook(curriculum, wandb_logger=wandb_logger))
            # Closure: reads live spawn_points from env's spawn manager each episode
            _cur_agent_ids = list(trainable_ids)
            def _make_spawn_plan_fn(_cur=curriculum, _env=env, _aids=_cur_agent_ids):
                def _fn():
                    sm = getattr(_env, "_spawn_manager", None)
                    pts = sm.spawn_points if sm is not None else {}
                    return _cur.next_spawn_plan(pts, _aids)
                return _fn
            spawn_plan_fn = _make_spawn_plan_fn()
            console.print_info(
                f"Curriculum: {len(phases)} phase(s) — "
                + ", ".join(f"'{p.name}' ({len(p.spawn_names)} spawns)" for p in phases)
            )

    action_constraints = agent_cfg.get("action_constraints", {})
    action_repeat = int(scenario.get("environment", {}).get("action_repeat", 1))
    render = bool(scenario.get("environment", {}).get("render", False))

    # DQN uses a discrete action set; all others use continuous denormalization.
    action_set = params.get("action_set") if algorithm == "dqn" else None
    action_composer = ActionComposer.from_config(
        action_low, action_high, action_constraints,
        action_set=np.asarray(action_set) if action_set is not None else None,
    )

    try:
        if algorithm in MARL_ALGOS:
            _run_mappo(
                env, trainable_ids, other_agents,
                obs_composers, reward_composers, action_composer, params,
                action_low, action_high, action_repeat, render,
                hooks, exp_cfg, output_dir, console,
                focal_agent_id=rl_agent_id,
            )
        elif algorithm in ON_POLICY_ALGOS:
            _run_on_policy(
                env, rl_agent_id, agent_cfg, other_agents,
                obs_composer, reward_composer, action_composer, params,
                action_repeat, render, hooks, exp_cfg, output_dir, console,
                run_id=run_id,
                spawn_plan_fn=spawn_plan_fn,
            )
        elif algorithm in OFF_POLICY_ALGOS:
            _run_off_policy(
                env, rl_agent_id, agent_cfg, other_agents,
                obs_composer, reward_composer, action_composer, params,
                action_low, action_high, action_repeat, render,
                hooks, exp_cfg, output_dir, console, algorithm,
                run_id=run_id,
                spawn_plan_fn=spawn_plan_fn,
            )
        else:
            console.print_error(f"Unknown algorithm: '{algorithm}'")
            sys.exit(1)
    finally:
        if wandb_logger:
            wandb_logger.finish()


def _run_on_policy(
    env, rl_agent_id, agent_cfg, other_agents,
    obs_composer, reward_composer, action_composer, params,
    action_repeat, render, hooks, exp_cfg, output_dir, console,
    run_id: str = "run",
    spawn_plan_fn=None,
) -> None:
    from agents.ppo import PPOAgent
    from training.on_policy_trainer import OnPolicyTrainer

    n_episodes = int(exp_cfg.get("episodes", 1000))

    action_space = env.action_spaces.get(rl_agent_id)
    agent = PPOAgent(
        obs_dim=obs_composer.obs_dim,
        action_low=action_space.low,
        action_high=action_space.high,
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
        action_composer=action_composer,
        action_repeat=action_repeat,
        hooks=hooks,
        render=render,
        run_id=run_id,
        spawn_plan_fn=spawn_plan_fn,
    )

    console.print_info(f"Starting PPO training for {n_episodes} episodes...")
    trainer.train(n_episodes=n_episodes)


def _run_off_policy(
    env, rl_agent_id, agent_cfg, other_agents,
    obs_composer, reward_composer, action_composer, params,
    action_low, action_high, action_repeat, render,
    hooks, exp_cfg, output_dir, console, algorithm,
    run_id: str = "run",
    spawn_plan_fn=None,
) -> None:
    from replay.replay_buffer import ReplayBuffer
    from training.off_policy_trainer import OffPolicyTrainer

    total_steps = int(exp_cfg.get("total_steps", 500_000))
    learning_starts = int(params.get("learning_starts", 10_000))
    train_freq = int(params.get("train_freq", 1))
    gradient_steps = int(params.get("gradient_steps", 1))
    batch_size = int(params.get("batch_size", 256))
    buffer_size = int(params.get("buffer_size", 1_000_000))

    from utils.torch_io import resolve_device
    device = resolve_device([params.get("device", "cpu")])

    # Action dim for replay buffer: DQN stores scalar index, others store full action vector
    if algorithm == "dqn":
        from agents.dqn import DQNAgent
        action_set = params.get("action_set", [])
        agent = DQNAgent(obs_dim=obs_composer.obs_dim, action_set=action_set, params=params)
        agent.set_total_steps(total_steps)
        buf_action_dim = 1
    elif algorithm in {"sac", "ddpg"}:
        from agents.sac import SACAgent
        agent = SACAgent(obs_dim=obs_composer.obs_dim, action_low=action_low, action_high=action_high, params=params)
        buf_action_dim = len(action_low)
    elif algorithm == "td3":
        from agents.td3 import TD3Agent
        agent = TD3Agent(obs_dim=obs_composer.obs_dim, action_low=action_low, action_high=action_high, params=params)
        buf_action_dim = len(action_low)
    else:
        console.print_error(f"Off-policy algorithm '{algorithm}' not implemented.")
        import sys; sys.exit(1)

    replay_buffer = ReplayBuffer(
        capacity=buffer_size,
        obs_dim=obs_composer.obs_dim,
        action_dim=buf_action_dim,
        device=device,
    )

    for hook in hooks:
        if hasattr(hook, "_agent") and hook._agent is None:
            hook._agent = agent

    trainer = OffPolicyTrainer(
        env=env,
        rl_agent_id=rl_agent_id,
        agent=agent,
        other_agents=other_agents,
        obs_composer=obs_composer,
        reward_composer=reward_composer,
        action_composer=action_composer,
        replay_buffer=replay_buffer,
        action_repeat=action_repeat,
        hooks=hooks,
        render=render,
        run_id=run_id,
        spawn_plan_fn=spawn_plan_fn,
    )

    console.print_info(
        f"Starting {algorithm.upper()} training for {total_steps:,} steps "
        f"(learning_starts={learning_starts:,})"
    )
    trainer.train(
        total_steps=total_steps,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        batch_size=batch_size,
    )


def _run_mappo(
    env, trainable_ids, other_agents,
    obs_composers, reward_composers, action_composer, params,
    action_low, action_high, action_repeat, render,
    hooks, exp_cfg, output_dir, console,
    focal_agent_id=None,
) -> None:
    from agents.mappo import MAPPOAgent
    from training.marl_trainer import MARLTrainer

    n_episodes = int(exp_cfg.get("episodes", 1000))
    focal_id = focal_agent_id or (trainable_ids[0] if trainable_ids else "")

    # obs_dim: all trainable agents share the same local observation spec
    obs_dim = obs_composers[focal_id].obs_dim

    # Probe global state dimension via one env reset (MARLTrainer will reset again per episode)
    _obs_dict, _info_dict = env.reset()
    global_state_dim = len(env.get_global_state().vector)

    # Merge training params for focal agent (already done by caller, but resolve again
    # to give MAPPOAgent the final merged dict).
    agent = MAPPOAgent(
        obs_dim=obs_dim,
        global_state_dim=global_state_dim,
        action_low=action_low,
        action_high=action_high,
        agent_ids=trainable_ids,
        params=params,
    )

    # Wire checkpoint hook (same pattern as single-agent trainers)
    for hook in hooks:
        if hasattr(hook, "_agent") and hook._agent is None:
            hook._agent = agent

    trainer = MARLTrainer(
        env=env,
        agent=agent,
        trainable_ids=trainable_ids,
        other_agents=other_agents,
        obs_composers=obs_composers,
        reward_composers=reward_composers,
        action_composer=action_composer,
        action_repeat=action_repeat,
        hooks=hooks,
        render=render,
        focal_agent_id=focal_id,
    )

    console.print_info(
        f"Starting MAPPO training for {n_episodes} episodes "
        f"| agents={trainable_ids} | obs_dim={obs_dim} | global_state_dim={global_state_dim}"
    )
    trainer.train(n_episodes=n_episodes)


if __name__ == "__main__":
    main()
