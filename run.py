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

from core.scenario import ScenarioError, load_and_expand_scenario, resolve_mappo_config
from core.setup import create_training_setup
from core.run_id import resolve_run_id, set_run_id_env
from core.provenance import build_run_provenance, provenance_mismatches
from src.core.agent_builder import get_trainable_agent_ids
from loggers.console import ConsoleLogger
from loggers.csv_logger import CSVLogger
from loggers.wandb_logger import WandbLogger
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer
from wrappers.actions.composer import ActionComposer
from training.hooks import CSVHook, CheckpointHook, ConsoleHook, CurriculumHook, WandbHook

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
    p.add_argument("--eval", action="store_true", help="Run evaluation instead of training")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for --eval")
    p.add_argument(
        "--allow-provenance-mismatch",
        action="store_true",
        help="Allow --eval with a checkpoint from a different scenario/config/map contract.",
    )
    p.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes; defaults to --episodes")
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


def _run_heuristic(
    scenario: Dict,
    args: argparse.Namespace,
    console: "ConsoleLogger",
) -> None:
    """Episode loop for scenarios where every agent is a heuristic/fixed policy.

    Replaces the RL training path when ``run.py`` detects no trainable agents.
    Supports the same ``--render``, ``--episodes``, ``--seed``, and ``--wandb``
    flags as the RL path.
    """
    from core.setup import create_training_setup

    agent_configs = scenario.get("agents", {})
    exp_cfg = scenario.get("experiment", {})
    env_cfg = scenario.get("environment", {})
    wandb_cfg = scenario.get("wandb", {})

    n_episodes = int(exp_cfg.get("episodes", 100))
    render = bool(env_cfg.get("render", False))
    agent_ids = list(agent_configs.keys())
    algos = {aid: agent_configs[aid].get("algorithm", "?") for aid in agent_ids}
    agents_str = "  ".join(f"{aid}={v}" for aid, v in algos.items())
    algorithm = "+".join(sorted({str(value) for value in algos.values()})) or "heuristic"
    run_id = args.run_id or resolve_run_id(
        scenario_name=exp_cfg.get("name"),
        algorithm=algorithm,
        seed=exp_cfg.get("seed"),
    )
    set_run_id_env(run_id)
    output_dir = args.output_dir or os.path.join(
        "outputs", exp_cfg.get("name", "unnamed"), run_id
    )
    provenance = build_run_provenance(
        scenario,
        scenario_path=args.scenario,
        run_id=run_id,
        algorithm=algorithm,
        trainable_agents=[],
    )
    csv_logger = CSVLogger(output_dir, scenario, provenance=provenance)

    console.print_header(
        f"Heuristic: {exp_cfg.get('name', 'unnamed')}",
        agents_str,
    )

    wandb_logger = None
    if wandb_cfg.get("enabled", False):
        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "f110"),
            name=wandb_cfg.get("name", exp_cfg.get("name")),
            config=scenario,
            tags=wandb_cfg.get("tags", []),
            group=wandb_cfg.get("group"),
            job_type=wandb_cfg.get("job_type", "heuristic"),
            entity=wandb_cfg.get("entity"),
            notes=wandb_cfg.get("notes"),
            mode=wandb_cfg.get("mode", "online"),
        )

    env, agents, _ = create_training_setup(
        scenario, mode="train", scenario_dir=Path(args.scenario).resolve().parent
    )
    for ag in agents.values():
        if hasattr(ag, "set_env"):
            ag.set_env(env)

    try:
        for episode in range(n_episodes):
            obs_dict, info = env.reset()
            for ag in agents.values():
                if hasattr(ag, "reset"):
                    ag.reset()

            step = 0
            done_set: set = set()
            any_collision = False
            timeout = False

            while True:
                if not obs_dict or done_set.issuperset(agent_ids):
                    break

                actions: Dict[str, np.ndarray] = {}
                for aid, obs in obs_dict.items():
                    if aid not in agents:
                        continue
                    try:
                        act = agents[aid].act(obs)
                    except Exception:
                        act = np.zeros(2, dtype=np.float32)
                    actions[aid] = np.asarray(act, dtype=np.float32)

                if not actions:
                    break

                obs_dict, _rewards, dones, truncs, info = env.step(actions)
                step += 1

                for aid in agent_ids:
                    if info.get(aid, {}).get("collision", False):
                        any_collision = True
                    if dones.get(aid, False) or truncs.get(aid, False):
                        done_set.add(aid)
                        if truncs.get(aid, False):
                            timeout = True

                if render:
                    env.render()

            status = "TIMEOUT" if timeout else ("COLLISION" if any_collision else "ok")
            console.print_info(f"ep {episode+1:4d}/{n_episodes}  steps={step:5d}  {status}")

            final_infos = {aid: dict(info.get(aid, {})) for aid in agent_ids}
            focal_info = final_infos.get(agent_ids[0], {}) if agent_ids else {}
            focal_info["outcome"] = focal_info.get("terminal_reason") or status.lower()
            csv_logger.log_training_episode(
                episode,
                reward=0.0,
                info=focal_info,
                metrics={
                    "episode_steps": step,
                    "collision": int(any_collision),
                    "timeout": int(timeout),
                    "agent_outcomes": {
                        aid: str(payload.get("terminal_reason") or status.lower())
                        for aid, payload in final_infos.items()
                    },
                    "agent_terminal_reasons": {
                        aid: payload.get("terminal_reason")
                        for aid, payload in final_infos.items()
                    },
                    "agent_finish_positions": {
                        aid: payload.get("finish_position")
                        for aid, payload in final_infos.items()
                    },
                    "agent_lap_counts": {
                        aid: int(payload.get("lap_count", 0))
                        for aid, payload in final_infos.items()
                    },
                },
            )

            if wandb_logger:
                wandb_logger.log({
                    "episode": episode,
                    "steps": step,
                    "collision": int(any_collision),
                    "timeout": int(timeout),
                })
    finally:
        csv_logger.close()
        if wandb_logger:
            wandb_logger.finish()

    console.print_info("Done.")


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

    if args.eval:
        _run_eval(scenario, args, console, scenario_dir)
        return

    # Pure heuristic scenario — no RL training, just run fixed-policy agents.
    if not trainable_ids and not (MARL_ALGOS & trainable_algos):
        _run_heuristic(scenario, args, console)
        return

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
    env, agents, _ = create_training_setup(
        scenario, mode="train", scenario_dir=scenario_dir
    )

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
    if algorithm in MARL_ALGOS:
        params = {**params, **resolve_mappo_config(scenario)}
        obs_dims = {aid: obs_composers[aid].obs_dim for aid in trainable_ids}
        if len(set(obs_dims.values())) != 1:
            raise ValueError(
                "Shared MAPPO actor requires identical local observation dimensions; "
                f"got {obs_dims}."
            )

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
    provenance = build_run_provenance(
        scenario,
        scenario_path=args.scenario,
        run_id=run_id,
        algorithm=algorithm,
        trainable_agents=trainable_ids,
    )
    csv_logger = CSVLogger(
        output_dir=output_dir,
        scenario_config=scenario,
        provenance=provenance,
    )

    # Hooks
    hooks = [
        ConsoleHook(
            logger=console,
            log_every=int(os.environ.get("F110_LOG_EVERY", "1")),
            summary_every=int(os.environ.get("F110_SUMMARY_EVERY", "25")),
        ),
        CSVHook(csv_logger),
        CheckpointHook(
            agent=None,
            output_dir=output_dir,
            save_every=int(params.get("checkpoint_every", os.environ.get("F110_CHECKPOINT_EVERY", 100))),
            provenance=provenance,
        ),  # agent set below
    ]
    if wandb_logger:
        hooks.append(WandbHook(wandb_logger))

    # Optional dataset recording
    dataset_writer = None
    if args.dataset_dir:
        from src.replay.dataset_writer import DatasetWriter, DatasetHook
        dataset_writer = DatasetWriter(
            output_dir=args.dataset_dir,
            chunk_size=args.dataset_chunk_size,
            metadata={
                "run_id": run_id,
                "algorithm": algorithm,
                "scenario": exp_cfg.get("name"),
                "scenario_hash": provenance["scenario_source_sha256"][:16],
                "scenario_source_sha256": provenance["scenario_source_sha256"],
                "resolved_config_hash": provenance["resolved_config_sha256"],
                "trainable_agents": trainable_ids,
                "episode_termination": scenario.get("environment", {}).get(
                    "episode_termination", {}
                ),
                "terminal_agents": scenario.get("environment", {}).get(
                    "terminal_agents", {}
                ),
                "target_laps": scenario.get("environment", {}).get("target_laps", 1),
                "map_protocols": provenance["map_protocols"],
                "provenance": provenance,
                "global_state_dim": len(env.get_global_state().vector),
                "observation_dims": {
                    aid: composer.obs_dim for aid, composer in obs_composers.items()
                },
                "lifecycle_contract_version": "1.0",
                "mappo": (
                    resolve_mappo_config(scenario)
                    if algorithm in MARL_ALGOS
                    else None
                ),
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
                run_id=run_id,
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
        csv_logger.close()
        if wandb_logger:
            wandb_logger.finish()


def _build_eval_actions(
    trainable_actions_phys: Dict[str, np.ndarray],
    other_agents: Dict[str, Any],
    obs_dict: Dict[str, Any],
    active_agents: Optional[set[str]] = None,
) -> Dict[str, np.ndarray]:
    """Assemble trainable and fixed-policy actions for one env decision."""
    active = active_agents if active_agents is not None else set(obs_dict)
    actions: Dict[str, np.ndarray] = {
        aid: np.asarray(action, dtype=np.float32)
        for aid, action in trainable_actions_phys.items()
        if aid in active
    }
    for aid, other_agent in other_agents.items():
        if aid not in active or aid not in obs_dict:
            continue
        try:
            act = other_agent.act(obs_dict[aid])
        except Exception:
            act = np.zeros(2, dtype=np.float32)
        actions[aid] = np.asarray(act, dtype=np.float32)
    return actions


def _build_eval_reward_context(
    env: Any,
    *,
    agent_id: str,
    info_dict: Dict[str, Any],
    obs_dict: Dict[str, Any],
    actions: Dict[str, np.ndarray],
    global_state: Optional[Any] = None,
) -> Dict[str, Any]:
    if global_state is None:
        try:
            global_state = env.get_global_state()
        except Exception:
            global_state = None
    global_vector = (
        global_state.vector
        if global_state is not None
        else np.zeros(0, dtype=np.float32)
    )
    return {
        "agent_id": agent_id,
        "all_infos": info_dict or {},
        "all_obs": obs_dict or {},
        "all_actions": actions or {},
        "global_state": global_vector,
        "last_step_facts": getattr(env, "last_step_facts", None),
    }


def _collect_eval_agent_states(env: Any, agent_ids: List[str]) -> Dict[str, Any]:
    states: Dict[str, Any] = {}
    for aid in agent_ids:
        try:
            states[aid] = env.get_agent_state(aid)
        except Exception:
            continue
    return states


def _run_eval(
    scenario: Dict,
    args: argparse.Namespace,
    console: "ConsoleLogger",
    scenario_dir: Path,
) -> None:
    """Evaluate a trained PPO or MAPPO checkpoint with deterministic actions."""
    from agents.mappo import MAPPOAgent
    from agents.ppo import PPOAgent
    from metrics.racing_eval import (
        aggregate_eval_episodes,
        create_episode_facts,
        finalize_episode_facts,
        update_agent_step_facts,
    )
    from training.marl_trainer import map_mappo_learning_rewards
    from utils.torch_io import resolve_device

    checkpoint = args.checkpoint
    if not checkpoint:
        console.print_error("--eval requires --checkpoint")
        sys.exit(1)

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        console.print_error(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    agent_configs = scenario.get("agents", {})
    trainable_ids = get_trainable_agent_ids(agent_configs)
    if not trainable_ids:
        console.print_error(
            "--eval requires at least one trainable agent in the scenario."
        )
        sys.exit(1)

    trainable_algos = {
        str(agent_configs[aid].get("algorithm", "")).strip().lower()
        for aid in trainable_ids
    }
    if trainable_algos == {"ppo"} and len(trainable_ids) == 1:
        algorithm = "ppo"
    elif trainable_algos == {"mappo"}:
        algorithm = "mappo"
    else:
        console.print_error(
            "--eval currently supports one PPO trainable agent or one MAPPO "
            f"trainable team; found algorithms={sorted(trainable_algos)} "
            f"trainable={trainable_ids}."
        )
        sys.exit(1)

    focal_agent_id = trainable_ids[0]
    focal_cfg = agent_configs[focal_agent_id]
    exp_cfg = scenario.get("experiment", {})
    env_cfg = scenario.get("environment", {})
    eval_episodes = (
        args.eval_episodes
        if args.eval_episodes is not None
        else int(exp_cfg.get("episodes", 1))
    )
    eval_episodes = max(1, int(eval_episodes))
    base_seed = int(exp_cfg.get("seed", 0) or 0)
    render = bool(env_cfg.get("render", False))
    action_repeat = int(env_cfg.get("action_repeat", 1))

    env, agents, _ = create_training_setup(
        scenario, mode="eval", scenario_dir=scenario_dir
    )
    action_space = env.action_spaces.get(focal_agent_id)
    if action_space is None:
        console.print_error(f"RL agent '{focal_agent_id}' not in env action_spaces.")
        sys.exit(1)

    action_low = action_space.low
    action_high = action_space.high
    action_dim = len(action_low)
    obs_composers = build_obs_composers(
        agent_configs, trainable_ids, env_cfg, scenario_dir, action_dim=action_dim
    )
    reward_composers = build_reward_composers(agent_configs, trainable_ids, scenario_dir)
    params = resolve_training_params(focal_cfg, scenario)
    if algorithm == "mappo":
        params = {**params, **resolve_mappo_config(scenario)}
    action_composers = {
        aid: ActionComposer.from_config(
            env.action_spaces[aid].low,
            env.action_spaces[aid].high,
            agent_configs[aid].get("action_constraints", {}),
        )
        for aid in trainable_ids
    }

    # Probe the env once so MAPPO can size the centralized critic before
    # loading the checkpoint.  Episode 0 is reset again below with the same seed.
    env.reset(seed=base_seed)
    global_state_dim = int(env.get_global_state().vector.shape[0])

    if algorithm == "ppo":
        agent = PPOAgent(
            obs_dim=obs_composers[focal_agent_id].obs_dim,
            action_low=action_low,
            action_high=action_high,
            params=params,
        )
    else:
        agent = MAPPOAgent(
            obs_dim=obs_composers[focal_agent_id].obs_dim,
            global_state_dim=global_state_dim,
            action_low=action_low,
            action_high=action_high,
            agent_ids=trainable_ids,
            params=params,
        )
    from utils.torch_io import safe_load
    checkpoint_payload = safe_load(str(checkpoint_path), map_location="cpu")
    stored_provenance = (
        checkpoint_payload.get("provenance")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    if isinstance(stored_provenance, dict):
        current_provenance = build_run_provenance(
            scenario,
            scenario_path=args.scenario,
            run_id="evaluation",
            algorithm=algorithm,
            trainable_agents=trainable_ids,
        )
        mismatches = provenance_mismatches(stored_provenance, current_provenance)
        if mismatches and not args.allow_provenance_mismatch:
            raise ValueError(
                "Checkpoint provenance does not match the evaluation scenario: "
                + "; ".join(mismatches)
                + ". Pass --allow-provenance-mismatch only for an intentional cross-scenario evaluation."
            )
        if mismatches:
            console.print_warning("Checkpoint provenance mismatch explicitly allowed: " + "; ".join(mismatches))
        else:
            console.print_info("Checkpoint provenance matches scenario/config/map hashes.")
    else:
        console.print_warning(
            "Checkpoint has no provenance block; scenario/config/map compatibility cannot be verified."
        )
    agent.load(str(checkpoint_path))
    agent.actor.eval()
    agent.critic.eval()

    trainable_set = set(trainable_ids)
    other_agents = {aid: ag for aid, ag in agents.items() if aid not in trainable_set}
    for ag in other_agents.values():
        if hasattr(ag, "set_env"):
            ag.set_env(env)

    device_str = str(resolve_device([params.get("device", "cpu")]))
    fixed_str = ", ".join(other_agents) or "none"
    trainable_str = ", ".join(trainable_ids)
    console.print_header(
        f"Evaluation: {exp_cfg.get('name', 'unnamed')}",
        f"algorithm={algorithm}  trainable=({trainable_str})  fixed=({fixed_str})",
    )
    console.print_info(
        f"checkpoint={checkpoint_path}  episodes={eval_episodes}  "
        f"seed={base_seed}  device={device_str}  obs_dim={obs_composers[focal_agent_id].obs_dim}  "
        f"action_dim={action_dim}"
    )

    all_agent_ids = list(getattr(env, "possible_agents", list(agent_configs)))
    opponent_ids = [aid for aid in all_agent_ids if aid not in trainable_set]
    target_id = str(focal_cfg.get("target_id", "") or "")
    opponent_agent_id = target_id if target_id in opponent_ids else (opponent_ids[0] if opponent_ids else None)
    eval_episodes_facts = []

    try:
        for episode in range(eval_episodes):
            obs_dict, info_dict = env.reset(seed=base_seed + episode)
            for composer in obs_composers.values():
                composer.reset()
            for composer in reward_composers.values():
                composer.reset()
            for ag in other_agents.values():
                if hasattr(ag, "reset"):
                    ag.reset()

            wrapped_obs: Dict[str, np.ndarray] = {
                aid: obs_composers[aid].wrap(
                    obs_dict.get(aid, {}),
                    info_dict.get(aid, {}),
                )
                for aid in trainable_ids
            }
            episode_facts = create_episode_facts(
                episode=episode,
                agent_ids=all_agent_ids,
                trainable_ids=trainable_ids,
                opponent_ids=opponent_ids,
            )
            env_steps = 0

            while True:
                active_agents = set(getattr(env, "agents", list(obs_dict)))
                if not active_agents:
                    break

                actions_norm: Dict[str, np.ndarray] = {}
                actions_phys: Dict[str, np.ndarray] = {}
                for aid in trainable_ids:
                    if aid not in active_agents or aid not in wrapped_obs:
                        continue
                    act_result = agent.act(wrapped_obs[aid], deterministic=True)
                    action_norm = np.asarray(act_result[0], dtype=np.float32)
                    actions_norm[aid] = action_norm
                    actions_phys[aid] = action_composers[aid].process(action_norm)

                actions = _build_eval_actions(
                    actions_phys,
                    other_agents,
                    obs_dict,
                    active_agents=active_agents,
                )
                if not actions:
                    break

                for _ in range(max(1, action_repeat)):
                    obs_dict, _rew_dict, term_dict, trunc_dict, info_dict = env.step(actions)
                    step_facts = getattr(env, "last_step_facts", None)
                    post_step_global_state = getattr(
                        step_facts, "global_state", None
                    )
                    if post_step_global_state is None:
                        post_step_global_state = env.get_global_state()
                    env_steps += 1
                    if render:
                        try:
                            env.render()
                        except Exception:
                            pass

                    update_agent_step_facts(
                        episode_facts,
                        step_idx=env_steps,
                        infos=info_dict,
                        terminations=term_dict,
                        truncations=trunc_dict,
                        agent_states=_collect_eval_agent_states(env, all_agent_ids),
                    )

                    substep_individual_rewards: Dict[str, float] = {}
                    for aid, action_norm in actions_norm.items():
                        if aid not in trainable_set:
                            continue
                        agent_done = bool(term_dict.get(aid, False) or trunc_dict.get(aid, False))
                        sub_step_info = {
                            "obs": wrapped_obs.get(aid, {}),
                            "next_obs": obs_dict.get(aid, {}),
                            "info": info_dict.get(aid, {}),
                            "done": agent_done,
                            "terminated": bool(term_dict.get(aid, False)),
                            "truncated": bool(trunc_dict.get(aid, False)),
                            "action": action_norm,
                            "timestep": env.timestep,
                        }
                        sub_step_info.update(
                            _build_eval_reward_context(
                                env,
                                agent_id=aid,
                                info_dict=info_dict,
                                obs_dict=obs_dict,
                                actions=actions,
                                global_state=post_step_global_state,
                            )
                        )
                        sub_reward, breakdown = reward_composers[aid].compute(sub_step_info)
                        facts = episode_facts.agents[aid]
                        facts.individual_reward_total += float(sub_reward)
                        substep_individual_rewards[aid] = float(sub_reward)
                        for name, value in breakdown.items():
                            facts.reward_components[name] = (
                                facts.reward_components.get(name, 0.0) + float(value)
                            )

                    learning_rewards = map_mappo_learning_rewards(
                        substep_individual_rewards,
                        trainable_ids=trainable_ids,
                        reward_mode=str(params.get("reward_mode", "individual")),
                        team_reward_reduction=str(
                            params.get("team_reward_reduction", "mean")
                        ),
                    )
                    for aid, learning_reward in learning_rewards.items():
                        episode_facts.agents[aid].reward_total += learning_reward

                    active_after_step = set(getattr(env, "agents", []))
                    if not active_after_step or not set(actions).issubset(active_after_step):
                        break

                for aid in trainable_ids:
                    if aid not in getattr(env, "agents", []):
                        continue
                    wrapped_obs[aid] = obs_composers[aid].wrap(
                        obs_dict.get(aid, {}),
                        info_dict.get(aid, {}),
                    )
                    if aid in actions_norm:
                        obs_composers[aid].update_prev_action(actions_norm[aid])

            finalize_episode_facts(episode_facts)
            eval_episodes_facts.append(episode_facts)
            episode_summary = aggregate_eval_episodes(
                [episode_facts],
                focal_agent_id=focal_agent_id,
                opponent_agent_id=opponent_agent_id,
            )
            focal_outcome = episode_facts.agents[focal_agent_id].outcome
            reward_total = sum(
                episode_facts.agents[aid].reward_total
                for aid in trainable_ids
                if aid in episode_facts.agents
            )
            if not opponent_ids and len(trainable_ids) > 1:
                win_value = episode_summary.get("team_both_finished_rate", 0.0)
            else:
                win_value = episode_summary.get(
                    "team_win_rate", episode_summary.get("win_rate", 0.0)
                )

            console.print_info(
                f"eval ep {episode + 1:4d}/{eval_episodes}  "
                f"reward={reward_total:+.2f}  steps={env_steps:5d}  "
                f"win={win_value:.0f}  outcome={focal_outcome}"
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary = aggregate_eval_episodes(
        eval_episodes_facts,
        focal_agent_id=focal_agent_id,
        opponent_agent_id=opponent_agent_id,
    )
    if not opponent_ids and len(trainable_ids) > 1:
        summary["success_rate"] = summary.get("team_both_finished_rate", 0.0)
    elif "team_win_rate" in summary:
        summary["success_rate"] = summary["team_win_rate"]
    elif "win_rate" in summary:
        summary["success_rate"] = summary["win_rate"]

    console.print_summary(summary)


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
    from src.replay.replay_buffer import ReplayBuffer
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
    run_id="run",
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
        run_id=run_id,
        reward_mode=str(params.get("reward_mode", "individual")),
        team_reward_reduction=str(params.get("team_reward_reduction", "mean")),
    )

    console.print_info(
        f"Starting MAPPO training for {n_episodes} episodes "
        f"| agents={trainable_ids} | obs_dim={obs_dim} | global_state_dim={global_state_dim}"
    )
    console.print_info(
        "MAPPO contract: "
        f"reward_mode={params.get('reward_mode')}  "
        f"critic_mode={params.get('critic_mode')}  "
        f"team_reward_reduction={params.get('team_reward_reduction')}"
    )
    trainer.train(n_episodes=n_episodes)


if __name__ == "__main__":
    main()
