#!/usr/bin/env python3
"""Unified training entry point — dispatches to the right trainer based on scenario algorithm."""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.scenario import ScenarioError, load_and_expand_scenario, resolve_evaluation_protocol, resolve_mappo_config, validate_scenario
from core.setup import (
    build_obs_composer, build_obs_composers,
    build_reward_composer, build_reward_composers,
    create_training_setup, resolve_training_params,
)
from core.run_id import resolve_run_id, set_run_id_env
from core.provenance import build_run_provenance, provenance_mismatches
from core.agent_builder import get_trainable_agent_ids
from loggers.console import ConsoleLogger
from loggers.csv_logger import CSVLogger
from loggers.wandb_logger import WandbLogger
from wrappers.actions.composer import ActionComposer
from training.hooks import (
    CSVHook,
    CheckpointHook,
    ConsoleHook,
    CurriculumHook,
    EvaluationCheckpointHook,
    WandbHook,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F110 RL training")
    p.add_argument("--scenario", required=True, help="Path to scenario YAML file")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    budget_args = p.add_mutually_exclusive_group()
    budget_args.add_argument("--episodes", type=int, default=None)
    budget_args.add_argument("--total-steps", type=int, default=None,
                             help="Aggregate environment-decision budget for PPO/MAPPO")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Positive per-episode physics-step limit for bounded testing; also caps evaluation")
    p.add_argument("--num-envs", type=int, default=None,
                   help="Parallel CPU environments for PPO or MAPPO training")
    p.add_argument("--num-workers", type=int, default=None,
                   help="PPO/MAPPO CPU worker processes; capped at num-envs")
    p.add_argument("--rollout-steps-per-env", type=int, default=None,
                   help="Decisions per environment per rollout; PPO pools num-envs times this value")
    p.add_argument("--collector-scheduling", choices=("synchronous", "ready"), default=None,
                   help="Serve all collectors together or serve ready workers without a per-step barrier")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="Parent PyTorch CPU threads; parallel collectors use one each")
    p.add_argument("--eval", action="store_true", help="Run evaluation instead of training")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint file or run directory (best_model.pt): evaluate with --eval, "
                        "or initialize PPO training weights with a fresh optimizer; "
                        "overrides experiment.checkpoint in the scenario")
    p.add_argument("--pretrained-actor", type=str, default=None,
                   help="PPO checkpoint file or run directory to initialize a MAPPO actor; fresh critic and optimizer")
    p.add_argument(
        "--allow-provenance-mismatch",
        action="store_true",
        help="Allow --eval with a checkpoint from a different scenario/config/map contract.",
    )
    p.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes; defaults to --episodes")
    p.add_argument("--eval-protocol", choices=("selection", "final"), default=None,
                   help="Use fixed scenario evaluation seeds, episodes, and horizon; requires --eval.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--record-races", action="store_true",
                   help="MAPPO: sample shared race frames and event clips (recording YAML settings); "
                        "write to --dataset-dir or OUTPUT_DIR/behavior")
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
        scenario["experiment"].pop("total_steps", None)
    if getattr(args, "total_steps", None) is not None:
        scenario.setdefault("experiment", {})["total_steps"] = args.total_steps
    max_steps = getattr(args, "max_steps", None)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        scenario.setdefault("environment", {})["max_steps"] = max_steps
        if scenario.get("evaluation"):
            scenario["evaluation"]["max_steps"] = max_steps
    if args.wandb:
        scenario.setdefault("wandb", {})["enabled"] = True
    elif args.no_wandb:
        scenario.setdefault("wandb", {})["enabled"] = False
    if args.render:
        scenario.setdefault("environment", {})["render"] = True
    elif args.no_render:
        scenario.setdefault("environment", {})["render"] = False
    for name in ("num_envs", "num_workers", "torch_threads", "collector_scheduling"):
        value = getattr(args, name, None)
        if value is not None:
            scenario.setdefault("experiment", {})[name] = value
    horizon = getattr(args, "rollout_steps_per_env", None)
    if horizon is not None:
        if horizon <= 0:
            raise ValueError("--rollout-steps-per-env must be positive")
        num_envs = int(scenario.get("experiment", {}).get("num_envs", 1))
        for cfg in scenario.get("agents", {}).values():
            if cfg.get("trainable", False) and cfg.get("algorithm") in {"ppo", "mappo"}:
                cfg.setdefault("params", {})["n_steps"] = horizon * (num_envs if cfg["algorithm"] == "ppo" else 1)
        scenario.setdefault("training_defaults", {})["rollout_steps_per_env"] = horizon
    return scenario


def _resolve_scenario_relative_path(value: str, scenario_dir: Path) -> Path:
    """Resolve checkpoint/config paths relative to the declaring scenario."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (scenario_dir / path).resolve()


def resolve_checkpoint_path(value: str) -> Path:
    """CLI paths are relative to cwd; a run directory selects its best model."""
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


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
    from core.setup import (
    build_obs_composer, build_obs_composers,
    build_reward_composer, build_reward_composers,
    create_training_setup, resolve_training_params,
)

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
    if args.eval_protocol and (not args.eval or args.eval_episodes is not None):
        raise ValueError("--eval-protocol requires --eval and uses fixed episodes; omit --eval-episodes.")

    try:
        # Validate the effective configuration after CLI overrides, so a local
        # --num-envs 1 check can override a scenario sized for a larger machine.
        scenario = load_and_expand_scenario(args.scenario, validate=False)
    except (ScenarioError, FileNotFoundError) as exc:
        console.print_error(f"Failed to load scenario: {exc}")
        sys.exit(1)

    scenario = apply_cli_overrides(scenario, args)
    try:
        validate_scenario(scenario)
    except ScenarioError as exc:
        console.print_error(f"Invalid scenario after CLI overrides: {exc}")
        sys.exit(1)
    if scenario["experiment"].get("torch_threads") is not None:
        import torch
        torch.set_num_threads(scenario["experiment"]["torch_threads"])
    scenario_dir = Path(args.scenario).resolve().parent
    if args.checkpoint is None:
        configured_checkpoint = scenario["experiment"].get("checkpoint")
        if configured_checkpoint is not None:
            args.checkpoint = str(_resolve_scenario_relative_path(
                configured_checkpoint, scenario_dir
            ))

    agent_configs = scenario.get("agents", {})
    exp_cfg = scenario.get("experiment", {})
    env_cfg = scenario.get("environment", {})

    # Scenario validation guarantees a single PPO agent or a homogeneous MAPPO team.
    trainable_ids = get_trainable_agent_ids(agent_configs)

    if args.pretrained_actor and (args.eval or not trainable_ids or any(
            str(agent_configs[aid]["algorithm"]).lower() != "mappo" for aid in trainable_ids)):
        raise ValueError("--pretrained-actor requires MAPPO training; use --checkpoint for evaluation.")
    if exp_cfg.get("evaluation_only") and not args.eval:
        raise ValueError("This scenario is evaluation-only; pass --eval and --checkpoint.")
    if scenario.get("two_team", {}).get("enabled", False):
        _run_two_team(scenario, args, console, scenario_dir)
        return
    if args.eval:
        _run_eval(scenario, args, console, scenario_dir)
        return

    initial_checkpoint = None
    if args.checkpoint:
        if len(trainable_ids) != 1 or str(
            agent_configs[trainable_ids[0]]["algorithm"]
        ).strip().lower() != "ppo":
            raise ValueError("--checkpoint for training currently supports one PPO learner only.")
        initial_checkpoint = resolve_checkpoint_path(args.checkpoint)

    if not trainable_ids:
        _run_heuristic(scenario, args, console)
        return

    rl_agent_id = trainable_ids[0]
    algorithm = str(agent_configs[rl_agent_id]["algorithm"]).strip().lower()
    selective_recording = args.record_races or scenario.get('recording', {}).get('enabled', False)
    if selective_recording:
        if algorithm != 'mappo':
            raise ValueError("Selective race recording requires MAPPO training")
        from replay.race_recorder import recording_config
        scenario['recording'] = recording_config({**scenario.get('recording', {}), 'enabled': True})

    agent_cfg = agent_configs[rl_agent_id]

    run_id = args.run_id or resolve_run_id(
        scenario_name=exp_cfg.get("name"),
        algorithm=algorithm,
        seed=exp_cfg.get("seed"),
    )
    output_dir = args.output_dir or os.path.join(
        "outputs", exp_cfg.get("name", "unnamed"), run_id
    )
    if initial_checkpoint is not None and Path(output_dir).resolve() == initial_checkpoint.parent:
        raise ValueError("Use a new --output-dir for PPO transfer to preserve the source checkpoints.")
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
    if algorithm != "mappo":
        trainable_ids = [rl_agent_id]
    # else: trainable_ids already holds the full list from get_trainable_agent_ids()

    obs_composers = build_obs_composers(agent_configs, trainable_ids, env_cfg, scenario_dir, action_dim=2)
    reward_composers = build_reward_composers(agent_configs, trainable_ids, scenario_dir)
    obs_composer = obs_composers[rl_agent_id]
    reward_composer = reward_composers[rl_agent_id]

    # Training params (needed before banner so we can show device)
    params = resolve_training_params(agent_cfg, scenario)
    params["_observation_contract"] = (
        obs_composer.contract if params["_physics_contract"] is not None else None)
    if initial_checkpoint is not None:
        params["_initial_checkpoint"] = str(initial_checkpoint)
        params["_initial_checkpoint_sha256"] = hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest()
    if algorithm == "mappo":
        params = {**params, **resolve_mappo_config(scenario)}
        obs_dims = {aid: obs_composers[aid].obs_dim for aid in trainable_ids}
        if len(set(obs_dims.values())) != 1:
            raise ValueError(
                "Shared MAPPO actor requires identical local observation dimensions; "
                f"got {obs_dims}."
            )
        for aid in trainable_ids:
            agent_action_space = env.action_spaces.get(aid)
            if agent_action_space is None:
                raise ValueError(f"MAPPO agent '{aid}' has no environment action space.")
            if (
                agent_action_space.n != action_dim
                or not np.allclose(agent_action_space.low, action_low)
                or not np.allclose(agent_action_space.high, action_high)
            ):
                raise ValueError(
                    "Shared MAPPO actor requires identical normalized-to-physical "
                    f"action contracts; agent '{aid}' differs from '{rl_agent_id}'."
                )

    pretrained_actor_path: Optional[Path] = None
    if getattr(args, "pretrained_actor", None):
        params["pretrained_actor_checkpoint"] = str(resolve_checkpoint_path(args.pretrained_actor))
    pretrained_actor_value = params.get("pretrained_actor_checkpoint")
    if algorithm == "mappo" and pretrained_actor_value:
        pretrained_actor_path = _resolve_scenario_relative_path(
            str(pretrained_actor_value), scenario_dir
        )
        if not pretrained_actor_path.is_file():
            raise FileNotFoundError(
                f"Pretrained PPO actor checkpoint not found: {pretrained_actor_path}"
            )
        params["_resolved_pretrained_actor_checkpoint"] = str(pretrained_actor_path)

    if params.get("lora") is not None:
        if algorithm != "mappo":
            raise ValueError("LoRA is supported for MAPPO actor transfer only")
        if pretrained_actor_path is None:
            raise ValueError("LoRA training requires --pretrained-actor or pretrained_actor_checkpoint")

    # --- Startup banner ---
    maps_raw = env_cfg.get(
        "maps", env_cfg.get("map_bundles", env_cfg.get("map", "?"))
    )
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

    provenance = build_run_provenance(
        scenario,
        scenario_path=args.scenario,
        run_id=run_id,
        algorithm=algorithm,
        trainable_agents=trainable_ids,
    )
    num_envs = int(exp_cfg.get("num_envs", 1))
    if initial_checkpoint is not None:
        provenance["initial_checkpoint"] = {
            "path": str(initial_checkpoint),
            "sha256": params["_initial_checkpoint_sha256"],
            "load_scope": "actor_and_critic",
            "observation_extension": params.get("pretrained_observation_extension"),
            "optimizer_restored": False,
            "training_progress_restored": False,
        }
    if num_envs > 1 and algorithm == "ppo":
        env_seed = env_cfg.get("seed")
        env_seed = exp_cfg["seed"] if env_seed is None else env_seed
        provenance["ppo_collection"] = {
            "mode": ("synchronous_grouped_workers_v1" if (int(exp_cfg.get("num_workers", num_envs)) < num_envs
                       or exp_cfg.get("collector_scheduling") == "ready")
                     else "synchronous_workers_v1"),
            "num_envs": num_envs, "worker_threads": 1,
            "collector_scheduling": exp_cfg.get("collector_scheduling", "synchronous"),
            "num_workers": min(num_envs, int(exp_cfg.get("num_workers", num_envs))),
            "worker_seeds": [(exp_cfg["seed"] + i) % (2 ** 32) for i in range(num_envs)],
            "environment_seeds": [(env_seed + i) % (2 ** 32) for i in range(num_envs)],
            "max_steps_per_worker_rollout": int(params.get("n_steps", 2048)) // num_envs,
        }
    if num_envs > 1 and algorithm == "mappo":
        provenance["mappo_collection"] = {
            "mode": "synchronous_grouped_workers_v1", "num_envs": num_envs,
            "num_workers": min(num_envs, int(exp_cfg.get("num_workers", num_envs))),
            "worker_threads": 1,
            "collector_scheduling": exp_cfg.get("collector_scheduling", "synchronous"),
            "rollout_steps_per_env": scenario.get("training_defaults", {}).get("rollout_steps_per_env", 256),
            "environment_step_unit": "joint_environment_decisions_including_opponent_only_steps",
            "policy_seeds": ("parent RNG; arrival ordering" if exp_cfg.get("collector_scheduling") == "ready"
                             else "parent RNG; deterministic worker/environment ordering"),
            "environment_seeds": [((env_cfg.get("seed") if env_cfg.get("seed") is not None
                                    else exp_cfg["seed"]) + i) % (2 ** 32) for i in range(num_envs)],
        }
    if pretrained_actor_path is not None:
        provenance["pretrained_actor"] = {
            "path": str(pretrained_actor_path),
            "sha256": hashlib.sha256(pretrained_actor_path.read_bytes()).hexdigest(),
            "load_scope": "actor_only",
            "observation_extension": params.get("pretrained_actor_observation_extension"),
            "lora": params.get("lora"),
        }
    csv_logger = CSVLogger(
        output_dir=output_dir,
        scenario_config=scenario,
        provenance=provenance,
    )

    # Hooks
    eval_cfg = scenario.get("evaluation", {}) or {}
    if not isinstance(eval_cfg, dict):
        raise ValueError("Scenario 'evaluation' must be a mapping when provided.")
    evaluation_selection_enabled = bool(eval_cfg.get("enabled", False)) and algorithm in {"ppo", "mappo"}

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
            save_best_training_reward=not evaluation_selection_enabled,
            save_final=algorithm == "mappo",
            save_every_steps=(int(params.get("checkpoint_every_steps", 4096000))
                              if exp_cfg.get("total_steps") is not None or
                              (algorithm == "mappo" and params.get("checkpoint_every_steps") is not None)
                              else None),
        ),  # agent set below
    ]
    if wandb_logger:
        hooks.append(WandbHook(wandb_logger))

    if params.get("_physics_contract") is not None:
        from training.hooks import PhysicsEpisodeHook
        hooks.append(PhysicsEpisodeHook(output_dir, transition_records=algorithm != 'mappo'))

    # Optional dataset recording
    dataset_writer = None
    if args.dataset_dir or selective_recording:
        from replay.dataset_writer import DatasetWriter, DatasetHook, RaceDatasetWriter, RaceDatasetHook
        writer_class = RaceDatasetWriter if selective_recording else DatasetWriter
        dataset_path = args.dataset_dir or str(Path(output_dir) / 'behavior')
        dataset_writer = writer_class(
            output_dir=dataset_path,
            **({'config': scenario['recording']} if selective_recording else {'chunk_size': args.dataset_chunk_size}),
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
                "physics_contract": params.get("_physics_contract"),
                "action_contract": params["_action_contract"],
                "observation_contract": params.get("_observation_contract"),
                "global_state_dim": len(env.get_global_state().vector),
                "global_state_contract_version": env.get_global_state().metadata.get(
                    "vector_contract_version"
                ),
                "global_state_centerline_fields": list(
                    env.get_global_state().metadata.get("centerline_fields", ())
                ),
                "observation_dims": {
                    aid: composer.obs_dim for aid, composer in obs_composers.items()
                },
                "lifecycle_contract_version": "1.0",
                "team_return_mode": params.get("team_return_mode", "per_agent"),
                "team_reward_contract": reward_composer.team_contract,
                "transition_contract": {
                    "version": "1.0",
                    "global_state": "pre_decision",
                    "lifecycle_fields": "post_decision",
                },
                "mappo": (
                    resolve_mappo_config(scenario)
                    if algorithm == "mappo"
                    else None
                ),
            },
        )
        hooks.append((RaceDatasetHook if selective_recording else DatasetHook)(dataset_writer))
        console.print_info(f"Dataset recording → {dataset_path}  "
                           f"mode={'sampled races + event clips' if selective_recording else 'all learner transitions'}")

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

    action_composer = ActionComposer.from_config(
        action_low, action_high, action_constraints,
        decision_dt=float(env_cfg.get("timestep", 0.01)) * action_repeat,
    )

    try:
        if algorithm == "mappo":
            _run_mappo(
                env, trainable_ids, other_agents,
                obs_composers, reward_composers, action_composer, params,
                action_low, action_high, action_repeat, render,
                hooks, exp_cfg, output_dir, console,
                focal_agent_id=rl_agent_id,
                run_id=run_id,
                scenario=scenario, scenario_dir=scenario_dir,
                provenance=provenance, wandb_logger=wandb_logger,
            )
        elif algorithm == "ppo":
            _run_on_policy(
                env, rl_agent_id, agent_cfg, other_agents,
                obs_composer, reward_composer, action_composer, params,
                action_repeat, render, hooks, exp_cfg, output_dir, console,
                run_id=run_id,
                spawn_plan_fn=spawn_plan_fn,
                scenario=scenario,
                scenario_dir=scenario_dir,
                provenance=provenance,
                wandb_logger=wandb_logger,
            )
        else:
            console.print_error(f"Unknown algorithm: '{algorithm}'")
            sys.exit(1)
    finally:
        if dataset_writer is not None:
            if selective_recording:
                # A successful trainer already closed the writer. Exceptions
                # preserve the retained prefix and mark this dataset incomplete.
                dataset_writer.close(complete=False)
            else:
                dataset_writer.close()
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
    from training.reward_context import build_reward_context
    return build_reward_context(
        env=env, agent_id=agent_id, info_dict=info_dict, obs_dict=obs_dict,
        actions=actions, global_state=global_state,
    )


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
        episode_race_record,
        capture_spawn_context,
        update_agent_step_facts,
    )
    from training.marl_trainer import map_mappo_learning_rewards
    from utils.torch_io import resolve_device

    checkpoint = args.checkpoint
    if not checkpoint:
        console.print_error("--eval requires --checkpoint or experiment.checkpoint in the scenario")
        sys.exit(1)

    checkpoint_path = resolve_checkpoint_path(checkpoint)

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
    provenance_scenario = scenario
    protocol_name = getattr(args, "eval_protocol", None)
    if protocol_name:
        protocol = resolve_evaluation_protocol(scenario, protocol_name)
        scenario = copy.deepcopy(scenario)
        scenario["experiment"]["seed"] = protocol["seed"]
        scenario["experiment"]["episodes"] = protocol["episodes"]
        scenario["environment"]["max_steps"] = protocol["max_steps"]
        scenario.setdefault("evaluation", {})["max_steps"] = protocol["max_steps"]
        if "target_laps" in protocol:
            scenario["evaluation"]["target_laps"] = protocol["target_laps"]
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
    params["_observation_contract"] = (
        obs_composers[focal_agent_id].contract if params["_physics_contract"] is not None else None)
    if algorithm == "mappo":
        params = {**params, **resolve_mappo_config(scenario)}
    action_composers = {
        aid: ActionComposer.from_config(
            env.action_spaces[aid].low,
            env.action_spaces[aid].high,
            agent_configs[aid].get("action_constraints", {}),
            decision_dt=float(env_cfg.get("timestep", 0.01)) * action_repeat,
        )
        for aid in trainable_ids
    }
    from training.reward_context import validate_team_reward_composers
    from metrics.racing_eval import team_finish_result
    has_team_rewards = validate_team_reward_composers(
        reward_composers, trainable_ids=trainable_ids,
        opponent_ids=[aid for aid in agent_configs if aid not in trainable_ids],
        reward_mode=params.get("reward_mode", "individual"),
        critic_mode=params.get("critic_mode", "agent_conditioned"),
        team_return_mode=params.get("team_return_mode", "per_agent"),
        action_repeat=action_repeat,
    )

    # Probe the env once so MAPPO can size the centralized critic before
    # loading the checkpoint.  Episode 0 is reset again below with the same seed.
    env.reset(seed=base_seed)
    global_snapshot = env.get_global_state()
    global_state_dim = int(global_snapshot.vector.shape[0])

    if algorithm == "ppo":
        agent = PPOAgent(
            obs_dim=obs_composers[focal_agent_id].obs_dim,
            action_low=action_low,
            action_high=action_high,
            params=params,
        )
    else:
        params = {
            **params,
            "_global_state_contract_version": global_snapshot.metadata.get(
                "vector_contract_version", "legacy_unspecified"
            ),
        }
        agent = MAPPOAgent(
            obs_dim=obs_composers[focal_agent_id].obs_dim,
            global_state_dim=global_state_dim,
            action_low=action_low,
            action_high=action_high,
            agent_ids=trainable_ids,
            params=params,
        )
    from utils.torch_io import safe_load
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint_payload = safe_load(str(checkpoint_path), map_location="cpu")
    selection = checkpoint_payload.get('checkpoint_selection') or {}
    checkpoint_steps = checkpoint_payload.get('environment_steps', selection.get('environment_steps'))
    checkpoint_version = checkpoint_payload.get('policy_version', selection.get('policy_version'))
    stored_provenance = (
        checkpoint_payload.get("provenance")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    mismatches = []
    if isinstance(stored_provenance, dict):
        current_provenance = build_run_provenance(
            provenance_scenario,
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
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading for evaluation; use a stable checkpoint file.")
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

    run_id = args.run_id or resolve_run_id(
        scenario_name=exp_cfg.get("name"), algorithm=f"{algorithm}-eval", seed=base_seed)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / exp_cfg.get("name", "unnamed") / "evaluation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_provenance = build_run_provenance(scenario, scenario_path=args.scenario,
        run_id=run_id, algorithm=algorithm, trainable_agents=trainable_ids)
    recorded_protocol = dict(name=protocol_name or 'custom', seeds=list(range(base_seed, base_seed+eval_episodes)),
        max_steps=env.max_steps, target_laps=getattr(env, 'target_laps', None),
        timestep_s=env.timestep, action_repeat=action_repeat)
    from replay.evaluation_recorder import EvaluationRecording, evaluation_recording_config
    recording_cfg = evaluation_recording_config(scenario, requested=bool(
        getattr(args, 'record_races', False) or getattr(args, 'dataset_dir', None)))
    recording = None
    if recording_cfg is not None:
        recording = EvaluationRecording(getattr(args, 'dataset_dir', None) or output_dir/'behavior',
            config=recording_cfg, env=env, trainable_ids=trainable_ids, run_id=run_id,
            action_repeat=action_repeat, metadata=dict(algorithm=algorithm, provenance=evaluation_provenance,
                physics_contract=params.get('_physics_contract'),
                action_contract=ActionComposer.contract_from_config(focal_cfg.get('action_constraints', {}), env.timestep*action_repeat),
                observation_contracts={aid: obs_composers[aid].contract for aid in trainable_ids},
                checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_hash))
        recording.begin(dict(evaluation_id='standalone', checkpoint=str(checkpoint_path.resolve()),
            checkpoint_sha256=checkpoint_hash, policy_version=checkpoint_version,
            environment_steps=checkpoint_steps))
    recording_complete = False

    all_agent_ids = list(getattr(env, "possible_agents", list(agent_configs)))
    opponent_ids = [aid for aid in all_agent_ids if aid not in trainable_set]
    target_id = str(focal_cfg.get("target_id", "") or "")
    opponent_agent_id = target_id if target_id in opponent_ids else (opponent_ids[0] if opponent_ids else None)
    eval_episodes_facts = []
    eval_physics = {}
    eval_maps = {}
    eval_spawns = {}
    team_results = []

    try:
        for episode in range(eval_episodes):
            obs_dict, info_dict = env.reset(
                seed=base_seed + episode,
                options={"map_episode_index": episode},
            )
            eval_maps[episode] = info_dict.get(focal_agent_id, {}).get("map_bundle") or getattr(
                env, "_map_bundle_active", None)
            eval_spawns[episode] = capture_spawn_context(env, all_agent_ids)
            if recording is not None:
                recording.start(episode, info_dict, protocol=recorded_protocol)
            for composer in obs_composers.values():
                composer.reset()
            for composer in reward_composers.values():
                composer.reset()
            for composer in action_composers.values():
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
            decision_index = 0
            episode_team_reward = 0.0

            while True:
                active_agents = set(getattr(env, "agents", list(obs_dict)))
                if not active_agents:
                    break

                actions_norm: Dict[str, np.ndarray] = {}
                actions_phys: Dict[str, np.ndarray] = {}
                active_trainable_ids = [
                    aid
                    for aid in trainable_ids
                    if aid in active_agents and aid in wrapped_obs
                ]
                if active_trainable_ids and algorithm == "mappo":
                    stacked_observations = np.stack(
                        [wrapped_obs[aid] for aid in active_trainable_ids], axis=0
                    )
                    actions_norm, _ = agent.act_batch(
                        active_trainable_ids,
                        stacked_observations,
                        deterministic=True,
                    )
                    actions_phys = {
                        aid: action_composers[aid].process(action)
                        for aid, action in actions_norm.items()
                    }
                else:
                    for aid in active_trainable_ids:
                        action_norm = agent.predict(wrapped_obs[aid])
                        actions_norm[aid] = action_norm
                        actions_phys[aid] = action_composers[aid].process(
                            action_norm
                        )

                actions = _build_eval_actions(
                    actions_phys,
                    other_agents,
                    obs_dict,
                    active_agents=active_agents,
                )
                if not actions:
                    break

                for substep in range(max(1, action_repeat)):
                    if recording is not None:
                        recording.before_step(info_dict, obs_dict, env_steps)
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
                    recorded_components, team_components = {}, {}
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
                        if recording is not None:
                            recorded_components[aid] = dict(breakdown)
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
                    if has_team_rewards:
                        team_context = _build_eval_reward_context(
                            env, agent_id=focal_agent_id, info_dict=info_dict,
                            obs_dict=obs_dict, actions=actions,
                            global_state=post_step_global_state,
                        )
                        bonus, team_components = reward_composers[focal_agent_id].compute(team_context, team=True)
                        for aid in learning_rewards:
                            learning_rewards[aid] += bonus
                    if learning_rewards and params.get("team_return_mode") == "joint":
                        episode_team_reward += next(iter(learning_rewards.values()))
                    for aid, learning_reward in learning_rewards.items():
                        episode_facts.agents[aid].reward_total += learning_reward

                    if recording is not None:
                        recording.step(infos=info_dict, obs=obs_dict, physical=actions, normalized=actions_norm,
                            wrapped=wrapped_obs, physics_index=env_steps-1, decision_index=decision_index, substep=substep,
                            terminated=term_dict, truncated=trunc_dict, rewards=learning_rewards,
                            individual_rewards=substep_individual_rewards, components=recorded_components,
                            team_components=team_components)

                    active_after_step = set(getattr(env, "agents", []))
                    if not active_after_step or not set(actions).issubset(active_after_step):
                        break

                decision_index += 1
                for aid in trainable_ids:
                    if aid not in getattr(env, "agents", []):
                        continue
                    if aid in actions_norm:
                        obs_composers[aid].update_prev_action(actions_norm[aid])
                    wrapped_obs[aid] = obs_composers[aid].wrap(
                        obs_dict.get(aid, {}),
                        info_dict.get(aid, {}),
                    )

            if recording is not None:
                recording.end()
            finalize_episode_facts(episode_facts)
            if info_dict.get(focal_agent_id, {}).get("physics") is not None:
                eval_physics[episode_facts.episode] = info_dict[focal_agent_id]["physics"]
            if has_team_rewards:
                team_results.append({
                    **team_finish_result(info_dict, trainable_ids, opponent_ids),
                    "team_episode_reward": episode_team_reward,
                })
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
            if has_team_rewards:
                objective = reward_composers[focal_agent_id].team_contract[0]["objective"]
                win_value = team_results[-1]["both_finished" if objective == "combined" else objective]

            console.print_info(
                f"eval ep {episode + 1:4d}/{eval_episodes}  "
                f"reward={reward_total:+.2f}  steps={env_steps:5d}  "
                f"win={win_value:.0f}  outcome={focal_outcome}"
            )
        recording_complete = True
    finally:
        if recording is not None:
            recording.close(complete=recording_complete)
        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary = aggregate_eval_episodes(
        eval_episodes_facts,
        focal_agent_id=focal_agent_id,
        opponent_agent_id=opponent_agent_id,
        timestep=float(env.timestep),
    )
    if not opponent_ids and len(trainable_ids) > 1:
        summary["success_rate"] = summary.get("team_both_finished_rate", 0.0)
    elif "team_win_rate" in summary:
        summary["success_rate"] = summary["team_win_rate"]
    elif "win_rate" in summary:
        summary["success_rate"] = summary["win_rate"]
    if has_team_rewards:
        objective = reward_composers[focal_agent_id].team_contract[0]["objective"]
        summary["team_objective"] = objective
        summary["team_result_means"] = {
            key: float(np.mean([row[key] for row in team_results])) for key in team_results[0]
        }
        summary["team_results_by_episode"] = team_results
        summary["success_rate"] = summary["team_result_means"][
            "both_finished" if objective == "combined" else objective
        ]

    if len(trainable_ids) == 2 and len(opponent_ids) == 2:
        # Historical win/success aliases can mean beating just one opponent.
        # Keep them in the report for compatibility, but use explicit headlines.
        console.print_summary({key + "_rate" if key in {"team_first_place", "team_sweep"} else key: summary[key] for key in (
            "race_count", "team_both_finished_rate", "both_finished_count",
            "at_least_one_finished_count", "team_first_place", "first_place_count",
            "team_sweep", "sweep_count", "team_rank_score", "any_learner_collision_dnf_count",
            "mean_clean_finish_time_s", "finish_time_sample_count", "valid_laps",
            "mean_valid_lap_time_s") if key in summary}, title="Evaluation Summary")
    else:
        console.print_summary(summary, title="Evaluation Summary")
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_provenance": stored_provenance,
        "provenance_mismatches": mismatches,
        "evaluation_provenance": evaluation_provenance,
        "protocol": protocol_name or "custom",
        "environment_steps": checkpoint_steps,
        "seeds": list(range(base_seed, base_seed + eval_episodes)),
        "max_steps": env.max_steps,
        "target_laps": getattr(env, 'target_laps', None),
        "timestep_s": float(env.timestep),
        "horizon_s": env.max_steps * float(env.timestep) if env.max_steps > 0 else None,
        "action_repeat": action_repeat,
        "summary": summary,
        "per_map": {
            map_name: aggregate_eval_episodes(
                [facts for facts in eval_episodes_facts if eval_maps[facts.episode] == map_name],
                focal_agent_id=focal_agent_id, opponent_agent_id=opponent_agent_id,
                timestep=float(env.timestep),
            ) for map_name in sorted({name for name in eval_maps.values() if name is not None})
        },
        "episode_results": [
            {"seed": base_seed + facts.episode,
             "map_bundle": eval_maps[facts.episode],
             "race_record": {**episode_race_record(facts, timestep=float(env.timestep)),
                             "phase": "evaluation", "run_id": run_id,
                             "environment_id": "evaluation", "environment_episode": facts.episode,
                             "environment_seed": base_seed + facts.episode,
                             "map_id": eval_maps[facts.episode],
                             "spawn_configuration": eval_spawns[facts.episode],
                             "action_repeat": action_repeat,
                             "episode_id": f"{run_id}_standalone_ep{facts.episode:06d}",
                             "checkpoint_sha256": checkpoint_hash},
             **({"physics": eval_physics[facts.episode]} if facts.episode in eval_physics else {}),
             **aggregate_eval_episodes(
                [facts], focal_agent_id=focal_agent_id,
                opponent_agent_id=opponent_agent_id, timestep=float(env.timestep),
            )}
            for facts in eval_episodes_facts
        ],
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print_info(f"Evaluation report: {report_path}")


def _configure_evaluation_recording(evaluator, scenario, output_dir, run_id, provenance):
    from replay.evaluation_recorder import EvaluationRecording, evaluation_recording_config
    config = evaluation_recording_config(scenario)
    if config is None:
        return
    ids = getattr(evaluator, 'trainable_ids', None) or [evaluator.rl_agent_id]
    composers = getattr(evaluator, 'obs_composers', None) or {ids[0]: evaluator.obs_composer}
    env = evaluator.env
    evaluator.recording = EvaluationRecording(Path(output_dir)/'evaluation_behavior', config=config,
        env=env, trainable_ids=ids, run_id=run_id, action_repeat=evaluator.action_repeat,
        metadata=dict(algorithm=scenario['agents'][ids[0]]['algorithm'], provenance=provenance or {},
            physics_contract=(provenance or {}).get('physics_contract'),
            action_contract=ActionComposer.contract_from_config(
                scenario['agents'][ids[0]].get('action_constraints', {}), env.timestep*evaluator.action_repeat),
            observation_contracts={aid: composer.contract for aid, composer in composers.items()},
            reward_availability='not_computed_by_selection_evaluator'))


def _run_on_policy(
    env, rl_agent_id, agent_cfg, other_agents,
    obs_composer, reward_composer, action_composer, params,
    action_repeat, render, hooks, exp_cfg, output_dir, console,
    run_id: str = "run",
    spawn_plan_fn=None,
    scenario: Optional[Dict[str, Any]] = None,
    scenario_dir: Optional[Path] = None,
    provenance: Optional[Dict[str, Any]] = None,
    wandb_logger: Optional[WandbLogger] = None,
) -> None:
    from agents.ppo import PPOAgent
    from training.on_policy_trainer import OnPolicyTrainer

    n_episodes = int(exp_cfg.get("episodes", 1000))

    action_space = env.action_spaces.get(rl_agent_id)
    evaluator = None
    eval_cfg = (scenario or {}).get("evaluation", {}) or {}
    if bool(eval_cfg.get("enabled", False)):
        from training.ppo_evaluator import DeterministicPPOEvaluator

        if scenario_dir is None:
            raise ValueError("Evaluation checkpoint selection requires a scenario directory.")
        eval_scenario = copy.deepcopy(scenario)
        selection_protocol = resolve_evaluation_protocol(scenario, "selection")
        eval_seed = selection_protocol["seed"]
        eval_scenario.setdefault("experiment", {})["seed"] = eval_seed
        eval_scenario.setdefault("environment", {})["max_steps"] = selection_protocol["max_steps"]
        # Setup seeds process-global RNGs. Preserve the training streams while
        # still giving the evaluation environment its own deterministic seed.
        numpy_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        import torch
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            eval_env, eval_agents, _ = create_training_setup(
                eval_scenario, mode="eval", scenario_dir=scenario_dir
            )
        finally:
            np.random.set_state(numpy_rng_state)
            random.setstate(python_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        eval_action_space = eval_env.action_spaces.get(rl_agent_id)
        if eval_action_space is None:
            raise ValueError(f"Evaluation env has no action space for '{rl_agent_id}'.")
        if not (
            np.allclose(eval_action_space.low, action_space.low)
            and np.allclose(eval_action_space.high, action_space.high)
        ):
            raise ValueError("Training and evaluation action bounds must match.")
        eval_obs_composer = build_obs_composer(
            agent_cfg,
            eval_scenario["environment"],
            scenario_dir,
            action_dim=len(eval_action_space.low),
        )
        if eval_obs_composer.obs_dim != obs_composer.obs_dim:
            raise ValueError("Training and evaluation observation dimensions must match.")
        eval_action_composer = ActionComposer.from_config(
            eval_action_space.low,
            eval_action_space.high,
            agent_cfg.get("action_constraints", {}),
            decision_dt=float(eval_scenario["environment"].get("timestep", 0.01))
            * int(eval_scenario["environment"].get("action_repeat", 1)),
        )
        eval_other_agents = {
            aid: controller for aid, controller in eval_agents.items() if aid != rl_agent_id
        }
        for controller in eval_other_agents.values():
            if hasattr(controller, "set_env"):
                controller.set_env(eval_env)
        evaluator = DeterministicPPOEvaluator(
            env=eval_env,
            rl_agent_id=rl_agent_id,
            other_agents=eval_other_agents,
            obs_composer=eval_obs_composer,
            action_composer=eval_action_composer,
            episodes=int(eval_cfg.get("episodes", 8)),
            base_seed=eval_seed,
            action_repeat=int(eval_scenario["environment"].get("action_repeat", 1)),
        )

    agent = PPOAgent(
        obs_dim=obs_composer.obs_dim,
        action_low=action_space.low,
        action_high=action_space.high,
        params=params,
    )
    initial_checkpoint = params.get("_initial_checkpoint")
    if initial_checkpoint:
        agent.load(initial_checkpoint, load_optimizer=False,
                   observation_extension=params.get("pretrained_observation_extension"))
        if hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest() != params["_initial_checkpoint_sha256"]:
            raise ValueError("Checkpoint changed while loading for training; use a stable checkpoint file.")
        console.print_info(
            f"Initialized PPO actor and critic from {initial_checkpoint}; "
            "fresh optimizer, episode budget, and learning-rate schedule."
        )

    # Wire checkpoint hook to the agent now that we have it
    for hook in hooks:
        if hasattr(hook, "_agent") and hook._agent is None:
            hook._agent = agent

    curriculum = None
    checkpoint_hook_type = EvaluationCheckpointHook
    if evaluator is not None:
        evaluator.bind_agent(agent)
        _configure_evaluation_recording(evaluator, scenario, output_dir, run_id, provenance)
        if (scenario or {}).get("map_curriculum") is not None:
            from training.map_curriculum import MapCurriculum, CurriculumEvaluator, MapCurriculumCheckpointHook
            curriculum = MapCurriculum(scenario["environment"]["map_bundles_eval"],
                scenario["environment"]["map_bundles_train"][0], **scenario["map_curriculum"])
            evaluator = CurriculumEvaluator(evaluator, curriculum, env._map_scheduler, console)
            checkpoint_hook_type = MapCurriculumCheckpointHook
        hooks.append(
            checkpoint_hook_type(
                agent=agent,
                output_dir=output_dir,
                evaluator=evaluator,
                evaluate_every=int(eval_cfg.get("every_episodes", 100)),
                selection_strategy=eval_cfg.get("selection_strategy", "completion_safety"),
                evaluate_every_steps=eval_cfg.get("every_steps"),
                provenance=provenance,
                console=console,
                wandb_logger=wandb_logger,
            )
        )
        console.print_info(
            "Best-model selection: deterministic evaluation "
            + (f"every {eval_cfg['every_steps']} transitions " if eval_cfg.get("every_steps")
               else f"every {int(eval_cfg.get('every_episodes', 100))} episodes ") +
            f"over {int(eval_cfg.get('episodes', 8))} fixed-seed episodes; "
            f"strategy={eval_cfg.get('selection_strategy', 'completion_safety')}."
        )

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

    num_envs = int(exp_cfg.get("num_envs", 1))
    total_steps = exp_cfg.get("total_steps")
    budget = f"{total_steps} environment transitions" if total_steps is not None else f"{n_episodes} episodes"
    console.print_info(f"Starting PPO training for {budget} | num_envs={num_envs}")
    try:
        if num_envs > 1:
            trainer.train_parallel(scenario, scenario_dir, num_envs, n_episodes,
                                   **({"total_steps": total_steps} if total_steps is not None else {}))
        else:
            trainer.train(n_episodes=n_episodes,
                          **({"total_steps": total_steps} if total_steps is not None else {}))
        if curriculum is not None and curriculum.complete:
            # Training stops at the gate, so the in-memory policy is exactly
            # curriculum_passed.pt. Final results never select or tune weights.
            final_protocol = resolve_evaluation_protocol(scenario, "final")
            final_scenario = copy.deepcopy(scenario)
            final_scenario["experiment"]["seed"] = final_protocol["seed"]
            final_scenario["evaluation"]["max_steps"] = final_protocol["max_steps"]
            final_scenario["evaluation"]["target_laps"] = final_protocol.get("target_laps", eval_cfg["target_laps"])
            final_env, final_agents, _ = create_training_setup(
                final_scenario, mode="eval", scenario_dir=scenario_dir)
            try:
                final_evaluator = DeterministicPPOEvaluator(
                    env=final_env, rl_agent_id=rl_agent_id,
                    other_agents={aid: ctrl for aid, ctrl in final_agents.items() if aid != rl_agent_id},
                    obs_composer=eval_obs_composer, action_composer=eval_action_composer,
                    episodes=final_protocol["episodes"], base_seed=final_protocol["seed"],
                    action_repeat=action_repeat).bind_agent(agent)
                report = final_evaluator.evaluate()
                report["evaluation_protocol"]["name"] = "final"
                report["checkpoint"] = str(Path(output_dir) / "curriculum_passed.pt")
                report["all_maps_passed"] = all(
                    report["per_map"].get(name, {}).get("episodes", 0) > 0
                    and report["per_map"][name]["strict_clean_finish_count"]
                    / report["per_map"][name]["episodes"] >= curriculum.threshold
                    for name in curriculum.maps)
                report_path = Path(output_dir) / "curriculum_final_evaluation.json"
                report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                console.print_info(f"Final endurance validation: all_maps_passed={report['all_maps_passed']}; {report_path}")
            finally:
                final_env.close()
    finally:
        env.close()
        if evaluator is not None:
            evaluator.close()


def _run_two_team(scenario, args, console, scenario_dir):
    """Independent team policies, paired checkpoints, and symmetric W&B metrics."""
    from agents.mappo import MAPPOAgent
    from training.two_team import (TwoTeamTrainer, evaluate_pair, preserve_rng,
                                   resolve_teams, selection_score)

    selective_recording = bool(args.record_races or args.dataset_dir or scenario.get('recording', {}).get('enabled'))
    if selective_recording:
        from replay.race_recorder import recording_config
        scenario['recording'] = recording_config({**scenario.get('recording', {}), 'enabled': True})
    if args.pretrained_actor or scenario.get("training_defaults", {}).get("pretrained_actor_checkpoint"):
        raise ValueError("The two-team actor adds team/race observations; this scenario starts from scratch")
    if args.checkpoint and not args.eval:
        raise ValueError("Two-team --checkpoint currently supports evaluation only")
    if args.eval and not args.checkpoint:
        raise ValueError("Two-team evaluation requires --checkpoint pointing to a paired checkpoint directory")
    teams = resolve_teams(scenario)
    exp, cfg = scenario["experiment"], scenario["environment"]
    trainable_ids = [aid for ids in teams.values() for aid in ids]
    run_id = args.run_id or resolve_run_id(scenario_name=exp["name"], algorithm="mappo", seed=exp["seed"])
    set_run_id_env(run_id)
    output = Path(args.output_dir or Path("outputs") / exp["name"] / run_id)
    output.mkdir(parents=True, exist_ok=True)
    provenance = build_run_provenance(scenario, scenario_path=args.scenario, run_id=run_id,
                                      algorithm="mappo", trainable_agents=trainable_ids)
    num_envs = int(exp.get("num_envs", 1))
    if num_envs > 1 and not args.eval:
        env_seed = cfg.get("seed") if cfg.get("seed") is not None else exp["seed"]
        provenance["two_team_collection"] = {
            "mode": "synchronous_grouped_two_team_v1", "num_envs": num_envs,
            "num_workers": min(num_envs, int(exp.get("num_workers", num_envs))),
            "worker_threads": 1, "teams": teams,
            "rollout_steps_per_env": scenario.get("training_defaults", {}).get("rollout_steps_per_env", 256),
            "environment_step_unit": "joint_environment_decisions",
            "environment_seeds": [(env_seed + i) % (2 ** 32) for i in range(num_envs)],
        }
    CSVLogger(str(output), scenario, provenance=provenance)
    protocol = resolve_evaluation_protocol(scenario, args.eval_protocol or "selection")
    if args.eval and not args.eval_protocol:
        protocol["episodes"] = args.eval_episodes or args.episodes or protocol["episodes"]
        if args.eval_episodes is not None or args.episodes is not None:
            protocol["name"] = "custom"
    env, _, _ = create_training_setup(scenario, mode="eval" if args.eval else "train", scenario_dir=scenario_dir)
    logger, eval_env, race_writer, eval_writer = None, None, None, None
    recording_complete = False
    try:
        observations = build_obs_composers(scenario["agents"], trainable_ids, cfg, scenario_dir)
        rewards = build_reward_composers(scenario["agents"], trainable_ids, scenario_dir)
        actions = {aid: ActionComposer.from_config(env.action_spaces[aid].low,
                    env.action_spaces[aid].high, scenario["agents"][aid].get("action_constraints", {}),
                    decision_dt=env.timestep) for aid in trainable_ids}
        snapshot = env.get_global_state()
        agents = {}
        for team, ids in teams.items():
            params = {**resolve_training_params(scenario["agents"][ids[0]], scenario),
                      **resolve_mappo_config(scenario)}
            params["_global_state_contract_version"] = str(snapshot.metadata.get("vector_contract_version")) + "+race_clock_v1"
            params["_observation_contract"] = {"base": observations[ids[0]].contract,
                "appended": ["remaining_time_fraction", "remaining_lap_fraction"], "version": 1}
            space = env.action_spaces[ids[0]]
            agents[team] = MAPPOAgent(observations[ids[0]].obs_dim + 2, len(snapshot.vector) + 1,
                                      space.low, space.high, ids, params)

        from core.scenario import load_yaml_config
        resolved_rewards = {aid: load_yaml_config(scenario_dir / scenario["agents"][aid]["reward"])
                            if isinstance(scenario["agents"][aid]["reward"], str)
                            else scenario["agents"][aid]["reward"] for aid in trainable_ids}
        checkpoint_contract = {"version": 1, "teams": teams, "two_team": scenario["two_team"],
            "target_laps": env.target_laps, "max_steps": env.max_steps,
            "terminal_agents": cfg.get("terminal_agents"),
            "rewards": resolved_rewards}

        def save_pair(name, episode, summary=None):
            directory = output / name
            directory.mkdir(parents=True, exist_ok=True)
            # Numeric filenames keep arbitrary team names out of filesystem paths.
            files = {team: f"team_{index}.pt" for index, team in enumerate(teams)}
            for team, agent in agents.items():
                agent.save(str(directory / files[team]))
            (directory / "pair.json").write_text(json.dumps({
                "contract": checkpoint_contract, "files": files, "episode": episode,
                "environment_steps": trainer.environment_steps, "evaluation": summary,
                "team_policy_versions": {team: trainer.updates for team in teams},
                "provenance": provenance,
            }, indent=2) + "\n")

        if args.eval:
            directory = Path(args.checkpoint).expanduser().resolve()
            if not (directory / "pair.json").is_file() and (directory / "best_pair" / "pair.json").is_file():
                directory = directory / "best_pair"
            manifest = json.loads((directory / "pair.json").read_text())
            if manifest["contract"] != checkpoint_contract and not args.allow_provenance_mismatch:
                raise ValueError("Paired checkpoint race contract differs; use matching scenario or --allow-provenance-mismatch")
            if manifest["contract"]["teams"] != teams:
                raise ValueError("Paired checkpoint team membership differs")
            for team, agent in agents.items():
                agent.load(str(directory / manifest["files"][team]))

        wb = scenario.get("wandb", {})
        if wb.get("enabled", False):
            logger = WandbLogger(project=wb.get("project", "marl-f110-selfplay"),
                config={**scenario, "two_team_contract": checkpoint_contract,
                        "actor_observation_contract": params["_observation_contract"]},
                name=wb.get("name", run_id), group=wb.get("group"), entity=wb.get("entity"),
                job_type="two-team-evaluation" if args.eval else wb.get("job_type", "two-team-training"),
                tags=wb.get("tags"), notes=wb.get("notes"), mode=wb.get("mode", "online"), run_id=run_id)
            if logger.run is not None:
                logger.run.define_metric("selfplay/environment_steps")
                logger.run.define_metric("selfplay/*", step_metric="selfplay/environment_steps")
                logger.run.define_metric("selfplay_eval/environment_steps")
                logger.run.define_metric("selfplay_eval/*", step_metric="selfplay_eval/environment_steps")
                logger.run.define_metric("perf/*", step_metric="train/environment_steps")
                console.print_info(f"W&B: {logger.wandb_url or 'offline'}")

        def emit(filename, row):
            with (output / filename).open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            if logger:
                logger.log_metrics(row)

        def recording_writer(directory, recording, phase):
            from replay.dataset_writer import RaceDatasetWriter
            writer = RaceDatasetWriter(directory, config=recording,
                metadata=dict(run_id=run_id, algorithm='mappo_two_team', scenario=exp['name'], phase=phase,
                    provenance=provenance, physics_contract=provenance['physics_contract'],
                    action_contract=ActionComposer.contract_from_config(
                        scenario['agents'][trainable_ids[0]].get('action_constraints', {}), env.timestep),
                    team_observation_contracts={team: agent.observation_contract for team, agent in agents.items()},
                    observation_dims={aid: observations[aid].obs_dim+2 for aid in trainable_ids},
                    agent_teams=cfg['agent_teams'], trainable_agents=trainable_ids,
                    paired_policy_contract=checkpoint_contract,
                    policy_version_rule='completed synchronous pair updates; zero is initialization',
                    episode_termination=cfg['episode_termination'], terminal_agents=cfg.get('terminal_agents'),
                    map_protocols=provenance.get('map_protocols')))
            console.print_info(f'Self-play {phase} recording → {directory}')
            return writer

        eval_recording = scenario.get('evaluation', {}).get('recording', {})
        record_evaluation = eval_recording.get('enabled', selective_recording)
        if selective_recording and not args.eval:
            race_writer = recording_writer(args.dataset_dir or output/'behavior', scenario['recording'], 'training')
        if record_evaluation and (args.eval or scenario.get('evaluation', {}).get('enabled', False)):
            from replay.race_recorder import recording_config
            # Small matched evaluations default to whole-race sampling. Their
            # global budget is independent of training's event-heavy stream.
            evaluation_config = recording_config({**scenario.get('recording', {}),
                'sample_probability': 1., 'windows': [], **eval_recording, 'enabled': True})
            evaluation_dir = (args.dataset_dir or output/'behavior') if args.eval else (
                Path(args.dataset_dir)/'evaluation' if args.dataset_dir else output/'evaluation_behavior')
            eval_writer = recording_writer(evaluation_dir, evaluation_config, 'evaluation')
        trainer = TwoTeamTrainer(env=env, teams=teams, agents=agents, observations=observations,
            rewards=rewards, actions=actions, event_config=scenario["two_team"].get("events"),
            render=bool(cfg.get("render")), on_update=lambda row: emit("updates.jsonl", row),
            run_id=run_id, race_writer=race_writer)
        if race_writer is not None and num_envs == 1:
            from replay.race_recorder import RaceRecorder
            trainer.race_recorder = RaceRecorder(race_writer.config, race_writer.add_event, run_id=run_id)
        eval_trainer = trainer
        if not args.eval and scenario.get("evaluation", {}).get("enabled", False):
            eval_scenario = copy.deepcopy(scenario)
            eval_scenario["experiment"]["seed"] = protocol["seed"]
            eval_scenario["environment"]["max_steps"] = protocol["max_steps"]
            with preserve_rng():
                eval_env, _, _ = create_training_setup(eval_scenario, mode="eval", scenario_dir=scenario_dir)
            eval_trainer = TwoTeamTrainer(env=eval_env, teams=teams, agents=agents,
                observations=copy.deepcopy(observations), rewards=copy.deepcopy(rewards),
                actions=copy.deepcopy(actions), event_config=scenario["two_team"].get("events"), run_id=run_id)

        if eval_writer is not None:
            from replay.race_recorder import RaceRecorder
            eval_trainer.race_writer = eval_writer
            eval_trainer.race_recorder = RaceRecorder(eval_writer.config, eval_writer.add_event, run_id=run_id)

        console.print_info(f"Two trainable teams: {teams}; actor/critic per team; three objectives: completion, own safety, opponent crashes")
        console.print_info(f"Outputs: {output}; reward aggregation=sum; gamma={params['gamma']}; actor inputs={agents[next(iter(teams))].obs_dim}")
        from collections import deque
        recent = deque(maxlen=100)
        best, eval_round, last_evaluation_step = None, 0, None

        def evaluate(episode):
            nonlocal best, eval_round, last_evaluation_step
            eval_round += 1
            evaluation_id = f"eval_{eval_round:06d}"
            versions = (manifest.get('team_policy_versions', {team: None for team in teams}) if args.eval
                        else {team: trainer.updates for team in teams})
            policy_version = next(iter(versions.values())) if len(set(versions.values())) == 1 else None
            checkpoint = directory if args.eval else None
            if not args.eval and eval_writer is not None and eval_writer.can_record(trainer.environment_steps):
                name = f'evaluation_pairs/{evaluation_id}'
                save_pair(name, episode)
                checkpoint = (output/name).resolve()
            files = {}
            if checkpoint is not None:
                pair = json.loads((checkpoint/'pair.json').read_text())
                files = {team: dict(file=filename,
                    sha256=hashlib.sha256((checkpoint/filename).read_bytes()).hexdigest())
                    for team, filename in pair['files'].items()}
            checkpoint_hash = (hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
                               if files else None)
            recorded_protocol = {**protocol, 'seeds': list(range(protocol['seed'], protocol['seed']+protocol['episodes'])),
                'deterministic': True, 'map_pick': cfg.get('map_pick'),
                'map_bundles': cfg.get('map_bundles_eval', cfg.get('map_bundles')),
                'max_steps': eval_trainer.env.max_steps, 'target_laps': eval_trainer.env.target_laps,
                'timestep_s': eval_trainer.env.timestep, 'action_repeat': 1,
                'episode_termination': eval_trainer.env.episode_termination_mode,
                'terminal_agents': cfg.get('terminal_agents'), 'random_spawn': cfg.get('random_spawn'),
                'terminate_on_collision': eval_trainer.env.terminate_on_collision,
                'track_limits_enabled': eval_trainer.env.track_limits_enabled,
                'terminate_on_track_boundary': eval_trainer.env.terminate_on_track_boundary,
                'agent_teams': cfg['agent_teams']}
            context = dict(phase='evaluation', evaluation_id=evaluation_id,
                protocol=protocol.get('name', 'selection'), evaluation_protocol=recorded_protocol,
                environment_steps=manifest.get('environment_steps') if args.eval else trainer.environment_steps,
                checkpoint_training_seed=(manifest.get('provenance', {}).get('seed') if args.eval else exp['seed']),
                team_policy_versions=versions, policy_version=policy_version,
                checkpoint=str(checkpoint) if checkpoint is not None else None,
                checkpoint_sha256=checkpoint_hash, checkpoint_files=files)
            summary, rows = evaluate_pair(eval_trainer, protocol, context=context)
            last_evaluation_step = trainer.environment_steps
            score = selection_score(summary, teams)
            is_best = not args.eval and (best is None or score > best)
            emit("evaluation_metrics.jsonl", {"selfplay_eval/round": eval_round,
                "selfplay_eval/training_episode": episode,
                **{f"selfplay_eval/{key}": value for key, value in context.items()},
                "selfplay_eval/is_best": is_best,
                **{f"selfplay_eval/{key}": value for key, value in summary.items()}})
            with (output / "evaluation_races.jsonl").open("a") as stream:
                for row in rows:
                    stream.write(json.dumps({"round": eval_round, **row}) + "\n")
            if is_best:
                best = score
                save_pair("best_pair", episode, summary)
            console.print_info(f"Pair evaluation {eval_round}: " + " | ".join(
                f"{team}: finish_rate={summary[f'{team}/finish_rate']:.3f}, crashes={summary[f'{team}/crash_count']:.3f}"
                for team in teams))
            return summary

        if args.eval:
            summary = evaluate(0)
        else:
            total_steps = exp.get("total_steps")
            limit = int(exp.get("episodes", 5000)) if total_steps is None else None
            episode, summary = 0, {}
            checkpoint_interval = params.get("checkpoint_every_steps")
            evaluation_interval = scenario.get("evaluation", {}).get("every_steps")
            for name, interval in (("checkpoint_every_steps", checkpoint_interval),
                                   ("evaluation.every_steps", evaluation_interval)):
                if interval is not None and (isinstance(interval, bool)
                        or not isinstance(interval, int) or interval <= 0):
                    raise ValueError(f"{name} must be a positive integer")
            next_checkpoint, next_evaluation = checkpoint_interval, evaluation_interval
            checkpoint_episodes = int(params.get("checkpoint_every", 100))
            evaluation_episodes = int(scenario.get("evaluation", {}).get("every_episodes", 100))
            next_checkpoint_episode, next_evaluation_episode = checkpoint_episodes, evaluation_episodes

            def on_update(row):
                nonlocal next_checkpoint, next_evaluation
                nonlocal next_checkpoint_episode, next_evaluation_episode
                if total_steps is not None:
                    row = {**row, "train/total_steps": total_steps,
                           "train/budget_fraction": trainer.environment_steps / total_steps}
                emit("updates.jsonl", row)
                steps = trainer.environment_steps
                # Run at the first policy-update boundary past each threshold,
                # including updates in the middle of an unfinished race.
                if next_checkpoint is not None and steps >= next_checkpoint:
                    save_pair(f"pair_step{steps:012d}", episode)
                    next_checkpoint = (steps // checkpoint_interval + 1) * checkpoint_interval
                if eval_env is not None and next_evaluation is not None and steps >= next_evaluation:
                    evaluate(episode)
                    next_evaluation = (steps // evaluation_interval + 1) * evaluation_interval
                if num_envs > 1:
                    if checkpoint_interval is None and episode >= next_checkpoint_episode:
                        save_pair(f"pair_ep{episode:06d}", episode)
                        next_checkpoint_episode = (episode // checkpoint_episodes + 1) * checkpoint_episodes
                    if (eval_env is not None and evaluation_interval is None
                            and episode >= next_evaluation_episode):
                        evaluate(episode)
                        next_evaluation_episode = (episode // evaluation_episodes + 1) * evaluation_episodes

            trainer.on_update = on_update
            budget = f"{total_steps:,} joint environment steps" if total_steps is not None else f"{limit} episodes"
            console.print_info(f"Training budget: {budget}; evaluation steps are excluded.")

            def report_episode(row, *, parallel=False):
                nonlocal episode, summary
                if parallel:
                    episode += 1
                summary = row
                if summary["completed"]:
                    recent.append(summary)
                rolling = {f"selfplay/rolling100/{team}/{key}": float(np.mean(
                    [row[f"{team}/{key}"] for row in recent]))
                    for team in teams if recent for key in ("win", "draw", "finish_rate", "both_finished",
                                                  "any_crash", "opponent_crash_count", "reward")}
                emit("team_metrics.jsonl", {"selfplay/episode": episode,
                    **(rolling if recent else {}),
                    **{f"selfplay/{key}": value for key, value in summary.items() if key != "environment_steps"},
                    **({"selfplay/environment_local_steps": summary["environment_steps"]} if parallel else {}),
                    "selfplay/environment_steps": trainer.environment_steps})
                if not parallel or exp.get("terminal_episode_detail", False):
                    console.print_info(f"ep {episode} steps={int(summary['steps'])}: " + " | ".join(
                        f"{team}: reward={summary[f'{team}/reward']:+.3f}, finishes={int(summary[f'{team}/finish_count'])}, crashes={int(summary[f'{team}/crash_count'])}"
                        for team in teams))

            if num_envs > 1:
                from training.parallel_two_team import train_parallel
                train_parallel(trainer, scenario, scenario_dir, console=console,
                               on_episode=lambda row: report_episode(row, parallel=True))
            else:
                while ((limit is None or episode < limit)
                       and (total_steps is None or trainer.environment_steps < total_steps)):
                    remaining = None if total_steps is None else total_steps - trainer.environment_steps
                    episode += 1
                    report_episode(trainer.episode(step_budget=remaining))
                    if checkpoint_interval is None and episode % int(params.get("checkpoint_every", 100)) == 0:
                        save_pair(f"pair_ep{episode:06d}", episode)
                    if (eval_env is not None and evaluation_interval is None
                            and episode % int(scenario["evaluation"].get("every_episodes", 100)) == 0):
                        evaluate(episode)
            save_pair("final_pair", episode)
            if eval_env is not None and last_evaluation_step != trainer.environment_steps:
                evaluate(episode)
        (output / "run_summary.json").write_text(json.dumps({"last_metrics": summary,
            "environment_steps": trainer.environment_steps, "updates": trainer.updates,
            "agent_steps": trainer.agent_steps, "total_steps": exp.get("total_steps"),
            "wandb_url": logger.wandb_url if logger else None}, indent=2) + "\n")
        recording_complete = True
    finally:
        if race_writer is not None:
            race_writer.close(complete=recording_complete)
        if eval_writer is not None:
            eval_writer.close(complete=recording_complete)
        env.close()
        if eval_env is not None:
            eval_env.close()
        if logger is not None:
            logger.finish()


def _run_mappo(
    env, trainable_ids, other_agents,
    obs_composers, reward_composers, action_composer, params,
    action_low, action_high, action_repeat, render,
    hooks, exp_cfg, output_dir, console,
    focal_agent_id=None,
    run_id="run",
    scenario=None, scenario_dir=None, provenance=None, wandb_logger=None,
) -> None:
    from agents.mappo import MAPPOAgent
    from training.marl_trainer import MARLTrainer
    from training.hooks import MAPPOConsoleHook

    n_episodes = int(exp_cfg.get("episodes", 1000))
    if int(exp_cfg.get("num_envs", 1)) > 1:
        if not exp_cfg.get("terminal_episode_detail", False):
            hooks[:] = [hook for hook in hooks if not isinstance(hook, ConsoleHook)]
        hooks.insert(0, MAPPOConsoleHook(console,
            window=int(exp_cfg.get("terminal_recent_episodes", 100)),
            every_updates=int(exp_cfg.get("terminal_every_updates", 10)),
            diagnostic_every=int(exp_cfg.get("terminal_diagnostic_every_updates", 100))))
    focal_id = focal_agent_id or (trainable_ids[0] if trainable_ids else "")

    # obs_dim: all trainable agents share the same local observation spec
    obs_dim = obs_composers[focal_id].obs_dim

    # Nonlinear state dimensions are available before reset. A sizing reset
    # would consume an unrecorded friction draw (and spawn) before episode zero.
    # Retain the historical reset sequence for legacy experiment compatibility.
    if params.get("_physics_contract") is None:
        env.reset()
    global_snapshot = env.get_global_state()
    global_state_dim = len(global_snapshot.vector)
    params = {
        **params,
        "_global_state_contract_version": global_snapshot.metadata.get(
            "vector_contract_version", "legacy_unspecified"
        ),
    }

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
    pretrained_actor = params.get("_resolved_pretrained_actor_checkpoint")
    if pretrained_actor:
        agent.load_pretrained_actor(str(pretrained_actor))
        console.print_info(
            f"Initialized MAPPO actor from PPO checkpoint: {pretrained_actor}"
        )
    if agent.lora_config is not None:
        trainable = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
        total = sum(p.numel() for p in agent.actor.parameters())
        console.print_info(
            f"LoRA mode={agent.lora_config['mode']} rank={agent.lora_config['rank']} "
            f"actor_trainable={trainable}/{total}; centralized critic fully trainable"
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

    total_steps = exp_cfg.get("total_steps")
    budget = f"{total_steps} joint environment decisions" if total_steps is not None else f"{n_episodes} episodes"
    trainer.console = console
    race_hooks = [h for h in hooks if hasattr(h, 'on_race_record')]
    if race_hooks and int(exp_cfg.get('num_envs', 1)) == 1:
        from replay.race_recorder import RaceRecorder
        trainer.race_recorder = RaceRecorder(race_hooks[0].recording_config,
                                             race_hooks[0].on_race_record, run_id=run_id)
    env_cfg = (scenario or {}).get("environment", {})
    console.print_info(
        f"Experiment={exp_cfg.get('name')} train_maps={env_cfg.get('map_bundles_train')} "
        f"eval_maps={env_cfg.get('map_bundles_eval')} seed={exp_cfg.get('seed')} "
        f"training_mode={'finite' if env.lifecycle.finish_on_laps else 'continuous'}; "
        "env_steps count joint decisions; agent_steps count learner transitions; physics_steps count simulator steps.")
    console.print_info(
        f"Starting MAPPO training for {budget} "
        f"| num_envs={exp_cfg.get('num_envs', 1)} | agents={trainable_ids} "
        f"| obs_dim={obs_dim} | global_state_dim={global_state_dim}"
    )
    console.print_info(
        "MAPPO contract: "
        f"reward_mode={params.get('reward_mode')}  "
        f"critic_mode={params.get('critic_mode')}  "
        f"team_reward_reduction={params.get('team_reward_reduction')}"
    )
    evaluator = None
    eval_cfg = (scenario or {}).get("evaluation", {}) or {}
    if eval_cfg.get("enabled", False):
        from training.mappo_evaluator import DeterministicMAPPOEvaluator
        import torch

        protocol = resolve_evaluation_protocol(scenario, "selection")
        if protocol["max_steps"] <= 0:
            raise ValueError("MAPPO checkpoint evaluation requires a finite max_steps")
        eval_scenario = copy.deepcopy(scenario)
        eval_scenario["experiment"]["seed"] = protocol["seed"]
        eval_scenario["environment"]["max_steps"] = protocol["max_steps"]
        np_state, py_state = np.random.get_state(), random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            eval_env, eval_agents, _ = create_training_setup(
                eval_scenario, mode="eval", scenario_dir=scenario_dir)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        try:
            eval_obs = build_obs_composers(scenario["agents"], trainable_ids,
                                           eval_scenario["environment"], scenario_dir)
            for aid in trainable_ids:
                space = eval_env.action_spaces[aid]
                if (eval_obs[aid].contract != obs_composers[aid].contract
                        or not np.array_equal(space.low, action_low)
                        or not np.array_equal(space.high, action_high)):
                    raise ValueError("MAPPO evaluation observation/action contracts must match training")
            for controller in eval_agents.values():
                if hasattr(controller, "set_env"):
                    controller.set_env(eval_env)
            evaluator = DeterministicMAPPOEvaluator(
                env=eval_env, trainable_ids=trainable_ids, other_agents=eval_agents,
                obs_composers=eval_obs, action_composer=action_composer,
                episodes=protocol["episodes"], base_seed=protocol["seed"],
                action_repeat=action_repeat,
            ).bind_agent(agent)
            _configure_evaluation_recording(evaluator, scenario, output_dir, run_id, provenance)
            trainer.hooks.append(EvaluationCheckpointHook(
                agent, str(output_dir), evaluator,
                evaluate_every=int(eval_cfg.get("every_episodes", 100)),
                selection_strategy=eval_cfg.get("selection_strategy", "team_completion"),
                evaluate_every_steps=eval_cfg.get("every_steps"),
                provenance=provenance, console=console, wandb_logger=wandb_logger,
            ))
            console.print_info(f"Selection evaluation: {protocol['episodes']} races, "
                               f"seeds {protocol['seed']}..{protocol['seed'] + protocol['episodes'] - 1}, "
                               f"target_laps={eval_env.target_laps}, max_physics_steps={eval_env.max_steps}.")
        except Exception:
            eval_env.close()
            raise
    try:
        num_envs = int(exp_cfg.get("num_envs", 1))
        if num_envs > 1:
            trainer.train_parallel(scenario, scenario_dir, num_envs, n_episodes,
                                   **({"total_steps": total_steps} if total_steps is not None else {}))
        else:
            trainer.train(n_episodes=n_episodes,
                          **({"total_steps": total_steps} if total_steps is not None else {}))
    finally:
        if evaluator is not None:
            evaluator.close()


if __name__ == "__main__":
    main()
