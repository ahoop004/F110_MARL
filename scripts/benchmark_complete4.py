#!/usr/bin/env python3
"""Fixed-work benchmark for the three four-agent MAPPO observation arms."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from agents.mappo import MAPPOAgent
from core.agent_builder import get_trainable_agent_ids
from core.map_selection import resolve_bundle_yaml
from core.provenance import build_run_provenance
from core.scenario import load_and_expand_scenario, resolve_mappo_config
from core.setup import create_training_setup
from env.types import SpawnPlan, SpawnState
from run import (
    build_obs_composers,
    build_reward_composers,
    resolve_training_params,
)
from training.hooks import TrainingHook
from training.marl_trainer import MARLTrainer
from wrappers.actions.composer import ActionComposer


SUPPORTED_SCENARIOS = (
    "scenarios/complete_4.yaml",
    "scenarios/complete_4_frenet.yaml",
    "scenarios/complete_4_frenet_neighbors.yaml",
)
RESULT_PREFIX = "F110_BENCHMARK_JSON="


class StageTimer:
    """Accumulate synchronized wall time and call counts by stage."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seconds: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def call(self, stage: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._sync()
        started = time.perf_counter()
        result = function(*args, **kwargs)
        self._sync()
        self.seconds[stage] += time.perf_counter() - started
        self.calls[stage] += 1
        return result


class CaptureHook(TrainingHook):
    def __init__(self) -> None:
        self.transitions = 0
        self.episode_metrics: Dict[str, Any] = {}

    def on_step(self, record: Any) -> None:
        self.transitions += 1

    def on_episode_end(
        self, episode: int, reward: float, info: Dict, metrics: Dict[str, Any]
    ) -> None:
        self.episode_metrics = dict(metrics)


def _scenario_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    supported = {(ROOT / item).resolve() for item in SUPPORTED_SCENARIOS}
    if resolved not in supported:
        choices = ", ".join(SUPPORTED_SCENARIOS)
        raise ValueError(f"Unsupported benchmark scenario {value!r}; choose one of: {choices}")
    return resolved


def _fixed_spawn_plan(map_name: str, agent_ids: Iterable[str]) -> SpawnPlan:
    yaml_path = resolve_bundle_yaml(ROOT / "maps", map_name)
    metadata = yaml.safe_load(yaml_path.read_text()) or {}
    entries = ((metadata.get("annotations") or {}).get("spawn_points") or [])
    points = {
        str(entry["name"]): np.asarray(entry["pose"], dtype=np.float32)
        for entry in entries
        if isinstance(entry, Mapping) and "name" in entry and "pose" in entry
    }
    def spawn_sort_key(name: str) -> tuple[int, int | str]:
        suffix = name.rsplit("_", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, name)

    ordered_names = sorted(points, key=spawn_sort_key)
    ids = list(agent_ids)
    if len(ordered_names) < len(ids):
        raise ValueError(
            f"Map {map_name!r} has {len(ordered_names)} named spawns for {len(ids)} agents."
        )
    states = tuple(
        SpawnState(agent_id=agent_id, pose=points[spawn_name], spawn_id=spawn_name)
        for agent_id, spawn_name in zip(ids, ordered_names)
    )
    return SpawnPlan(states=states, plan_id="complete4_fixed_grid", map_id=map_name)


def _configure_fixed_work(
    scenario: Dict[str, Any], *, map_name: str, seed: int, physics_substeps: int, device: str
) -> None:
    scenario["experiment"]["seed"] = int(seed)
    scenario.setdefault("wandb", {})["enabled"] = False
    environment = scenario["environment"]
    environment["map_bundles"] = [map_name]
    environment["map_bundles_train"] = [map_name]
    environment["map_bundles_eval"] = [map_name]
    environment["map_bundle_active"] = map_name
    environment["map_cycle"] = "per_episode"
    environment["map_pick"] = "first"
    environment["max_steps"] = int(physics_substeps)
    environment["target_laps"] = 1_000_000
    environment["terminate_on_collision"] = False
    environment["episode_termination"] = {"mode": "all_agents"}
    scenario.setdefault("training_defaults", {})["device"] = device


def _wrap_method(
    obj: Any,
    name: str,
    timer: StageTimer,
    stage: str,
    *,
    transform: Callable[[Callable[..., Any], tuple, dict], Any] | None = None,
) -> Callable[..., Any]:
    original = getattr(obj, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if transform is None:
            return timer.call(stage, original, *args, **kwargs)
        return timer.call(stage, transform, original, args, kwargs)

    setattr(obj, name, wrapped)
    return original


def _warm_inference(
    agent: MAPPOAgent,
    observations: Mapping[str, np.ndarray],
    global_state: np.ndarray,
    iterations: int,
) -> None:
    for _ in range(max(iterations, 0)):
        for agent_id, observation in observations.items():
            agent.act(observation, deterministic=True)
            agent.evaluate_state(global_state, agent_id)
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    scenario_path = _scenario_path(args.scenario)
    setup_started = time.perf_counter()
    scenario = load_and_expand_scenario(str(scenario_path))
    _configure_fixed_work(
        scenario,
        map_name=args.map,
        seed=args.seed,
        physics_substeps=args.physics_substeps,
        device=args.device,
    )
    scenario_dir = scenario_path.parent
    trainable_ids = get_trainable_agent_ids(scenario["agents"])
    spawn_plan = _fixed_spawn_plan(args.map, trainable_ids)
    env, fixed_agents, _ = create_training_setup(
        scenario, mode="train", scenario_dir=scenario_dir
    )
    if fixed_agents:
        raise ValueError("complete_4 benchmark expects every agent to be trainable")

    action_space = env.action_spaces[trainable_ids[0]]
    obs_composers = build_obs_composers(
        scenario["agents"], trainable_ids, scenario["environment"], scenario_dir
    )
    reward_composers = build_reward_composers(
        scenario["agents"], trainable_ids, scenario_dir
    )
    params = resolve_training_params(scenario["agents"][trainable_ids[0]], scenario)
    params = {**params, **resolve_mappo_config(scenario), "device": args.device}

    original_reset = env.reset
    initial_obs, initial_info = original_reset(
        seed=args.seed, options={"spawn_plan": spawn_plan}
    )
    wrapped_initial = {
        agent_id: obs_composers[agent_id].wrap(
            initial_obs[agent_id], initial_info[agent_id]
        )
        for agent_id in trainable_ids
    }
    initial_global = env.get_global_state().vector
    agent = MAPPOAgent(
        obs_dim=obs_composers[trainable_ids[0]].obs_dim,
        global_state_dim=len(initial_global),
        action_low=action_space.low,
        action_high=action_space.high,
        agent_ids=trainable_ids,
        params=params,
    )
    action_composer = ActionComposer.from_config(
        action_space.low,
        action_space.high,
        scenario["agents"][trainable_ids[0]].get("action_constraints", {}),
    )
    _warm_inference(agent, wrapped_initial, initial_global, args.warmup_iterations)
    agent.clear_buffers()

    timer = StageTimer(agent.device)
    action_digest = hashlib.sha256()
    optimizer_samples = 0

    def reset_transform(original: Callable[..., Any], call_args: tuple, kwargs: dict) -> Any:
        kwargs = dict(kwargs)
        kwargs["seed"] = args.seed
        kwargs["options"] = {"spawn_plan": spawn_plan}
        return original(*call_args, **kwargs)

    def act_transform(original: Callable[..., Any], call_args: tuple, kwargs: dict) -> Any:
        action, log_probability = original(*call_args, deterministic=True)
        action_digest.update(np.asarray(action, dtype=np.float32).tobytes())
        return action, log_probability

    def update_transform(original: Callable[..., Any], call_args: tuple, kwargs: dict) -> Any:
        nonlocal optimizer_samples
        pooled = sum(buffer.size() for buffer in agent.buffers.values())
        optimizer_samples += pooled * agent.n_epochs
        return original(*call_args, **kwargs)

    _wrap_method(env, "reset", timer, "environment_reset", transform=reset_transform)
    _wrap_method(env, "step", timer, "environment_step")
    _wrap_method(agent, "act", timer, "policy_action", transform=act_transform)
    _wrap_method(agent, "evaluate_state", timer, "centralized_value")
    _wrap_method(agent, "store", timer, "rollout_storage")
    _wrap_method(agent, "update", timer, "ppo_update", transform=update_transform)
    for composer in obs_composers.values():
        _wrap_method(composer, "wrap", timer, "observation_composition")
    for composer in reward_composers.values():
        _wrap_method(composer, "compute", timer, "reward_composition")

    import training.marl_trainer as trainer_module

    original_reward_context = trainer_module.build_reward_context

    def timed_reward_context(*call_args: Any, **kwargs: Any) -> Any:
        return timer.call(
            "reward_context", original_reward_context, *call_args, **kwargs
        )

    capture = CaptureHook()
    trainer = MARLTrainer(
        env=env,
        agent=agent,
        trainable_ids=trainable_ids,
        other_agents={},
        obs_composers=obs_composers,
        reward_composers=reward_composers,
        action_composer=action_composer,
        action_repeat=int(scenario["environment"].get("action_repeat", 1)),
        hooks=[capture],
        render=False,
        focal_agent_id=trainable_ids[0],
        run_id="benchmark",
        reward_mode=params["reward_mode"],
        team_reward_reduction=params["team_reward_reduction"],
    )

    if agent.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(agent.device)
    profiler = cProfile.Profile() if args.profile else None
    trainer_module.build_reward_context = timed_reward_context
    try:
        if profiler is not None:
            profiler.enable()
        total_started = time.perf_counter()
        trainer.train(n_episodes=1)
        if agent.device.type == "cuda":
            torch.cuda.synchronize(agent.device)
        total_seconds = time.perf_counter() - total_started
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(args.profile)
    finally:
        trainer_module.build_reward_context = original_reward_context
        env.close()

    setup_seconds = total_started - setup_started
    decisions = int(capture.episode_metrics.get("episode_steps", 0))
    substeps = int(timer.calls.get("environment_step", 0))
    peak_cuda = (
        int(torch.cuda.max_memory_allocated(agent.device))
        if agent.device.type == "cuda"
        else 0
    )
    provenance = build_run_provenance(
        scenario,
        scenario_path=scenario_path,
        run_id="benchmark",
        algorithm="mappo",
        trainable_agents=trainable_ids,
    )
    return {
        "repetition": args.repetition,
        "scenario": str(scenario_path.relative_to(ROOT)),
        "map": args.map,
        "spawn_plan": {
            state.agent_id: state.spawn_id for state in spawn_plan.states
        },
        "seed": args.seed,
        "device": str(agent.device),
        "action_source": "deterministic_shared_actor",
        "action_sequence_sha256": action_digest.hexdigest(),
        "counts": {
            "joint_decisions": decisions,
            "physics_substeps": substeps,
            "transitions": capture.transitions,
            "optimizer_samples": optimizer_samples,
            "ppo_updates": int(timer.calls.get("ppo_update", 0)),
        },
        "seconds": {
            "setup_and_map_load": setup_seconds,
            "total_measured": total_seconds,
            **dict(timer.seconds),
        },
        "throughput": {
            "decisions_per_second": decisions / total_seconds,
            "physics_substeps_per_second": substeps / total_seconds,
            "update_samples_per_second": (
                optimizer_samples / timer.seconds["ppo_update"]
                if timer.seconds.get("ppo_update", 0.0) > 0.0
                else 0.0
            ),
        },
        "memory": {
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "peak_cuda_bytes": peak_cuda,
        },
        "metadata": {
            "git": provenance["git"],
            "scenario_source_sha256": provenance["scenario_source_sha256"],
            "resolved_config_sha256": provenance["resolved_config_sha256"],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(agent.device)
                if agent.device.type == "cuda"
                else None
            ),
            "cpu": platform.processor() or platform.machine(),
        },
    }


def summarize_results(results: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("At least one measured benchmark result is required")

    summary: Dict[str, Any] = {"repetitions": len(results), "metrics": {}}
    fields = (
        ("throughput", "decisions_per_second"),
        ("throughput", "physics_substeps_per_second"),
        ("throughput", "update_samples_per_second"),
        ("seconds", "total_measured"),
        ("memory", "peak_rss_kib"),
        ("memory", "peak_cuda_bytes"),
    )
    for section, name in fields:
        values = [float(result[section][name]) for result in results]
        median = statistics.median(values)
        summary["metrics"][name] = {
            "median": median,
            "min": min(values),
            "max": max(values),
            "spread": max(values) - min(values),
        }

    count_signatures = {
        tuple(sorted(result["counts"].items())) for result in results
    }
    action_hashes = {str(result["action_sequence_sha256"]) for result in results}
    summary["fixed_work_verified"] = len(count_signatures) == 1
    summary["action_sequence_verified"] = len(action_hashes) == 1
    return summary


def _worker_command(args: argparse.Namespace, repetition: int, profile: str | None) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--scenario",
        args.scenario,
        "--map",
        args.map,
        "--seed",
        str(args.seed),
        "--physics-substeps",
        str(args.physics_substeps),
        "--device",
        args.device,
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--repetition",
        str(repetition),
    ]
    if profile:
        command.extend(("--profile", profile))
    return command


def run_coordinator(args: argparse.Namespace) -> Dict[str, Any]:
    if args.profile:
        Path(args.profile).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )

    def invoke_worker(repetition: int, profile: str | None = None) -> Dict[str, Any]:
        completed = subprocess.run(
            _worker_command(args, repetition, profile),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYGLET_HEADLESS": "true"},
        )
        payload_line = next(
            (line for line in reversed(completed.stdout.splitlines()) if line.startswith(RESULT_PREFIX)),
            None,
        )
        if payload_line is None:
            raise RuntimeError(
                f"Benchmark worker emitted no result. stderr:\n{completed.stderr}"
            )
        return json.loads(payload_line[len(RESULT_PREFIX) :])

    results = [invoke_worker(repetition) for repetition in range(args.repetitions)]
    profiled_result = invoke_worker(-1, args.profile) if args.profile else None

    report = {
        "benchmark_version": "1.0",
        "workload": {
            "scenario": args.scenario,
            "map": args.map,
            "seed": args.seed,
            "physics_substeps": args.physics_substeps,
            "device": args.device,
            "warmup_iterations": args.warmup_iterations,
        },
        "summary": summarize_results(results),
        "results": results,
    }
    if profiled_result is not None:
        report["profiled_result_excluded_from_summary"] = profiled_result
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.profile:
        profile_path = Path(args.profile).expanduser().resolve()
        stream = io.StringIO()
        pstats.Stats(str(profile_path), stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(50)
        profile_path.with_suffix(profile_path.suffix + ".txt").write_text(
            stream.getvalue()
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=SUPPORTED_SCENARIOS[0], choices=SUPPORTED_SCENARIOS)
    parser.add_argument("--map", default="Budapest_map")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-substeps", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default="/tmp/f110_complete4_benchmark.json")
    parser.add_argument("--profile", default=None, help="Optional cProfile output for repetition zero")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.physics_substeps <= 0:
        parser.error("--physics-substeps must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        print(RESULT_PREFIX + json.dumps(run_worker(args), sort_keys=True))
        return
    report = run_coordinator(args)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"results: {Path(args.output).expanduser().resolve()}")
    if args.profile:
        print(f"profile: {Path(args.profile).expanduser().resolve()}")
        profile_path = Path(args.profile).expanduser().resolve()
        print(f"profile report: {profile_path.with_suffix(profile_path.suffix + '.txt')}")


if __name__ == "__main__":
    main()
