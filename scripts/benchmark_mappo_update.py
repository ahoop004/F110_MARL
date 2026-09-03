#!/usr/bin/env python3
"""Benchmark MAPPO PPO updates on one deterministic stored rollout."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from agents.mappo import MAPPOAgent


AGENT_IDS = ("car_0", "car_1", "car_2", "car_3")


def _params(args: argparse.Namespace, batch_size: int, *, n_steps: int | None = None,
            n_epochs: int | None = None) -> dict[str, Any]:
    return {
        "device": args.device,
        "hidden_dims": [256, 256],
        "n_steps": args.n_steps if n_steps is None else n_steps,
        "n_epochs": args.n_epochs if n_epochs is None else n_epochs,
        "batch_size": batch_size,
        "critic_mode": "agent_conditioned",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }


def _make_agent(args: argparse.Namespace, batch_size: int, *, n_steps: int | None = None,
                n_epochs: int | None = None) -> MAPPOAgent:
    return MAPPOAgent(
        obs_dim=args.obs_dim,
        global_state_dim=args.global_state_dim,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=list(AGENT_IDS),
        params=_params(args, batch_size, n_steps=n_steps, n_epochs=n_epochs),
    )


def _rollout(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed + 1)
    row_dim = args.obs_dim + args.global_state_dim + 2 + 5
    rollout = rng.normal(
        size=(len(AGENT_IDS), args.n_steps, row_dim)
    ).astype(np.float32)
    action_start = args.obs_dim + args.global_state_dim
    action_end = action_start + 2
    rollout[:, :, action_start:action_end] = np.tanh(
        rollout[:, :, action_start:action_end]
    ) * 0.8
    rollout[:, :, action_end + 3 :] = 0.0
    rollout[:, -1, action_end + 4] = 1.0
    next_global_state = rng.normal(size=args.global_state_dim).astype(np.float32)
    return rollout, next_global_state


def _load_rollout(agent: MAPPOAgent, rollout: np.ndarray) -> None:
    agent._rollout_storage.copy_(torch.from_numpy(rollout).to(agent.device))
    for buffer in agent.buffers.values():
        buffer.ptr = agent.n_steps


def _tensor_digest(parameters: tuple[torch.nn.Parameter, ...]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _l2(parameters: tuple[torch.nn.Parameter, ...], *, gradients: bool) -> float:
    total = 0.0
    for parameter in parameters:
        value = parameter.grad if gradients else parameter.detach()
        if value is not None:
            total += float(torch.sum(value.detach().double().square()).cpu())
    return total ** 0.5


def _new_loaded_agent(
    args: argparse.Namespace,
    batch_size: int,
    initial: Mapping[str, Any],
    rollout: np.ndarray,
) -> MAPPOAgent:
    torch.manual_seed(args.seed)
    agent = _make_agent(args, batch_size)
    agent.actor.load_state_dict(copy.deepcopy(initial["actor"]))
    agent.critic.load_state_dict(copy.deepcopy(initial["critic"]))
    agent.optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
    _load_rollout(agent, rollout)
    return agent


def _run_update(
    agent: MAPPOAgent,
    next_global_state: np.ndarray,
    *,
    shuffle_seed: int,
) -> tuple[float, dict[str, float]]:
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
        torch.cuda.reset_peak_memory_stats(agent.device)
    torch.manual_seed(shuffle_seed)
    started = time.perf_counter()
    metrics = agent.update(next_global_state)
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
    return time.perf_counter() - started, metrics


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    template = _make_agent(args, args.batch_sizes[0])
    initial = {
        "actor": copy.deepcopy(template.actor.state_dict()),
        "critic": copy.deepcopy(template.critic.state_dict()),
        "optimizer": copy.deepcopy(template.optimizer.state_dict()),
    }
    rollout, next_global_state = _rollout(args)

    # Warm lazy CUDA modules and optimizer kernels outside measured updates.
    warm_steps = min(args.n_steps, 64)
    warm = _make_agent(args, min(args.batch_sizes), n_steps=warm_steps, n_epochs=1)
    _load_rollout(warm, rollout[:, :warm_steps])
    _run_update(warm, next_global_state, shuffle_seed=args.seed + 2)
    del warm

    optimizer_samples = len(AGENT_IDS) * args.n_steps * args.n_epochs
    results = []
    for batch_size in args.batch_sizes:
        repetitions = []
        for repetition in range(args.repetitions):
            agent = _new_loaded_agent(args, batch_size, initial, rollout)
            elapsed, metrics = _run_update(
                agent,
                next_global_state,
                shuffle_seed=args.seed + 10,
            )
            repetitions.append(
                {
                    "repetition": repetition,
                    "seconds": elapsed,
                    "optimizer_samples_per_second": optimizer_samples / elapsed,
                    "peak_cuda_bytes": (
                        int(torch.cuda.max_memory_allocated(agent.device))
                        if agent.device.type == "cuda"
                        else 0
                    ),
                    "metrics": metrics,
                    "parameter_sha256": _tensor_digest(agent._optim_parameters),
                    "parameter_l2": _l2(agent._optim_parameters, gradients=False),
                    "final_gradient_l2": _l2(
                        agent._optim_parameters, gradients=True
                    ),
                }
            )
            del agent
        results.append(
            {
                "batch_size": batch_size,
                "optimizer_steps_per_update": args.n_epochs
                * ((len(AGENT_IDS) * args.n_steps + batch_size - 1) // batch_size),
                "seconds": _summary([item["seconds"] for item in repetitions]),
                "optimizer_samples_per_second": _summary(
                    [item["optimizer_samples_per_second"] for item in repetitions]
                ),
                "peak_cuda_bytes": _summary(
                    [float(item["peak_cuda_bytes"]) for item in repetitions]
                ),
                "deterministic_repetitions": len(
                    {item["parameter_sha256"] for item in repetitions}
                )
                == 1,
                "repetitions": repetitions,
            }
        )

    return {
        "benchmark_version": "1.0",
        "workload": {
            "agent_ids": list(AGENT_IDS),
            "obs_dim": args.obs_dim,
            "global_state_dim": args.global_state_dim,
            "n_steps_per_agent": args.n_steps,
            "n_epochs": args.n_epochs,
            "optimizer_samples": optimizer_samples,
            "rollout_sha256": hashlib.sha256(rollout.tobytes()).hexdigest(),
            "shuffle_seed": args.seed + 10,
            "batch_sizes": args.batch_sizes,
            "device": str(template.device),
        },
        "metadata": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(template.device)
                if template.device.type == "cuda"
                else None
            ),
        },
        "results": results,
    }


def write_profile(args: argparse.Namespace, output: Path) -> None:
    if args.profile_batch_size is None:
        return
    torch.manual_seed(args.seed)
    template = _make_agent(args, args.profile_batch_size)
    initial = {
        "actor": copy.deepcopy(template.actor.state_dict()),
        "critic": copy.deepcopy(template.critic.state_dict()),
        "optimizer": copy.deepcopy(template.optimizer.state_dict()),
    }
    rollout, next_global_state = _rollout(args)
    agent = _new_loaded_agent(
        args, args.profile_batch_size, initial, rollout
    )
    activities = [ProfilerActivity.CPU]
    if agent.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    torch.manual_seed(args.seed + 10)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        agent.update(next_global_state)
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
    trace_path = output.with_suffix(output.suffix + ".trace.json")
    profiler.export_chrome_trace(str(trace_path))
    tables = [
        profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
    ]
    if agent.device.type == "cuda":
        tables.append(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=50
            )
        )
    output.with_suffix(output.suffix + ".profile.txt").write_text(
        "\n\n".join(tables)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--obs-dim", type=int, default=115)
    parser.add_argument("--global-state-dim", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile-batch-size", type=int)
    parser.add_argument("--output", default="/tmp/f110_mappo_update_benchmark.json")
    args = parser.parse_args()
    if any(value <= 0 for value in args.batch_sizes):
        parser.error("--batch-sizes must be positive")
    if args.n_steps <= 0 or args.n_epochs <= 0 or args.repetitions <= 0:
        parser.error("steps, epochs, and repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = benchmark(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_profile(args, output)
    for result in report["results"]:
        print(
            f"batch={result['batch_size']:4d} "
            f"samples/s={result['optimizer_samples_per_second']['median']:.0f} "
            f"seconds={result['seconds']['median']:.3f} "
            f"peak_cuda_mib={result['peak_cuda_bytes']['median'] / 2**20:.1f}"
        )
    print(f"results: {output}")


if __name__ == "__main__":
    main()
