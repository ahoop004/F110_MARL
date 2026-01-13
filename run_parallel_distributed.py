#!/usr/bin/env python3
"""Launch multiple parallel training runs with distributed experience sharing.

This script:
1. Starts a shared distributed buffer registry
2. Launches N parallel training processes
3. Each process samples from the shared buffer pool
4. Automatically cleans up when runs complete

Usage:
    # Launch 4 parallel SAC runs with distributed buffers
    python run_parallel_distributed.py \
        --scenario scenarios/sac.yaml \
        --n_parallel 4 \
        --cross_sample_ratio 0.2 \
        --strategy self_heavy

    # Launch with different algorithms
    python run_parallel_distributed.py \
        --scenarios scenarios/sac.yaml scenarios/td3.yaml scenarios/ppo.yaml \
        --cross_sample_ratio 0.3

    # Disable distributed sharing (baseline comparison)
    python run_parallel_distributed.py \
        --scenario scenarios/sac.yaml \
        --n_parallel 4 \
        --strategy local_only
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import IO, List, Optional

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from replay.distributed_buffer import start_registry_server


def launch_training_process(
    scenario: str,
    run_id: str,
    registry_host: Optional[str],
    registry_port: Optional[int],
    registry_authkey: Optional[str],
    buffer_id: str,
    cross_sample_ratio: float,
    strategy: str,
    seed: int,
    gpu_id: Optional[int] = None,
    wandb: bool = True,
    extra_args: Optional[List[str]] = None,
    log_file: Optional[IO[str]] = None,
):
    """Launch a single training process.

    Args:
        scenario: Path to scenario YAML
        run_id: Unique run identifier
        registry_host: Host for registry server
        registry_port: Port for registry server
        registry_authkey: Auth key (hex-encoded) for registry server
        buffer_id: Buffer ID for this run
        cross_sample_ratio: Fraction of samples from distributed pool
        strategy: Sampling strategy
        seed: Random seed for this run (for exploration diversity)
        gpu_id: GPU to use (None = CPU)
        wandb: Enable WandB logging
        extra_args: Additional arguments to pass to training script
    """
    env = os.environ.copy()

    # Set GPU
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Set distributed buffer parameters via environment
    env['DISTRIBUTED_BUFFER_ID'] = buffer_id
    env['DISTRIBUTED_CROSS_SAMPLE_RATIO'] = str(cross_sample_ratio)
    env['DISTRIBUTED_STRATEGY'] = strategy
    env['RUN_ID'] = run_id
    if registry_host and registry_port and registry_authkey:
        env['DISTRIBUTED_REGISTRY_HOST'] = registry_host
        env['DISTRIBUTED_REGISTRY_PORT'] = str(registry_port)
        env['DISTRIBUTED_REGISTRY_AUTHKEY'] = registry_authkey

    # Build command
    cmd = [
        'python3',
        'run_v2.py',
        '--scenario', scenario,
        '--run_id', run_id,
        '--seed', str(seed),
    ]

    if wandb:
        cmd.append('--wandb')

    if extra_args:
        cmd.extend(extra_args)

    print(f"🚀 Launching {run_id} with buffer {buffer_id}")
    print(f"   Strategy: {strategy}, Cross-sample ratio: {cross_sample_ratio}, Seed: {seed}")
    print(f"   Command: {' '.join(cmd)}")

    # Launch process
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=log_file,
        text=True,
        bufsize=1,
    )

    return proc


def _tail_file(path: Path, max_lines: int = 40) -> List[str]:
    lines: deque[str] = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lines.append(line.rstrip())
    except FileNotFoundError:
        return []
    return list(lines)


def monitor_processes(
    processes: List[subprocess.Popen],
    run_ids: List[str],
    log_paths: List[Path],
):
    """Monitor training processes and report status.

    Args:
        processes: List of subprocess handles
        run_ids: Corresponding run IDs
    """
    print("\n" + "="*80)
    print("PARALLEL TRAINING MONITOR")
    print("="*80)

    active = set(range(len(processes)))

    while active:
        for i in list(active):
            proc = processes[i]
            retcode = proc.poll()

            if retcode is not None:
                # Process finished
                active.remove(i)
                if retcode == 0:
                    print(f"✓ {run_ids[i]} completed successfully")
                else:
                    print(f"✗ {run_ids[i]} failed with code {retcode}")
                    log_path = log_paths[i] if i < len(log_paths) else None
                    if log_path:
                        tail_lines = _tail_file(log_path, max_lines=40)
                        if tail_lines:
                            print(f"  Log tail: {log_path}")
                            for line in tail_lines:
                                print(f"    {line}")

        if active:
            print(f"\r📊 Active runs: {len(active)}/{len(processes)}", end='', flush=True)
            time.sleep(5)

    print("\n" + "="*80)
    print("ALL TRAINING RUNS COMPLETE")
    print("="*80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch parallel training runs with distributed experience sharing"
    )

    # Scenario configuration
    parser.add_argument(
        '--scenario',
        type=str,
        help='Single scenario for all runs',
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        help='Multiple scenarios (one per run)',
    )
    parser.add_argument(
        '--n_parallel',
        type=int,
        default=4,
        help='Number of parallel runs (default: 4)',
    )

    # Distributed buffer configuration
    parser.add_argument(
        '--cross_sample_ratio',
        type=float,
        default=0.2,
        help='Fraction of samples from distributed pool (0.0-1.0, default: 0.2)',
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='self_heavy',
        choices=['uniform', 'weighted', 'self_heavy', 'newest', 'local_only'],
        help='Sampling strategy (default: self_heavy)',
    )
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=1000000,
        help='Max size per buffer (default: 1000000)',
    )

    # GPU configuration
    parser.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        help='GPU IDs to use (round-robin assignment)',
    )

    # Seed configuration
    parser.add_argument(
        '--base_seed',
        type=int,
        default=42,
        help='Base random seed (each run gets base_seed + run_index, default: 42)',
    )

    # Logging
    parser.add_argument(
        '--wandb',
        action='store_true',
        default=True,
        help='Enable WandB logging (default: True)',
    )
    parser.add_argument(
        '--no-wandb',
        action='store_false',
        dest='wandb',
        help='Disable WandB logging',
    )

    # Additional args to pass through
    parser.add_argument(
        '--extra_args',
        nargs=argparse.REMAINDER,
        help='Additional arguments to pass to training script',
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.scenario and not args.scenarios:
        print("Error: Must provide --scenario or --scenarios")
        sys.exit(1)

    # Determine scenarios for each run
    if args.scenarios:
        scenarios = args.scenarios
        n_runs = len(scenarios)
    else:
        scenarios = [args.scenario] * args.n_parallel
        n_runs = args.n_parallel

    print("="*80)
    print("DISTRIBUTED PARALLEL TRAINING SETUP")
    print("="*80)
    print(f"Number of parallel runs: {n_runs}")
    print(f"Scenarios: {scenarios}")
    print(f"Cross-sample ratio: {args.cross_sample_ratio}")
    print(f"Strategy: {args.strategy}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Base seed: {args.base_seed} (runs get seeds {args.base_seed} to {args.base_seed + n_runs - 1})")
    print(f"GPUs: {args.gpus if args.gpus else 'CPU only'}")
    print("="*80)

    # Create distributed registry
    registry_host = None
    registry_port = None
    registry_authkey = None
    if args.strategy != 'local_only':
        print("\n🔧 Starting distributed buffer registry...")
        registry, address, authkey = start_registry_server(
            max_buffer_size=args.buffer_size,
        )
        registry_host, registry_port = address
        registry_authkey = authkey.hex()
    else:
        print("\n⚠ Distributed sharing disabled (local_only mode)")
        registry = None

    # Register buffers for each run
    buffer_ids = []
    run_ids = []

    for i in range(n_runs):
        scenario_name = Path(scenarios[i]).stem
        run_id = f"{scenario_name}_run{i+1}_{int(time.time())}"
        run_ids.append(run_id)

        if registry:
            buffer_id = registry.register_buffer(run_id=run_id, algorithm=scenario_name)
            buffer_ids.append(buffer_id)
        else:
            buffer_ids.append(None)

    # Launch training processes
    print(f"\n🚀 Launching {n_runs} training processes...")
    processes = []
    log_paths: List[Path] = []
    log_files: List[IO[str]] = []
    log_dir = ROOT_DIR / "outputs" / "parallel_runs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_runs):
        gpu_id = args.gpus[i % len(args.gpus)] if args.gpus else None
        seed = args.base_seed + i  # Different seed for each run
        log_path = log_dir / f"{run_ids[i]}.log"
        log_file = log_path.open("w", encoding="utf-8")

        proc = launch_training_process(
            scenario=scenarios[i],
            run_id=run_ids[i],
            registry_host=registry_host,
            registry_port=registry_port,
            registry_authkey=registry_authkey,
            buffer_id=buffer_ids[i],
            cross_sample_ratio=args.cross_sample_ratio,
            strategy=args.strategy,
            seed=seed,
            gpu_id=gpu_id,
            wandb=args.wandb,
            extra_args=args.extra_args,
            log_file=log_file,
        )

        processes.append(proc)
        log_paths.append(log_path)
        log_files.append(log_file)
        time.sleep(2)  # Stagger launches

    # Monitor processes
    try:
        monitor_processes(processes, run_ids, log_paths)
    except KeyboardInterrupt:
        print("\n⚠ Interrupt received, terminating processes...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait(timeout=10)
    finally:
        for log_file in log_files:
            log_file.close()

    # Cleanup
    if registry:
        print("\n🧹 Cleaning up registry...")
        for buffer_id in buffer_ids:
            if buffer_id:
                registry.deregister_buffer(buffer_id)
        registry.stop()

    print("\n✅ All done!")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
