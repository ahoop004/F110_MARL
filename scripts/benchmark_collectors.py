"""Compare collector layouts at a fixed environment count and rollout horizon.

Runs use the real training entry point, fresh processes, and separate output
folders. Ready scheduling changes stochastic sampling order; these are matched
work budgets, not identical policy trajectories or learning-quality comparisons.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--scenario', default='scenarios/ppo_lap_completion_pretrain.yaml')
    parser.add_argument('--set', dest='parameter_overrides', action='append', default=[], metavar='KEY=YAML')
    parser.add_argument('--num-envs', type=int, default=12)
    parser.add_argument('--workers', nargs='+', type=int, default=[4, 12])
    parser.add_argument('--scheduling', nargs='+', choices=['synchronous', 'ready'], default=['synchronous', 'ready'])
    parser.add_argument('--rollout-steps-per-env', type=int, default=128)
    parser.add_argument('--total-steps', type=int, default=6144)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if min(args.num_envs, *args.workers, args.rollout_steps_per_env, args.repetitions) < 1:
        parser.error('environment, worker, horizon and repetition counts must be positive')
    if args.total_steps < 2 * args.num_envs * args.rollout_steps_per_env:
        parser.error('use at least two full rounds so startup/warmup can be separated')
    output = args.output_dir or Path(tempfile.mkdtemp(prefix='f110_collectors_'))
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYGLET_HEADLESS': 'true', **dict.fromkeys(
        ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'], '1')}
    results = []
    for workers in args.workers:
        for scheduling in args.scheduling:
            for repetition in range(args.repetitions):
                run_dir = output / f'w{workers}_{scheduling}_r{repetition}'
                run_dir.mkdir()  # Never overwrite a previous benchmark.
                command = [sys.executable, str(ROOT / 'run.py'), '--scenario', args.scenario,
                           '--num-envs', str(args.num_envs), '--num-workers', str(workers),
                           '--collector-scheduling', scheduling, '--rollout-steps-per-env', str(args.rollout_steps_per_env),
                           '--total-steps', str(args.total_steps), '--seed', str(args.seed),
                           '--torch-threads', '1', '--no-render', '--no-wandb', '--quiet',
                           '--output-dir', str(run_dir.resolve())]
                for override in args.parameter_overrides:
                    command.extend(('--set', override))
                started = time.perf_counter()
                with (run_dir / 'console.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                wall_s = time.perf_counter() - started
                with (run_dir / 'update_metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                measured = [row for row in rows if row.get('perf/collection_seconds')]
                if len(measured) < 2:
                    raise RuntimeError(f'Need at least two timed rounds: {run_dir}')
                steps = float(measured[-1]['train/environment_steps']) - float(measured[0]['train/environment_steps'])
                collection_s = sum(float(row['perf/collection_seconds']) for row in measured[1:])
                update_s = sum(float(row['perf/update_seconds']) for row in measured[1:])
                result = dict(workers=min(workers, args.num_envs), num_envs=args.num_envs,
                              scheduling=scheduling, repetition=repetition, command=command,
                              process_wall_seconds=wall_s, startup_seconds=float(measured[0]['perf/startup_seconds']),
                              measured_rounds=len(measured)-1, measured_steps=steps,
                              collection_steps_per_second=steps / collection_s,
                              round_steps_per_second=steps / (collection_s + update_s),
                              update_seconds=update_s, output_dir=str(run_dir))
                results.append(result)
                (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps(result), flush=True)
    print(f'Results: {output / "results.json"}')


if __name__ == '__main__':
    main()
