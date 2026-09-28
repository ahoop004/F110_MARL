"""Time fixed MPC traffic and physics with a stationary ego, without PPO or rendering.

Compare the action/state hashes across revisions as well as warmed-up timings.
This is a controller microbenchmark, not a training-throughput benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup


def benchmark(scenario_path: Path, *, steps: int, warmup: int, seed: int,
              traffic: bool = False) -> dict:
    overrides = []
    if traffic:
        overrides = [
            'environment.spawn.policy="centerline_random"',
            'environment.spawn.centerline.min_distance=2.0',
            'environment.respawn_agents=["car_1","car_2","car_3","car_4","car_5","car_6"]',
            'environment.respawn_on_vehicle_collision=true',
            'experiment.name="ppo_1v1_mpc_traffic_circle"',
        ]
        for index, (algorithm, target, maximum) in enumerate([
            ("kinematic_mpc", 1.5, 2.0),
            ("obstacle_aware_mpc", 2.0, 2.5),
            ("defensive_mpc", 2.0, 2.5),
            ("cbf_mpc", 2.5, 3.0),
            ("mpcc", 2.5, 3.0),
        ], start=2):
            config = {
                "algorithm": algorithm, "trainable": False, "role": "traffic",
                "target_id": "car_0", "action_adapter": "rolling_speed_to_wheel_v1",
                "params": {"dt": .05, "target_speed": target,
                           "min_speed": .5, "max_speed": maximum},
            }
            overrides.append(f"agents.car_{index}={json.dumps(config)}")
    scenario = load_and_expand_scenario(str(scenario_path), overrides=overrides)
    scenario["environment"]["render"] = False
    scenario_hash = hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(1)
    env, controllers, _ = create_training_setup(scenario, scenario_dir=scenario_path.parent)
    timings = {aid: [] for aid in controllers}
    timings["env.step"] = []
    actions_hash, states_hash = hashlib.sha256(), hashlib.sha256()
    resets = 0
    try:
        obs, _ = env.reset(seed=seed)
        for controller in controllers.values():
            controller.set_env(env)
            controller.reset()
        for step in range(warmup + steps):
            actions = {"car_0": np.zeros(2, dtype=np.float32)}
            for aid, controller in controllers.items():
                start = time.perf_counter()
                actions[aid] = controller.act(obs[aid])
                elapsed = time.perf_counter() - start
                if step >= warmup:
                    timings[aid].append(elapsed)
                actions_hash.update(actions[aid].tobytes())
            start = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(actions)
            elapsed = time.perf_counter() - start
            if step >= warmup:
                timings["env.step"].append(elapsed)
            states_hash.update(env.sim.agent_poses.tobytes())
            if any(terminated.values()) or any(truncated.values()):
                resets += 1
                obs, _ = env.reset(seed=seed + resets)
                for controller in controllers.values():
                    controller.reset()
        measured_s = sum(sum(values) for values in timings.values())
        return {
            "scenario": str(scenario_path), "seed": seed,
            "scenario_sha256": scenario_hash,
            "steps": steps, "warmup": warmup, "resets": resets,
            "controller_and_physics_steps_per_second": steps / measured_s,
            "mean_ms": {key: float(np.mean(values) * 1000) for key, values in timings.items()},
            "p95_ms": {key: float(np.percentile(values, 95) * 1000) for key, values in timings.items()},
            "actions_sha256": actions_hash.hexdigest(),
            "states_sha256": states_hash.hexdigest(),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scenario", type=Path, default=ROOT / "scenarios/ppo_1v1_racing_mpc_circle.yaml")
    parser.add_argument("--traffic", action="store_true",
                        help="Add the five mixed MPC traffic cars and randomized spawns")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("steps and warmup must be positive")
    result = benchmark(args.scenario.resolve(), steps=args.steps, warmup=args.warmup,
                       seed=args.seed, traffic=args.traffic)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
