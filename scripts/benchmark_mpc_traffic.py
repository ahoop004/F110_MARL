"""Time fixed MPC traffic and physics with a stationary ego, without PPO.

Compare the action/state hashes across revisions as well as warmed-up timings.
Use --render to include desktop drawing. This is a controller microbenchmark,
not a training-throughput benchmark; learner inference and updates are excluded.
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
              traffic: bool = False, render: bool = False, overrides=None) -> dict:
    scenario_overrides = []
    if traffic:
        scenario_overrides = [
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
            scenario_overrides.append(f"agents.car_{index}={json.dumps(config)}")
    scenario_overrides.extend(overrides or [])
    scenario = load_and_expand_scenario(str(scenario_path), overrides=scenario_overrides)
    scenario["environment"]["render"] = render
    scenario_hash = hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(1)
    setup_start = time.perf_counter()
    env, controllers, _ = create_training_setup(scenario, scenario_dir=scenario_path.parent)
    setup_s = time.perf_counter() - setup_start
    timings = {aid: [] for aid in controllers}
    timings["env.step"] = []
    if render:
        timings["env.render"] = []
    actions_hash, states_hash = hashlib.sha256(), hashlib.sha256()
    resets = 0
    reset_s = 0.
    try:
        if render and env._headless:
            raise RuntimeError("--render needs a display; unset PYGLET_HEADLESS for desktop timing")
        obs, _ = env.reset(seed=seed)
        for controller in controllers.values():
            controller.set_env(env)
            controller.reset()
        for step in range(warmup + steps):
            if step == warmup:
                measured_start = time.perf_counter()
            actions = {"car_0": np.zeros(2, dtype=np.float32)}
            for aid, controller in controllers.items():
                if aid not in env.agents:
                    continue
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
            if render:
                start = time.perf_counter()
                env.render()
                if step >= warmup:
                    timings["env.render"].append(time.perf_counter() - start)
            states_hash.update(env.sim.agent_poses.tobytes())
            if any(terminated.values()) or any(truncated.values()):
                start = time.perf_counter()
                resets += 1
                obs, _ = env.reset(seed=seed + resets)
                for controller in controllers.values():
                    controller.reset()
                if step >= warmup:
                    reset_s += time.perf_counter() - start
        measured_wall_s = time.perf_counter() - measured_start
        measured_s = sum(sum(values) for key, values in timings.items() if key != "env.render")
        return {
            "scenario": str(scenario_path), "seed": seed,
            "scenario_sha256": scenario_hash,
            "steps": steps, "warmup": warmup, "resets": resets,
            "render": render, "setup_seconds": setup_s,
            "measured_wall_seconds": measured_wall_s, "reset_seconds": reset_s,
            "measured_steps_per_second": steps / measured_wall_s,
            "controller_and_physics_steps_per_second": steps / measured_s,
            "mean_ms": {key: float(np.mean(values) * 1000) for key, values in timings.items() if values},
            "p95_ms": {key: float(np.percentile(values, 95) * 1000) for key, values in timings.items() if values},
            "actions_sha256": actions_hash.hexdigest(),
            "states_sha256": states_hash.hexdigest(),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scenario", type=Path, default=ROOT / "scenarios/ppo_1v1_racing_mpc.yaml")
    parser.add_argument("--render", action="store_true", help="Include desktop rendering (requires a display)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=YAML")
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
                       seed=args.seed, traffic=args.traffic, render=args.render, overrides=args.overrides)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
