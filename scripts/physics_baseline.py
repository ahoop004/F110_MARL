#!/usr/bin/env python3
"""Capture/check frozen scripted physics rollouts; no training or checkpoint load."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.agent_builder import get_trainable_agent_ids
from core.provenance import build_run_provenance
from core.scenario import load_and_expand_scenario, load_yaml_config
from core.setup import create_training_setup
from env.types import SpawnPlan, SpawnState
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer

SCENARIOS = (
    "legacy/ppo.yaml",
    "ppo_lap_completion_pretrain.yaml",
    "ppo_lap_completion_transfer.yaml",
    "legacy/mappo_gaplock.yaml",
)
DEFAULT_OUTPUT = ROOT / "tests/fixtures/physics_baseline"
PHYSICS_SOURCES = (
    "src/physics/dynamic_models.py", "src/physics/vehicle.py",
    "src/physics/simulaton.py", "src/physics/integration.py",
    "src/env/f110ParallelEnv.py", "src/wrappers/actions/composer.py",
    "src/wrappers/observations/track.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset_hashes(env) -> dict:
    # Hash geometry and occupancy assets, not only the map YAML.
    directory = Path(env.yaml_path).parent
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in {".yaml", ".csv", ".png", ".pgm", ".jpg"}
    }


def rollout(case: dict, *, explicit_model: bool = False) -> tuple[dict, dict]:
    # Freeze inputs independently of future scenario and include edits.
    scenario = json.loads(json.dumps(case["scenario"]))
    if explicit_model:
        scenario["environment"].setdefault("vehicle_params", {}).update(
            model="legacy_st", model_version=1
        )
    env, _, _ = create_training_setup(scenario, mode="eval", scenario_dir=ROOT / "scenarios")
    try:
        ids = list(env.possible_agents)
        trainable = get_trainable_agent_ids(scenario["agents"])
        observations = {
            aid: ObservationComposer.from_config(case["observations"][aid], scenario["environment"])
            for aid in trainable
        }
        repeat = int(scenario["environment"].get("action_repeat", 1))
        dt = env.timestep * repeat
        actions = {
            aid: ActionComposer.from_config(
                env.action_spaces[aid].low, env.action_spaces[aid].high,
                scenario["agents"][aid].get("action_constraints", {}), decision_dt=dt,
            ) for aid in ids
        }
        if "spawn_plan" in case:
            plan = SpawnPlan(
                states=tuple(SpawnState(agent_id=aid, pose=np.array(entry["pose"]),
                                       spawn_id=entry["spawn_id"])
                             for aid, entry in case["spawn_plan"].items()),
                plan_id="physics_baseline", map_id=case["map"],
            )
            raw, info = env.reset(seed=case["seed"], options={"spawn_plan": plan})
        else:
            raw, info = env.reset(seed=case["seed"])
        spawn = {
            aid: {"pose": np.asarray(raw[aid]["pose"]).tolist(),
                  "spawn_id": info[aid].get("spawn_id")}
            for aid in ids
        }
        series = {"state": [], "global": [], "physical_actions": [],
                  "normalized_actions": [], "terminated": [], "truncated": []}
        series.update({f"obs_{aid}": [] for aid in trainable})

        def snapshot():
            series["state"].append(np.stack([car.state.copy() for car in env.sim.agents]))
            series["global"].append(env.get_global_state().vector.copy())
            for aid in trainable:
                series[f"obs_{aid}"].append(observations[aid].wrap(raw[aid], info[aid]))

        snapshot()
        commands = np.asarray(case["commands"], dtype=np.float32)
        for command in commands:
            active = list(env.agents)
            physical = {aid: actions[aid].process(command[index])
                        for index, aid in enumerate(ids) if aid in active}
            for aid in trainable:
                if aid in active:
                    observations[aid].update_prev_action(command[ids.index(aid)])
            for _ in range(repeat):
                raw, _, terminated, truncated, info = env.step(physical)
                series["physical_actions"].append(np.stack([
                    physical.get(aid, np.zeros(2, dtype=np.float32)) for aid in ids]))
                series["normalized_actions"].append(command.copy())
                series["terminated"].append([terminated[aid] for aid in ids])
                series["truncated"].append([truncated[aid] for aid in ids])
                snapshot()
                if env.episode_done:
                    break
            if env.episode_done:
                break
        details = {
            "spawn_plan": spawn, "map": env._map_bundle_active or Path(env.yaml_path).stem,
            "assets": asset_hashes(env), "vehicle_params": env.params,
            "timestep": env.timestep, "integrator": str(env.integrator),
            "action_contracts": {
                aid: {"low": env.action_spaces[aid].low.tolist(),
                      "high": env.action_spaces[aid].high.tolist(),
                      "constraints": scenario["agents"][aid].get("action_constraints", {}),
                      "contract": ActionComposer.contract_from_config(
                          scenario["agents"][aid].get("action_constraints", {}), dt)}
                for aid in ids},
            "observation_dimensions": {aid: obs.obs_dim for aid, obs in observations.items()},
            "physics_substeps": len(series["physical_actions"]),
        }
        return {key: np.asarray(value) for key, value in series.items()}, details
    finally:
        env.close()


def capture(output: Path, decisions: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    sources = {relative: sha256(ROOT / relative) for relative in PHYSICS_SOURCES}
    cases = {}
    for filename in SCENARIOS:
        path = ROOT / "scenarios" / filename
        scenario = load_and_expand_scenario(str(path))
        ids = list(scenario["agents"])
        # All cars follow stored commands, including normally fixed opponents.
        # This measures physics/composers, not controller or trainer quality.
        commands = np.zeros((decisions, len(ids), 2), dtype=np.float32)
        for index, aid in enumerate(ids):
            commands[:, index, 0] = 0.12 * np.sin(np.arange(decisions) * 0.08 + index)
            constraints = scenario["agents"][aid].get("action_constraints", {})
            if constraints.get("speed_control", "direct") == "acceleration":
                commands[:, index, 1] = np.resize(np.repeat([0.8, 0.0, -0.7, 0.3], 40), decisions)
            else:
                commands[:, index, 1] = np.resize(np.repeat([0.12, 0.22, -1.0, 0.1], 40), decisions)
        trainable = get_trainable_agent_ids(scenario["agents"])
        case = {
            "scenario": scenario, "seed": 42, "commands": commands.tolist(),
            "observations": {aid: (load_yaml_config(path.parent / scenario["agents"][aid]["observation"])
                if isinstance(scenario["agents"][aid]["observation"], str)
                else scenario["agents"][aid]["observation"])
                             for aid in trainable},
            "rewards": {aid: (load_yaml_config(path.parent / scenario["agents"][aid]["reward"])
                if isinstance(scenario["agents"][aid]["reward"], str)
                else scenario["agents"][aid]["reward"])
                        for aid in trainable},
            "provenance": build_run_provenance(
                scenario, scenario_path=path, run_id="physics_baseline",
                algorithm=scenario["agents"][trainable[0]]["algorithm"], trainable_agents=trainable),
        }
        arrays, details = rollout(case)
        case.update(details)
        np.savez_compressed(output / f"{path.stem}.npz", **arrays)
        cases[path.stem] = case
    manifest = {"version": 1, "python": platform.python_version(),
                "numpy": np.__version__,
                "libraries": {name: version(name) for name in ("numba", "scipy", "torch")},
                "physics_sources": {"origin": "workspace_at_capture", "sha256": sources},
                "cases": cases}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def check(output: Path, *, explicit_model: bool = False) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, case in manifest["cases"].items():
        for relative, digest in case["assets"].items():
            if sha256(ROOT / relative) != digest:
                raise ValueError(f"Baseline map asset changed: {relative}")
        actual, details = rollout(case, explicit_model=explicit_model)
        for field in ("physics_substeps", "timestep", "integrator", "action_contracts",
                      "observation_dimensions", "map"):
            assert details[field] == case[field], f"{name}/{field}: contract changed"
        actual_params = {key: value for key, value in details["vehicle_params"].items()
                         if key not in {"model", "model_version"}}
        expected_params = {key: value for key, value in case["vehicle_params"].items()
                           if key not in {"model", "model_version"}}
        assert actual_params == expected_params, f"{name}: vehicle parameters changed"
        with np.load(output / f"{name}.npz", allow_pickle=False) as expected:
            assert set(actual) == set(expected.files), name
            for key, value in actual.items():
                assert value.shape == expected[key].shape, f"{name}/{key}: shape changed"
                assert value.dtype == expected[key].dtype, f"{name}/{key}: dtype changed"
                if value.dtype.kind == "b":
                    np.testing.assert_array_equal(value, expected[key], err_msg=f"{name}/{key}")
                else:
                    np.testing.assert_allclose(value, expected[key], rtol=1e-6, atol=1e-7,
                                               err_msg=f"{name}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "check"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decisions", type=int, default=160)
    parser.add_argument("--explicit-model", action="store_true")
    args = parser.parse_args()
    if args.decisions <= 0:
        parser.error("--decisions must be positive")
    if args.mode == "capture":
        capture(args.output, args.decisions)
    else:
        check(args.output, explicit_model=args.explicit_model)


if __name__ == "__main__":
    main()
