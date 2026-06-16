#!/usr/bin/env python3
"""Probe runner for pure heuristic (non-RL) scenarios.

Discovers all valid map bundles, creates one env per map, runs N episodes each,
and prints a per-map summary: avg steps, collision rate, timeout rate.

Usage:
    python probe.py --scenario scenarios/nrl_1car.yaml
    python probe.py --scenario scenarios/nrl_4car.yaml --episodes 20 --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.scenario import ScenarioError, load_and_expand_scenario
from core.map_selection import (
    discover_map_bundles,
    apply_map_bundle,
    normalize_maps_key,
    coerce_bundle_list,
)
from core.agent_builder import build_fixed_policy_agents, get_trainable_agent_ids
from core.config import register_builtin_agents
from core.env_builder import create_environment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe non-RL heuristic controllers across all maps")
    p.add_argument("--scenario", required=True, help="Path to scenario YAML")
    p.add_argument("--episodes", type=int, default=None, help="Total episodes (split across maps)")
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def resolve_all_maps(scenario: Dict[str, Any]) -> List[str]:
    """Return every valid map bundle from the scenario without applying train/eval split."""
    env_cfg = dict(scenario["environment"])
    env_cfg = normalize_maps_key(env_cfg)
    raw = env_cfg.get("map_bundles")
    if raw is True or (isinstance(raw, str) and raw.strip().lower() in {"auto", "all"}):
        return discover_map_bundles(env_cfg)
    if raw:
        return coerce_bundle_list(raw) or []
    # Single-map scenario: already resolved by normalize_maps_key
    bundle = env_cfg.get("map_bundle_active") or env_cfg.get("map_bundle")
    return [bundle] if bundle else []


def build_env_for_bundle(
    scenario: Dict[str, Any], bundle: str
) -> Any:
    """Create a fresh env configured for one specific map bundle."""
    import copy

    s = copy.deepcopy(scenario)
    env_cfg = s["environment"]
    env_cfg = normalize_maps_key(env_cfg)

    # Strip cycling / split keys — we manage maps manually
    for key in ("map_bundles", "map_bundles_train", "map_bundles_eval",
                "map_bundle_active", "map_cycle", "map_pick", "maps"):
        env_cfg.pop(key, None)

    env_cfg = apply_map_bundle(env_cfg, bundle)
    s["environment"] = env_cfg

    seed = s.get("experiment", {}).get("seed")
    return create_environment(env_cfg, s["agents"], seed)


def run_episode(
    env,
    agents: Dict[str, Any],
    render: bool,
) -> Dict[str, Any]:
    obs_dict, _info = env.reset()
    for ag in agents.values():
        if hasattr(ag, "reset"):
            ag.reset()

    step = 0
    any_collision = False
    timeout = False
    done_set: set = set()
    agent_collided: Dict[str, bool] = {aid: False for aid in agents}
    last_info: Dict = {}

    while True:
        if not obs_dict or done_set.issuperset(agents.keys()):
            break

        actions: Dict[str, np.ndarray] = {}
        for aid, obs in obs_dict.items():
            if aid not in agents or aid in done_set:
                continue
            try:
                act = agents[aid].act(obs)
            except Exception:
                act = np.zeros(2, dtype=np.float32)
            actions[aid] = np.asarray(act, dtype=np.float32)

        if not actions:
            break

        obs_dict, _rewards, dones, truncs, last_info = env.step(actions)
        step += 1

        for aid in agents:
            if last_info.get(aid, {}).get("collision", False):
                agent_collided[aid] = True
                any_collision = True
            if dones.get(aid, False) or truncs.get(aid, False):
                done_set.add(aid)
                if truncs.get(aid, False):
                    timeout = True

        if render:
            env.render()

    return {
        "steps": step,
        "collision": any_collision,
        "agent_collided": agent_collided,
        "timeout": timeout,
    }


def main() -> None:
    args = parse_args()

    try:
        scenario = load_and_expand_scenario(args.scenario)
    except (ScenarioError, FileNotFoundError) as exc:
        print(f"Error loading scenario: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        scenario.setdefault("experiment", {})["seed"] = args.seed
    if args.render:
        scenario.setdefault("environment", {})["render"] = True

    agent_configs = scenario.get("agents", {})
    exp_cfg = scenario.get("experiment", {})

    trainable = get_trainable_agent_ids(agent_configs)
    if trainable:
        print(
            f"Error: scenario has trainable RL agents {trainable}.\n"
            "  Use run.py for RL training. probe.py is for heuristic-only scenarios.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_maps = resolve_all_maps(scenario)
    if not all_maps:
        print("No map bundles found. Check environment.maps or map_bundles.", file=sys.stderr)
        sys.exit(1)

    n_total = args.episodes or int(exp_cfg.get("episodes", 45))
    eps_per_map = max(1, n_total // len(all_maps))
    agent_ids = list(agent_configs.keys())

    print(f"Scenario : {exp_cfg.get('name', Path(args.scenario).stem)}")
    agents_str = ", ".join(f"{aid}={agent_configs[aid].get('algorithm', '?')}" for aid in agent_ids)
    print(f"Agents   : {agents_str}")
    print(f"Maps     : {', '.join(all_maps)}")
    print(f"Episodes : {eps_per_map} per map  ({eps_per_map * len(all_maps)} total)")
    print()

    register_builtin_agents()

    # Per-map stats accumulator: steps, collision count, timeout count
    map_stats: Dict[str, Tuple[List[int], int, int]] = {}

    for bundle in all_maps:
        env = build_env_for_bundle(scenario, bundle)
        agents = build_fixed_policy_agents(agent_configs)
        for ag in agents.values():
            if hasattr(ag, "set_env"):
                ag.set_env(env)

        steps_list: List[int] = []
        n_col = 0
        n_timeout = 0
        agent_col_counts: Dict[str, int] = {aid: 0 for aid in agent_ids}

        for ep in range(eps_per_map):
            result = run_episode(env, agents, args.render)
            steps_list.append(result["steps"])
            n_col += int(result["collision"])
            n_timeout += int(result["timeout"])
            for aid in agent_ids:
                if result["agent_collided"].get(aid, False):
                    agent_col_counts[aid] += 1
            if not args.quiet:
                col_tag = "COLLISION" if result["collision"] else "ok"
                end_tag = "TIMEOUT" if result["timeout"] else "done"
                per_agent = "  ".join(
                    f"{aid}:{'X' if result['agent_collided'].get(aid) else 'ok'}"
                    for aid in agent_ids
                )
                print(
                    f"  [{bundle:<22}] ep{ep+1:02d}  "
                    f"steps={result['steps']:4d}  {col_tag}  {end_tag}  [{per_agent}]"
                )

        try:
            env.close()
        except Exception:
            pass

        map_stats[bundle] = (steps_list, n_col, n_timeout, agent_col_counts)

    # Summary table
    col_w = 22
    per_agent_header = "  ".join(f"{aid}COL%" for aid in agent_ids)
    total_w = col_w + 32 + 10 * len(agent_ids)
    print()
    print("=" * total_w)
    print(f"{'MAP':<{col_w}} {'AVG_STEPS':>10} {'ANY_COL%':>9} {'TIMEOUT%':>9}  {per_agent_header}")
    print("-" * total_w)
    for bundle, (steps_list, n_col, n_timeout, agent_col_counts) in map_stats.items():
        n = len(steps_list)
        agent_col_str = "  ".join(f"{100*agent_col_counts[aid]/n:>7.1f}%" for aid in agent_ids)
        print(
            f"{bundle:<{col_w}} "
            f"{np.mean(steps_list):>10.1f} "
            f"{100*n_col/n:>8.1f}% "
            f"{100*n_timeout/n:>8.1f}%  "
            f"{agent_col_str}"
        )
    print("=" * total_w)


if __name__ == "__main__":
    main()
