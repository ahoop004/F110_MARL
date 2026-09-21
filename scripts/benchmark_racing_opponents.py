"""Bounded, reproducible fixed-controller races on the current physics.

No training or experiment defaults are modified. Write each completed race
immediately so an interrupted benchmark retains its observations.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import yaml
from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup
from scipy.spatial import cKDTree
from utils.track_preview import _resample_uniform


def scenario_for(controller, map_name, mode, seed, max_steps, laps):
    scenario = load_and_expand_scenario(str(ROOT/'scenarios/mappo_2v2_penalties_scratch.yaml'))
    # The experiment opponents may change; keep the named hybrid comparison
    # independent so it can never silently become a mislabeled MPC baseline.
    hybrid = yaml.safe_load((ROOT/'configs/controllers/hybrid_pp_ftg.yaml').read_text())
    candidate = yaml.safe_load((ROOT/'configs/controllers/racing_mpc.yaml').read_text())
    scenario['agents'] = {'car_0': deepcopy(candidate if controller == 'racing_mpc' else hybrid)}
    if mode in ('traffic', 'pair'):
        for i in range(1, 4):
            scenario['agents'][f'car_{i}'] = deepcopy(hybrid)
        if mode == 'pair':
            scenario['agents']['car_1'] = deepcopy(scenario['agents']['car_0'])
    if mode == 'passing':
        scenario['agents']['car_1'] = deepcopy(hybrid)
        scenario['agents']['car_1']['params']['max_speed'] = 1.2
        scenario['agents']['car_1']['params']['ftg']['max_speed'] = 1.2
    scenario['experiment'].update(seed=seed, episodes=1, num_envs=1, torch_threads=1)
    env = scenario['environment']
    for key in ['map_bundles', 'map_bundles_train', 'map_bundles_eval']:
        env[key] = [map_name]
    env.update(max_steps=max_steps, target_laps=laps, agent_teams={},
               random_spawn={'enabled': True, 'allow_reuse': False},
               episode_termination={'mode': 'all_agents', 'lap_completion': True})
    env['lap_counting']['require_finish_line'] = True
    scenario['evaluation'].update(max_steps=max_steps, target_laps=laps)
    return scenario


def run_race(controller, map_name, mode, seed, max_steps, laps):
    scenario = scenario_for(controller, map_name, mode, seed, max_steps, laps)
    env, agents, _ = create_training_setup(scenario, mode='eval', scenario_dir=ROOT/'scenarios')
    try:
        obs, info = env.reset(seed=seed)
        path, _ = _resample_uniform(env.centerline_points[:, :2], .1)
        tree = cKDTree(path)
        length = len(path)*.1
        if mode == 'passing':
            # Same deterministic centerline start for both controllers, slow car 3m ahead.
            idx = (seed*37) % len(path)
            poses = []
            for offset in (0, 30):
                i = (idx+offset) % len(path)
                delta = path[(i+1) % len(path)]-path[i]
                poses.append([*path[i], np.arctan2(delta[1], delta[0])])
            obs, info = env.reset(seed=seed, options={'poses': np.array(poses)})
        for agent in agents.values():
            agent.set_env(env)
            agent.reset()
        # Warm compilation/map setup outside latency timing. Reset policy memory.
        agents['car_0'].act(obs['car_0'], aid='car_0')
        agents['car_0'].reset()
        latency, fallbacks, timed_laps, previous_laps = [], 0, [], 0
        start = time.perf_counter()
        previous_s = {aid: float(tree.query(env.get_agent_state(aid).pose[:2])[1])*.1
                      for aid in env.possible_agents}
        total_s = {aid: (s-previous_s['car_0']+length/2) % length-length/2
                   for aid, s in previous_s.items()}
        eligible_passes = {aid for aid, gap in total_s.items() if aid != 'car_0' and 0 < gap < 10}
        passed = set()
        final = {}
        for step in range(max_steps):
            actions = {}
            for aid in env.agents:
                t = time.perf_counter()
                actions[aid] = agents[aid].act(obs[aid], deterministic=True, aid=aid)
                if aid == 'car_0':
                    latency.append(time.perf_counter()-t)
                    fallbacks += bool(getattr(agents[aid], 'last_plan', {}).get('brake_fallback', False))
            obs, _, terminated, truncated, info = env.step(actions)
            for aid in env.possible_agents:
                s = float(tree.query(env.get_agent_state(aid).pose[:2])[1])*.1
                total_s[aid] += (s-previous_s[aid]+length/2) % length-length/2
                previous_s[aid] = s
                if aid in eligible_passes and total_s['car_0']-total_s[aid] > env.params['length']:
                    # Separate passing an active car from driving past its wreck.
                    if env.get_agent_state(aid).metadata.get('status') == 'active':
                        passed.add(aid)
            if 'car_0' in info:
                final = info['car_0']
                count = int(final.get('lap_count', 0))
                if count > previous_laps:
                    timed_laps.append(float(final['lap_time_steps'])*env.timestep)
                    previous_laps = count
            if mode == 'pair':
                finished = all(aid not in env.agents for aid in ('car_0', 'car_1'))
            else:
                finished = bool(terminated.get('car_0') or truncated.get('car_0'))
            if finished:
                break
            if (step+1) % 1000 == 0:
                print(f'{controller} {map_name} {mode} seed={seed}: '
                      f'{step+1} steps, laps={previous_laps}', file=sys.stderr, flush=True)
        result = {
            'controller': controller, 'map': map_name, 'mode': mode, 'seed': seed,
            'steps': step+1, 'simulation_s': (step+1)*env.timestep,
            'wall_s': time.perf_counter()-start, 'laps': int(final.get('lap_count', 0)),
            'outcome': final.get('terminal_reason') or ('timeout' if step+1 == max_steps else 'unknown'),
            'completed': bool(final.get('race_completed', False)),
            'collision': bool(final.get('collision', False)), 'lap_times_s': timed_laps,
            'decision_ms_mean': float(np.mean(latency)*1000),
            'decision_ms_p95': float(np.percentile(latency, 95)*1000),
            'brake_fallbacks': fallbacks,
            'active_cars_passed': sorted(passed),
            'pair_both_finished': (all(env.get_agent_state(aid).progress.finished for aid in ('car_0', 'car_1'))
                                   if mode == 'pair' else None),
            'scenario_sha256': hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            'controller_config': scenario['agents']['car_0'],
            'other_agents_at_stop': {
                aid: {'laps': int(env.get_agent_state(aid).progress.lap_count),
                      'collision': bool(env.get_agent_state(aid).collision),
                      'finished': bool(env.get_agent_state(aid).progress.finished)}
                for aid in env.possible_agents if aid != 'car_0'},
            'controller_source_sha256': hashlib.sha256((ROOT/'src/agents/mpc/racing.py'
                if controller == 'racing_mpc' else ROOT/'src/agents/waypoint.py').read_bytes()).hexdigest(),
        }
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--controllers', nargs='+', choices=['hybrid_pp_ftg', 'racing_mpc'], default=['hybrid_pp_ftg', 'racing_mpc'])
    parser.add_argument('--maps', nargs='+', default=['circle_map', 'Budapest_map'])
    parser.add_argument('--modes', nargs='+', choices=['solo', 'traffic', 'pair', 'passing'], default=['solo', 'traffic'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[10042, 10043, 10044])
    parser.add_argument('--max-steps', type=int, default=16000)
    parser.add_argument('--laps', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps < 1 or args.laps < 1:
        parser.error('max-steps and laps must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite evidence from earlier runs.
    with args.output.open('x') as handle:
        for map_name in args.maps:
            for mode in args.modes:
                for seed in args.seeds:
                    for controller in args.controllers:
                        result = run_race(controller, map_name, mode, seed, args.max_steps, args.laps)
                        line = json.dumps(result, allow_nan=False)
                        handle.write(line+'\n')
                        handle.flush()
                        print(line, flush=True)


if __name__ == '__main__':
    main()
