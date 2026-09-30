"""Compare serial and concurrent fixed-policy checkpoint evaluation on one node.

Races are capped for this timing probe; its scores are not checkpoint-selection
scores. The first evaluation includes cold worker startup. Later evaluations
reuse workers. No training, checkpoint writing, recording, or W&B is performed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
os.environ['PYGLET_HEADLESS'] = 'true'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'

import torch

from agents.mappo import MAPPOAgent
from core.scenario import load_and_expand_scenario, resolve_mappo_config
from core.setup import create_training_setup, build_obs_composers, resolve_training_params
from training.mappo_evaluator import DeterministicMAPPOEvaluator
from training.parallel_mappo_evaluator import ParallelMAPPOEvaluator
from wrappers.actions.composer import ActionComposer


def build_evaluator(args, workers):
    path = args.scenario.resolve()
    scenario = load_and_expand_scenario(str(path), overrides=args.overrides)
    ids = [aid for aid, cfg in scenario['agents'].items() if cfg.get('trainable')]
    if (not ids or any(scenario['agents'][aid]['algorithm'] != 'mappo' for aid in ids)
            or scenario.get('two_team') or scenario.get('skill_curriculum')):
        raise ValueError('This benchmark requires fixed-opponent MAPPO without a curriculum')
    scenario['experiment']['seed'] = scenario['evaluation']['seed']
    scenario['environment']['render'] = False
    scenario['environment']['max_steps'] = scenario['evaluation']['max_steps'] = args.max_steps
    env, opponents, _ = create_training_setup(scenario, mode='eval', scenario_dir=path.parent)
    try:
        for controller in opponents.values():
            if hasattr(controller, 'set_env'):
                controller.set_env(env)
        obs = build_obs_composers(scenario['agents'], ids, scenario['environment'], path.parent)
        space = env.action_spaces[ids[0]]
        snapshot = env.get_global_state()
        params = {**resolve_training_params(scenario['agents'][ids[0]], scenario), **resolve_mappo_config(scenario)}
        params.update(_observation_contract=obs[ids[0]].contract,
            _observation_contracts={aid: obs[aid].contract for aid in ids},
            _observation_dims={aid: obs[aid].obs_dim for aid in ids},
            _global_state_contract_version=snapshot.metadata['vector_contract_version'])
        agent = MAPPOAgent(max(o.obs_dim for o in obs.values()), len(snapshot.vector),
                           space.low, space.high, ids, params)
        if args.checkpoint:
            agent.load(str(args.checkpoint.resolve()))
        else:
            source = params.get('pretrained_actor_checkpoint')
            if not source:
                raise ValueError('Supply --checkpoint or configure a pretrained_actor_checkpoint')
            agent.load_pretrained_actor(str((path.parent / source).resolve()))
        action = ActionComposer.from_config(space.low, space.high,
            scenario['agents'][ids[0]].get('action_constraints', {}),
            decision_dt=env.timestep * scenario['environment'].get('action_repeat', 1))
        kwargs = dict(env=env, trainable_ids=ids, other_agents=opponents, obs_composers=obs,
            action_composer=action, episodes=scenario['evaluation']['episodes'],
            base_seed=scenario['evaluation']['seed'], action_repeat=scenario['environment'].get('action_repeat', 1),
            focal_agent_id=scenario['evaluation'].get('progress_agent_id'))
        evaluator = (ParallelMAPPOEvaluator(scenario=scenario, scenario_dir=path.parent,
            num_workers=workers, **kwargs) if workers > 1 else DeterministicMAPPOEvaluator(**kwargs))
        return evaluator.bind_agent(agent)
    except BaseException:
        env.close()
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--scenario', type=Path, default=ROOT / 'scenarios/mappo_1v1_attack.yaml')
    parser.add_argument('--checkpoint', type=Path, help='MAPPO checkpoint; otherwise use the configured PPO actor')
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--max-steps', type=int, default=128, help='Timing-only episode cap')
    parser.add_argument('--repetitions', type=int, default=2, help='Warm evaluations after the first cold evaluation')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=YAML')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(*args.workers, args.max_steps, args.repetitions) < 1:
        parser.error('workers, max-steps, and repetitions must be positive')
    if args.output and args.output.exists():
        parser.error('output already exists; choose a new filename')
    torch.set_num_threads(1)
    results, reference = [], None
    for workers in args.workers:
        evaluator = build_evaluator(args, workers)
        try:
            for iteration in range(args.repetitions + 1):
                started = time.perf_counter()
                summary = evaluator.evaluate()
                elapsed = time.perf_counter() - started
                if reference is None:
                    reference = summary
                row = dict(workers=min(workers, evaluator.episodes), iteration=iteration,
                    cold=iteration == 0, seconds=elapsed,
                    physics_steps=sum(r['physics_steps'] for r in summary['episode_results']),
                    matches_first_summary=summary == reference)
                results.append(row)
                print(json.dumps(row), flush=True)
                if args.output:
                    args.output.write_text(json.dumps(dict(scenario=str(args.scenario),
                        checkpoint=str(args.checkpoint) if args.checkpoint else 'configured pretrained actor',
                        max_steps=args.max_steps, overrides=args.overrides, results=results), indent=2) + '\n')
        finally:
            evaluator.close()


if __name__ == '__main__':
    main()
