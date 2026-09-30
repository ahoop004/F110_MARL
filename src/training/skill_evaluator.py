"""Fixed tactical benchmarks and paired solo retention, isolated from training."""
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from core.scenario import load_and_expand_scenario, resolve_evaluation_protocol
from core.setup import create_training_setup, build_obs_composers
from training.mappo_evaluator import DeterministicMAPPOEvaluator
from training.two_team import preserve_rng
from wrappers.actions.composer import ActionComposer


def summarize_skill(rows, ego):
    facts = [row['agents'][ego]['skill'] for row in rows]
    passes = [f['pass_time_s'] for f in facts if f['pass_time_s'] is not None]
    return dict(episodes=len(rows), success_rate=float(np.mean([f['success'] for f in facts])),
                ego_failure_rate=float(np.mean([f['ego_failed'] for f in facts])),
                opponent_failure_rate=float(np.mean([f['outcome'] == 'opponent_failure' for f in facts])),
                mean_pass_time_s=float(np.mean(passes)) if passes else None,
                mean_progress=float(np.mean([f['ego_progress'] for f in facts])),
                lead_retention=float(np.mean([f['lead_retention'] for f in facts])))


def retention_result(base_rows, adapted_rows, ego, config):
    """Pair only identical map/seed trials; missing clean timing cannot pass."""
    def keyed(rows):
        result = {(r['map_id'], r['seed']): r['agents'][ego] for r in rows}
        if not rows or len(result) != len(rows):
            raise ValueError('Retention requires nonempty unique map/seed trials')
        return result
    base, adapted = keyed(base_rows), keyed(adapted_rows)
    if base.keys() != adapted.keys():
        raise ValueError('Retention requires identical base and adapter map/seed trials')
    def clean(row):
        return bool(row['finished'] and not row['collision_dnf'] and not row['boundary_dnf']
                    and not row['boundary_respawns'] and not row['collision_respawns'])
    base_rate = sum(clean(r) for r in base.values()) / len(base)
    adapted_rate = sum(clean(r) for r in adapted.values()) / len(adapted)
    pairs = [(base[k]['mean_valid_lap_time_s'], adapted[k]['mean_valid_lap_time_s'])
             for k in base if clean(base[k]) and clean(adapted[k])]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None and a > 0 and b > 0]
    ratio = float(np.mean([b for a, b in pairs]) / np.mean([a for a, b in pairs])) if pairs else None
    passed = (adapted_rate + 1e-12 >= base_rate - config.get('max_completion_drop', .05)
              and ratio is not None and ratio <= 1 + config.get('max_lap_time_increase', .1) + 1e-12)
    return dict(retention_passed=passed, retention_base_completion=base_rate,
                retention_completion=adapted_rate, retention_lap_time_ratio=ratio,
                retention_paired_laps=len(pairs))


def base_policy(agent):
    """Deterministic frozen network from a self-contained adapter checkpoint."""
    base = copy(agent)
    base.actor = deepcopy(agent.actor)
    base.last_raw_actions = {}
    with torch.no_grad():
        for bank in base.actor.adapters:
            for residual in bank.values():
                residual.B.zero_()
    base.actor.requires_grad_(False)
    return base


class SkillEvaluator:
    def __init__(self, *, scenario, scenario_dir, agent, output_dir, curriculum=None,
                 protocol='selection', render=False):
        self.scenario = deepcopy(scenario)
        self.directory = Path(scenario_dir)
        self.agent = agent
        self.base = base_policy(agent)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.curriculum = curriculum
        self.protocol = protocol
        self.render = render
        self.progress_callback = None
        self.recording = None
        self._evaluators = {}
        self._baseline = {}

    def set_progress_callback(self, callback):
        previous, self.progress_callback = self.progress_callback, callback
        for evaluator in self._evaluators.values():
            evaluator.set_progress_callback(callback)
        return previous

    def _get_evaluator(self, scenario, stage=None):
        skill = scenario['environment']['skill_task']['skill']
        key = (skill, stage) if stage is not None else ('solo',)
        if key in self._evaluators:
            return self._evaluators[key]
        config = deepcopy(scenario)
        config['environment']['render'] = self.render
        ego = config['environment']['skill_task']['ego_id']
        protocol = resolve_evaluation_protocol(scenario, self.protocol)
        factor = 2 if self.protocol == 'final' else 1
        retention = config['skill_curriculum'].get('retention', {})
        if stage is None:
            config.pop('skill_curriculum')
            config['environment'].pop('skill_task')
            learner = deepcopy(config['agents'][ego])
            learner.pop('target_id', None)
            # Environment requests only driving geometry. The actor composer below
            # still includes its five target fields, which are zero when absent.
            learner['observation']['observation'].pop('target_frenet')
            config['agents'] = {ego: learner}
            config['environment']['spawn'] = dict(policy='centerline_random',
                centerline=dict(min_distance=1.), ego=dict(speed=0.))
            config['environment']['target_laps'] = 1
            config['environment']['episode_termination'] = dict(mode='all_trainable', lap_completion=True)
            config['evaluation'].update(target_laps=1, lap_completion=True,
                max_steps=retention.get('max_steps', 16000))
            episodes = factor * retention.get('episodes_per_map', 5) * len(config['environment']['map_bundles_eval'])
            seed = protocol['seed'] + 200000
        else:
            config['evaluation']['max_steps'] = protocol['max_steps']
            stages = config['skill_curriculum']['stages']
            episodes = factor * stages[stage]['evaluation_episodes_per_map'] * len(stages[stage]['maps'])
            offset = sum(s['evaluation_episodes_per_map'] * len(s['maps']) for s in stages[:stage]) * factor
            seed = protocol['seed'] + (100000 if skill == 'defend' else 0) + offset
        config['experiment']['seed'] = seed
        config['environment']['max_steps'] = config['evaluation']['max_steps']
        env = None
        try:
            with preserve_rng():
                env, opponents, _ = create_training_setup(config, mode='eval', scenario_dir=self.directory)
                if stage is not None:
                    env.set_skill_stage(stage)
                obs = build_obs_composers(scenario['agents'], [ego], config['environment'], self.directory)
                space = env.action_spaces[ego]
                if (obs[ego].contract != self.agent.observation_contract or
                        not np.array_equal(space.low, self.agent.action_low) or
                        not np.array_equal(space.high, self.agent.action_high)):
                    raise ValueError('Skill evaluation must preserve actor observation/action contracts')
                for controller in opponents.values():
                    controller.set_env(env)
                actions = ActionComposer.from_config(space.low, space.high,
                    scenario['agents'][ego].get('action_constraints', {}), decision_dt=env.timestep)
                evaluator = DeterministicMAPPOEvaluator(env=env, trainable_ids=[ego],
                    other_agents=opponents, obs_composers=obs, action_composer=actions,
                    episodes=episodes, base_seed=seed, protocol_name=self.protocol).bind_agent(self.agent)
                evaluator.render = self.render
                evaluator.set_progress_callback(self.progress_callback)
                self._evaluators[key] = evaluator
                return evaluator
        except BaseException:
            if env is not None:
                env.close()
            raise

    def _run(self, scenario, agent, *, solo=False):
        ego = scenario['environment']['skill_task']['ego_id']
        if solo:
            return self._get_evaluator(scenario).bind_agent(agent).evaluate()
        stages, rows = [], []
        for index, stage in enumerate(scenario['skill_curriculum']['stages']):
            summary = self._get_evaluator(scenario, index).bind_agent(agent).evaluate()
            episodes = summary['episode_results']
            for row in episodes:
                row['skill_stage'] = stage['name']
            stages.append(dict(name=stage['name'], **summarize_skill(episodes, ego),
                per_map={name: summarize_skill([r for r in episodes if r['map_id'] == name], ego)
                         for name in stage['maps']}))
            rows.extend(episodes)
        passing = scenario['environment']['skill_task']['skill'] == 'pass'
        times = [s['mean_pass_time_s'] for s in stages if s['mean_pass_time_s'] is not None]
        progress = float(np.mean([s['mean_progress'] for s in stages]))
        return dict(skill=scenario['environment']['skill_task']['skill'],
                    benchmark=dict(task=deepcopy(scenario['environment']['skill_task']),
                        stages=deepcopy(scenario['skill_curriculum']['stages']),
                        protocol=resolve_evaluation_protocol(scenario, self.protocol)),
                    stages=stages, episode_results=rows, episodes=len(rows),
                    skill_success_rate=float(np.mean([s['success_rate'] for s in stages])),
                    skill_ego_failure_rate=float(np.mean([s['ego_failure_rate'] for s in stages])),
                    skill_mean_pass_time_s=float(np.mean(times)) if times else None,
                    skill_mean_progress=progress,
                    skill_lead_retention=float(np.mean([s['lead_retention'] for s in stages])),
                    skill_tiebreak=(-float(np.mean(times)) if times else -1e30) if passing else progress)

    def _base_run(self, scenario, *, solo=False):
        key = 'solo' if solo else scenario['environment']['skill_task']['skill']
        if key not in self._baseline:
            self._baseline[key] = self._run(scenario, self.base, solo=solo)
            payload = dict(protocol=self.protocol, source=self.agent.pretrained_actor_source,
                           configuration_sha256=hashlib.sha256(json.dumps(self.scenario, sort_keys=True).encode()).hexdigest(),
                           results=self._baseline)
            (self.output / f'skill_base_{self.protocol}.json').write_text(json.dumps(payload, indent=2) + '\n')
        return self._baseline[key]

    def evaluate(self):
        # Base measurements are cached once per fixed protocol, never selected
        # using final-test data or changed by the curriculum's active stage.
        base_skill = self._base_run(self.scenario)
        base_solo = self._base_run(self.scenario, solo=True)
        summary = self._run(self.scenario, self.agent)
        solo = self._run(self.scenario, self.agent, solo=True)
        summary.update(retention_result(base_solo['episode_results'], solo['episode_results'],
            self.scenario['environment']['skill_task']['ego_id'], self.scenario['skill_curriculum'].get('retention', {})))
        summary['solo'] = solo
        ego = self.scenario['environment']['skill_task']['ego_id']
        summary['retention_per_map'] = {
            name: retention_result(
                [r for r in base_solo['episode_results'] if r['map_id'] == name],
                [r for r in solo['episode_results'] if r['map_id'] == name], ego,
                self.scenario['skill_curriculum'].get('retention', {}))
            for name in sorted({r['map_id'] for r in solo['episode_results']})}
        summary['base_skill_success_rate'] = base_skill['skill_success_rate']
        summary['evaluation_protocol'] = dict(name=self.protocol, deterministic=True,
            tactical_seed_offset={'pass': 0, 'defend': 100000}, solo_seed_offset=200000,
            seed=resolve_evaluation_protocol(self.scenario, self.protocol)['seed'])
        if self.curriculum is not None:
            state = self.curriculum.observe(summary)
            self.agent.skill_curriculum_state = state
            summary['curriculum'] = state
            summary['curriculum_stage'] = state['stage']
            summary['curriculum_streak'] = state['streak']
        return summary

    def evaluate_final(self):
        if self.protocol != 'final' or self.curriculum is not None:
            raise ValueError('Final skill evaluation must be isolated from curriculum/selection')
        own = self.scenario['environment']['skill_task']['skill']
        other = 'defend' if own == 'pass' else 'pass'
        scenario = load_and_expand_scenario(str(self.directory / f'mappo_1v1_{other}_lora.yaml'))
        # Cross-skill testing uses the requested source run's physical and actor
        # contracts and protocol seeds, with the other skill's task/curriculum.
        for key in ('vehicle_params', 'timestep', 'lidar_beams', 'lidar_range', 'track_preview', 'friction'):
            scenario['environment'][key] = deepcopy(self.scenario['environment'][key])
        ego = scenario['environment']['skill_task']['ego_id']
        for key in ('observation', 'action_constraints'):
            scenario['agents'][ego][key] = deepcopy(self.scenario['agents'][ego][key])
        scenario['evaluation']['final_test']['seed'] = self.scenario['evaluation']['final_test']['seed']
        result = self.evaluate()
        result['cross_skill'] = self._run(scenario, self.agent)
        result['cross_skill']['skill'] = other
        result['base'] = {own: self._base_run(self.scenario), other: self._base_run(scenario),
                          'solo': self._base_run(self.scenario, solo=True)}
        return result

    def close(self):
        for evaluator in self._evaluators.values():
            evaluator.close()
