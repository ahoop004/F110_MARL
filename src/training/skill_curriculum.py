"""Parent-owned, evaluation-gated curricula for independent tactical adapters."""
from copy import deepcopy
import math

from env.skills import SKILL_SEED_OFFSETS, validate_skill_task


def _number(value, name, *, low=0., high=math.inf):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f'{name} must be finite in [{low}, {high}]')


def validate_skill_curriculum(scenario):
    env, agents = scenario['environment'], scenario['agents']
    raw = env.get('skill_task')
    curriculum = scenario.get('skill_curriculum')
    if raw is None and curriculum is None:
        return
    if raw is None or curriculum is None:
        raise ValueError('skill_task and skill_curriculum must be configured together')
    task = validate_skill_task(raw, agents)
    ego, target = task['ego_id'], task.get('target_id')
    recovery = task['skill'] == 'recovery'
    learners = [aid for aid, cfg in agents.items() if cfg.get('trainable')]
    params = {**scenario.get('training_defaults', {}), **agents[ego].get('params', {})}
    if (learners != [ego] or agents[ego]['algorithm'] != 'mappo'
            or (params.get('lora') or {}).get('mode') != 'shared'):
        raise ValueError('Skills require one shared-LoRA MAPPO learner')
    if recovery:
        if len(agents) != 1 or agents[ego].get('target_id') is not None:
            raise ValueError('Recovery requires a single learner without a target')
    elif (len(agents) != 2 or agents[target]['algorithm'] != 'racing_mpc'
          or agents[ego].get('target_id') != target):
        raise ValueError('Interaction skills require a learner targeting one fixed racing_mpc')
    limits = env.get('track_limits', {})
    if (env.get('attack_task') or env.get('respawn') or env.get('respawn_agents') or
            env.get('respawn_on_vehicle_collision') or scenario.get('curriculum') or scenario.get('map_curriculum')):
        raise ValueError('Skills cannot combine attack, respawn, or other curricula')
    if (env.get('episode_termination', {}).get('lap_completion', True) or
            not limits.get('enabled') or not limits.get('terminate') or
            env.get('terminate_on_collision') is not True or env.get('action_repeat', 1) != 1):
        raise ValueError('Skills require no lap completion, collision/boundary termination and action_repeat=1')
    if (env.get('max_steps', 0) <= 0 or env.get('map_cycle') != 'per_episode' or
            env.get('map_pick') != 'round_robin' or env.get('epoch_shuffle', False)):
        raise ValueError('Skills require bounded episodes and per_episode round_robin maps')
    allowed = {'stages', 'success_threshold', 'max_ego_failure_rate', 'required_evaluations', 'retention'}
    if not isinstance(curriculum, dict) or set(curriculum) - allowed:
        raise ValueError(f'skill_curriculum accepts {sorted(allowed)}')
    stages = curriculum.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('skill_curriculum requires stages')
    configured = set(env.get('map_bundles', []))
    eval_maps = env.get('map_bundles_eval', [])
    if not eval_maps or len(set(eval_maps)) != len(eval_maps) or set(eval_maps) != configured:
        raise ValueError('Skill retention requires each configured map exactly once in map_bundles_eval')
    names = set()
    for stage in stages:
        ranges = ('lateral_offset', 'initial_speed', 'heading_error') if recovery else (
            'gap', 'lateral_offset', 'initial_speed', 'opponent_speed')
        fields = {'name', 'maps', 'evaluation_episodes_per_map', *ranges}
        if not isinstance(stage, dict) or set(stage) != fields:
            raise ValueError(f'Skill stages require {sorted(fields)}')
        name = stage['name']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Skill stage names must be unique nonempty strings')
        names.add(name)
        maps = stage['maps']
        if (not isinstance(maps, list) or not maps or len(set(maps)) != len(maps)
                or not set(maps) <= configured):
            raise ValueError('Stage maps must be unique configured bundles')
        for field in ranges:
            pair = stage[field]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f'Stage {field} requires [minimum, maximum]')
            for value in pair:
                _number(value, field, low=(-math.pi if field == 'heading_error' else
                                          -math.inf if field == 'lateral_offset' else 0.),
                        high=math.pi if field == 'heading_error' else math.inf)
            if pair[0] > pair[1] or (field in {'gap', 'opponent_speed'} and pair[0] <= 0):
                raise ValueError(f'Invalid stage {field} range')
        count = stage['evaluation_episodes_per_map']
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError('evaluation_episodes_per_map must be a positive integer')
    if env.get('map_bundles_train') != stages[0]['maps']:
        raise ValueError('Initial skill training maps must match the first curriculum stage')
    for key, default in (('success_threshold', .8), ('max_ego_failure_rate', .1)):
        _number(curriculum.get(key, default), key, high=1.)
    count = curriculum.get('required_evaluations', 2)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('required_evaluations must be a positive integer')
    retention = curriculum.get('retention', {})
    if not isinstance(retention, dict) or set(retention) - {
            'episodes_per_map', 'max_steps', 'max_completion_drop', 'max_lap_time_increase'}:
        raise ValueError('Invalid skill retention configuration')
    for key, default in (('episodes_per_map', 5), ('max_steps', 16000)):
        value = retention.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'retention.{key} must be a positive integer')
    for key, default in (('max_completion_drop', .05), ('max_lap_time_increase', .1)):
        _number(retention.get(key, default), key, high=1.)
    evaluation = scenario.get('evaluation', {})
    if evaluation.get('enabled') and not evaluation.get('every_steps'):
        raise ValueError('Skill curricula require step-based evaluation')
    if (evaluation.get('selection_strategy') != 'skill' or evaluation.get('lap_completion') is not False
            or evaluation.get('terminate_on_collision') is not True
            or evaluation.get('terminate_on_track_limit') is not True):
        raise ValueError('Skill evaluation requires skill selection and matching task/safety termination')
    # The public episode totals count tactical trials; solo retention is additional.
    total = sum(len(s['maps']) * s['evaluation_episodes_per_map'] for s in stages)
    if evaluation.get('episodes') != total or evaluation.get('final_test', {}).get('episodes') != 2 * total:
        raise ValueError('Skill selection episodes must equal stage totals; final_test must double them')
    # Each skill and solo retention have disjoint seed blocks for both protocols.
    span = max(total * 2, len(configured) * retention.get('episodes_per_map', 5) * 2)
    if span >= 100000:
        raise ValueError('Skill evaluation protocols must fit within their 100000-seed blocks')
    selection_seed, final_seed = evaluation['seed'], evaluation['final_test']['seed']
    offsets = SKILL_SEED_OFFSETS.values()
    if (selection_seed + max(offsets) + span >= 2**32 or final_seed + max(offsets) + span >= 2**32 or
            any(max(selection_seed + a, final_seed + b) < min(selection_seed + a + span, final_seed + b + span)
                for a in offsets for b in offsets)):
        raise ValueError('Skill selection/final tactical and solo seed blocks must be disjoint')


class SkillCurriculum:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.stage = 0
        self.streak = 0
        self.complete = False
        self.last_evaluation = None

    def observe(self, summary):
        rows = summary['stages']
        threshold = self.config.get('success_threshold', .8)
        current = rows[self.stage]
        passed = (summary['retention_passed'] and current['success_rate'] >= threshold
                  and current['ego_failure_rate'] <= self.config.get('max_ego_failure_rate', .1)
                  and all(row['success_rate'] >= threshold for row in rows[:self.stage]))
        self.streak = self.streak + 1 if passed else 0
        self.last_evaluation = dict(retention_passed=summary['retention_passed'],
                                    stages=deepcopy(rows))
        if self.streak >= self.config.get('required_evaluations', 2):
            if self.stage < len(self.config['stages']) - 1:
                self.stage += 1
                self.streak = 0
            else:
                self.complete = True
        return self.state_dict()

    def state_dict(self):
        return dict(version=1, stage=self.stage, stage_name=self.config['stages'][self.stage]['name'],
                    streak=self.streak, complete=self.complete, last_evaluation=self.last_evaluation)
