"""Skill rewards use the same authoritative events as evaluation."""
import math

from wrappers.rewards.base import RewardComponent


class SkillRewardComponent(RewardComponent):
    def __init__(self, config):
        defaults = dict(progress_weight=.1, relative_progress_weight=.5,
                        lead_per_second=.2, success_bonus=10.,
                        ego_failure_penalty=-20., lead_lost_penalty=-10.,
                        lateral_weight=1., heading_weight=1., opponent_progress_weight=.2,
                        idle_per_second=-1.)
        self.weights = {key: float(config.get(key, value)) for key, value in defaults.items()}
        if not all(math.isfinite(v) for v in self.weights.values()):
            raise ValueError('Skill reward weights must be finite')

    def compute(self, step_info):
        facts = step_info['info']['skill']
        w = self.weights
        if facts['event'] and facts['ego_failed']:
            return {'skill/ego_failure': w['ego_failure_penalty']}
        if facts['skill'] == 'pressure' and facts['outcome'] == 'opponent_failure':
            return {}  # Opponent crashes never provide pressure credit.
        reward = {'skill/progress': w['progress_weight'] * facts['progress_delta']}
        if facts['skill'] == 'pass':
            reward['skill/relative_progress'] = w['relative_progress_weight'] * facts['relative_progress_delta']
        elif facts['skill'] == 'defend':
            reward['skill/lead'] = w['lead_per_second'] * facts['lead_reward_s']
        elif facts['skill'] == 'recovery':
            reward['skill/lateral'] = -w['lateral_weight'] * facts['lateral_error'] * facts['dt']
            reward['skill/heading'] = -w['heading_weight'] * facts['heading_error'] * facts['dt']
        elif facts['skill'] == 'pressure':
            reward['skill/opponent_progress'] = -w['opponent_progress_weight'] * facts['opponent_progress_delta']
            reward['skill/idle'] = w['idle_per_second'] * facts['idle_s']
        if facts['event'] and facts['success']:
            reward['skill/success'] = w['success_bonus']
        if facts['event'] and facts['outcome'] == 'lead_lost':
            reward['skill/lead_lost'] = w['lead_lost_penalty']
        return reward
