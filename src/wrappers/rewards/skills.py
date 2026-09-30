"""Skill rewards use the same authoritative events as evaluation."""
import math

from wrappers.rewards.base import RewardComponent


class SkillRewardComponent(RewardComponent):
    def __init__(self, config):
        defaults = dict(progress_weight=.1, relative_progress_weight=.5,
                        lead_per_second=.2, success_bonus=10.,
                        ego_failure_penalty=-20., lead_lost_penalty=-10.)
        self.weights = {key: float(config.get(key, value)) for key, value in defaults.items()}
        if not all(math.isfinite(v) for v in self.weights.values()):
            raise ValueError('Skill reward weights must be finite')

    def compute(self, step_info):
        facts = step_info['info']['skill']
        w = self.weights
        if facts['event'] and facts['ego_failed']:
            return {'skill/ego_failure': w['ego_failure_penalty']}
        reward = {'skill/progress': w['progress_weight'] * facts['progress_delta']}
        if facts['skill'] == 'pass':
            reward['skill/relative_progress'] = w['relative_progress_weight'] * facts['relative_progress_delta']
        else:
            reward['skill/lead'] = w['lead_per_second'] * facts['lead_reward_s']
        if facts['event'] and facts['success']:
            reward['skill/success'] = w['success_bonus']
        if facts['event'] and facts['outcome'] == 'lead_lost':
            reward['skill/lead_lost'] = w['lead_lost_penalty']
        return reward
