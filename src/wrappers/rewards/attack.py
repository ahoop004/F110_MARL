"""Reward repeated target crashes with the environment's survival-qualified facts."""
import math

from wrappers.rewards.base import RewardComponent


class AttackRewardComponent(RewardComponent):
    def __init__(self, config):
        self.bonus = float(config.get("success_bonus", 10.))
        self.penalty = float(config.get("ego_crash_penalty", -20.))
        self.approach = float(config.get("approach_weight", .1))
        self.pressure = float(config.get("edge_weight", .5))
        if (not all(math.isfinite(x) for x in (self.bonus, self.penalty, self.approach, self.pressure))
                or self.bonus <= 0 or self.penalty >= 0 or min(self.approach, self.pressure) < 0):
            raise ValueError("attack reward requires positive bonus, negative crash penalty and nonnegative shaping")

    def compute(self, step_info):
        facts = (step_info.get("info") or {}).get("attack")
        if facts is None:
            raise ValueError("attack reward requires environment.attack_task")
        return {"attack/success": self.bonus * facts["success"],
                "attack/ego_crash": self.penalty if facts["ego_failed"] else 0.,
                "attack/approach": self.approach * facts["approach_delta"],
                "attack/edge_pressure": self.pressure * facts["edge_delta"]}
