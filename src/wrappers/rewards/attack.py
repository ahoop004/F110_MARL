"""Reward repeated target crashes with the environment's survival-qualified facts."""
import math

from wrappers.rewards.base import RewardComponent


class AttackRewardComponent(RewardComponent):
    def __init__(self, config):
        self.bonus = float(config.get("success_bonus", 10.))
        self.penalty = float(config.get("ego_crash_penalty", -20.))
        self.approach = float(config.get("approach_weight", .1))
        self.pressure = float(config.get("edge_weight", .5))
        self.position = float(config.get("position_weight", 0.))
        self.safety = float(config.get("safety_weight", 0.))
        self.margin = float(config.get("safety_margin", .15))
        self.gamma = float(config.get("shaping_gamma", .99))
        self.use_geometry = "shaping_gamma" in config or bool(self.position or self.safety)
        if (not all(math.isfinite(x) for x in (self.bonus, self.penalty, self.approach, self.pressure,
                                              self.position, self.safety, self.margin, self.gamma))
                or self.bonus <= 0 or self.penalty >= 0 or self.margin <= 0 or not 0 < self.gamma <= 1
                or min(self.approach, self.pressure, self.position, self.safety) < 0):
            raise ValueError("attack reward requires positive bonus, negative crash penalty and nonnegative shaping")
        self.reset()

    def reset(self):
        self._previous = None

    def compute(self, step_info):
        facts = (step_info.get("info") or {}).get("attack")
        if facts is None:
            raise ValueError("attack reward requires environment.attack_task")
        rewards = {"attack/success": self.bonus * facts["success"],
                   "attack/ego_crash": self.penalty if facts["ego_failed"] else 0.,
                   "attack/approach": self.approach * facts["approach_delta"],
                   "attack/edge_pressure": self.pressure * facts["edge_delta"]}
        if not self.use_geometry:
            return rewards
        g = facts.get("shaping")
        if g is None:
            raise ValueError("attack positioning requires pre-respawn footprint geometry")
        safe = max(0., min(1., min(g["ego_clearance"], g["vehicle_clearance"]) / self.margin))
        approach = math.exp(-max(0., g["distance"] - 1.) / 3.) * safe
        # Broad preference for being 0.25 m ahead and alongside, with room between cars.
        position = math.exp(-(g["delta_s"] + .25) ** 2
                            - ((abs(g["delta_d"]) - g["width"] - 2 * self.margin) / .5) ** 2) * safe
        side = (max(0., min(1., math.copysign(1., g["target_d"]) * g["delta_d"] / (g["width"] + self.margin)))
                if g["target_d"] else 0.)
        pressure = position * side * max(0., 1. - g["target_clearance"] / .75)
        ended = (facts["target_crash"] or facts["ego_failed"] or step_info.get("done")
                 or step_info.get("terminated") or step_info.get("truncated"))
        current = (approach, position, pressure) if g["moving"] and not ended else (0., 0., 0.)
        # Close the old potential at crashes/termination; initialize a fresh
        # baseline after respawn so teleportation cannot earn shaping reward.
        for i, (name, weight) in enumerate((("approach", self.approach), ("position", self.position),
                                            ("edge_pressure", self.pressure))):
            rewards["attack/" + name] = (weight * (self.gamma * current[i] - self._previous[i])
                                         if self._previous is not None else 0.)
        rewards["attack/safety"] = -self.safety * (1. - safe) ** 2 * float(step_info["timestep"])
        self._previous = None if ended else current
        return rewards
