"""Target proximity and track-edge pressure rewards."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.rewards.base import RewardComponent


class TargetProximityComponent(RewardComponent):
    """Reward staying near a target at a preferred distance."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 1.0))
        self.preferred_distance = float(config.get("preferred_distance", 1.5))
        self.distance_tolerance = float(config.get("distance_tolerance", 0.5))

    def compute(self, step_info: dict) -> Dict[str, float]:
        obs = step_info.get("next_obs") or step_info.get("obs") or {}
        target_pose = np.asarray(obs.get("target_pose", [0.0, 0.0, 0.0]), dtype=np.float64)
        ego_pose = np.asarray(obs.get("pose", [0.0, 0.0, 0.0]), dtype=np.float64)
        dist = float(np.linalg.norm(target_pose[:2] - ego_pose[:2]))
        deviation = abs(dist - self.preferred_distance)
        bonus = max(0.0, 1.0 - deviation / max(self.distance_tolerance, 1e-6))
        return {"target_proximity/bonus": self.weight * bonus}


class TargetEdgePressureComponent(RewardComponent):
    """Reward proportional to an environment-supplied target edge-pressure fact."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 0.5))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        pressure_val = float(info.get("forcing_reward", 0.0))
        return {"target_edge_pressure/bonus": self.weight * pressure_val}


class RacePursuitComponent(RewardComponent):
    """Penalize trailing by unwrapped progress; reward each isolated respawn."""

    def __init__(self, config):
        self.penalty = float(config.get("behind_penalty", -0.01))
        self.bonus = float(config.get("respawn_bonus", 1.0))
        self.target_id = str(config.get("target_id", "car_1"))
        self.reset()

    def reset(self):
        self._gap = None

    def compute(self, step_info):
        info = step_info.get("info") or {}
        other = (step_info.get("all_infos") or {}).get(self.target_id, {})
        ego = info.get("centerline", {})
        target = other.get("centerline", {})
        if not ego or not target:
            raise ValueError("race_pursuit requires both agents' centerline facts")
        if self._gap is None:
            # Initial grid is local; subsequent progress stays unwrapped, so
            # lapping the opponent cannot flip the ordering at the seam.
            self._gap = (target["progress"] - ego["progress"] + .5) % 1.0 - .5
        else:
            self._gap += target["progress_delta"] - ego["progress_delta"]
        return {"race_pursuit/behind": self.penalty if self._gap > 0 else 0.0,
                "race_pursuit/respawn": self.bonus if info.get("target_respawned") else 0.0}


class TeamSupportComponent(RewardComponent):
    """Credit teammate progress and local, non-contact blocking outcomes.

    Blocking credit is the teammate's signed progress advantage over an active
    opponent following behind the ego in the same corridor. There is no reward
    for proximity alone, parked/crashed opponents, reversing, or a collision.
    This is an outcome proxy, not a claim that ego caused the opponent's delay.
    """

    def __init__(self, config):
        self.teammate_id = str(config.get('teammate_id', 'car_0'))
        self.progress_weight = float(config.get('progress_weight', 1.0))
        self.blocking_weight = float(config.get('blocking_weight', 1.0))
        self.max_distance = float(config.get('max_distance', 6.0))
        self.min_distance = float(config.get('min_distance', 0.6))
        self.lateral_distance = float(config.get('lateral_distance', 0.6))
        self.max_step_progress = float(config.get('max_step_progress', 1.0))
        values = [self.progress_weight, self.blocking_weight, self.max_distance,
                  self.min_distance, self.lateral_distance, self.max_step_progress]
        if (not all(np.isfinite(v) and v >= 0 for v in values)
                or not self.max_distance > self.min_distance
                or self.lateral_distance <= 0 or self.max_step_progress <= 0):
            raise ValueError('team_support requires finite nonnegative weights and positive distance/progress bounds')

    def compute(self, step_info):
        infos = step_info.get('all_infos') or {}
        ego = step_info.get('info') or {}
        mate = infos.get(self.teammate_id, {})
        # No bonus for a mutual crash or for continuing after the racer is gone.
        if (ego.get('terminal_reason') or mate.get('terminal_reason')
                or (ego.get('track_limits') or {}).get('exceeded')
                or (mate.get('track_limits') or {}).get('exceeded')):
            return {}
        length = step_info.get('track_length')
        if not length or not np.isfinite(length) or length <= 0:
            raise ValueError('team_support requires a positive track_length')
        if not mate.get('centerline') or not ego.get('centerline'):
            raise ValueError('team_support requires ego and teammate centerline facts')
        def delta(info):
            return float(np.clip(info['centerline'].get('progress_delta', 0.0) * length,
                                 -self.max_step_progress, self.max_step_progress))
        mate_delta = delta(mate)
        rewards = {'team_support/progress': self.progress_weight * mate_delta}
        opponents = set(step_info.get('opponent_agent_ids') or ())
        advantages = []
        if delta(ego) >= 0 and mate_delta > 0:
            for neighbor in ego.get('frenet_neighbors', []):
                aid = neighbor['agent_id']
                other = infos.get(aid, {})
                if (aid not in opponents or not other.get('centerline')
                        or other.get('terminal_reason')
                        or (other.get('track_limits') or {}).get('exceeded')):
                    continue
                if (-self.max_distance <= neighbor['delta_s'] <= -self.min_distance
                        and abs(neighbor['delta_d']) <= self.lateral_distance):
                    # Negative opponent motion cannot manufacture blocking credit.
                    advantages.append(mate_delta - max(0.0, delta(other)))
        if advantages:
            rewards['team_support/blocking'] = self.blocking_weight * sum(advantages) / max(1, len(opponents))
        return rewards
