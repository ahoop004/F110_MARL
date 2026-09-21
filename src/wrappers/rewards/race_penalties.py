"""Shared race penalties, independent of either team's controller rewards."""
from metrics.race_penalties import POLICY, penalty_totals, terminal_penalty_event
from wrappers.rewards.base import RewardComponent


class TeamRacePenaltiesComponent(RewardComponent):
    scope = "team"

    def __init__(self, config):
        if config.get("policy") != POLICY:
            raise ValueError(f"team_race_penalties requires policy: {POLICY}")
        if set(config) - {"enabled", "policy"}:
            raise ValueError("Penalty weights are fixed by the versioned race policy")
        self.contract = {"component": "team_race_penalties", "policy": POLICY}
        self.reset()

    def reset(self):
        self._events = set()

    def compute(self, step_info):
        team = step_info["trainable_agent_ids"]
        opponents = step_info["opponent_agent_ids"]
        infos = step_info["all_infos"]
        new = set()
        for aid in [*team, *opponents]:
            event = terminal_penalty_event(aid, infos.get(aid, {}))
            if event is not None and event not in self._events:
                new.add(event)
        self._events.update(new)
        totals = penalty_totals(new, team, opponents)
        return {key: value for key, value in {
            "race_penalties/own": totals["own_score"],
            "race_penalties/opponents": totals["opponent_score"],
        }.items() if value != 0.0}
