"""Versioned race penalties derived from authoritative terminal facts.

Version 1 scores terminal involvement, never blame. It cannot classify vehicle
versus wall contacts, nonterminal contacts, or nonterminal boundary excursions.
"""
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

POLICY = "terminal_incidents_v1"
OWN_WEIGHT = 1.0
OPPONENT_WEIGHT = 0.25
PENALTY_KINDS = {"collision": "collision_dnf", "track_boundary": "boundary_dnf"}


@dataclass(frozen=True)
class RacePenaltyEvent:
    agent_id: str
    kind: str
    step: int
    points: float = 1.0

    def to_dict(self):
        return asdict(self)


def terminal_penalty_event(agent_id: str, info: Mapping) -> RacePenaltyEvent | None:
    """One event per penalized vehicle's terminal incident, not per info read."""
    kind = PENALTY_KINDS.get(info.get("terminal_reason"))
    step = info.get("terminal_step")
    if kind is None or step is None:
        return None
    return RacePenaltyEvent(str(agent_id), kind, int(step))


def penalty_totals(events: Sequence[RacePenaltyEvent], team_ids: Sequence[str],
                   opponent_ids: Sequence[str]) -> dict:
    """Deduplicate persisted events and normalize by configured team sizes."""
    team, opponents = set(team_ids), set(opponent_ids)
    if not team or not opponents or team & opponents:
        raise ValueError("Penalty scoring requires disjoint, nonempty teams")
    events = set(events)
    own = sum(event.points for event in events if event.agent_id in team)
    other = sum(event.points for event in events if event.agent_id in opponents)
    return {"own_points": own, "opponent_points": other,
            "own_score": -OWN_WEIGHT * own / len(team),
            "opponent_score": OPPONENT_WEIGHT * other / len(opponents)}
