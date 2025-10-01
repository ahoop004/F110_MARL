"""Helper utilities for upcoming multi-agent training paths.

These helpers are guarded by feature flags so that existing single-attacker
workflows continue to behave exactly as before until the multi-attacker stack
is ready to roll out.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

FeatureSource = Optional[Mapping[str, Any]]


def _parse_env_flags() -> Dict[str, bool]:
    raw = os.environ.get("F110_MULTI_AGENT_FLAGS", "")
    flags: Dict[str, bool] = {}
    for token in raw.split(","):
        name = token.strip().lower()
        if not name:
            continue
        if name.startswith("!"):
            flags[name[1:]] = False
        else:
            flags[name] = True
    return flags


_ENV_FLAGS = _parse_env_flags()


def feature_enabled(name: str, *, source: FeatureSource = None, default: bool = False) -> bool:
    """Return True if a named feature flag is enabled for the current run."""

    key = name.strip().lower()
    if not key:
        return default

    if key in _ENV_FLAGS:
        return _ENV_FLAGS[key]

    if source and isinstance(source, Mapping):
        flags = source.get("feature_flags") or source.get("features")
        if isinstance(flags, Mapping) and key in flags:
            value = flags[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1", "on"}:
                    return True
                if lowered in {"false", "no", "0", "off"}:
                    return False
    return default


def gather_role_ids(roster: Any, role: str, *, default: Optional[Sequence[str]] = None) -> List[str]:
    """Collect every agent identifier bound to a semantic role."""

    if not role:
        return list(default or [])

    role_key = role.strip()
    collected: List[str] = []

    # Support AgentTeam / RosterLayout with a "roles" mapping.
    mapping = getattr(roster, "roles", None)
    if isinstance(mapping, Mapping) and role_key in mapping:
        value = mapping[role_key]
        if isinstance(value, (list, tuple, set)):
            collected.extend(str(item) for item in value)
        else:
            collected.append(str(value))

    # Fall back to scanning declared assignments when available.
    assignments = getattr(roster, "assignments", None)
    if isinstance(assignments, Iterable):
        for assignment in assignments:
            spec = getattr(assignment, "spec", None)
            agent_id = getattr(assignment, "agent_id", None)
            assignment_role = getattr(spec, "role", None)
            if assignment_role == role_key and agent_id is not None:
                collected.append(str(agent_id))

    if collected:
        # Preserve declaration order but remove duplicates.
        seen = set()
        ordered: List[str] = []
        for item in collected:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    return list(default or [])


@dataclass
class JointExperience:
    """Light-weight container for joint observations/actions used in shared critics."""

    attacker_ids: Sequence[str]
    defender_id: Optional[str]
    observations: Mapping[str, Any]
    actions: Mapping[str, Any]
    rewards: Mapping[str, float]
    dones: Mapping[str, bool]
    info: Mapping[str, Any] = field(default_factory=dict)

    def attacker_actions(self) -> List[Any]:
        return [self.actions.get(agent_id) for agent_id in self.attacker_ids]


class JointReplayStub:
    """Minimal replay buffer scaffold to unblock multi-agent trainer prototyping."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = max(int(capacity), 1)
        self._items: List[JointExperience] = []

    def __len__(self) -> int:
        return len(self._items)

    def append(self, item: JointExperience) -> None:
        self._items.append(item)
        if len(self._items) > self.capacity:
            self._items.pop(0)

    def clear(self) -> None:
        self._items.clear()

    def sample(self, count: int) -> List[JointExperience]:
        if count <= 0 or not self._items:
            return []
        if count >= len(self._items):
            return list(self._items)
        step = max(len(self._items) // count, 1)
        return [self._items[i] for i in range(0, len(self._items), step)][:count]


__all__ = [
    "feature_enabled",
    "gather_role_ids",
    "JointExperience",
    "JointReplayStub",
]
