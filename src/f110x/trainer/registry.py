"""Lookup utilities for constructing trainers by algorithm key."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from f110x.trainer.base import Trainer
from f110x.trainer.off_policy import DQNTrainer, SACTrainer, TD3Trainer
from f110x.trainer.on_policy import PPOTrainer, RecurrentPPOTrainer

TrainerFactory = Callable[[str, Any, Optional[Mapping[str, Any]]], Trainer]


def _normalize(key: str) -> str:
    return key.lower().strip()


_REGISTRY: Dict[str, TrainerFactory] = {}


def register_trainer(name: str, factory: TrainerFactory, *, aliases: Iterable[str] = ()) -> None:
    """Register a trainer factory under the provided name and optional aliases."""

    keys = {_normalize(name), *(_normalize(alias) for alias in aliases)}
    for key in keys:
        _REGISTRY[key] = factory


def resolve_trainer(name: str) -> TrainerFactory:
    key = _normalize(name)
    if key not in _REGISTRY:
        raise KeyError(f"Trainer '{name}' has not been registered")
    return _REGISTRY[key]


def create_trainer(
    name: str,
    agent_id: str,
    controller: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Trainer:
    """Instantiate a trainer for the requested algorithm key."""

    factory = resolve_trainer(name)
    return factory(agent_id, controller, config)


def registered_trainers() -> Dict[str, TrainerFactory]:
    """Return a copy of the current registry contents."""

    return dict(_REGISTRY)


def _class_factory(cls: type[Trainer]) -> TrainerFactory:
    def factory(agent_id: str, controller: Any, _: Optional[Mapping[str, Any]] = None) -> Trainer:
        return cls(agent_id, controller)

    return factory


# Default registrations --------------------------------------------------------
register_trainer("ppo", _class_factory(PPOTrainer))
register_trainer(
    "rec_ppo",
    _class_factory(RecurrentPPOTrainer),
    aliases=("recurrent_ppo", "ppo_recurrent"),
)
register_trainer("td3", _class_factory(TD3Trainer))
register_trainer("sac", _class_factory(SACTrainer))
register_trainer("dqn", _class_factory(DQNTrainer))


__all__ = [
    "TrainerFactory",
    "register_trainer",
    "resolve_trainer",
    "create_trainer",
    "registered_trainers",
]
