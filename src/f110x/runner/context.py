"""Runner-tailored context model for training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from f110x.engine.reward import CurriculumSchedule
from f110x.envs import F110ParallelEnv
from f110x.trainer.base import Trainer
from f110x.utils.builders import AgentBundle, AgentTeam
from f110x.utils.config_models import ExperimentConfig
from f110x.utils.map_loader import MapData
from f110x.utils.start_pose import StartPoseOption
from f110x.utils.logger import Logger


@dataclass
class RunnerContext:
    """Aggregates runtime artefacts needed by training/eval runners."""

    cfg: ExperimentConfig
    env: F110ParallelEnv
    map_data: MapData
    start_pose_options: Optional[List[StartPoseOption]]
    team: AgentTeam
    reward_cfg: Dict[str, Any]
    curriculum_schedule: CurriculumSchedule
    output_root: Path
    start_pose_back_gap: float = 0.0
    start_pose_min_spacing: float = 0.0
    render_interval: int = 0
    eval_interval: int = 0
    eval_episodes: int = 1
    update_after: int = 1
    trainer_map: Dict[str, Trainer] = field(default_factory=dict)
    trainable_ids: List[str] = field(default_factory=list)
    primary_agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    logger: Logger = field(default_factory=Logger)

    # Accessors -----------------------------------------------------------------

    @property
    def trainable_agent_ids(self) -> List[str]:
        return list(self.trainable_ids)

    @property
    def trainable_bundles(self) -> List[AgentBundle]:
        return [self.team.by_id[aid] for aid in self.trainable_ids if aid in self.team.by_id]

    def get_bundle(self, agent_id: str) -> AgentBundle:
        return self.team.by_id[agent_id]

    def get_trainer(self, agent_id: str) -> Optional[Trainer]:
        return self.trainer_map.get(agent_id)

    def iter_trainers(self) -> Iterator[tuple[str, Trainer]]:
        return iter(self.trainer_map.items())

    @property
    def primary_bundle(self) -> AgentBundle:
        if not self.primary_agent_id:
            raise RuntimeError("RunnerContext has no primary agent configured")
        if self.primary_agent_id not in self.team.by_id:
            raise KeyError(f"Primary agent '{self.primary_agent_id}' not present in team roster")
        return self.team.by_id[self.primary_agent_id]

    @property
    def primary_trainer(self) -> Trainer:
        if not self.primary_agent_id:
            raise RuntimeError("RunnerContext has no primary agent configured")
        trainer = self.trainer_map.get(self.primary_agent_id)
        if trainer is None:
            raise RuntimeError(
                f"Primary agent '{self.primary_agent_id}' is not associated with a trainer"
            )
        return trainer

    # Mutation helpers ----------------------------------------------------------

    def update_metadata(self, **updates: Any) -> None:
        self.metadata.update({key: value for key, value in updates.items()})

    def set_primary_agent(self, agent_id: Optional[str]) -> None:
        if agent_id is not None and agent_id not in self.team.by_id:
            raise KeyError(f"Agent '{agent_id}' not present in team roster")
        self.primary_agent_id = agent_id

    def register_trainable(self, agent_ids: Iterable[str]) -> None:
        for aid in agent_ids:
            if aid not in self.trainable_ids and aid in self.team.by_id:
                self.trainable_ids.append(aid)


__all__ = ["RunnerContext"]
