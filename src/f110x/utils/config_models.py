from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EnvConfig:
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvConfig":
        return cls(data=dict(data))

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def start_pose_options(self):
        return self.data.get("start_pose_options")


@dataclass
class PPOConfig:
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PPOConfig":
        return cls(data=dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)


@dataclass
class RewardConfig:
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardConfig":
        return cls(data=dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)


@dataclass
class MainConfig:
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MainConfig":
        return cls(data=dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def mode(self) -> str:
        return self.data.get("mode", "train")

    @property
    def wandb(self) -> Any:
        return self.data.get("wandb", {})

    @property
    def checkpoint(self) -> Optional[str]:
        return self.data.get("checkpoint")


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    main: MainConfig = field(default_factory=MainConfig)
    ppo_agent_idx: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        import yaml

        with path.open("r") as f:
            data = yaml.safe_load(f)

        return cls(
            env=EnvConfig.from_dict(data.get("env", {})),
            ppo=PPOConfig.from_dict(data.get("ppo", {})),
            reward=RewardConfig.from_dict(data.get("reward", {})),
            main=MainConfig.from_dict(data.get("main", {})),
            ppo_agent_idx=int(data.get("ppo_agent_idx", 0)),
            raw=data,
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

