from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Mapping, Sequence

from f110x.utils.config_schema import (
    EnvSchema,
    DQNConfigSchema,
    MainSchema,
    PPOConfigSchema,
    RecPPOConfigSchema,
    RewardSchema,
    TD3ConfigSchema,
    SACConfigSchema,
)


def _coerce_mapping(value: Any, *, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{name} must be a mapping")


def _coerce_sequence(value: Any, *, name: str) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise TypeError(f"{name} must be a sequence")


def _normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _normalize_string_list(value: Any, *, name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]

@dataclass
class AgentWrapperSpec:
    """Declarative wrapper configuration for per-agent observation pipelines."""

    factory: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "AgentWrapperSpec":
        if isinstance(data, str):
            return cls(factory=data, params={})
        if not isinstance(data, dict):
            raise TypeError(f"Wrapper spec must be dict or str, received {type(data)!r}")

        factory = data.get("factory") or data.get("name") or data.get("type")
        if not factory:
            raise ValueError("Wrapper spec requires a 'factory'/'name' key")
        params = _coerce_mapping(data.get("params"), name="Wrapper params")
        return cls(factory=str(factory), params=params)


@dataclass
class AgentSpecConfig:
    """Configuration describing how to build a single agent/controller."""

    algo: str
    slot: Optional[int] = None
    agent_id: Optional[str] = None
    role: Optional[str] = None
    config_ref: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    wrappers: List[AgentWrapperSpec] = field(default_factory=list)
    trainable: Optional[bool] = None
    target_roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpecConfig":
        if not isinstance(data, dict):
            raise TypeError(f"Agent spec must be a mapping, received {type(data)!r}")

        wrappers = [AgentWrapperSpec.from_dict(wrapper) for wrapper in _coerce_sequence(data.get("wrappers"), name="Agent 'wrappers'")]

        params = _coerce_mapping(data.get("params"), name="Agent params")

        target_roles_list = _normalize_string_list(data.get("target_roles"), name="Agent target_roles")

        slot = data.get("slot")
        if slot is None:
            slot = data.get("index")
        if slot is not None:
            slot = int(slot)

        trainable = data.get("trainable")
        if trainable is not None:
            trainable = bool(trainable)

        metadata = _coerce_mapping(data.get("metadata"), name="Agent metadata")

        algo = data.get("algo")
        if not algo:
            raise ValueError("Agent spec requires an 'algo' key")
        agent_id = _normalize_string(data.get("agent_id"))
        role = _normalize_string(data.get("role"))
        config_ref = _normalize_string(data.get("config_ref"))

        return cls(
            algo=str(algo),
            slot=slot,
            agent_id=agent_id,
            role=role,
            config_ref=config_ref,
            params=dict(params),
            wrappers=wrappers,
            trainable=trainable,
            target_roles=target_roles_list,
            metadata=dict(metadata),
        )


@dataclass
class AgentRosterConfig:
    roster: List[AgentSpecConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "AgentRosterConfig":
        if data is None:
            return cls([])

        if isinstance(data, dict):
            roster_raw = data.get("roster") or data.get("agents") or []
        elif isinstance(data, list):
            roster_raw = data
        else:
            raise TypeError("Agents config must be a list or mapping containing a 'roster' key")

        roster_list = _coerce_sequence(roster_raw, name="Agent roster")
        roster: List[AgentSpecConfig] = []
        for entry in roster_list:
            roster.append(AgentSpecConfig.from_dict(entry))
        return cls(roster=roster)

    @classmethod
    def legacy_default(cls, raw_experiment: Dict[str, Any]) -> "AgentRosterConfig":
        env_cfg = raw_experiment.get("env", {}) or {}
        n_agents = int(env_cfg.get("n_agents", 2) or 2)
        ppo_idx = int(raw_experiment.get("ppo_agent_idx", 0) or 0)

        roster: List[AgentSpecConfig] = []
        for slot in range(n_agents):
            if slot == ppo_idx:
                wrapper_spec = AgentWrapperSpec(
                    factory="obs",
                    params={"max_scan": 30.0, "normalize": True, "target_role": "opponent"},
                )
                roster.append(
                    AgentSpecConfig(
                        algo="ppo",
                        slot=slot,
                        role="ego",
                        config_ref="ppo",
                        wrappers=[wrapper_spec],
                        trainable=True,
                        target_roles=["opponent"],
                    )
                )
            else:
                if n_agents == 2:
                    role = "opponent"
                else:
                    role = f"opponent_{slot}"
                roster.append(
                    AgentSpecConfig(
                        algo="follow_gap",
                        slot=slot,
                        role=role,
                        trainable=False,
                    )
                )
        return cls(roster=roster)

    def __bool__(self) -> bool:
        return bool(self.roster)



@dataclass
class EnvConfig:
    schema: EnvSchema = field(default_factory=EnvSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvConfig":
        return cls(schema=EnvSchema.from_dict(data))

    def to_kwargs(self) -> Dict[str, Any]:
        return self.schema.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.schema.get(key, default)

    @property
    def start_pose_options(self):
        return self.schema.start_pose_options


@dataclass
class PPOConfig:
    schema: PPOConfigSchema = field(default_factory=PPOConfigSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PPOConfig":
        return cls(schema=PPOConfigSchema.from_dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.schema.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if hasattr(self.schema, key):
            setattr(self.schema, key, value)
        else:
            self.schema.extras[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return self.schema.to_dict()


@dataclass
class RecPPOConfig:
    schema: RecPPOConfigSchema = field(default_factory=RecPPOConfigSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecPPOConfig":
        return cls(schema=RecPPOConfigSchema.from_dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.schema.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.schema.to_dict()


@dataclass
class RewardConfig:
    schema: RewardSchema = field(default_factory=RewardSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardConfig":
        return cls(schema=RewardSchema.from_dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.schema.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.schema.to_dict()


@dataclass
class MainConfig:
    schema: MainSchema = field(default_factory=MainSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MainConfig":
        return cls(schema=MainSchema.from_dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.schema.get(key, default)

    @property
    def mode(self) -> str:
        return self.schema.mode

    @property
    def wandb(self) -> Any:
        return self.schema.wandb

    @property
    def checkpoint(self) -> Optional[str]:
        return self.schema.checkpoint


@dataclass
class TD3Config:
    schema: TD3ConfigSchema = field(default_factory=TD3ConfigSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TD3Config":
        return cls(schema=TD3ConfigSchema.from_dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.schema.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.schema.to_dict()


@dataclass
class SACConfig:
    schema: SACConfigSchema = field(default_factory=SACConfigSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SACConfig":
        return cls(schema=SACConfigSchema.from_dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.schema.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.schema.to_dict()


@dataclass
class DQNConfig:
    schema: DQNConfigSchema = field(default_factory=DQNConfigSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DQNConfig":
        return cls(schema=DQNConfigSchema.from_dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.schema.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.schema.to_dict()


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rec_ppo: RecPPOConfig = field(default_factory=RecPPOConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    sac: SACConfig = field(default_factory=SACConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    main: MainConfig = field(default_factory=MainConfig)
    agents: AgentRosterConfig = field(default_factory=AgentRosterConfig)
    ppo_agent_idx: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, experiment: Optional[str] = None) -> "ExperimentConfig":
        import yaml

        with path.open("r") as f:
            raw_doc = yaml.safe_load(f) or {}

        if not isinstance(raw_doc, dict):
            raise TypeError("Configuration root must be a mapping")

        if "experiments" in raw_doc:
            experiments = raw_doc.get("experiments") or {}
            if not isinstance(experiments, dict):
                raise TypeError("'experiments' section must be a mapping")
            selected = experiment or raw_doc.get("default_experiment")
            if not selected:
                raise KeyError("No experiment provided and 'default_experiment' missing")
            if selected not in experiments:
                raise KeyError(f"Experiment '{selected}' not found in config")
            data = experiments[selected] or {}
            if not isinstance(data, dict):
                raise TypeError(f"Experiment '{selected}' must be a mapping")
            data = dict(data)
            data.setdefault("main", {})
            if isinstance(data["main"], dict):
                data["main"].setdefault("experiment_name", selected)
        else:
            data = raw_doc

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        data_map = dict(data)

        raw_agents = data_map.get("agents")
        if raw_agents is not None:
            agents = AgentRosterConfig.from_dict(raw_agents)
            if not agents:
                agents = AgentRosterConfig.legacy_default(data_map)
        else:
            agents = AgentRosterConfig.legacy_default(data_map)

        return cls(
            env=EnvConfig.from_dict(data_map.get("env", {})),
            ppo=PPOConfig.from_dict(data_map.get("ppo", {})),
            rec_ppo=RecPPOConfig.from_dict(data_map.get("rec_ppo", {})),
            td3=TD3Config.from_dict(data_map.get("td3", {})),
            sac=SACConfig.from_dict(data_map.get("sac", {})),
            dqn=DQNConfig.from_dict(data_map.get("dqn", {})),
            reward=RewardConfig.from_dict(data_map.get("reward", {})),
            main=MainConfig.from_dict(data_map.get("main", {})),
            agents=agents,
            ppo_agent_idx=int(data_map.get("ppo_agent_idx", 0)),
            raw=data_map,
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def get_section(self, key: str) -> Optional[Dict[str, Any]]:
        value = self.raw.get(key)
        if isinstance(value, dict):
            return value
        return None
