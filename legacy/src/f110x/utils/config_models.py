from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Mapping, Sequence
import warnings

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
    reward: Dict[str, Any] = field(default_factory=dict)
    policy_curriculum: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpecConfig":
        if not isinstance(data, dict):
            raise TypeError(f"Agent spec must be a mapping, received {type(data)!r}")


        wrappers = [AgentWrapperSpec.from_dict(wrapper) for wrapper in _coerce_sequence(data.get("wrappers"), name="Agent 'wrappers'")]

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
        curriculum_payload: Dict[str, Any] = {}
        for key in ("policy_curriculum", "controller_curriculum", "defender_curriculum"):
            if key in data:
                curriculum_payload = _coerce_mapping(data.get(key), name=f"Agent {key}")
                break

        reward_cfg = _coerce_mapping(data.get("reward"), name="Agent reward config")

        algo_section = data.get("algorithm")
        params: Dict[str, Any] = {}
        config_ref = data.get("config_ref")

        algo = data.get("algo")
        if algo_section is not None:
            if not isinstance(algo_section, Mapping):
                raise TypeError("Agent 'algorithm' section must be a mapping")
            algo_map = dict(algo_section)
            algo_params = _coerce_mapping(algo_map.get("params"), name="Agent algorithm params")
            extra_params = {
                key: value
                for key, value in algo_map.items()
                if key not in {"name", "algo", "type", "params", "config_ref"}
            }
            if algo_params:
                params.update(algo_params)
            if extra_params:
                params.update(extra_params)
            candidate_name = algo_map.get("name") or algo_map.get("algo") or algo_map.get("type")
            if candidate_name:
                algo = candidate_name
            if "config_ref" in algo_map:
                config_ref = algo_map.get("config_ref")

        params.update(_coerce_mapping(data.get("params"), name="Agent params"))

        def _flatten_nested(block: Dict[str, Any]) -> Dict[str, Any]:
            result = dict(block)
            for key in list(block.keys()):
                value = block[key]
                if isinstance(value, Mapping):
                    nested_params = value.get("params") if isinstance(value.get("params"), Mapping) else None
                    if nested_params:
                        for inner_key, inner_value in nested_params.items():
                            result.setdefault(inner_key, inner_value)
                        if len(value.keys()) == 1 or set(value.keys()) == {"params"}:
                            result.pop(key, None)
            return result

        params = _flatten_nested(params)

        arch = params.pop("architecture", None)
        if not algo and arch:
            algo = arch

        if not algo:
            raise ValueError("Agent spec requires an algorithm identifier ('algo' or algorithm.name)")

        agent_id = _normalize_string(data.get("agent_id"))
        role = _normalize_string(data.get("role"))
        config_ref = _normalize_string(config_ref)

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
            reward=dict(reward_cfg),
            policy_curriculum=dict(curriculum_payload),
        )


@dataclass
class AgentRosterConfig:
    roster: List[AgentSpecConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "AgentRosterConfig":
        if data is None:
            return cls([])

        if isinstance(data, dict):
            if "roster" in data or "agents" in data:
                roster_raw = data.get("roster") or data.get("agents") or []
            else:
                roster_raw = []
                for agent_id, spec in data.items():
                    if not isinstance(spec, Mapping):
                        raise TypeError("Agent definition must be a mapping")
                    entry = dict(spec)
                    entry.setdefault("agent_id", agent_id)
                    roster_raw.append(entry)
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
    _federated_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _collect_cache: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MainConfig":
        instance = cls(schema=MainSchema.from_dict(data))
        instance._federated_cache = instance._sanitize_federated()
        instance._collect_cache = instance._sanitize_collect()
        return instance

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

    @property
    def federated(self) -> Dict[str, Any]:
        cache = getattr(self, "_federated_cache", None)
        if not cache:
            cache = self._sanitize_federated()
            self._federated_cache = cache
        return dict(cache)

    @property
    def collect_settings(self) -> Dict[str, int]:
        cache = getattr(self, "_collect_cache", None)
        if not cache:
            cache = self._sanitize_collect()
            self._collect_cache = cache
        return dict(cache)

    @property
    def collect_workers(self) -> int:
        return int(self.collect_settings.get("collect_workers", 1))

    @property
    def collect_prefetch(self) -> int:
        return int(self.collect_settings.get("collect_prefetch", 2))

    @property
    def collect_seed_stride(self) -> int:
        return int(self.collect_settings.get("collect_seed_stride", 1))

    # ------------------------------------------------------------------
    def _sanitize_federated(self) -> Dict[str, Any]:
        raw = self.schema.federated or {}
        if not isinstance(raw, Mapping):
            raw = {}

        enabled = bool(raw.get("enabled", False))

        interval_raw = raw.get("interval", 100)
        try:
            interval = int(interval_raw)
        except (TypeError, ValueError):
            raise ValueError("main.federated.interval must be an integer")
        if interval <= 0:
            if enabled:
                raise ValueError("main.federated.interval must be a positive integer when federated training is enabled")
            interval = 1

        agents_raw = raw.get("agents", [])
        if agents_raw is None:
            agents_iter: Iterable[Any] = ()
        elif isinstance(agents_raw, (list, tuple, set)):
            agents_iter = agents_raw
        elif isinstance(agents_raw, Mapping):
            agents_iter = agents_raw.keys()
        else:
            agents_iter = (agents_raw,)
        agents: List[str] = []
        for entry in agents_iter:
            text = str(entry).strip()
            if text:
                agents.append(text)
        if enabled and not agents:
            raise ValueError("main.federated.agents must list at least one trainable agent when federated training is enabled")

        root_raw = raw.get("root")
        root = None
        if root_raw is not None:
            text = str(root_raw).strip()
            root = text or None

        mode_raw = raw.get("mode", raw.get("strategy", "mean"))
        mode = str(mode_raw).strip().lower() if mode_raw is not None else "mean"

        timeout_raw = raw.get("timeout", 600.0)
        try:
            timeout = float(timeout_raw)
        except (TypeError, ValueError):
            raise ValueError("main.federated.timeout must be numeric")
        if timeout <= 0.0:
            raise ValueError("main.federated.timeout must be positive")

        weights_raw = raw.get("weights")
        weights: Any
        if isinstance(weights_raw, Mapping):
            weights = {
                str(key).strip(): float(value)
                for key, value in weights_raw.items()
                if str(key).strip()
            }
        elif isinstance(weights_raw, (list, tuple)):
            weights = [float(value) for value in weights_raw]
        elif weights_raw is None:
            weights = None
        else:
            weights = float(weights_raw)

        checkpoint_flag = bool(raw.get("checkpoint_after_sync", True))
        optimizer_strategy_raw = raw.get("optimizer_strategy", raw.get("optimizer_mode", "local"))
        optimizer_strategy = str(optimizer_strategy_raw).strip().lower() if optimizer_strategy_raw is not None else "local"
        if optimizer_strategy not in {"local", "average", "reset"}:
            raise ValueError("main.federated.optimizer_strategy must be one of: 'local', 'average', 'reset'")

        sanitized: Dict[str, Any] = {
            "enabled": enabled,
            "interval": interval,
            "agents": agents,
            "root": root,
            "mode": mode or "mean",
            "timeout": timeout,
            "weights": weights,
            "checkpoint_after_sync": checkpoint_flag,
            "optimizer_strategy": optimizer_strategy,
        }

        for key, value in raw.items():
            if key not in sanitized:
                sanitized[key] = value

        self.schema.federated = dict(sanitized)
        return sanitized

    def _sanitize_collect(self) -> Dict[str, int]:
        def _positive_int(value: Any, default: int, name: str) -> int:
            if value is None:
                return default
            try:
                result = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"main.{name} must be an integer") from None
            if result <= 0:
                raise ValueError(f"main.{name} must be greater than zero")
            return result

        workers = _positive_int(self.schema.collect_workers, 1, "collect_workers")
        prefetch = _positive_int(self.schema.collect_prefetch, 2, "collect_prefetch")
        stride = _positive_int(self.schema.collect_seed_stride, 1, "collect_seed_stride")

        sanitized = {
            "collect_workers": workers,
            "collect_prefetch": prefetch,
            "collect_seed_stride": stride,
        }

        self.schema.collect_workers = workers
        self.schema.collect_prefetch = prefetch
        self.schema.collect_seed_stride = stride
        return sanitized


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

        if "scenario" in raw_doc and isinstance(raw_doc["scenario"], Mapping):
            scenario_block = dict(raw_doc["scenario"] or {})
            if "meta" in raw_doc and "meta" not in scenario_block:
                scenario_block["meta"] = raw_doc["meta"]
            data = scenario_block
        else:
            data = raw_doc

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
            data = dict(data)

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

        reward_section = data_map.get("reward")
        if not reward_section:
            for spec in agents.roster:
                if spec.reward:
                    reward_section = spec.reward
                    break

        env_cfg = EnvConfig.from_dict(data_map.get("env", {}))

        roster_size = len(agents.roster)
        env_raw = data_map.get("env") if isinstance(data_map.get("env"), Mapping) else {}
        provided_n_agents: Optional[int] = None
        if isinstance(env_raw, Mapping) and "n_agents" in env_raw:
            try:
                provided_n_agents = int(env_raw["n_agents"])
            except (TypeError, ValueError):
                provided_n_agents = None

        if roster_size:
            if provided_n_agents is None:
                env_cfg.schema.n_agents = roster_size
            elif provided_n_agents != roster_size:
                warnings.warn(
                    f"Environment declares n_agents={provided_n_agents} but {roster_size} agents are configured; keeping explicit n_agents",
                    UserWarning,
                )

        return cls(
            env=env_cfg,
            ppo=PPOConfig.from_dict(data_map.get("ppo", {})),
            rec_ppo=RecPPOConfig.from_dict(data_map.get("rec_ppo", {})),
            td3=TD3Config.from_dict(data_map.get("td3", {})),
            sac=SACConfig.from_dict(data_map.get("sac", {})),
            dqn=DQNConfig.from_dict(data_map.get("dqn", {})),
            reward=RewardConfig.from_dict(reward_section or {}),
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
