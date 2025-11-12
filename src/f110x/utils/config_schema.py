"""Typed configuration schema definitions with centralised defaults."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin

T = TypeVar("T", bound="BaseSchema")


def _default_vehicle_params() -> Dict[str, float]:
    """Default vehicle dynamics parameters used across experiments."""

    return {
        "mu": 1.0489,
        "C_Sf": 4.718,
        "C_Sr": 5.4562,
        "lf": 0.15875,
        "lr": 0.17145,
        "h": 0.074,
        "m": 3.74,
        "I": 0.04712,
        "s_min": -0.4189,
        "s_max": 0.4189,
        "sv_min": -3.2,
        "sv_max": 3.2,
        "v_switch": 7.319,
        "a_max": 9.51,
        "v_min": -5.0,
        "v_max": 10.0,
        "width": 0.225,
        "length": 0.32,
    }


class SchemaError(ValueError):
    """Raised when configuration coercion fails."""


@dataclass
class BaseSchema:
    """Base class handling typed coercion and extra-key capture."""

    extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls: Type[T], data: Optional[Mapping[str, Any]]) -> T:
        instance = cls()  # type: ignore[call-arg]
        if data:
            instance.update_from_dict(data)
        return instance

    # Conversion helpers -------------------------------------------------
    def update_from_dict(self, data: Mapping[str, Any]) -> None:
        field_map = {f.name: f for f in fields(self)}
        for key, value in data.items():
            if key == "extras":
                continue
            field_info = field_map.get(key)
            if field_info is None:
                self.extras[key] = value
                continue
            coerced = _coerce_value(field_info.type, value)
            setattr(self, key, coerced)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extras":
                continue
            value = getattr(self, f.name)
            payload[f.name] = _safe_copy(value)
        payload.update(self.extras)
        return payload

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extras.get(key, default)


def _safe_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _safe_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_copy(item) for item in value)
    return value


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]  # noqa: E721
        if not args:
            return Any
        if len(args) == 1:
            return args[0]
        return Union[tuple(args)]
    return annotation


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    annotation = _strip_optional(annotation)
    origin = get_origin(annotation)

    if origin in (list, List, Sequence, tuple, Tuple):
        args = get_args(annotation)
        elem_type = args[0] if args else Any
        return [
            _coerce_value(elem_type, item) if elem_type is not Any else item
            for item in value
        ]

    if origin in (dict, Dict, MutableMapping):
        key_type, val_type = (get_args(annotation) + (Any, Any))[:2]
        return {
            _coerce_value(key_type, k) if key_type is not Any else k:
            _coerce_value(val_type, v) if val_type is not Any else v
            for k, v in value.items()
        }

    if origin is Union:
        # Fall back to raw value if coercion is ambiguous
        return value

    if annotation in (int, float):
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"Could not coerce value {value!r} to {annotation}") from exc

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1", "on"}:
                return True
            if lowered in {"false", "no", "n", "0", "off"}:
                return False
            raise SchemaError(f"Cannot coerce string '{value}' to bool")
        return bool(value)

    if annotation is str:
        return str(value)

    return value


@dataclass
class EnvSchema(BaseSchema):
    seed: Optional[int] = 42
    n_agents: int = 2
    max_steps: int = 5000
    timestep: float = 0.01
    integrator: str = "RK4"
    render_interval: int = 0
    update: int = 1
    map_dir: str = "maps"
    map_bundle: Optional[str] = None
    map_yaml: Optional[str] = None
    map: Optional[str] = None
    map_ext: str = ".png"
    render_mode: str = "human"
    start_poses: List[List[float]] = field(default_factory=list)
    start_pose_options: List[List[List[float]]] = field(default_factory=list)
    start_pose_back_gap: Optional[float] = None
    start_pose_min_spacing: Optional[float] = None
    spawn_points: Optional[Any] = None
    spawn_point_sets: List[Any] = field(default_factory=list)
    spawn_point_randomize: Any = None
    spawn_profile: Optional[str] = None
    spawn_curriculum: Dict[str, Any] = field(default_factory=dict)
    lidar_beams: int = 1080
    lidar_range: float = 30.0
    lidar_dist: float = 0.0
    start_thresh: float = 0.5
    target_laps: int = 1
    terminate_on_collision: bool = True
    centerline_autoload: bool = True
    centerline_csv: Optional[str] = None
    centerline_render: bool = True
    centerline_features: bool = True
    vehicle_params: Dict[str, float] = field(default_factory=_default_vehicle_params)

    def update_from_dict(self, data: Mapping[str, Any]) -> None:  # type: ignore[override]
        merged = dict(data)
        vehicle_params = merged.pop("vehicle_params", merged.pop("params", None))
        super().update_from_dict(merged)
        if vehicle_params is not None:
            if not isinstance(vehicle_params, Mapping):
                raise SchemaError("env.vehicle_params must be a mapping")
            params = _default_vehicle_params()
            params.update({str(k): float(v) for k, v in vehicle_params.items()})
            self.vehicle_params = params


@dataclass
class RewardSchema(BaseSchema):
    mode: str = "gaplock"
    components: Dict[str, Any] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)
    fastest_lap: Dict[str, Any] = field(default_factory=dict)
    target_crash_reward: float = 1.0
    ego_collision_penalty: float = 0.0
    truncation_penalty: float = 0.0
    success_once: bool = True
    reward_horizon: Optional[float] = None
    reward_clip: Optional[float] = None
    idle_speed_threshold: float = 0.4
    idle_patience_steps: int = 120


@dataclass
class PPOConfigSchema(BaseSchema):
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    device: str = "cpu"
    gamma: float = 0.99
    lam: float = 0.95
    ent_coef: float = 0.01
    clip_eps: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 64
    train_episodes: int = 5000
    eval_episodes: int = 5
    save_dir: str = "checkpoints/"
    checkpoint_name: Optional[str] = None
    ent_coef_schedule: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecPPOConfigSchema(BaseSchema):
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    device: str = "cpu"
    gamma: float = 0.99
    lam: float = 0.95
    ent_coef: float = 0.01
    clip_eps: float = 0.2
    update_epochs: int = 5
    sequence_batch_size: int = 1
    max_grad_norm: float = 0.5
    rnn_type: str = "lstm"
    rnn_hidden_size: int = 128
    rnn_layers: int = 1
    rnn_dropout: float = 0.0
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    train_episodes: int = 5000
    eval_episodes: int = 5
    save_dir: str = "checkpoints/"
    checkpoint_name: Optional[str] = None
    ent_coef_schedule: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MainSchema(BaseSchema):
    mode: str = "train"
    wandb: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[str] = None
    output_root: str = "outputs"
    eval_interval: int = 0
    eval_episodes: int = 5
    federated: Dict[str, Any] = field(default_factory=dict)
    collect_workers: int = 1
    collect_prefetch: int = 2
    collect_seed_stride: int = 1


@dataclass
class TD3ConfigSchema(BaseSchema):
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    batch_size: int = 128
    buffer_size: int = 100_000
    warmup_steps: int = 1_000
    exploration_noise: float = 0.1
    initial_speed: float = 0.0
    initial_speed_warmup_steps: int = 0
    initial_speed_warmup_throttle: float = 0.0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    device: str = "cpu"


@dataclass
class SACConfigSchema(BaseSchema):
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha: float = 0.2
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 100_000
    warmup_steps: int = 10_000
    auto_alpha: bool = True
    target_entropy: Optional[float] = None
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    device: str = "cpu"


@dataclass
class DQNConfigSchema(BaseSchema):
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    target_update_interval: int = 500
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 20_000
    learning_starts: int = 1_000
    max_grad_norm: float = 0.0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_final: float = 1.0
    per_beta_increment: float = 0.0001
    per_min_priority: float = 0.001
    per_epsilon: float = 1e-6
    action_mode: str = "absolute"
    steering_rate: float = 0.5
    accel_rate: float = 2.0
    brake_rate: float = 4.0
    rate_dt: Optional[float] = None
    rate_initial_steer: float = 0.0
    rate_initial_speed: float = 0.0
    rate_prevent_reverse: bool = True
    rate_stop_speed: float = 0.0
    rate_speed_index: int = 1
    action_set: List[List[float]] = field(default_factory=list)
    device: str = "cpu"
