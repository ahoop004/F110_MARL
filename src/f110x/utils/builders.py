"""Factory helpers for building environments and agent teams from configs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from f110x.envs import F110ParallelEnv
from f110x.utils.config_models import (
    AgentRosterConfig,
    AgentSpecConfig,
    AgentWrapperSpec,
    ExperimentConfig,
)
from f110x.utils.map_loader import MapData, MapLoader
from f110x.utils.start_pose import StartPoseOption, parse_start_pose_options
from f110x.wrappers.observation import ObsWrapper
from f110x.wrappers.action import (
    ActionRepeatWrapper,
    DiscreteActionWrapper,
    DeltaDiscreteActionWrapper,
    PreventReverseContinuousWrapper,
)
from f110x.wrappers.common import to_numpy
from f110x.policies.gap_follow import FollowTheGapPolicy
from f110x.policies.blocker import BlockingPolicy
from f110x.policies.ftg_centerline import FollowTheGapCenterlinePolicy
from f110x.policies.secondary_vicon import SecondaryViconPolicy
from f110x.policies.ppo.ppo import PPOAgent
from f110x.policies.ppo.rec_ppo import RecurrentPPOAgent
from f110x.policies.random_policy import random_policy
from f110x.policies.simple_heuristic import simple_heuristic
from f110x.policies.centerline_pursuit import CenterlinePursuitPolicy
from f110x.policies.td3.td3 import TD3Agent
from f110x.policies.sac.sac import SACAgent
from f110x.policies.dqn.dqn import DQNAgent
from f110x.policies.rainbow import RainbowDQNAgent
from f110x.render import EnvRenderer
from f110x.trainer.base import Trainer
from f110x.trainer import registry as trainer_registry


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------


def build_env(cfg: ExperimentConfig) -> Tuple[F110ParallelEnv, MapData, Optional[List[StartPoseOption]]]:
    """Instantiate the simulator, returning env + loaded map artefacts."""

    loader = MapLoader()
    env_cfg_dict = cfg.env.to_kwargs()
    map_data = loader.load(env_cfg_dict)
    env_cfg = dict(env_cfg_dict)

    raw_env_cfg = cfg.raw.get("env", {})
    if not isinstance(raw_env_cfg, Mapping):
        raw_env_cfg = {}
    env_cfg["_centerline_render_user_override"] = "centerline_render" in raw_env_cfg
    env_cfg["_centerline_features_user_override"] = "centerline_features" in raw_env_cfg
    env_cfg["_centerline_autoload_user_override"] = "centerline_autoload" in raw_env_cfg

    _apply_spawn_point_config(env_cfg, map_data)
    env_cfg["map_meta"] = map_data.metadata
    env_cfg["map_image_path"] = str(map_data.image_path)
    env_cfg["map_image_size"] = map_data.image_size
    env_cfg["map_yaml_path"] = str(map_data.yaml_path)
    env_cfg["map_ext"] = map_data.image_path.suffix or env_cfg.get("map_ext")

    finish_line_meta = map_data.metadata.get("finish_line")
    if isinstance(finish_line_meta, Mapping):
        env_cfg["finish_line"] = dict(finish_line_meta)

    map_root_raw = env_cfg.get("map_dir")
    map_root_path: Optional[Path] = None
    if map_root_raw:
        map_root_path = Path(str(map_root_raw)).expanduser().resolve()
    try:
        if map_root_path is not None:
            relative_yaml = map_data.yaml_path.relative_to(map_root_path)
        else:
            relative_yaml = map_data.yaml_path
    except ValueError:
        relative_yaml = map_data.yaml_path
    yaml_value = str(relative_yaml)
    env_cfg["map_yaml"] = yaml_value
    env_cfg["map"] = yaml_value

    # Allow map metadata to override start position defaults when provided.
    for meta_key in ("start_poses", "start_pose_options"):
        if meta_key in map_data.metadata:
            env_cfg[meta_key] = map_data.metadata[meta_key]

    env = F110ParallelEnv(map_data=map_data, **env_cfg)
    env.set_centerline(map_data.centerline, path=map_data.centerline_path)
    start_pose_options = parse_start_pose_options(env_cfg.get("start_pose_options"))
    return env, map_data, start_pose_options


def _apply_spawn_point_config(env_cfg: Dict[str, Any], map_data: MapData) -> None:
    """Translate spawn point configuration into explicit pose options."""

    spawn_points = map_data.spawn_points
    if not spawn_points:
        return

    n_agents = int(env_cfg.get("n_agents", 0) or len(spawn_points))
    agent_ids = [f"car_{idx}" for idx in range(n_agents)]

    def _names_to_option(names: Iterable[Any], option_id: Optional[str] = None, extra_meta: Optional[Dict[str, Any]] = None):
        name_list = list(names)
        if len(name_list) != n_agents:
            raise ValueError(
                f"spawn point selection requires {n_agents} entries for map {map_data.yaml_path}, "
                f"received {len(name_list)}"
            )

        pose_rows = []
        metadata: Dict[str, Any] = {"spawn_points": {}}
        if extra_meta:
            metadata.update(extra_meta)

        for idx, name in enumerate(name_list):
            if name is None:
                raise ValueError(
                    f"spawn point index {idx} missing for selection on map {map_data.yaml_path}"
                )
            key = str(name)
            if key not in spawn_points:
                raise KeyError(
                    f"spawn point '{key}' not found in annotations for {map_data.yaml_path}. "
                    f"Available: {sorted(spawn_points)}"
                )
            pose_rows.append(spawn_points[key].tolist())
            metadata.setdefault("spawn_points", {})[agent_ids[idx]] = key

        if option_id is not None:
            metadata.setdefault("spawn_option_id", option_id)

        return {"poses": pose_rows, "metadata": metadata}

    start_pose_options_cfg: List[Any] = list(env_cfg.get("start_pose_options", []) or [])

    if not env_cfg.get("start_poses") and env_cfg.get("spawn_points") is None:
        available_names = list(spawn_points.keys())
        if len(available_names) >= n_agents:
            default_option = _names_to_option(available_names[:n_agents], option_id="map_default")
            env_cfg["start_poses"] = default_option["poses"]
            start_pose_options_cfg.append(default_option)

    spawn_points_cfg = env_cfg.pop("spawn_points", None)
    if spawn_points_cfg is not None:
        if isinstance(spawn_points_cfg, (list, tuple)):
            names = spawn_points_cfg
        elif isinstance(spawn_points_cfg, dict):
            if "names" in spawn_points_cfg:
                names = spawn_points_cfg["names"]
            else:
                names = [None] * n_agents
                for key, value in spawn_points_cfg.items():
                    try:
                        slot = int(key)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= slot < n_agents:
                        names[slot] = value
        else:
            raise TypeError("env.spawn_points must be a list or mapping of agent slots to spawn names")

        default_option = _names_to_option(names, option_id="default_spawn")
        env_cfg["start_poses"] = default_option["poses"]
        start_pose_options_cfg.append(default_option)

    spawn_sets_cfg = env_cfg.pop("spawn_point_sets", []) or []
    for idx, entry in enumerate(spawn_sets_cfg):
        extra_meta: Dict[str, Any] = {}
        option_id: Optional[str] = None

        if isinstance(entry, dict):
            names = entry.get("names") or entry.get("spawn_points") or entry.get("points") or entry.get("pose_names")
            extra_meta = dict(entry.get("metadata", {}))
            option_id = entry.get("id") or entry.get("name") or f"spawn_set_{idx}"
        else:
            names = entry
            option_id = f"spawn_set_{idx}"

        if names is None:
            continue

        option_payload = _names_to_option(names, option_id=option_id, extra_meta=extra_meta)
        start_pose_options_cfg.append(option_payload)

    random_cfg = env_cfg.pop("spawn_point_randomize", None)
    if random_cfg:
        if isinstance(random_cfg, dict):
            pool = random_cfg.get("pool") or random_cfg.get("names") or list(spawn_points.keys())
            allow_reuse = bool(random_cfg.get("allow_reuse", False))
            option_id = random_cfg.get("id") or "spawn_random"
        else:
            pool = list(spawn_points.keys())
            allow_reuse = False
            option_id = "spawn_random"

        pool = [str(name) for name in pool if str(name) in spawn_points]
        if not pool:
            raise ValueError("spawn_point_randomize requires at least one valid spawn point in the pool")

        placeholder = np.zeros((n_agents, 3), dtype=np.float32).tolist()
        metadata = {
            "spawn_random_pool": pool,
            "spawn_random_allow_reuse": allow_reuse,
            "spawn_random_count": n_agents,
            "spawn_option_id": option_id,
        }
        start_pose_options_cfg.append({"poses": placeholder, "metadata": metadata})

    profile_id_raw = env_cfg.pop("spawn_profile", None)
    if profile_id_raw and start_pose_options_cfg:
        profile_id = str(profile_id_raw).strip().lower()
        preferred_ids = [profile_id]
        if profile_id in {"baseline", "default", "default_spawn"}:
            preferred_ids.append("default_spawn")
        selected_option = None
        for option in start_pose_options_cfg:
            metadata = option.get("metadata") or {}
            option_id_raw = metadata.get("spawn_option_id") or metadata.get("id")
            option_id = str(option_id_raw).strip().lower() if option_id_raw is not None else ""
            if option_id and option_id in preferred_ids:
                selected_option = option
                break
        if selected_option is None:
            available_ids = [
                (opt.get("metadata") or {}).get("spawn_option_id")
                for opt in start_pose_options_cfg
            ]
            raise ValueError(
                f"spawn_profile '{profile_id_raw}' not found; available options: {available_ids}"
            )
        env_cfg["start_poses"] = selected_option["poses"]

    if start_pose_options_cfg:
        env_cfg["start_pose_options"] = start_pose_options_cfg


# ---------------------------------------------------------------------------
# Agent roster layout helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentAssignment:
    """Materialised view of an agent spec bound to an env slot."""

    spec: AgentSpecConfig
    slot: int
    agent_id: str


@dataclass
class RosterLayout:
    """Resolved roster with convenient lookup utilities."""

    assignments: List[AgentAssignment]

    def __post_init__(self) -> None:
        by_id: Dict[str, AgentAssignment] = {}
        by_slot: Dict[int, AgentAssignment] = {}
        by_role: Dict[str, List[AgentAssignment]] = {}
        for assignment in self.assignments:
            by_id[assignment.agent_id] = assignment
            by_slot[assignment.slot] = assignment
            role = assignment.spec.role
            if role:
                by_role.setdefault(role, []).append(assignment)
        self._by_id = by_id
        self._by_slot = by_slot
        self._by_role = {role: list(assignments) for role, assignments in by_role.items()}

    # Lookups ---------------------------------------------------------------
    def resolve_slot(self, slot: int) -> Optional[str]:
        assignment = self._by_slot.get(slot)
        return assignment.agent_id if assignment else None

    def resolve_role(
        self,
        role: str,
        *,
        index: int = 0,
        exclude: Optional[str] = None,
    ) -> Optional[str]:
        if not role:
            return None

        assignments = list(self._by_role.get(role, ()))
        if exclude is not None:
            assignments = [assn for assn in assignments if assn.agent_id != exclude]

        if not assignments:
            return None

        count = len(assignments)
        normalized_index = index % count if count else 0
        return assignments[normalized_index].agent_id

    def first_other(self, agent_id: str) -> Optional[str]:
        for other_id in self._by_id.keys():
            if other_id != agent_id:
                return other_id
        return None

    @property
    def agent_ids(self) -> List[str]:
        return [assignment.agent_id for assignment in self.assignments]

    @property
    def roles(self) -> Dict[str, List[str]]:
        return {
            role: [assn.agent_id for assn in assignments]
            for role, assignments in self._by_role.items()
        }

    def role_ids(self, role: str) -> List[str]:
        if not role:
            return []
        return [assn.agent_id for assn in self._by_role.get(role, [])]


def _compile_roster(cfg: ExperimentConfig, env: F110ParallelEnv) -> RosterLayout:
    """Bind agent specs to environment slots, filling gaps with heuristics."""

    possible_agents = list(env.possible_agents)
    n_slots = len(possible_agents)

    roster_cfg: AgentRosterConfig = cfg.agents
    specs: List[AgentSpecConfig] = list(roster_cfg.roster)

    if len(specs) > n_slots:
        raise ValueError(
            f"Agent roster defines {len(specs)} agents but env only exposes {n_slots} slots"
        )

    assignments: List[AgentAssignment] = []
    id_to_slot = {aid: idx for idx, aid in enumerate(possible_agents)}
    occupied: Dict[int, AgentAssignment] = {}
    deferred: List[AgentSpecConfig] = []

    # First pass: honour explicit slot / agent identifiers
    for spec in specs:
        slot = spec.slot
        agent_id = spec.agent_id

        if agent_id is not None:
            if agent_id not in id_to_slot:
                raise ValueError(
                    f"Agent spec for algo '{spec.algo}' references unknown agent_id '{agent_id}'"
                )
            inferred_slot = id_to_slot[agent_id]
            if slot is None:
                slot = inferred_slot
            elif slot != inferred_slot:
                raise ValueError(
                    f"Agent spec slot mismatch: requested slot {slot} but agent_id '{agent_id}' maps to {inferred_slot}"
                )

        if slot is not None:
            if slot < 0 or slot >= n_slots:
                raise ValueError(f"Agent slot index {slot} out of range for {n_slots} env agents")
            if slot in occupied:
                conflict = occupied[slot]
                raise ValueError(
                    "Duplicate agent slot assignment: "
                    f"slot {slot} already bound to '{conflict.spec.algo}'"
                )
            agent_id = agent_id or possible_agents[slot]
            assignment = AgentAssignment(spec=spec, slot=slot, agent_id=agent_id)
            assignments.append(assignment)
            occupied[slot] = assignment
        else:
            deferred.append(spec)

    # Second pass: allocate remaining specs to free slots in declaration order
    free_slots = [idx for idx in range(n_slots) if idx not in occupied]
    if len(deferred) > len(free_slots):
        raise ValueError(
            f"Agent roster has {len(deferred)} unallocated specs but only {len(free_slots)} remaining slots"
        )

    for spec, slot in zip(deferred, free_slots):
        agent_id = spec.agent_id or possible_agents[slot]
        assignment = AgentAssignment(spec=spec, slot=slot, agent_id=agent_id)
        assignments.append(assignment)
        occupied[slot] = assignment

    # Fill any still-empty slots with benign heuristics to keep env satisfied
    remaining = [idx for idx in range(n_slots) if idx not in occupied]
    for slot in remaining:
        agent_id = possible_agents[slot]
        fallback_spec = AgentSpecConfig(
            algo="follow_gap",
            slot=slot,
            agent_id=agent_id,
            role=f"opponent_{slot}",
            trainable=False,
        )
        assignment = AgentAssignment(spec=fallback_spec, slot=slot, agent_id=agent_id)
        assignments.append(assignment)

    assignments.sort(key=lambda assn: assn.slot)
    return RosterLayout(assignments)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


@dataclass
class ObservationAdapter:
    """Wraps a callable observation transform with target resolution metadata."""

    name: str
    wrapper: Any
    target_role: Optional[str] = None
    target_slot: Optional[int] = None
    target_agent: Optional[str] = None

    def __call__(
        self,
        raw_obs: Dict[str, Any],
        agent_id: str,
        roster: RosterLayout,
    ) -> Any:
        target_id = self._resolve_target(agent_id, roster)
        return self.wrapper(raw_obs, agent_id, target_id)

    def _resolve_target(self, agent_id: str, roster: RosterLayout) -> Optional[str]:
        explicit = self.target_agent
        if explicit is not None:
            return explicit

        if self.target_role is not None:
            resolved = roster.resolve_role(self.target_role, exclude=agent_id)
            if resolved is not None:
                return resolved

        if self.target_slot is not None:
            resolved = roster.resolve_slot(self.target_slot)
            if resolved is not None and resolved != agent_id:
                return resolved

        fallback = roster.first_other(agent_id)
        return fallback


class ObservationPipeline(Sequence[ObservationAdapter]):
    """Utility to compose ordered observation adapters."""

    def __init__(self, stages: Iterable[ObservationAdapter]):
        self._stages: List[ObservationAdapter] = list(stages)

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self):
        return iter(self._stages)

    def __getitem__(self, index: int) -> ObservationAdapter:
        return self._stages[index]

    @property
    def stages(self) -> List[ObservationAdapter]:
        return list(self._stages)

    def transform(
        self,
        raw_obs: Dict[str, Any],
        agent_id: str,
        roster: RosterLayout,
    ) -> Any:
        result: Any = raw_obs
        for stage in self._stages:
            result = stage(raw_obs, agent_id, roster)
        return result

    def to_vector(self, raw_obs: Dict[str, Any], agent_id: str, roster: RosterLayout) -> np.ndarray:
        output = self.transform(raw_obs, agent_id, roster)
        vector = to_numpy(output, flatten=True)
        return vector.astype(np.float32, copy=False)


WrapperBuilder = Callable[[AgentWrapperSpec, "AgentBuildContext", AgentAssignment, RosterLayout], ObservationAdapter]


def _infer_map_xy_scale(map_data: MapData) -> float:
    metadata = map_data.metadata or {}
    resolution = float(metadata.get("resolution", 0.05))
    origin = metadata.get("origin", (0.0, 0.0, 0.0))
    try:
        ox = float(origin[0])
        oy = float(origin[1])
    except (TypeError, ValueError, IndexError):
        ox = 0.0
        oy = 0.0
    width, height = map_data.image_size
    x_vals = [ox, ox + width * resolution]
    y_vals = [oy, oy + height * resolution]
    max_abs = max(abs(val) for val in (x_vals + y_vals))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return 30.0
    return max_abs


def _build_obs_wrapper(
    wrapper_spec: AgentWrapperSpec,
    ctx: "AgentBuildContext",
    assignment: AgentAssignment,
    roster: RosterLayout,
) -> ObservationAdapter:
    params = dict(wrapper_spec.params)

    target_role = params.pop("target_role", None)
    target_slot = params.pop("target_slot", None)
    target_agent = params.pop("target_agent", None) or params.pop("target_agent_id", None)

    if target_slot is not None:
        target_slot = int(target_slot)

    env_lidar = getattr(ctx.env, "lidar_beams", None)
    if "lidar_beams" not in params and env_lidar:
        params["lidar_beams"] = int(env_lidar)

    def _requests_centerline(data: Dict[str, Any]) -> bool:
        components = data.get("components")
        if isinstance(components, list):
            for entry in components:
                if not isinstance(entry, dict):
                    continue
                comp_type = str(entry.get("type") or "").lower()
                comp_id = str(entry.get("id") or "").lower()
                if comp_type == "centerline" or comp_id == "centerline":
                    return True
        if any(key in data for key in ("centerline", "centerline_features")):
            return True
        return False

    wrapper_requests_centerline = _requests_centerline(params)

    # Centerline-aware features
    centerline_points = ctx.map_data.centerline
    use_centerline = params.pop("use_centerline_features", None)
    if use_centerline is None:
        use_centerline = ctx.env.centerline_features_enabled or wrapper_requests_centerline
    use_centerline = bool(use_centerline)
    if use_centerline and centerline_points is not None:
        params.setdefault("centerline", centerline_points)
        params.setdefault("centerline_features", True)
        ctx.env.register_centerline_usage(require_features=True)
    else:
        params.setdefault("centerline_features", False)

    if target_agent is not None:
        params.setdefault("legacy_target_agent", str(target_agent))

    default_pose_scale = _infer_map_xy_scale(ctx.map_data)

    components = params.get("components")
    if isinstance(components, list):
        for entry in components:
            if not isinstance(entry, dict):
                continue
            comp_type = str(entry.get("type") or entry.get("name") or entry.get("id") or "").strip().lower()
            comp_params = entry.setdefault("params", {})
            if comp_type in {"ego_pose", "pose", "target_pose", "relative_pose"}:
                if comp_params.get("normalize_xy") in (None, 0, 0.0):
                    comp_params["normalize_xy"] = default_pose_scale

    obs_wrapper = ObsWrapper(**params)
    return ObservationAdapter(
        name=wrapper_spec.factory,
        wrapper=obs_wrapper,
        target_role=str(target_role) if target_role is not None else None,
        target_slot=target_slot,
        target_agent=str(target_agent) if target_agent is not None else None,
    )


WRAPPER_BUILDERS: Dict[str, WrapperBuilder] = {
    "obs": _build_obs_wrapper,
    "observation": _build_obs_wrapper,
    "obs_wrapper": _build_obs_wrapper,
}


# ---------------------------------------------------------------------------
# Agent builders & registry
# ---------------------------------------------------------------------------


@dataclass
class AgentBundle:
    assignment: AgentAssignment
    algo: str
    controller: Any
    obs_pipeline: ObservationPipeline
    trainable: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    trainer: Optional[Trainer] = None
    action_wrapper: Optional[Any] = None

    @property
    def agent_id(self) -> str:
        return self.assignment.agent_id

    @property
    def role(self) -> Optional[str]:
        return self.assignment.spec.role

    @property
    def slot(self) -> int:
        return self.assignment.slot

    @property
    def wrappers(self) -> ObservationPipeline:
        """Backwards-compatible access to the observation pipeline stages."""

        return self.obs_pipeline


class FunctionPolicy:
    """Light wrapper giving function-based heuristics a class interface."""

    def __init__(self, fn: Callable[[Any, Any], Any], name: str) -> None:
        self._fn = fn
        self.name = name

    def get_action(self, action_space, obs):
        return self._fn(action_space, obs)


@dataclass
class AgentBuildContext:
    env: F110ParallelEnv
    cfg: ExperimentConfig
    roster: RosterLayout
    map_data: MapData
    sample_obs: Optional[Dict[str, Any]] = None
    shared_algorithms: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def ensure_sample(self) -> Dict[str, Any]:
        if self.sample_obs is None:
            seed = self.cfg.env.get("seed")
            try:
                reset_seed = None if seed is None else int(seed)
            except (TypeError, ValueError):
                reset_seed = None
            sample_obs, _ = self.env.reset(seed=reset_seed)
            self.sample_obs = sample_obs
        return self.sample_obs


AgentBuilderFn = Callable[[AgentAssignment, AgentBuildContext, RosterLayout, ObservationPipeline], AgentBundle]


def _resolve_algorithm_config(ctx: AgentBuildContext, spec: AgentSpecConfig) -> Dict[str, Any]:
    base: Dict[str, Any] = {}

    if spec.config_ref:
        section = ctx.cfg.get_section(spec.config_ref)
        if section:
            base.update(section)
    else:
        section = ctx.cfg.get_section(spec.algo)
        if section:
            base.update(section)

    base.update(spec.params)
    return base


def _is_trainable(spec: AgentSpecConfig, default: bool) -> bool:
    if spec.trainable is None:
        return default
    return bool(spec.trainable)


def _config_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _continuous_prevent_reverse_factory(action_space: spaces.Box, algo_cfg: Dict[str, Any]) -> Optional[Any]:
    flag = algo_cfg.get("prevent_reverse")
    if flag is None:
        flag = algo_cfg.get("rate_prevent_reverse")
    if not _config_flag(flag):
        return None

    min_speed_raw = algo_cfg.get("prevent_reverse_min_speed")
    if min_speed_raw is None:
        min_speed_raw = algo_cfg.get("rate_stop_speed", 0.0)
    try:
        min_speed_val = float(min_speed_raw)
    except (TypeError, ValueError):
        min_speed_val = 0.0

    speed_index_raw = algo_cfg.get("prevent_reverse_speed_index")
    if speed_index_raw is None:
        speed_index_raw = algo_cfg.get("rate_speed_index", 1)
    try:
        speed_index_val = int(speed_index_raw)
    except (TypeError, ValueError):
        speed_index_val = 1

    algo_cfg["prevent_reverse"] = True
    algo_cfg["prevent_reverse_min_speed"] = min_speed_val
    algo_cfg["prevent_reverse_speed_index"] = speed_index_val

    return PreventReverseContinuousWrapper(
        action_space.low,
        action_space.high,
        min_speed=min_speed_val,
        speed_index=speed_index_val,
    )


def _build_continuous_algo(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
    *,
    algo_key: str,
    controller_factory: Callable[[Dict[str, Any]], Any],
    trainer_key: Optional[str] = None,
    post_init: Optional[Callable[[Any], None]] = None,
    action_wrapper_factory: Optional[Callable[[spaces.Box, Dict[str, Any]], Any]] = None,
    trainable_default: bool = True,
) -> AgentBundle:
    """Shared builder for algorithms that require Box action spaces."""

    agent_id = assignment.agent_id
    action_space = ctx.env.action_space(agent_id)
    if not isinstance(action_space, spaces.Box):
        raise TypeError(
            f"{algo_key} builder requires a continuous Box action space; "
            f"received {type(action_space)!r} for agent '{agent_id}'"
        )

    if not pipeline:
        raise ValueError(
            f"{algo_key} agent '{agent_id}' requires at least one observation wrapper to define obs_dim"
        )

    sample_obs = ctx.ensure_sample()
    obs_vector = pipeline.to_vector(sample_obs, agent_id, roster)
    algo_cfg = _resolve_algorithm_config(ctx, assignment.spec)
    algo_cfg["obs_dim"] = int(obs_vector.size)
    algo_cfg["act_dim"] = int(action_space.shape[0])
    algo_cfg["action_low"] = action_space.low.astype(np.float32).tolist()
    algo_cfg["action_high"] = action_space.high.astype(np.float32).tolist()

    initial_speed_raw = algo_cfg.get("initial_speed", 0.0)
    try:
        initial_speed = float(initial_speed_raw)
    except (TypeError, ValueError):
        initial_speed = 0.0
    algo_cfg["initial_speed"] = initial_speed

    controller = controller_factory(algo_cfg)
    if post_init is not None:
        post_init(controller)

    trainer_name = trainer_key or algo_key
    trainer = trainer_registry.create_trainer(trainer_name, agent_id, controller, config=algo_cfg)

    action_wrapper = None
    if action_wrapper_factory is not None:
        action_wrapper = action_wrapper_factory(action_space, algo_cfg)

    return AgentBundle(
        assignment=assignment,
        algo=algo_key,
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=trainable_default),
        metadata={
            "config": algo_cfg,
            "initial_speed": initial_speed,
        },
        trainer=trainer,
        action_wrapper=action_wrapper,
    )


def _build_algo_ppo(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    return _build_continuous_algo(
        assignment,
        ctx,
        roster,
        pipeline,
        algo_key="ppo",
        controller_factory=PPOAgent,
        trainer_key="ppo",
        action_wrapper_factory=_continuous_prevent_reverse_factory,
    )


def _build_algo_td3(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    return _build_continuous_algo(
        assignment,
        ctx,
        roster,
        pipeline,
        algo_key="td3",
        controller_factory=TD3Agent,
        trainer_key="td3",
        action_wrapper_factory=_continuous_prevent_reverse_factory,
    )


def _build_algo_sac(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    return _build_continuous_algo(
        assignment,
        ctx,
        roster,
        pipeline,
        algo_key="sac",
        controller_factory=SACAgent,
        trainer_key="sac",
        action_wrapper_factory=_continuous_prevent_reverse_factory,
    )


def _build_algo_rec_ppo(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    return _build_continuous_algo(
        assignment,
        ctx,
        roster,
        pipeline,
        algo_key="rec_ppo",
        controller_factory=RecurrentPPOAgent,
        trainer_key="rec_ppo",
        post_init=lambda controller: getattr(controller, "reset_hidden_state", lambda: None)(),
        action_wrapper_factory=_continuous_prevent_reverse_factory,
    )


def _build_dqn_family_algo(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
    *,
    algo_name: str,
    controller_factory: Callable[[Dict[str, Any]], Any],
    trainer_key: str,
) -> AgentBundle:
    agent_id = assignment.agent_id
    action_space = ctx.env.action_space(agent_id)
    if not isinstance(action_space, spaces.Box):
        raise TypeError(
            "DQN builder currently requires a Box action space for action templates; "
            f"received {type(action_space)!r} for agent '{agent_id}'"
        )

    if not pipeline:
        raise ValueError(
            f"DQN agent '{agent_id}' requires at least one observation wrapper to define obs_dim"
        )

    sample_obs = ctx.ensure_sample()
    obs_vector = pipeline.to_vector(sample_obs, agent_id, roster)
    dqn_cfg = _resolve_algorithm_config(ctx, assignment.spec)
    action_mode = str(dqn_cfg.get("action_mode", "absolute")).lower()

    if action_mode == "rate":
        timestep = float(dqn_cfg.get("rate_dt") or ctx.cfg.env.schema.timestep or 0.01)
        if timestep <= 0:
            raise ValueError("rate-based DQN requires a positive timestep")
        steering_rate = abs(float(dqn_cfg.get("steering_rate", 0.5)))
        accel_rate = abs(float(dqn_cfg.get("accel_rate", 2.0)))
        brake_rate = abs(float(dqn_cfg.get("brake_rate", 4.0)))
        prevent_reverse = bool(dqn_cfg.get("rate_prevent_reverse", True))
        stop_threshold = float(dqn_cfg.get("rate_stop_speed", 0.0))
        speed_index = int(dqn_cfg.get("rate_speed_index", 1))
        steer_delta = steering_rate * timestep
        accel_delta = accel_rate * timestep
        brake_delta = -brake_rate * timestep
        steer_options = (-steer_delta, 0.0, steer_delta)
        pedal_options = (brake_delta, 0.0, accel_delta)
        action_deltas = [[s, p] for s in steer_options for p in pedal_options]
        dqn_cfg["action_set"] = action_deltas
        dqn_cfg["action_deltas"] = action_deltas
        rate_initial = [
            float(dqn_cfg.get("rate_initial_steer", 0.0)),
            float(dqn_cfg.get("rate_initial_speed", 0.0)),
        ]
    else:
        if "action_set" not in dqn_cfg:
            raise ValueError(
                f"DQN agent '{agent_id}' requires an 'action_set' list in config or spec params"
            )
        rate_initial = None

    obs_dim = int(obs_vector.size)
    dqn_cfg["obs_dim"] = obs_dim

    shared_policy_key = dqn_cfg.get("shared_policy") or dqn_cfg.get("shared_policy_id")
    controller: Any
    shared_store = ctx.shared_algorithms.setdefault(algo_name.lower(), {})
    if shared_policy_key:
        policy_id = str(shared_policy_key)
        shared_entry = shared_store.get(policy_id)
        if shared_entry is None:
            controller = controller_factory(dqn_cfg)
            shared_store[policy_id] = {
                "agent": controller,
                "obs_dim": controller.obs_dim,
                "config": dict(dqn_cfg),
            }
        else:
            controller = shared_entry["agent"]
            stored_obs_dim = int(shared_entry.get("obs_dim", controller.obs_dim))
            if stored_obs_dim != obs_dim:
                raise ValueError(
                    f"Shared {algo_name} policy '{policy_id}' expects obs_dim={stored_obs_dim}, received {obs_dim} for agent '{agent_id}'"
                )
    else:
        controller = controller_factory(dqn_cfg)

    trainer = trainer_registry.create_trainer(trainer_key, agent_id, controller, config=dqn_cfg)
    if action_mode == "delta":
        action_wrapper = DeltaDiscreteActionWrapper(
            dqn_cfg.get("action_deltas", dqn_cfg["action_set"]),
            action_space.low,
            action_space.high,
        )
    elif action_mode == "rate":
        action_wrapper = DeltaDiscreteActionWrapper(
            dqn_cfg["action_deltas"],
            action_space.low,
            action_space.high,
            initial_action=rate_initial,
            prevent_reverse=prevent_reverse,
            stop_threshold=stop_threshold,
            speed_index=speed_index,
        )
    else:
        action_wrapper = DiscreteActionWrapper(dqn_cfg["action_set"])

    repeat_steps = int(dqn_cfg.get("action_repeat", 1) or 1)
    if repeat_steps > 1:
        action_wrapper = ActionRepeatWrapper(action_wrapper, repeat_steps)
        dqn_cfg["action_repeat"] = repeat_steps
    return AgentBundle(
        assignment=assignment,
        algo=algo_name,
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=True),
        metadata={"config": dqn_cfg},
        trainer=trainer,
        action_wrapper=action_wrapper,
    )

def _build_algo_dqn(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    return _build_dqn_family_algo(
        assignment,
        ctx,
        roster,
        pipeline,
        algo_name="dqn",
        controller_factory=DQNAgent,
        trainer_key="dqn",
    )


def _build_algo_r_dqn(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    return _build_dqn_family_algo(
        assignment,
        ctx,
        roster,
        pipeline,
        algo_name="r_dqn",
        controller_factory=RainbowDQNAgent,
        trainer_key="r_dqn",
    )


def _build_algo_follow_gap(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = FollowTheGapPolicy.from_config(assignment.spec.params)
    if ctx.map_data.centerline is not None:
        setattr(controller, "centerline", ctx.map_data.centerline)

    # Convey roster context so convoy-style controllers can locate peers.
    if hasattr(controller, "agent_slot"):
        controller.agent_slot = assignment.slot

    target_slot = getattr(controller, "secondary_target_slot", None)
    if isinstance(target_slot, bool):  # guard against bool-as-int
        target_slot = int(target_slot)
    target_agent_id = getattr(controller, "secondary_target_agent", None)
    resolved_target_slot = None
    if target_agent_id:
        for assn in roster.assignments:
            if assn.agent_id == target_agent_id:
                resolved_target_slot = assn.slot
                break
    if resolved_target_slot is not None:
        controller.secondary_target_slot = resolved_target_slot
    else:
        try:
            target_slot_int = int(target_slot) if target_slot is not None else None
        except (TypeError, ValueError):
            target_slot_int = None
        if target_slot_int is None or target_slot_int == assignment.slot or target_slot_int < 0:
            other_slots = [assn.slot for assn in roster.assignments if assn.slot != assignment.slot]
            if other_slots:
                controller.secondary_target_slot = other_slots[0]
            else:
                controller.secondary_target_slot = assignment.slot
        else:
            controller.secondary_target_slot = target_slot_int

    metadata: Dict[str, Any] = {"centerline": ctx.map_data.centerline}
    if assignment.spec.policy_curriculum:
        metadata["policy_curriculum"] = dict(assignment.spec.policy_curriculum)
    return AgentBundle(
        assignment=assignment,
        algo="follow_gap",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata=metadata,
    )


def _build_algo_ftg_centerline(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = FollowTheGapCenterlinePolicy.from_config(assignment.spec.params)
    if ctx.map_data.centerline is not None:
        controller.set_centerline(ctx.map_data.centerline)
    metadata: Dict[str, Any] = {"centerline": ctx.map_data.centerline}
    return AgentBundle(
        assignment=assignment,
        algo="ftg_c",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata=metadata,
    )


def _build_algo_secondary_vicon(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = SecondaryViconPolicy.from_config(assignment.spec.params)

    if hasattr(controller, "agent_slot"):
        controller.agent_slot = assignment.slot

    target_agent_id = getattr(controller, "target_agent", None)
    resolved_target_slot = None
    if target_agent_id:
        for assn in roster.assignments:
            if assn.agent_id == target_agent_id:
                resolved_target_slot = assn.slot
                break

    if resolved_target_slot is not None:
        controller.target_slot = resolved_target_slot
    else:
        try:
            raw_slot = getattr(controller, "target_slot", None)
            target_slot_int = int(raw_slot) if raw_slot is not None else None
        except (TypeError, ValueError):
            target_slot_int = None
        if target_slot_int is None or target_slot_int == assignment.slot or target_slot_int < 0:
            other_slots = [assn.slot for assn in roster.assignments if assn.slot != assignment.slot]
            controller.target_slot = other_slots[0] if other_slots else assignment.slot
        else:
            controller.target_slot = target_slot_int

    if hasattr(ctx.env, "add_render_callback"):
        cache: Dict[str, Any] = {"fn": None, "params": None}

        def lane_border_callback(renderer: EnvRenderer, _cache=cache) -> None:
            params = (
                float(getattr(controller, "lane_center", 0.0)),
                float(getattr(controller, "warning_border", 0.0)),
                float(getattr(controller, "hard_border", 0.0)),
            )
            if _cache["fn"] is None or _cache["params"] != params:
                _cache["fn"] = EnvRenderer.make_lane_border_callback(
                    lane_center=params[0],
                    warning_border=params[1],
                    hard_border=params[2],
                )
                _cache["params"] = params
            callback_fn = _cache["fn"]
            if callable(callback_fn):
                callback_fn(renderer)

        ctx.env.add_render_callback(lane_border_callback)

    metadata: Dict[str, Any] = {}
    if assignment.spec.policy_curriculum:
        metadata["policy_curriculum"] = dict(assignment.spec.policy_curriculum)

    return AgentBundle(
        assignment=assignment,
        algo="secondary_vicon",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata=metadata,
    )


def _build_algo_random(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = FunctionPolicy(random_policy, name="random")
    if ctx.map_data.centerline is not None:
        setattr(controller, "centerline", ctx.map_data.centerline)
    return AgentBundle(
        assignment=assignment,
        algo="random",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata={"centerline": ctx.map_data.centerline},
    )


def _build_algo_waypoint(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = FunctionPolicy(simple_heuristic, name="waypoint")
    if ctx.map_data.centerline is not None:
        setattr(controller, "centerline", ctx.map_data.centerline)
    return AgentBundle(
        assignment=assignment,
        algo="waypoint",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata={
            "note": "Placeholder waypoint heuristic",
            "centerline": ctx.map_data.centerline,
        },
    )


def _build_algo_centerline(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    centerline_points = ctx.map_data.centerline
    params = dict(assignment.spec.params)

    if centerline_points is not None:
        allowed_keys = {
            "lookahead_distance",
            "base_speed",
            "min_speed",
            "max_speed",
            "heading_gain",
            "lateral_gain",
            "turn_slowdown",
        }
        kwargs = {key: params[key] for key in allowed_keys if key in params}
        controller = CenterlinePursuitPolicy(centerline=centerline_points, **kwargs)
    else:
        controller = FunctionPolicy(simple_heuristic, name="centerline")

    if centerline_points is not None:
        ctx.env.register_centerline_usage(require_features=True, require_render=True)
    return AgentBundle(
        assignment=assignment,
        algo="centerline",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata={
            "note": "Centerline pursuit heuristic" if centerline_points is not None else "Placeholder centerline heuristic",
            "centerline": centerline_points,
        },
    )


def _build_algo_blocker(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = BlockingPolicy.from_config(assignment.spec.params)
    controller.total_agents = ctx.env.n_agents
    controller.agent_slot = assignment.slot
    params = assignment.spec.params if isinstance(assignment.spec.params, dict) else {}
    target_id = params.get("target_agent")
    target_slot = None
    if target_id is not None:
        for assn in roster.assignments:
            if assn.agent_id == str(target_id):
                target_slot = assn.slot
                break
    else:
        other_slots = [assn.slot for assn in roster.assignments if assn.slot != assignment.slot]
        target_slot = other_slots[0] if other_slots else None
    controller.target_slot = target_slot
    return AgentBundle(
        assignment=assignment,
        algo="blocker",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata={"target_slot": target_slot},
    )

AGENT_BUILDERS: Dict[str, AgentBuilderFn] = {
    "ppo": _build_algo_ppo,
    "rec_ppo": _build_algo_rec_ppo,
    "follow_gap": _build_algo_follow_gap,
    "gap_follow": _build_algo_follow_gap,
    "followthegap": _build_algo_follow_gap,
    "ftg_c": _build_algo_ftg_centerline,
    "blocker": _build_algo_blocker,
    "secondary_vicon": _build_algo_secondary_vicon,
    "random": _build_algo_random,
    "waypoint": _build_algo_waypoint,
    "centerline": _build_algo_centerline,
    "td3": _build_algo_td3,
    "sac": _build_algo_sac,
    "dqn": _build_algo_dqn,
    "r_dqn": _build_algo_r_dqn,
    "rainbow_dqn": _build_algo_r_dqn,
}


# ---------------------------------------------------------------------------
# Agent team facade
# ---------------------------------------------------------------------------


@dataclass
class AgentTeam:
    env: F110ParallelEnv
    cfg: ExperimentConfig
    roster: RosterLayout
    map_data: MapData
    agents: List[AgentBundle]

    def __post_init__(self) -> None:
        self.by_id: Dict[str, AgentBundle] = {bundle.agent_id: bundle for bundle in self.agents}
        role_map: Dict[str, List[str]] = {}
        for bundle in self.agents:
            if not bundle.role:
                continue
            role_map.setdefault(bundle.role, []).append(bundle.agent_id)
        self.roles: Dict[str, List[str]] = role_map
        self._legacy_tuple: Optional[Tuple[Any, Any, str, str, ObsWrapper]] = self._compute_legacy_tuple()
        self.trainers: Dict[str, Trainer] = {
            bundle.agent_id: bundle.trainer
            for bundle in self.agents
            if bundle.trainer is not None
        }

    # Legacy unpacking ------------------------------------------------------
    def _compute_legacy_tuple(self) -> Optional[Tuple[Any, Any, str, str, ObsWrapper]]:
        if len(self.agents) != 2:
            return None

        ppo_candidates = [bundle for bundle in self.agents if bundle.algo.lower() in {"ppo", "rec_ppo"}]
        if len(ppo_candidates) != 1:
            return None
        ppo_bundle = ppo_candidates[0]

        other_bundle = next((b for b in self.agents if b is not ppo_bundle), None)
        if other_bundle is None or other_bundle.algo.lower() not in {"follow_gap", "gap_follow", "followthegap"}:
            return None

        obs_wrapper = None
        for adapter in ppo_bundle.wrappers:
            if isinstance(adapter.wrapper, ObsWrapper):
                obs_wrapper = adapter.wrapper
                break

        if obs_wrapper is None:
            return None

        return (
            ppo_bundle.controller,
            other_bundle.controller,
            ppo_bundle.agent_id,
            other_bundle.agent_id,
            obs_wrapper,
        )

    def __iter__(self) -> Iterable[Any]:
        if self._legacy_tuple is None:
            raise RuntimeError(
                "AgentTeam no longer supports tuple unpacking for this roster; update the caller to use the new API"
            )
        return iter(self._legacy_tuple)

    # Convenience accessors -------------------------------------------------
    def observation(self, agent_id: str, raw_obs: Dict[str, Any]) -> Any:
        bundle = self.by_id[agent_id]
        return bundle.obs_pipeline.transform(raw_obs, agent_id, self.roster)

    def action(self, agent_id: str, action: Any, *, return_info: bool = False) -> Any:
        bundle = self.by_id[agent_id]
        wrapper = bundle.action_wrapper
        meta = None
        if wrapper is None:
            transformed = action
        else:
            transform_with_info = getattr(wrapper, "transform_with_info", None)
            if callable(transform_with_info):
                transformed, meta = transform_with_info(agent_id, action)
            else:
                transformed = wrapper.transform(agent_id, action)

        if return_info:
            return transformed, meta
        return transformed

    def apply_initial_conditions(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust the environment to honour any per-agent metadata initialisers."""
        speed_map: Dict[str, float] = {}
        for bundle in self.agents:
            initial = bundle.metadata.get("initial_speed")
            if initial is None:
                config = bundle.metadata.get("config")
                if isinstance(config, dict):
                    initial = config.get("initial_speed")
            if initial is None:
                continue
            try:
                speed_val = float(initial)
            except (TypeError, ValueError):
                continue
            if np.isfinite(speed_val) and abs(speed_val) > 0.0:
                speed_map[bundle.agent_id] = speed_val
        if not speed_map:
            return obs

        updated = self.env.apply_initial_speeds(speed_map)
        if updated is None:
            return obs
        return updated

    def reset_actions(self) -> None:
        for bundle in self.agents:
            wrapper = bundle.action_wrapper
            if wrapper is None:
                continue
            reset_fn = getattr(wrapper, "reset", None)
            if callable(reset_fn):
                reset_fn(bundle.agent_id)
        for bundle in self.agents:
            controller_reset = getattr(bundle.controller, "reset_hidden_state", None)
            if callable(controller_reset):
                controller_reset()

    # Role helpers ---------------------------------------------------------
    def role_ids(self, role: str) -> List[str]:
        """Return all agent identifiers bound to a given role."""

        if not role:
            return []
        return list(self.roles.get(role, []))

    def primary_role(self, role: str) -> Optional[str]:
        """Return the first declared agent identifier for a role, if unique."""

        members = self.role_ids(role)
        if len(members) == 1:
            return members[0]
        return None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_agents(
    env: F110ParallelEnv,
    cfg: ExperimentConfig,
    map_data: MapData,
    *,
    ensure_sample: bool = True,
) -> AgentTeam:
    """Build all controller instances declared in the config.

    Returns an :class:`AgentTeam` facade which exposes convenient lookups for
    policies, observation pipelines, and role metadata. When ``ensure_sample`` is
    true (the default) the environment will be reset once to probe the
    observation space so policy builders can infer network dimensions.
    """

    roster = _compile_roster(cfg, env)
    context = AgentBuildContext(env=env, cfg=cfg, roster=roster, map_data=map_data)
    if ensure_sample:
        context.ensure_sample()

    bundles: List[AgentBundle] = []

    for assignment in roster.assignments:
        stages: List[ObservationAdapter] = []
        for wrapper_spec in assignment.spec.wrappers:
            builder = WRAPPER_BUILDERS.get(wrapper_spec.factory.lower())
            if builder is None:
                raise KeyError(
                    f"Unknown wrapper factory '{wrapper_spec.factory}' for agent '{assignment.agent_id}'"
                )
            stages.append(builder(wrapper_spec, context, assignment, roster))

        pipeline = ObservationPipeline(stages)

        builder_fn = AGENT_BUILDERS.get(assignment.spec.algo.lower())
        if builder_fn is None:
            raise KeyError(
                f"Unknown agent algorithm '{assignment.spec.algo}' for agent '{assignment.agent_id}'"
            )
        bundles.append(builder_fn(assignment, context, roster, pipeline))

    return AgentTeam(env=env, cfg=cfg, roster=roster, map_data=map_data, agents=bundles)
