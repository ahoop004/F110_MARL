"""Factory helpers for building environments and agent teams from configs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
from f110x.utils.start_pose import parse_start_pose_options
from f110x.wrappers.observation import ObsWrapper
from f110x.wrappers.action import (
    DiscreteActionWrapper,
    DeltaDiscreteActionWrapper,
)
from f110x.wrappers.common import to_numpy
from f110x.policies.gap_follow import FollowTheGapPolicy
from f110x.policies.ppo.ppo import PPOAgent
from f110x.policies.ppo.rec_ppo import RecurrentPPOAgent
from f110x.policies.random_policy import random_policy
from f110x.policies.simple_heuristic import simple_heuristic
from f110x.policies.td3.td3 import TD3Agent
from f110x.policies.sac.sac import SACAgent
from f110x.policies.dqn.dqn import DQNAgent
from f110x.trainers.base import Trainer
from f110x.trainers.ppo_guided import PPOTrainer
from f110x.trainers.rec_ppo_trainer import RecurrentPPOTrainer
from f110x.trainers.td3_trainer import TD3Trainer
from f110x.trainers.dqn_trainer import DQNTrainer
from f110x.trainers.sac_trainer import SACTrainer


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------


def build_env(cfg: ExperimentConfig) -> Tuple[F110ParallelEnv, MapData, Optional[List[np.ndarray]]]:
    """Instantiate the simulator, returning env + loaded map artefacts."""

    loader = MapLoader()
    env_cfg_dict = cfg.env.to_kwargs()
    map_data = loader.load(env_cfg_dict)
    env_cfg = dict(env_cfg_dict)
    env_cfg["map_meta"] = map_data.metadata
    env_cfg["map_image_path"] = str(map_data.image_path)
    env_cfg["map_image_size"] = map_data.image_size
    env_cfg["map_yaml_path"] = str(map_data.yaml_path)

    # Allow map metadata to override start position defaults when provided.
    for meta_key in ("start_poses", "start_pose_options"):
        if meta_key in map_data.metadata:
            env_cfg[meta_key] = map_data.metadata[meta_key]

    env = F110ParallelEnv(**env_cfg)
    start_pose_options = parse_start_pose_options(env_cfg.get("start_pose_options"))
    return env, map_data, start_pose_options


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
        by_role: Dict[str, AgentAssignment] = {}
        for assignment in self.assignments:
            by_id[assignment.agent_id] = assignment
            by_slot[assignment.slot] = assignment
            role = assignment.spec.role
            if role:
                by_role[role] = assignment
        self._by_id = by_id
        self._by_slot = by_slot
        self._by_role = by_role

    # Lookups ---------------------------------------------------------------
    def resolve_slot(self, slot: int) -> Optional[str]:
        assignment = self._by_slot.get(slot)
        return assignment.agent_id if assignment else None

    def resolve_role(self, role: str) -> Optional[str]:
        assignment = self._by_role.get(role)
        return assignment.agent_id if assignment else None

    def first_other(self, agent_id: str) -> Optional[str]:
        for other_id in self._by_id.keys():
            if other_id != agent_id:
                return other_id
        return None

    @property
    def agent_ids(self) -> List[str]:
        return [assignment.agent_id for assignment in self.assignments]

    @property
    def roles(self) -> Dict[str, str]:
        return {
            assignment.spec.role: assignment.agent_id
            for assignment in self.assignments
            if assignment.spec.role
        }


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
        current: Any = None,
    ) -> Any:
        target_id = self._resolve_target(agent_id, roster)
        return self.wrapper(raw_obs, agent_id, target_id)

    def _resolve_target(self, agent_id: str, roster: RosterLayout) -> Optional[str]:
        explicit = self.target_agent
        if explicit is not None:
            return explicit

        if self.target_role is not None:
            resolved = roster.resolve_role(self.target_role)
            if resolved is not None and resolved != agent_id:
                return resolved

        if self.target_slot is not None:
            resolved = roster.resolve_slot(self.target_slot)
            if resolved is not None and resolved != agent_id:
                return resolved

        fallback = roster.first_other(agent_id)
        if fallback is None:
            raise ValueError(
                f"Observation adapter '{self.name}' could not identify a target for agent '{agent_id}'"
            )
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

    @property
    def names(self) -> List[str]:
        return [stage.name for stage in self._stages]

    def transform(
        self,
        raw_obs: Dict[str, Any],
        agent_id: str,
        roster: RosterLayout,
        *,
        initial: Any = None,
    ) -> Any:
        result: Any = raw_obs if initial is None else initial
        for stage in self._stages:
            result = stage(raw_obs, agent_id, roster, result)
        return result

    def to_vector(self, raw_obs: Dict[str, Any], agent_id: str, roster: RosterLayout) -> np.ndarray:
        output = self.transform(raw_obs, agent_id, roster)
        vector = to_numpy(output, flatten=True)
        return vector.astype(np.float32, copy=False)


WrapperBuilder = Callable[[AgentWrapperSpec, "AgentBuildContext", AgentAssignment, RosterLayout], ObservationAdapter]


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

    algo = assignment.spec.algo.lower()
    if algo in {"ppo", "rec_ppo", "td3", "sac"} and "lidar_beams" not in params:
        env_lidar = getattr(ctx.env, "lidar_beams", None)
        if env_lidar:
            params["lidar_beams"] = int(env_lidar)

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
    sample_obs: Optional[Dict[str, Any]] = None
    sample_infos: Optional[Dict[str, Any]] = None

    def ensure_sample(self) -> Dict[str, Any]:
        if self.sample_obs is None:
            seed = self.cfg.env.get("seed")
            try:
                reset_seed = None if seed is None else int(seed)
            except (TypeError, ValueError):
                reset_seed = None
            self.sample_obs, self.sample_infos = self.env.reset(seed=reset_seed)
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


def _build_algo_ppo(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    agent_id = assignment.agent_id
    action_space = ctx.env.action_space(agent_id)
    if not isinstance(action_space, spaces.Box):
        raise TypeError(
            "PPO builder currently requires a continuous Box action space; "
            f"received {type(action_space)!r} for agent '{agent_id}'"
        )

    if not pipeline:
        raise ValueError(
            f"PPO agent '{agent_id}' requires at least one observation wrapper to define obs_dim"
        )

    sample_obs = ctx.ensure_sample()
    obs_vector = pipeline.to_vector(sample_obs, agent_id, roster)
    ppo_cfg = _resolve_algorithm_config(ctx, assignment.spec)
    ppo_cfg["obs_dim"] = int(obs_vector.size)
    ppo_cfg["act_dim"] = int(action_space.shape[0])
    ppo_cfg["action_low"] = action_space.low.astype(np.float32).tolist()
    ppo_cfg["action_high"] = action_space.high.astype(np.float32).tolist()

    controller = PPOAgent(ppo_cfg)
    trainer = PPOTrainer(agent_id, controller)
    # PPOAgent already scales actions to environment bounds internally.
    action_wrapper = None
    return AgentBundle(
        assignment=assignment,
        algo="ppo",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=True),
        metadata={"config": ppo_cfg},
        trainer=trainer,
        action_wrapper=action_wrapper,
    )


def _build_algo_td3(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    agent_id = assignment.agent_id
    action_space = ctx.env.action_space(agent_id)
    if not isinstance(action_space, spaces.Box):
        raise TypeError(
            "TD3 builder requires a continuous Box action space; "
            f"received {type(action_space)!r} for agent '{agent_id}'"
        )

    if not pipeline:
        raise ValueError(
            f"TD3 agent '{agent_id}' requires at least one observation wrapper to define obs_dim"
        )

    sample_obs = ctx.ensure_sample()
    obs_vector = pipeline.to_vector(sample_obs, agent_id, roster)
    td3_cfg = _resolve_algorithm_config(ctx, assignment.spec)
    td3_cfg["obs_dim"] = int(obs_vector.size)
    td3_cfg["act_dim"] = int(action_space.shape[0])
    td3_cfg["action_low"] = action_space.low.astype(np.float32).tolist()
    td3_cfg["action_high"] = action_space.high.astype(np.float32).tolist()

    controller = TD3Agent(td3_cfg)
    trainer = TD3Trainer(agent_id, controller)
    # TD3Agent actions are already emitted in environment units, so skip rescaling wrapper.
    action_wrapper = None
    return AgentBundle(
        assignment=assignment,
        algo="td3",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=True),
        metadata={"config": td3_cfg},
        trainer=trainer,
        action_wrapper=action_wrapper,
    )


def _build_algo_sac(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    agent_id = assignment.agent_id
    action_space = ctx.env.action_space(agent_id)
    if not isinstance(action_space, spaces.Box):
        raise TypeError(
            "SAC builder requires a continuous Box action space; "
            f"received {type(action_space)!r} for agent '{agent_id}'"
        )

    if not pipeline:
        raise ValueError(
            f"SAC agent '{agent_id}' requires at least one observation wrapper to define obs_dim"
        )

    sample_obs = ctx.ensure_sample()
    obs_vector = pipeline.to_vector(sample_obs, agent_id, roster)
    sac_cfg = _resolve_algorithm_config(ctx, assignment.spec)
    sac_cfg["obs_dim"] = int(obs_vector.size)
    sac_cfg["act_dim"] = int(action_space.shape[0])
    sac_cfg["action_low"] = action_space.low.astype(np.float32).tolist()
    sac_cfg["action_high"] = action_space.high.astype(np.float32).tolist()

    controller = SACAgent(sac_cfg)
    trainer = SACTrainer(agent_id, controller)
    # SACAgent outputs environment-scaled actions directly; avoid double scaling.
    action_wrapper = None
    return AgentBundle(
        assignment=assignment,
        algo="sac",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=True),
        metadata={"config": sac_cfg},
        trainer=trainer,
        action_wrapper=action_wrapper,
    )


def _build_algo_rec_ppo(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    agent_id = assignment.agent_id
    action_space = ctx.env.action_space(agent_id)
    if not isinstance(action_space, spaces.Box):
        raise TypeError(
            "Recurrent PPO builder requires a continuous Box action space; "
            f"received {type(action_space)!r} for agent '{agent_id}'"
        )

    if not pipeline:
        raise ValueError(
            f"Recurrent PPO agent '{agent_id}' requires at least one observation wrapper to define obs_dim"
        )

    sample_obs = ctx.ensure_sample()
    obs_vector = pipeline.to_vector(sample_obs, agent_id, roster)
    rec_cfg = _resolve_algorithm_config(ctx, assignment.spec)
    rec_cfg["obs_dim"] = int(obs_vector.size)
    rec_cfg["act_dim"] = int(action_space.shape[0])
    rec_cfg["action_low"] = action_space.low.astype(np.float32).tolist()
    rec_cfg["action_high"] = action_space.high.astype(np.float32).tolist()

    controller = RecurrentPPOAgent(rec_cfg)
    controller.reset_hidden_state()
    trainer = RecurrentPPOTrainer(agent_id, controller)
    # Recurrent PPO emits environment-scaled actions directly.
    action_wrapper = None
    return AgentBundle(
        assignment=assignment,
        algo="rec_ppo",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=True),
        metadata={"config": rec_cfg},
        trainer=trainer,
        action_wrapper=action_wrapper,
    )


def _build_algo_dqn(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
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
    if "action_set" not in dqn_cfg:
        raise ValueError(
            f"DQN agent '{agent_id}' requires an 'action_set' list in config or spec params"
        )

    action_mode = str(dqn_cfg.get("action_mode", "absolute")).lower()
    dqn_cfg["obs_dim"] = int(obs_vector.size)

    controller = DQNAgent(dqn_cfg)
    trainer = DQNTrainer(agent_id, controller)
    if action_mode == "delta":
        action_wrapper = DeltaDiscreteActionWrapper(
            dqn_cfg.get("action_deltas", dqn_cfg["action_set"]),
            action_space.low,
            action_space.high,
        )
    else:
        action_wrapper = DiscreteActionWrapper(dqn_cfg["action_set"])
    return AgentBundle(
        assignment=assignment,
        algo="dqn",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=True),
        metadata={"config": dqn_cfg},
        trainer=trainer,
        action_wrapper=action_wrapper,
    )


def _build_algo_follow_gap(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = FollowTheGapPolicy.from_config(assignment.spec.params)
    return AgentBundle(
        assignment=assignment,
        algo="follow_gap",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
    )


def _build_algo_random(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = FunctionPolicy(random_policy, name="random")
    return AgentBundle(
        assignment=assignment,
        algo="random",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
    )


def _build_algo_waypoint(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = FunctionPolicy(simple_heuristic, name="waypoint")
    return AgentBundle(
        assignment=assignment,
        algo="waypoint",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata={"note": "Placeholder waypoint heuristic"},
    )


def _build_algo_centerline(
    assignment: AgentAssignment,
    ctx: AgentBuildContext,
    roster: RosterLayout,
    pipeline: ObservationPipeline,
) -> AgentBundle:
    controller = FunctionPolicy(simple_heuristic, name="centerline")
    return AgentBundle(
        assignment=assignment,
        algo="centerline",
        controller=controller,
        obs_pipeline=pipeline,
        trainable=_is_trainable(assignment.spec, default=False),
        metadata={"note": "Placeholder centerline heuristic"},
    )


def _unsupported_builder(name: str) -> AgentBuilderFn:
    def builder(
        assignment: AgentAssignment,
        ctx: AgentBuildContext,
        roster: RosterLayout,
        pipeline: ObservationPipeline,
    ) -> AgentBundle:
        raise NotImplementedError(
            f"Agent algorithm '{name}' is registered but does not have an implementation yet"
        )

    return builder


AGENT_BUILDERS: Dict[str, AgentBuilderFn] = {
    "ppo": _build_algo_ppo,
    "rec_ppo": _build_algo_rec_ppo,
    "follow_gap": _build_algo_follow_gap,
    "gap_follow": _build_algo_follow_gap,
    "followthegap": _build_algo_follow_gap,
    "random": _build_algo_random,
    "waypoint": _build_algo_waypoint,
    "centerline": _build_algo_centerline,
    "td3": _build_algo_td3,
    "sac": _build_algo_sac,
    "dqn": _build_algo_dqn,
}


# ---------------------------------------------------------------------------
# Agent team facade
# ---------------------------------------------------------------------------


@dataclass
class AgentTeam:
    env: F110ParallelEnv
    cfg: ExperimentConfig
    roster: RosterLayout
    agents: List[AgentBundle]

    def __post_init__(self) -> None:
        self.by_id: Dict[str, AgentBundle] = {bundle.agent_id: bundle for bundle in self.agents}
        self.roles: Dict[str, str] = {
            bundle.role: bundle.agent_id for bundle in self.agents if bundle.role
        }
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
    def policy(self, agent_id: str) -> Any:
        return self.by_id[agent_id].controller

    def observation(self, agent_id: str, raw_obs: Dict[str, Any]) -> Any:
        bundle = self.by_id[agent_id]
        return bundle.obs_pipeline.transform(raw_obs, agent_id, self.roster)

    def action(self, agent_id: str, action: Any) -> Any:
        bundle = self.by_id[agent_id]
        wrapper = bundle.action_wrapper
        if wrapper is None:
            return action
        return wrapper.transform(agent_id, action)

    @property
    def trainable_agents(self) -> List[str]:
        return [bundle.agent_id for bundle in self.agents if bundle.trainable]

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


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_agents(env: F110ParallelEnv, cfg: ExperimentConfig, *, ensure_sample: bool = True) -> AgentTeam:
    """Build all controller instances declared in the config.

    Returns an :class:`AgentTeam` facade which exposes convenient lookups for
    policies, observation pipelines, and role metadata. When ``ensure_sample`` is
    true (the default) the environment will be reset once to probe the
    observation space so policy builders can infer network dimensions.
    """

    roster = _compile_roster(cfg, env)
    context = AgentBuildContext(env=env, cfg=cfg, roster=roster)
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

    return AgentTeam(env=env, cfg=cfg, roster=roster, agents=bundles)
