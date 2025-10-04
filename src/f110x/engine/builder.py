"""Helper routines for assembling runner contexts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from f110x.engine.reward import CurriculumSchedule, build_curriculum_schedule
from f110x.runner.context import RunnerContext
from f110x.utils.builders import AgentBundle, AgentTeam, build_agents, build_env
from f110x.utils.config_models import ExperimentConfig
from f110x.utils.logger import Logger


def _select_primary_bundle(
    trainable_bundles: Iterable[AgentBundle],
    prefer_algorithms: Sequence[str],
) -> Optional[AgentBundle]:
    bundles: List[AgentBundle] = list(trainable_bundles)
    if not bundles:
        return None

    preferred = [algo.lower() for algo in prefer_algorithms]
    if preferred:
        for algo in preferred:
            for bundle in bundles:
                if bundle.algo.lower() == algo:
                    return bundle

    return bundles[0]


def build_runner_context(
    cfg: ExperimentConfig,
    *,
    logger: Optional[Logger] = None,
    prefer_algorithms: Sequence[str] = ("ppo", "rec_ppo"),
    ensure_trainable: bool = True,
) -> RunnerContext:
    """Compose a :class:`RunnerContext` from the provided configuration."""

    env, map_data, start_pose_options = build_env(cfg)
    team = build_agents(env, cfg, map_data)

    trainer_map = {
        bundle.agent_id: bundle.trainer
        for bundle in team.agents
        if bundle.trainer is not None
    }

    trainable_bundles: List[AgentBundle] = []
    for bundle in team.agents:
        if bundle.trainer is None:
            continue
        trainable = bundle.trainable
        if trainable is None:
            trainable = True
        if trainable:
            trainable_bundles.append(bundle)

    if ensure_trainable and not trainable_bundles:
        raise RuntimeError("No trainable agent with a trainer adapter found in roster")

    primary_bundle = _select_primary_bundle(trainable_bundles, prefer_algorithms)
    primary_agent_id = primary_bundle.agent_id if primary_bundle else None

    reward_cfg = cfg.reward.to_dict()
    raw_curriculum = cfg.get("reward_curriculum", [])
    if not isinstance(raw_curriculum, Iterable):
        raw_curriculum = []
    curriculum_schedule: CurriculumSchedule = build_curriculum_schedule(raw_curriculum)

    render_interval = int(cfg.env.get("render_interval", 0) or 0)
    update_after = int(cfg.env.get("update", 1) or 1)
    start_pose_back_gap = float(cfg.env.get("start_pose_back_gap", 0.0) or 0.0)
    start_pose_min_spacing = float(cfg.env.get("start_pose_min_spacing", 0.0) or 0.0)

    output_root = Path(cfg.main.get("output_root", "outputs")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    cfg.main.schema.output_root = str(output_root)

    ctx = RunnerContext(
        cfg=cfg,
        env=env,
        map_data=map_data,
        start_pose_options=start_pose_options,
        team=team,
        reward_cfg=reward_cfg,
        curriculum_schedule=curriculum_schedule,
        output_root=output_root,
        start_pose_back_gap=start_pose_back_gap,
        start_pose_min_spacing=start_pose_min_spacing,
        render_interval=render_interval,
        update_after=update_after,
        trainer_map=trainer_map,
        trainable_ids=[bundle.agent_id for bundle in trainable_bundles],
        primary_agent_id=primary_agent_id,
        metadata={
            "preferred_primary_algorithms": list(prefer_algorithms),
        },
        logger=logger or Logger(),
    )

    return ctx


__all__ = ["build_runner_context"]
