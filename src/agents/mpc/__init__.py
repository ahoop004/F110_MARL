"""Shared MPC utilities for fixed-policy racing controllers."""

from agents.mpc.base import (
    MPCConfig,
    MPCPlanResult,
    evaluate_action_sequences,
    generate_action_sequences,
    make_action_grid,
    plan_grid_search,
)
from agents.mpc.costs import (
    CostWeights,
    MPCCWeights,
    control_effort_cost,
    heading_error_cost,
    mpcc_geometry_cost,
    path_tracking_cost,
    progress_reward,
    steering_smoothness_cost,
    target_speed_cost,
    trajectory_cost,
)
from agents.mpc.cbf import CBFMPCAgent, soft_barrier_cost
from agents.mpc.defensive import (
    DefensiveMPCAgent,
    build_defensive_centerline,
    defensive_bias_active,
    defensive_lateral_offset,
    extract_target_pose,
    target_safety_cost,
)
from agents.mpc.kinematic import KinematicMPCAgent
from agents.mpc.mpcc import MPCCAgent
from agents.mpc.obstacles import (
    ObstacleAwareMPCAgent,
    extract_lidar_scan,
    lidar_points_world,
    obstacle_proximity_cost,
)
from agents.mpc.rollout import (
    DEFAULT_DT,
    DEFAULT_WHEELBASE,
    kinematic_bicycle_step,
    normalize_actions,
    normalize_pose,
    rollout_kinematic_bicycle,
)

__all__ = [
    "CostWeights",
    "CBFMPCAgent",
    "DEFAULT_DT",
    "DEFAULT_WHEELBASE",
    "DefensiveMPCAgent",
    "KinematicMPCAgent",
    "MPCConfig",
    "MPCCAgent",
    "MPCPlanResult",
    "MPCCWeights",
    "ObstacleAwareMPCAgent",
    "build_defensive_centerline",
    "control_effort_cost",
    "defensive_bias_active",
    "defensive_lateral_offset",
    "evaluate_action_sequences",
    "extract_lidar_scan",
    "extract_target_pose",
    "generate_action_sequences",
    "heading_error_cost",
    "kinematic_bicycle_step",
    "lidar_points_world",
    "make_action_grid",
    "mpcc_geometry_cost",
    "normalize_actions",
    "normalize_pose",
    "obstacle_proximity_cost",
    "path_tracking_cost",
    "plan_grid_search",
    "progress_reward",
    "rollout_kinematic_bicycle",
    "soft_barrier_cost",
    "steering_smoothness_cost",
    "target_safety_cost",
    "target_speed_cost",
    "trajectory_cost",
]
