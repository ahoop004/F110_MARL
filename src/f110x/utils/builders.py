"""Factory helpers for building envs and agents from ExperimentConfig."""
from __future__ import annotations

from typing import Tuple, Optional, List
import numpy as np

from f110x.utils.config_models import ExperimentConfig
from f110x.utils.map_loader import MapLoader, MapData
from f110x.utils.start_pose import parse_start_pose_options
from f110x.envs import F110ParallelEnv
from policies.gap_follow import FollowTheGapPolicy
from policies.ppo.ppo import PPOAgent
from f110x.wrappers.observation import ObsWrapper


def build_env(cfg: ExperimentConfig) -> Tuple[F110ParallelEnv, MapData, Optional[List[np.ndarray]]]:
    loader = MapLoader()
    env_cfg_dict = cfg.env.to_kwargs()
    map_data = loader.load(env_cfg_dict)
    env_cfg = dict(env_cfg_dict)
    env_cfg["map_meta"] = map_data.metadata
    env_cfg["map_image_path"] = str(map_data.image_path)
    env_cfg["map_image_size"] = map_data.image_size
    env_cfg["map_yaml_path"] = str(map_data.yaml_path)
    env = F110ParallelEnv(**env_cfg)
    start_pose_options = parse_start_pose_options(env_cfg.get("start_pose_options"))
    return env, map_data, start_pose_options


def build_agents(env: F110ParallelEnv, cfg: ExperimentConfig):
    obs_wrapper = ObsWrapper(max_scan=30.0, normalize=True)
    gap_policy = FollowTheGapPolicy()

    obs, _ = env.reset()
    agent_ids = env.agents
    ppo_idx = cfg.ppo_agent_idx
    gap_idx = 1 - ppo_idx
    ppo_agent_id = agent_ids[ppo_idx]
    gap_agent_id = agent_ids[gap_idx]

    action_space = env.action_space(ppo_agent_id)
    ppo_cfg = cfg.ppo.to_dict()
    ppo_cfg["obs_dim"] = len(obs_wrapper(obs, ppo_agent_id, gap_agent_id))
    ppo_cfg["act_dim"] = action_space.shape[0]
    ppo_cfg["action_low"] = action_space.low.astype(np.float32).tolist()
    ppo_cfg["action_high"] = action_space.high.astype(np.float32).tolist()

    ppo_agent = PPOAgent(ppo_cfg)
    return ppo_agent, gap_policy, ppo_agent_id, gap_agent_id, obs_wrapper
