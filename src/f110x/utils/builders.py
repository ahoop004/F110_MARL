from typing import Tuple
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from f110x.utils.config_models import ExperimentConfig
from f110x.envs import F110ParallelEnv
from policies.gap_follow import FollowTheGapPolicy
from policies.ppo.ppo import PPOAgent
from f110x.wrappers.observation import ObsWrapper


def build_env_and_agents(cfg: ExperimentConfig):
    env_cfg = cfg.env.to_kwargs()

    map_dir = Path(cfg.env.get("map_dir", ""))
    map_yaml_name = env_cfg.get("map_yaml") or env_cfg.get("map")
    if map_yaml_name is None:
        raise ValueError("config.env must define map_yaml or map")

    map_yaml_path = (map_dir / map_yaml_name).expanduser().resolve()
    with open(map_yaml_path, "r") as map_file:
        map_meta = yaml.safe_load(map_file)

    image_rel = map_meta.get("image")
    fallback_image = env_cfg.get("map_image")
    if image_rel:
        image_path = (map_yaml_path.parent / image_rel).resolve()
    elif fallback_image:
        image_path = (map_dir / fallback_image).expanduser().resolve()
    else:
        map_ext = env_cfg.get("map_ext", ".png")
        image_path = map_yaml_path.with_suffix(map_ext)

    with Image.open(image_path) as map_img:
        image_size = map_img.size

    env_cfg["map_meta"] = map_meta
    env_cfg["map_image_path"] = str(image_path)
    env_cfg["map_image_size"] = image_size

    env = F110ParallelEnv(**env_cfg)

    start_pose_options = env_cfg.get("start_pose_options")
    processed_options = None
    if start_pose_options:
        processed_options = []
        for option in start_pose_options:
            arr = np.asarray(option, dtype=np.float32)
            if arr.ndim == 1:
                arr = np.expand_dims(arr, axis=0)
            processed_options.append(arr)

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

    return env, env_cfg, processed_options, ppo_agent, gap_policy, obs_wrapper, ppo_agent_id, gap_agent_id
