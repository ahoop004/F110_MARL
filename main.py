from pathlib import Path

import yaml
import numpy as np
from PIL import Image
from f110x.envs import F110ParallelEnv
from policies.random_policy import random_policy
from policies.simple_heuristic import simple_heuristic


POLICIES = {
    "random": random_policy,
    "heuristic": simple_heuristic,
}



with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

scenario = cfg["agents"]
env_cfg = cfg["env"]
map_dir = Path(env_cfg.get("map_dir", ""))
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

env = F110ParallelEnv(**cfg["env"])

def run_eval(env, policy_fn, episodes=5):
    results = []
    for ep in range(episodes):
        obs, infos = env.reset()
        done = {aid: False for aid in env.agents}
        totals = {aid: 0.0 for aid in env.agents}

        while not all(done.values()):
            actions = {}
            for aid in env.agents:
                if not done[aid]:
                    actions[aid] = policy_fn(env.action_space(aid), obs[aid])
            obs, rewards, terms, truncs, infos = env.step(actions)
            for aid, r in rewards.items():
                totals[aid] += r
            done = {aid: terms[aid] or truncs[aid] for aid in env.possible_agents}
            env.render()
        results.append(totals)
    return results


for name, fn in POLICIES.items():
    scores = run_eval(env, fn, episodes=3)
    print(name, scores)
