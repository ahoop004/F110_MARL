from pathlib import Path

import yaml
import numpy as np
from PIL import Image
from f110x.envs import F110ParallelEnv
from policies.random_policy import random_policy
from policies.simple_heuristic import simple_heuristic
from policies.gap_follow import FollowTheGapPolicy
from policies.ppo.ppo import PPOAgent
from f110x.wrappers.observation import ObsWrapper
from f110x.wrappers.reward import RewardWrapper



with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

env_cfg = cfg["env"]

# Map setup
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

# Environment
env = F110ParallelEnv(**env_cfg)

# -------------------------------------------------------------------
# Initialize wrappers & policies
# -------------------------------------------------------------------
obs_wrapper = ObsWrapper(max_scan=30.0, normalize=True)
reward_cfg = cfg.get("reward", {})
reward_wrapper = RewardWrapper(**reward_cfg)

gap_policy = FollowTheGapPolicy()

# Reset once to probe obs space
obs, infos = env.reset()
agent_ids = env.agents  # e.g. ["A", "B"]

ppo_agent_idx = cfg.get("ppo_agent_idx", 0)
gap_agent_idx = 1 - ppo_agent_idx

PPO_AGENT = agent_ids[ppo_agent_idx]
GAP_AGENT = agent_ids[gap_agent_idx]

# Build PPO agent config
obs_dim = len(obs_wrapper(obs, PPO_AGENT, GAP_AGENT))
act_dim = env.action_space(PPO_AGENT).shape[0]

ppo_cfg = cfg["ppo"].copy()
ppo_cfg["obs_dim"] = obs_dim
ppo_cfg["act_dim"] = act_dim
ppo_agent = PPOAgent(ppo_cfg)

print(f"[INFO] PPO agent: {PPO_AGENT}, Gap-follow agent: {GAP_AGENT}")
print(f"[INFO] Obs dim: {obs_dim}, Act dim: {act_dim}")

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def run_mixed(env, episodes=5):
    results = []
    for ep in range(episodes):
        obs, infos = env.reset()
        done = {aid: False for aid in env.agents}
        totals = {aid: 0.0 for aid in env.agents}
        reward_wrapper.reset()

        while not all(done.values()):
            actions = {}

            # PPO-controlled agent
            if PPO_AGENT in obs and not done.get(PPO_AGENT, False):
                o_wrapped = obs_wrapper(obs, PPO_AGENT, GAP_AGENT)
                actions[PPO_AGENT] = ppo_agent.act(o_wrapped, PPO_AGENT)

            # Gap-follow agent
            if GAP_AGENT in obs and not done.get(GAP_AGENT, False):
                actions[GAP_AGENT] = gap_policy.get_action(
                    env.action_space(GAP_AGENT), obs[GAP_AGENT]
                )

            # Step
            obs, rewards, terms, truncs, infos = env.step(actions)

            if all(terms.get(aid, False) or truncs.get(aid, False) for aid in done.keys()):
                break

            # Rewards & storage for surviving agents
            for aid in obs.keys():
                shaped = reward_wrapper(
                    obs, aid,
                    rewards.get(aid, 0.0),
                    done=terms.get(aid, False) or truncs.get(aid, False),
                    info=infos.get(aid, {})
                )
                totals[aid] = totals.get(aid, 0.0) + shaped

                if (
                    aid == PPO_AGENT
                    and aid in actions
                    and not (terms.get(aid, False) or truncs.get(aid, False))
                ):
                    o_wrapped = obs_wrapper(obs, PPO_AGENT, GAP_AGENT)
                    ppo_agent.store(
                        o_wrapped,
                        actions[aid],
                        shaped,
                        terms.get(aid, False) or truncs.get(aid, False)
                    )

            # Update done flags safely
            done.update({aid: terms.get(aid, False) or truncs.get(aid, False)
                         for aid in obs.keys()})

            env.render()

        results.append(totals)
        ppo_agent.update()  # PPO learns after each episode

    return results

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
episodes = cfg["ppo"].get("train_episodes", 10)
scores = run_mixed(env, episodes=episodes)
print("Mixed PPO/Gap-Follow scores:", scores)
ppo_agent.save("checkpoints/ppo_latest.pt")