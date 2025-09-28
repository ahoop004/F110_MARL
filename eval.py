from pathlib import Path
import yaml
import numpy as np
from PIL import Image
from collections import deque

from f110x.envs import F110ParallelEnv
from policies.gap_follow import FollowTheGapPolicy
from policies.ppo.ppo import PPOAgent
from f110x.wrappers.observation import ObsWrapper
from f110x.wrappers.reward import RewardWrapper

# -------------------------------------------------------------------
# Config and environment setup (shared with train.py)
# -------------------------------------------------------------------
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

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

env = F110ParallelEnv(**env_cfg)
render_enabled = env_cfg.get("render_mode", "") == "human"

start_pose_back_gap = float(env_cfg.get("start_pose_back_gap", 0.0))
start_pose_min_spacing = float(env_cfg.get("start_pose_min_spacing", 0.0))
start_pose_options = env_cfg.get("start_pose_options")
if start_pose_options:
    processed = []
    for option in start_pose_options:
        arr = np.asarray(option, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        processed.append(arr)
    start_pose_options = processed


def _adjust_start_poses(poses: np.ndarray) -> np.ndarray:
    if poses.shape[0] < 2:
        return poses

    adjusted = poses.copy()
    leader = adjusted[0]
    heading = np.array([np.cos(leader[2]), np.sin(leader[2])], dtype=np.float32)

    if start_pose_back_gap > 0.0:
        for idx in range(1, adjusted.shape[0]):
            rel = adjusted[idx, :2] - leader[:2]
            proj = float(np.dot(rel, heading))
            if proj < 0 and abs(proj) < start_pose_back_gap:
                delta = start_pose_back_gap + proj
                adjusted[idx, :2] -= heading * delta

    if start_pose_min_spacing > 0.0:
        for idx in range(1, adjusted.shape[0]):
            rel = adjusted[idx, :2] - leader[:2]
            dist = float(np.linalg.norm(rel))
            if dist < start_pose_min_spacing and dist > 1e-6:
                direction = heading if np.dot(rel, heading) >= 0 else -heading
                adjusted[idx, :2] += direction * (start_pose_min_spacing - dist)

    return adjusted


def reset_environment(environment: F110ParallelEnv):
    if not start_pose_options:
        return environment.reset()

    indices = np.random.permutation(len(start_pose_options))
    for idx in indices:
        poses = np.array(start_pose_options[idx], copy=True)
        poses = _adjust_start_poses(poses)
        obs, infos = environment.reset(options={"poses": poses})
        collisions = [obs.get(aid, {}).get("collision", False) for aid in obs.keys()]
        if not any(collisions):
            return obs, infos

    return environment.reset()

# Wrappers and policies
obs_wrapper = ObsWrapper(max_scan=30.0, normalize=True)
reward_cfg = cfg.get("reward", {})
gap_policy = FollowTheGapPolicy()

# Agent IDs
obs, infos = reset_environment(env)
agent_ids = env.agents
ppo_agent_idx = cfg.get("ppo_agent_idx", 0)
gap_agent_idx = 1 - ppo_agent_idx
PPO_AGENT = agent_ids[ppo_agent_idx]
GAP_AGENT = agent_ids[gap_agent_idx]

# PPO setup
obs_dim = len(obs_wrapper(obs, PPO_AGENT, GAP_AGENT))
action_space = env.action_space(PPO_AGENT)
act_dim = action_space.shape[0]
ppo_cfg = cfg["ppo"].copy()
ppo_cfg["obs_dim"] = obs_dim
ppo_cfg["act_dim"] = act_dim
ppo_cfg["action_low"] = action_space.low.astype(np.float32).tolist()
ppo_cfg["action_high"] = action_space.high.astype(np.float32).tolist()

checkpoint_dir = Path(ppo_cfg.get("save_dir", "checkpoints")).expanduser()
checkpoint_name = ppo_cfg.get("checkpoint_name", "ppo_best.pt")
checkpoint_path = checkpoint_dir / checkpoint_name
ppo_agent = PPOAgent(ppo_cfg)

if checkpoint_path.exists():
    ppo_agent.load(str(checkpoint_path))
    print(f"[INFO] Loaded PPO checkpoint from {checkpoint_path}")
else:
    raise FileNotFoundError(f"No PPO checkpoint found at {checkpoint_path}")

print(f"[INFO] PPO agent: {PPO_AGENT}, Gap agent: {GAP_AGENT}")
print(f"[INFO] Obs dim: {obs_dim}, Act dim: {act_dim}")

# -------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------
def evaluate(env, episodes=20):
    results = []
    render_live = render_enabled
    for ep in range(episodes):
        obs, infos = reset_environment(env)
        done = {aid: False for aid in env.possible_agents}
        totals = {aid: 0.0 for aid in env.possible_agents}
        reward_wrapper = RewardWrapper(**reward_cfg)
        reward_wrapper.reset()

        steps = 0
        collision_history = []
        terms, truncs = {}, {}

        while True:
            actions = {}
            if PPO_AGENT in obs and not done.get(PPO_AGENT, False):
                ppo_obs = obs_wrapper(obs, PPO_AGENT, GAP_AGENT)
                actions[PPO_AGENT] = ppo_agent.act_deterministic(ppo_obs, PPO_AGENT)

            if GAP_AGENT in obs and not done.get(GAP_AGENT, False):
                actions[GAP_AGENT] = gap_policy.get_action(
                    env.action_space(GAP_AGENT), obs[GAP_AGENT]
                )

            if not actions:
                break

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            steps += 1

            for aid in totals:
                shaped = rewards.get(aid, 0.0)
                if next_obs.get(aid) is not None:
                    shaped = reward_wrapper(
                        next_obs,
                        aid,
                        rewards.get(aid, 0.0),
                        done=terms.get(aid, False) or truncs.get(aid, False),
                        info=infos.get(aid, {}),
                        all_obs=next_obs,
                    )
                totals[aid] += shaped

            collision_agents = [
                aid for aid in totals if bool(next_obs.get(aid, {}).get("collision", 0))
            ]
            if collision_agents:
                collision_history.extend(collision_agents)

            obs = next_obs
            done = {
                aid: terms.get(aid, False) or truncs.get(aid, False) or (aid in collision_agents)
                for aid in done
            }
            if render_live:
                try:
                    env.render()
                except Exception as exc:
                    print(f"[WARN] Disabling render during eval: {exc}")
                    render_live = False
            if all(done.values()):
                break

        end_cause = []
        if collision_history:
            end_cause.append("collision")
        for aid in done:
            if terms.get(aid, False):
                end_cause.append(f"term:{aid}")
            if truncs.get(aid, False):
                end_cause.append(f"trunc:{aid}")
        if not end_cause:
            end_cause.append("unknown")

        cause_str = ",".join(end_cause)
        print(
            f"[EVAL {ep+1:03d}/{episodes}] steps={steps} cause={cause_str} "
            f"return_ppo={totals[PPO_AGENT]:.2f} return_gap={totals[GAP_AGENT]:.2f}"
        )
        results.append(dict(episode=ep+1, steps=steps, cause=cause_str,
                            return_ppo=totals[PPO_AGENT], return_gap=totals[GAP_AGENT]))
    return results

if __name__ == "__main__":
    episodes = cfg["ppo"].get("eval_episodes", 20)
    eval_results = evaluate(env, episodes=episodes)
    print("Evaluation finished. Results:")
    for r in eval_results:
        print(r)
