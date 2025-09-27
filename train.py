from pathlib import Path

import yaml
import numpy as np
from PIL import Image
from f110x.envs import F110ParallelEnv
from policies.gap_follow import FollowTheGapPolicy
from policies.ppo.ppo import PPOAgent
from f110x.wrappers.observation import ObsWrapper
from f110x.wrappers.reward import RewardWrapper

from collections import deque

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

env_cfg = cfg["env"]
render_interval = env_cfg.get("render_interval", 0) 
update_after = env_cfg.get('update',1)

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

base_start_poses = None
start_pose_jitter_cfg = env_cfg.get("start_pose_jitter", {})
jitter_pos = float(start_pose_jitter_cfg.get("pos", 0.0))
jitter_heading = float(start_pose_jitter_cfg.get("heading", 0.0))
min_sep = float(start_pose_jitter_cfg.get("min_distance", 0.0))
jitter_max_attempts = max(1, int(start_pose_jitter_cfg.get("max_attempts", 5)))
if env_cfg.get("start_poses"):
    base_start_poses = np.asarray(env_cfg["start_poses"], dtype=np.float32)
    if base_start_poses.ndim == 1:
        base_start_poses = np.expand_dims(base_start_poses, axis=0)


def sample_start_poses():
    if base_start_poses is None or base_start_poses.size == 0:
        return None

    for _ in range(jitter_max_attempts):
        jittered = base_start_poses.copy()
        if jitter_pos > 0.0:
            noise = np.random.uniform(-jitter_pos, jitter_pos, size=jittered[:, :2].shape).astype(np.float32)
            jittered[:, :2] = base_start_poses[:, :2] + noise
        if jitter_heading > 0.0:
            theta_noise = np.random.uniform(-jitter_heading, jitter_heading, size=(jittered.shape[0],)).astype(np.float32)
            jittered[:, 2] = base_start_poses[:, 2] + theta_noise
        if jitter_heading > 0.0:
            jittered[:, 2] = np.mod(jittered[:, 2] + np.pi, 2 * np.pi) - np.pi

        if min_sep <= 0.0 or jittered.shape[0] <= 1:
            return jittered

        diffs = jittered[:, None, :2] - jittered[None, :, :2]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        if np.all(dists >= min_sep):
            return jittered

    return base_start_poses.copy()


def reset_env_with_jitter(environment):
    for _ in range(jitter_max_attempts):
        jittered = sample_start_poses()
        options = {"poses": jittered} if jittered is not None else None
        obs, infos = environment.reset(options=options)
        collisions = [obs.get(aid, {}).get("collision", False) for aid in obs.keys()]
        if not any(collisions):
            return obs, infos
    return environment.reset()

# -------------------------------------------------------------------
# Initialize wrappers & policies
# -------------------------------------------------------------------
obs_wrapper = ObsWrapper(max_scan=30.0, normalize=True)
reward_cfg = cfg.get("reward", {})

raw_curriculum = cfg.get("reward_curriculum", [])
curriculum_schedule = []
if isinstance(raw_curriculum, list):
    for stage in raw_curriculum:
        if not isinstance(stage, dict):
            continue
        mode = stage.get("mode")
        if mode is None:
            continue
        upper = stage.get("until")
        if upper is None:
            upper = stage.get("episodes")
        if upper is not None:
            try:
                upper = int(upper)
            except (TypeError, ValueError):
                upper = None
        curriculum_schedule.append((upper, mode))

curriculum_schedule.sort(key=lambda item: float("inf") if item[0] is None else item[0])

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
action_space = env.action_space(PPO_AGENT)
act_dim = action_space.shape[0]

ppo_cfg = cfg["ppo"].copy()
ppo_cfg["obs_dim"] = obs_dim
ppo_cfg["act_dim"] = act_dim
ppo_cfg["action_low"] = action_space.low.astype(np.float32).tolist()
ppo_cfg["action_high"] = action_space.high.astype(np.float32).tolist()

checkpoint_dir = Path(ppo_cfg.get("save_dir", "checkpoints")).expanduser()
checkpoint_name = ppo_cfg.get("checkpoint_name", "ppo_best.pt")
default_checkpoint = checkpoint_dir / checkpoint_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
load_override = ppo_cfg.get("load_path")
checkpoint_path = Path(load_override).expanduser() if load_override else default_checkpoint

ppo_agent = PPOAgent(ppo_cfg)

if checkpoint_path.exists():
    try:
        ppo_agent.load(str(checkpoint_path))
        print(f"[INFO] Loaded PPO checkpoint from {checkpoint_path}")
    except Exception as exc:  # pragma: no cover - defensive load guard
        print(f"[WARN] Failed to load PPO checkpoint at {checkpoint_path}: {exc}")
best_path = default_checkpoint
print(f"[INFO] PPO agent: {PPO_AGENT}, Gap-follow agent: {GAP_AGENT}")
print(f"[INFO] Obs dim: {obs_dim}, Act dim: {act_dim}")

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def resolve_reward_mode(episode_idx: int) -> str:
    if curriculum_schedule:
        for threshold, mode in curriculum_schedule:
            if threshold is None or episode_idx < threshold:
                return mode
        return curriculum_schedule[-1][1]

    if episode_idx < 1000:
        return "basic"
    if episode_idx < 2000:
        return "pursuit"
    return "adversarial"


def get_curriculum_reward(episode_idx: int) -> RewardWrapper:
    # TODO: Move curriculum selection into shared utility so eval/inference can reuse schedule.
    wrapper_cfg = reward_cfg.copy()
    wrapper_cfg["mode"] = resolve_reward_mode(episode_idx)
    wrapper = RewardWrapper(**wrapper_cfg)
    wrapper.reset()
    return wrapper

def run_mixed(env, episodes=5):
    results = []
    window = max(1, int(cfg["ppo"].get("rolling_avg_window", 10)))
    recent_returns = deque(maxlen=window)
    best_return = float("-inf")

    for ep in range(episodes):
        obs, infos = reset_env_with_jitter(env)
        done = {aid: False for aid in env.possible_agents}
        totals = {aid: 0.0 for aid in env.possible_agents}
        reward_wrapper = get_curriculum_reward(ep)
        episode_steps = 0
        collision_history = []

        terms = {}
        truncs = {}

        while True:
            actions = {}
            ppo_obs = None
            ppo_action = None

            if PPO_AGENT in obs and not done.get(PPO_AGENT, False):
                ppo_obs = obs_wrapper(obs, PPO_AGENT, GAP_AGENT)
                ppo_action = ppo_agent.act(ppo_obs, PPO_AGENT)
                actions[PPO_AGENT] = ppo_action

            if GAP_AGENT in obs and not done.get(GAP_AGENT, False):
                actions[GAP_AGENT] = gap_policy.get_action(
                    env.action_space(GAP_AGENT), obs[GAP_AGENT]
                )

            if not actions:
                break

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            episode_steps += 1

            shaped_rewards = {}
            for aid in totals.keys():
                agent_obs = next_obs.get(aid)
                shaped = rewards.get(aid, 0.0)
                if agent_obs is not None:
                    shaped = reward_wrapper(
                        next_obs,
                        aid,
                        rewards.get(aid, 0.0),
                        done=terms.get(aid, False) or truncs.get(aid, False),
                        info=infos.get(aid, {}),
                        all_obs=next_obs,
                    )
                totals[aid] = totals.get(aid, 0.0) + shaped
                shaped_rewards[aid] = shaped

            collision_agents = [
                aid for aid in totals.keys()
                if bool(next_obs.get(aid, {}).get("collision", 0.0))
            ]
            collision_happened = bool(collision_agents)
            if collision_happened:
                collision_history.extend(collision_agents)

            if ppo_action is not None:
                terminated = terms.get(PPO_AGENT, False) or collision_happened
                truncated = truncs.get(PPO_AGENT, False)

                next_obs_for_wrap = next_obs
                if PPO_AGENT not in next_obs_for_wrap:
                    terminal_obs = infos.get(PPO_AGENT, {}).get("terminal_observation")
                    if terminal_obs is not None:
                        next_obs_for_wrap = dict(next_obs_for_wrap)
                        next_obs_for_wrap[PPO_AGENT] = terminal_obs
                    else:
                        next_obs_for_wrap = obs
                try:
                    next_wrapped = obs_wrapper(next_obs_for_wrap, PPO_AGENT, GAP_AGENT)
                except Exception:
                    next_wrapped = ppo_obs

                if truncated and not terminated:
                    ppo_agent.record_final_value(next_wrapped)

                ppo_agent.store(
                    next_wrapped,
                    ppo_action,
                    shaped_rewards.get(PPO_AGENT, rewards.get(PPO_AGENT, 0.0)),
                    terminated,
                )

            for aid in done.keys():
                done_flag = terms.get(aid, False) or truncs.get(aid, False)
                if aid in collision_agents:
                    done_flag = True
                done[aid] = done_flag

            obs = next_obs

            if collision_happened or all(done.values()):
                break

            if render_interval and (ep % render_interval == 0):
                env.render()

        results.append(totals)

        if ppo_agent.rew_buf and ((ep + 1) % update_after == 0):
            ppo_agent.update()

        ppo_return = totals.get(PPO_AGENT, 0.0)
        recent_returns.append(ppo_return)
        if len(recent_returns) == recent_returns.maxlen:
            avg_return = sum(recent_returns) / len(recent_returns)
            if avg_return > best_return:
                best_return = avg_return
                ppo_agent.save(str(best_path))
                print(
                    f"[INFO] New best model saved at episode {ep + 1} "
                    f"(avg_return={avg_return:.2f})"
                )

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

        # TODO: Route episode telemetry through a logging helper configurable via YAML.
        print(
            f"[EP {ep+1:03d}/{episodes}] mode={reward_wrapper.mode} "
            f"steps={episode_steps} cause={cause_str} "
            f"return_ppo={totals.get(PPO_AGENT, 0.0):.2f} "
            f"return_gap={totals.get(GAP_AGENT, 0.0):.2f}"
        )

    if ppo_agent.rew_buf:
        ppo_agent.update()

    return results

if __name__ == "__main__":
    # -------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------
    episodes = cfg["ppo"].get("train_episodes", 10)
    scores = run_mixed(env, episodes=episodes)
    print("Mixed PPO/Gap-Follow scores:", scores)
    ppo_agent.save(str(default_checkpoint))
