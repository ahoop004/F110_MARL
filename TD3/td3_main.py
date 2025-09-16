

from __future__ import annotations
import os
import argparse
from typing import Optional, Tuple
import random
from collections import deque
import math

import numpy as np
import torch
import yaml
import gymnasium as gym

from obs import ObservationWrapper
from act import ActionWrapper
from agents import TD3Agent, TD3Config
from replay_buffer import PrioritizedReplayBuffer
from rewards import RewardWrapper
from map_utils import get_map_bounds
from gap_follow import gap_follow_action

from ..src.f110x.envs.f110ParallelEnv import F110Env

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():
    cnf_path = '/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/TD3/config.yaml'
    
    with open(cnf_path, "r") as f:
        cfg = yaml.safe_load(f)


    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)
    
    model_path = cfg['env'].get('model')
    
    
    action_low = np.array(cfg["env"]["action_low"], dtype=np.float32)
    action_high = np.array(cfg["env"]["action_high"], dtype=np.float32)
    act_wrap = ActionWrapper( float(action_low[0]),
                             float(action_high[0]),
                             float(action_low[1]),
                             float(action_high[1]),
                             clip=True,
                             allow_reverse=False
                             )
    map_bounds = get_map_bounds(cfg['env'].get('map_path')+'.yaml')
    lidar_max = cfg["obs"]["lidar_max"]
    obs_w = ObservationWrapper(lidar_max,map_bounds,action_high[1],
    lidar_reduce_factor=20,)
    
    
    reward_w = RewardWrapper()

    max_steps = int(cfg["env"].get("max_steps", 5000))

    env = F110Env(
                cfg["env"]["id"],
                max_steps=max_steps,
                render_mode=cfg["env"].get("render_mode", None),
                map_dir=cfg["env"]["map_dir"],
                map=cfg["env"]["map"],
                map_ext=cfg["env"]["map_ext"],
                num_agents=int(cfg["env"]["num_agents"]),
                render_fps=30
            )
    
    
    
    
    # start_poses = cfg["env"].get("start_poses", None)
    # if start_poses is not None:
    #     start_poses = np.array(start_poses, dtype=np.float32)
    all_start_poses = cfg["env"].get("start_poses", None)
    if all_start_poses is not None:
        ego_pose, opp_pose = random.choice(all_start_poses)
        start_poses = np.array([ego_pose, opp_pose], dtype=np.float32)
    else:
        start_poses = None


    obs_dict, info = env.reset(seed=seed,options=start_poses)

    obs_vec= obs_w.build(obs_dict)
    obs_dim = obs_vec.size
    act_dim = int(len(action_low))  # expect 2


    td3_cfg = TD3Config(
        actor_hidden=tuple(cfg["td3"].get("actor_hidden", (256,256))),
        critic_hidden=tuple(cfg["td3"].get("critic_hidden", (256,256))),
        gamma=cfg["td3"]["gamma"],
        tau=cfg["td3"]["tau"],
        actor_lr=cfg["td3"]["actor_lr"],
        critic_lr=cfg["td3"]["critic_lr"],
        policy_freq=cfg["td3"]["policy_freq"],
        policy_noise=cfg["action"]["policy_noise"],
        noise_clip=cfg["action"]["noise_clip"],
        expl_noise_std=cfg["action"]["train_action_noise_std"],
        expl_noise_clip=cfg["action"].get("train_action_noise_clip", 0.5),
        per_alpha=cfg["per"]["alpha"],
        per_beta_init=cfg["per"]["beta_init"],
        per_beta_final=cfg["per"]["beta_final"],
        per_eps=cfg["per"]["priority_epsilon"],
    )
    agent = TD3Agent(obs_dim, act_dim, cfg=td3_cfg)
    
    if os.path.isfile(str(model_path + "best.pt")):
        agent.load(str(model_path + "best.pt"))
        print('loaded')

    buffer = PrioritizedReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        capacity=int(cfg["per"]["capacity"]),
        alpha=cfg["per"]["alpha"],
        priority_eps=cfg["per"]["priority_epsilon"],
        seed=seed,
    )


    total_steps = int(cfg["train"]["total_steps"])
    warmup_steps = int(cfg["train"]["warmup_steps"])
    update_after = int(cfg["train"]["update_after"])
    batch_size = int(cfg["train"]["batch_size"])
    updates_per_step = int(cfg["train"]["updates_per_step"])

    global_steps = 0
    episode = 0
    best_eval_return = -1e9
    
    # --- logging state ---
    train_ret_hist = deque(maxlen=100)   # MA-100 of train returns

    term_counts = {"opp_crash": 0, "ego_crash": 0, "timeout": 0}


    def run_episode(eval_mode: bool = False) -> Tuple[float, int, dict]:
        nonlocal global_steps, episode

        total_r = 0.0
        steps = 0
        done = False
        terminated = False
        truncated = False
        
        large_steer_cnt = 0
        idle_cnt = 0
     
        
        
        if all_start_poses is not None:
            ego_pose, opp_pose = random.choice(all_start_poses)
            start_poses = np.array([ego_pose, opp_pose], dtype=np.float32)
        else:
            start_poses = None
        
        obs_dict, _ = env.reset(options=start_poses)
        ego = int(obs_dict["ego_idx"])
        act_wrap.reset()
        reward_w.reset(obs_dict)  

        agent.reset_action_state()
        
        while not done and steps < max_steps:

            obs_vec_local = obs_w.build(obs_dict)
            
            if not eval_mode and global_steps < warmup_steps:
                act_norm = np.random.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)
            else:
                act_norm = agent.select_action(obs_vec_local, eval_mode=eval_mode)
                
            act_norm = np.clip(act_norm, -1, 1)

            ego_action = act_wrap.build(act_norm)
         

            opp = 1 - ego
            opp_scan = np.asarray(obs_dict["scans"][opp], dtype=np.float32)
            opp_action_env = gap_follow_action(opp_scan).astype(np.float32)

            actions_env = np.zeros((2, 2), dtype=np.float32)
            actions_env[ego] = ego_action
            actions_env[opp] = opp_action_env
            actions_env = actions_env.astype(np.float32)

            next_obs_dict, env_rew, terminated, truncated, info = env.step(actions_env)
            
            episode_ended = bool(terminated or truncated)  # for control/printing/etc.
            done_for_td = bool(terminated)
            done = bool(terminated or truncated)

            next_obs_vec = obs_w.build(next_obs_dict)

            r = reward_w.compute(next_obs_dict,act_norm)
            if getattr(reward_w, "opp_crashed_now", False):
                done_for_td = True           # terminal for replay/bootstrapping
                done = True                  # end the control loop
                truncated = False            # not a timeout
                terminated = True            # mark as terminal transition
                # print("Crash", start_poses[1])
            elif truncated and not terminated:  # timed out (no crash)
                r += -1.0

            if not eval_mode:
                buffer.add(obs_vec_local, act_norm, r, next_obs_vec, done_for_td)

                # Learn (after update_after)
                if global_steps >= update_after:
                    for _ in range(updates_per_step):
                        progress = min(1.0, global_steps / max(1, total_steps))
                        _metrics = agent.update(buffer, batch_size, progress=progress)

            # Tally
            total_r += r
            steps += 1
            global_steps += (0 if eval_mode else 1)
            obs_dict = next_obs_dict
            # env.render()


            if episode_ended or steps >= max_steps:
                break
        
        # derive termination cause for logging
        if getattr(reward_w, "opp_crashed_now", False):
            term = "opp_crash"
        elif terminated:
            term = "ego_crash"
        elif truncated:
            term = "timeout"
        else:
            term = "unknown"

        metrics = {
            "term": term,
            "pct_large_steer": (100.0 * large_steer_cnt / max(1, steps)),
            "pct_idle": (100.0 * idle_cnt / max(1, steps)),
        }
        return total_r, steps, metrics



    # --- Training loop
    print("[TD3] Starting training...")
    sr10 = deque(maxlen=10)
    sr100 = deque(maxlen=100)
    while episode < 10000:
      
        episode += 1
        ep_ret, ep_steps, m = run_episode(eval_mode=False)
        train_ret_hist.append(ep_ret)
       
        term_counts[m["term"]] = term_counts.get(m["term"], 0) + 1


        if episode % 1000 == 0:
            eval_rets, eval_steps_list = [], []
            for _ in range(3):
                er, es, _ = run_episode(eval_mode=True)
                eval_rets.append(er); eval_steps_list.append(es)
            eval_ret = float(np.mean(eval_rets))
            eval_steps = int(np.mean(eval_steps_list))
            
            # Save best
            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                best_path = str(model_path + "best.pt")
                agent.save(best_path)
                print(f"[SAVE] New best eval return {best_eval_return:.3f}  ")

            print(f"[EVAL] ret={eval_ret:.3f} steps={eval_steps} best={best_eval_return:.3f}")


        sr10.append(1 if m["term"] == "opp_crash" else 0)
        sr100.append(1 if m["term"] == "opp_crash" else 0)
        print(f"Ep {episode:04d} [TRAIN] | R: {ep_ret:.2f} | steps: {ep_steps} | term: {m['term']} "
            f"| SR@10: {100*sum(sr10)/max(1,len(sr10)):.0f}% SR@100: {100*sum(sr100)/max(1,len(sr100)):.0f}%")
        


    env.close()


if __name__ == "__main__":
    main()
