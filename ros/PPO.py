#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS1 PPO actor node aligned with scenarios/gaplock_ppo.yaml.
Place alongside your PPO checkpoint (ppo_gaplock.pt) inside your ROS package.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict

import numpy as np
import rospy
import torch
import torch.nn as nn
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan

from gaplock_utils import (
    ACTION_HIGH,
    ACTION_LOW,
    OBS_DIM,
    build_observation,
    init_agent_state,
    scale_continuous_action,
    update_agent_state,
)


def _infer_hidden_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
    dims = []
    idx = 0
    while True:
        key = f"body.{idx}.weight"
        tensor = state_dict.get(key)
        if tensor is None:
            break
        dims.append(int(tensor.shape[0]))
        idx += 2  # skip ReLU layers
    return tuple(dims)


class PPOActor(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dims: Tuple[int, ...] = (256, 256),
        act_dim: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        prev = obs_dim
        for hid in hidden_dims:
            layers.extend([nn.Linear(prev, hid), nn.ReLU()])
            prev = hid
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        mu = self.mu_head(features)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return mu, std


def load_ppo_actor(ckpt_path: str, *, device: str) -> PPOActor:
    ckpt = torch.load(ckpt_path, map_location=device)
    state: Optional[Dict[str, torch.Tensor]]
    if isinstance(ckpt, nn.Module):
        state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        actor_state = ckpt.get("actor")
        if isinstance(actor_state, dict):
            state = actor_state
        elif all(isinstance(k, str) for k in ckpt):
            state = ckpt  # flat state dict
        else:
            state = None
    else:
        state = None
    if state is None:
        raise RuntimeError(f"PPO checkpoint format unsupported: keys={list(ckpt.keys())}")

    hidden_dims = _infer_hidden_dims(state)
    if not hidden_dims:
        hidden_dims = (256, 256)
    model = PPOActor(hidden_dims=hidden_dims).to(device)
    model.load_state_dict(state, strict=False)
    return model


class RLActorNode:
    def __init__(self, default_ckpt: str = "ppo_gaplock.pt") -> None:
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.primary_topic = rospy.get_param("~primary_topic", "/vicon/Limo_04/Limo_04")
        self.secondary_topic = rospy.get_param("~secondary_topic", "/vicon/Limo_02/Limo_02")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_ckpt = rospy.get_param("~default_ckpt", default_ckpt)
        if not os.path.isabs(default_ckpt):
            default_ckpt = os.path.join(script_dir, default_ckpt)
        self.ckpt_path = rospy.get_param("~ckpt", default_ckpt)

        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.use_safety = rospy.get_param("~use_safety", True)
        self.hard_border = float(rospy.get_param("~hard_border", 1.0))
        self.prevent_reverse = rospy.get_param("~prevent_reverse", True)
        self.min_throttle = float(rospy.get_param("~prevent_reverse_min_speed", 0.01))
        self.max_pose_age = float(rospy.get_param("~max_pose_age", 0.25))

        self.last_scan: Optional[np.ndarray] = None
        self.primary_state = init_agent_state()
        self.secondary_state = init_agent_state()

        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)
        self.sub_primary = rospy.Subscriber(self.primary_topic, TransformStamped, self.on_primary, queue_size=1)
        self.sub_secondary = rospy.Subscriber(self.secondary_topic, TransformStamped, self.on_secondary, queue_size=1)
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)

        if not os.path.isfile(self.ckpt_path):
            rospy.logfatal("PPO checkpoint not found: %s", self.ckpt_path)
            raise FileNotFoundError(self.ckpt_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = load_ppo_actor(self.ckpt_path, device=self.device).eval()

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.on_tick)
        rospy.loginfo("PPO ROS actor ready | ckpt=%s | device=%s", self.ckpt_path, self.device)

    def on_scan(self, msg: LaserScan) -> None:
        self.last_scan = np.asarray(msg.ranges, dtype=np.float32)

    def on_primary(self, msg: TransformStamped) -> None:
        update_agent_state(self.primary_state, msg)

    def on_secondary(self, msg: TransformStamped) -> None:
        update_agent_state(self.secondary_state, msg)

    def on_tick(self, _evt) -> None:
        if self.last_scan is None or self.primary_state["pose"] is None or self.secondary_state["pose"] is None:
            self.pub_cmd.publish(Twist())
            return

        now = rospy.Time.now().to_sec()
        for label, state in (("primary", self.primary_state), ("secondary", self.secondary_state)):
            stamp = state["stamp"]
            if stamp is None or (now - stamp) > self.max_pose_age:
                rospy.logwarn_throttle(2.0, "%s pose stale -> zero command", label)
                self.pub_cmd.publish(Twist())
                return

        if self.use_safety:
            sec_y = float(self.secondary_state["pose"][1])
            if abs(sec_y) > self.hard_border:
                rospy.logwarn_throttle(2.0, "Target |y|=%.2f exceeds %.2f -> stopping", sec_y, self.hard_border)
                self.pub_cmd.publish(Twist())
                return

        obs_vec = build_observation(self.last_scan, self.primary_state, self.secondary_state)
        obs_t = torch.from_numpy(obs_vec.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            mu, _ = self.actor(obs_t)
            squashed = torch.tanh(mu)
            raw_action = squashed.cpu().numpy()[0]

        action = scale_continuous_action(raw_action)
        steer = float(np.clip(action[0], ACTION_LOW[0], ACTION_HIGH[0]))
        throttle = float(np.clip(action[1], ACTION_LOW[1], ACTION_HIGH[1]))
        if self.prevent_reverse:
            throttle = max(throttle, self.min_throttle)

        cmd = Twist()
        cmd.angular.z = steer
        cmd.linear.x = throttle
        self.pub_cmd.publish(cmd)


def main():
    rospy.init_node("rl_actor_ppo_node", anonymous=False)
    RLActorNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
