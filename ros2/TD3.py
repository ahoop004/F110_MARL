#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 TD3 actor node aligned with scenarios/gaplock_td3.yaml.
Drop this script plus td3_gaplock.pt into your ROS package scripts directory.
"""

#from __future__ import annotations

import os
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy

from car_crashers.gaplock_utils import (
    ACTION_HIGH,
    ACTION_LOW,
    OBS_DIM,
    build_observation,
    init_agent_state,
    scale_continuous_action,
    update_agent_state,
)


class TD3Actor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden_dims=(512, 256, 128), act_dim: int = 2) -> None:
        super().__init__()
        layers = []
        prev = obs_dim
        for hid in hidden_dims:
            layers.extend([nn.Linear(prev, hid), nn.ReLU()])
            prev = hid
        layers.append(nn.Linear(prev, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def load_td3_actor(ckpt_path: str, model: nn.Module, device: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, nn.Module):
        model.load_state_dict(ckpt.state_dict(), strict=False)
        return model
    if isinstance(ckpt, dict):
        if "actor" in ckpt and isinstance(ckpt["actor"], dict):
            model.load_state_dict(ckpt["actor"], strict=False)
            return model
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            model.load_state_dict(ckpt["state_dict"], strict=False)
            return model
        if all(isinstance(k, str) for k in ckpt):
            model.load_state_dict(ckpt, strict=False)
            return model
    raise RuntimeError(f"TD3 checkpoint format unsupported: keys={list(ckpt.keys())}")


class RLActorNode(Node):
    def __init__(self, default_ckpt: str = "td3_gaplock.pt") -> None:
        super().__init__('rl_actor_td3_node')
        self.scan_topic = "/scan"
        self.primary_topic = "/vicon/Limo_04/Limo_04"
        self.secondary_topic = "/vicon/Limo_02/Limo_02"
        self.cmd_topic = "/cmd_vel"

        self.logger = self.get_logger()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.isabs(default_ckpt):
            default_ckpt = os.path.join(script_dir, default_ckpt)
        self.ckpt_path = default_ckpt

        #TODO: ROS puts everyhing in an /install folder, where the TD3file isn't
        self.ckpt_path = "/home/agilex/agilex_ros2_ws/src/car_crashers/car_crashers/td3_gaplock_young-sweep-2.pt"

        self.rate_hz = 20.0
        self.use_safety = True
        self.hard_border = 1.0
        self.prevent_reverse = True
        self.min_throttle = 0.01
        self.max_pose_age = 0.25

        self.last_scan: Optional[np.ndarray] = None
        self.primary_state = init_agent_state()
        self.secondary_state = init_agent_state()


        qos = QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos)

        self.sub_primary = self.create_subscription(TransformStamped, self.primary_topic, self.on_primary, 20)
        self.sub_secondary = self.create_subscription(TransformStamped, self.secondary_topic, self.on_secondary, 20)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 20)

        if not os.path.isfile(self.ckpt_path):
            self.logger.fatal('Checkpoint not found: %s' % self.ckpt_path)
            raise FileNotFoundError(self.ckpt_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = load_td3_actor(self.ckpt_path, TD3Actor().to(self.device), device=self.device).eval()

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_tick)
        self.logger.info("TD3 ROS actor ready | ckpt=%s | device=%s" % (self.ckpt_path, self.device))

    def on_scan(self, msg: LaserScan) -> None:
        self.last_scan = np.asarray(msg.ranges, dtype=np.float32)

    def on_primary(self, msg: TransformStamped) -> None:
        update_agent_state(self.primary_state, msg)

    def on_secondary(self, msg: TransformStamped) -> None:
        update_agent_state(self.secondary_state, msg)

    def on_tick(self) -> None:
        if self.last_scan is None or self.primary_state["pose"] is None or self.secondary_state["pose"] is None:
            self.pub_cmd.publish(Twist())
            self.logger.info("TD3 Waiting for scan/primary/secondary info")
            return

        now = self.get_clock().now()
        for label, state in (("primary", self.primary_state), ("secondary", self.secondary_state)):
            stamp = state["stamp"]
            if stamp is None: # or (now - stamp) > self.max_pose_age:
                self.logger.warn("%s pose stale -> zero command" % label)
                self.pub_cmd.publish(Twist())
                return

        if self.use_safety:
            sec_y = float(self.secondary_state["pose"][1])
            if abs(sec_y) > self.hard_border:
                self.logger.warn("Target |y|=%.2f exceeds %.2f -> stopping" % (sec_y, self.hard_border))
                self.pub_cmd.publish(Twist())
                return

        obs_vec = build_observation(self.last_scan, self.primary_state, self.secondary_state)
        obs_t = torch.from_numpy(obs_vec.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            raw = self.actor(obs_t).cpu().numpy()[0]

        action = scale_continuous_action(raw)
        steer = float(np.clip(action[0], ACTION_LOW[0], ACTION_HIGH[0]))
        throttle = float(np.clip(action[1], ACTION_LOW[1], ACTION_HIGH[1]))
        if self.prevent_reverse:
            throttle = max(throttle, self.min_throttle)

        cmd = Twist()
        cmd.angular.z = steer
        cmd.linear.x = throttle
        self.pub_cmd.publish(cmd)


def main():
    #rospy.init_node("rl_actor_td3_node", anonymous=False)
    rclpy.init()
    node = RLActorNode()
    rclpy.spin(node)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
