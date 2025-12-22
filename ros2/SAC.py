#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 SAC actor node aligned with scenarios/gaplock_sac.yaml.
Place alongside sac_gaplock.pt inside your ROS package scripts directory.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import torch
import torch.nn as nn
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan

try:
    from car_crashers.gaplock_utils import (  # type: ignore
        ACTION_HIGH,
        ACTION_LOW,
        OBS_DIM,
        build_observation,
        init_agent_state,
        scale_continuous_action,
        update_agent_state,
    )
except ImportError:  # pragma: no cover
    from gaplock_utils import (  # type: ignore
        ACTION_HIGH,
        ACTION_LOW,
        OBS_DIM,
        build_observation,
        init_agent_state,
        scale_continuous_action,
        update_agent_state,
    )


def _stamp_to_sec(stamp: Any) -> Optional[float]:
    if stamp is None:
        return None
    if isinstance(stamp, (float, int)):
        return float(stamp)
    nanoseconds = getattr(stamp, "nanoseconds", None)
    if nanoseconds is not None:
        return float(nanoseconds) * 1e-9
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is not None and nanosec is not None:
        return float(sec) + float(nanosec) * 1e-9
    return None


def _infer_actor_arch(state_dict: Dict[str, torch.Tensor]) -> Tuple[Tuple[int, ...], int]:
    dims = []  # type: List[int]
    idx = 0
    while True:
        key = f"net.{idx}.weight"
        tensor = state_dict.get(key)
        if tensor is None:
            break
        dims.append(int(tensor.shape[0]))
        idx += 2  # skip ReLU layers
    if not dims:
        raise RuntimeError("Unable to infer SAC actor architecture from checkpoint")
    final_out = dims[-1]
    if final_out % 2 != 0:
        raise RuntimeError(
            f"SAC actor final layer size {final_out} is not even; expected mu/log_std pairs"
        )
    act_dim = final_out // 2
    hidden_dims = tuple(dims[:-1])
    return hidden_dims, act_dim


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        act_dim: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        prev = obs_dim
        for hid in hidden_dims:
            layers.extend([nn.Linear(prev, hid), nn.ReLU()])
            prev = hid
        layers.append(nn.Linear(prev, act_dim * 2))
        self.net = nn.Sequential(*layers)
        self.act_dim = int(act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(obs)
        mu, log_std = torch.split(raw, self.act_dim, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mu, log_std


def load_sac_actor(ckpt_path: str, *, device: str) -> GaussianPolicy:
    ckpt = torch.load(ckpt_path, map_location=device)
    actor_state = ckpt.get("actor") if isinstance(ckpt, dict) else None
    if not isinstance(actor_state, dict):
        raise RuntimeError(f"SAC checkpoint format unsupported: keys={list(ckpt.keys())}")
    hidden_dims, act_dim = _infer_actor_arch(actor_state)
    if not hidden_dims:
        hidden_dims = (512, 256, 128)
    policy = GaussianPolicy(hidden_dims=hidden_dims, act_dim=act_dim).to(device)
    policy.load_state_dict(actor_state, strict=True)
    return policy


class RLActorNode(Node):
    def __init__(self, default_ckpt: str = "sac_gaplock.pt") -> None:
        super().__init__("rl_actor_sac_node")
        self.logger = self.get_logger()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.isabs(default_ckpt):
            default_ckpt = os.path.join(script_dir, default_ckpt)

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("primary_topic", "/vicon/Limo_04/Limo_04")
        self.declare_parameter("secondary_topic", "/vicon/Limo_02/Limo_02")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("ckpt", default_ckpt)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("use_safety", True)
        self.declare_parameter("hard_border", 1.0)
        self.declare_parameter("prevent_reverse", True)
        self.declare_parameter("prevent_reverse_min_speed", 0.01)
        self.declare_parameter("max_pose_age", 0.25)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.primary_topic = str(self.get_parameter("primary_topic").value)
        self.secondary_topic = str(self.get_parameter("secondary_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.ckpt_path = str(self.get_parameter("ckpt").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.use_safety = bool(self.get_parameter("use_safety").value)
        self.hard_border = float(self.get_parameter("hard_border").value)
        self.prevent_reverse = bool(self.get_parameter("prevent_reverse").value)
        self.min_throttle = float(self.get_parameter("prevent_reverse_min_speed").value)
        self.max_pose_age = float(self.get_parameter("max_pose_age").value)

        self.last_scan: Optional[np.ndarray] = None
        self.primary_state = init_agent_state()
        self.secondary_state = init_agent_state()

        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, scan_qos)
        self.sub_primary = self.create_subscription(TransformStamped, self.primary_topic, self.on_primary, 20)
        self.sub_secondary = self.create_subscription(TransformStamped, self.secondary_topic, self.on_secondary, 20)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 20)

        if not os.path.isfile(self.ckpt_path):
            self.logger.fatal("SAC checkpoint not found: %s" % self.ckpt_path)
            raise FileNotFoundError(self.ckpt_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = load_sac_actor(self.ckpt_path, device=self.device).eval()

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_tick)
        self.logger.info("SAC ROS actor ready | ckpt=%s | device=%s" % (self.ckpt_path, self.device))

    def on_scan(self, msg: LaserScan) -> None:
        self.last_scan = np.asarray(msg.ranges, dtype=np.float32)

    def on_primary(self, msg: TransformStamped) -> None:
        update_agent_state(self.primary_state, msg)

    def on_secondary(self, msg: TransformStamped) -> None:
        update_agent_state(self.secondary_state, msg)

    def on_tick(self) -> None:
        if self.last_scan is None or self.primary_state["pose"] is None or self.secondary_state["pose"] is None:
            self.pub_cmd.publish(Twist())
            return

        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        for label, state in (("primary", self.primary_state), ("secondary", self.secondary_state)):
            stamp = state["stamp"]
            stamp_sec = _stamp_to_sec(stamp)
            if stamp_sec is None or (now_sec - stamp_sec) > self.max_pose_age:
                self.logger.warning("%s pose stale -> zero command" % label)
                self.pub_cmd.publish(Twist())
                return

        if self.use_safety:
            sec_y = float(self.secondary_state["pose"][1])
            if abs(sec_y) > self.hard_border:
                self.logger.warning("Target |y|=%.2f exceeds %.2f -> stopping" % (sec_y, self.hard_border))
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
    rclpy.init()
    node = None
    try:
        node = RLActorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
