#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 TD3 actor node aligned with scenarios/gaplock_td3.yaml.
Drop this script plus td3_gaplock.pt into your ROS package scripts directory.
"""

#from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Optional

import numpy as np
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy

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


def _apply_numpy_pickle_compat() -> None:
    """Allow loading checkpoints pickled with numpy>=2 on numpy<2."""

    if "numpy._core" in sys.modules:
        return

    try:
        core_mod = importlib.import_module("numpy.core")
    except Exception:
        return

    sys.modules.setdefault("numpy._core", core_mod)
    for name in ("multiarray", "_multiarray_umath", "numeric", "umath", "overrides"):
        alias_name = "numpy._core.%s" % name
        if alias_name in sys.modules:
            continue
        try:
            sub_mod = importlib.import_module("numpy.core.%s" % name)
        except Exception:
            continue
        sys.modules.setdefault(alias_name, sub_mod)


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
    _apply_numpy_pickle_compat()
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
        super().__init__("rl_actor_td3_node")
        self.logger = self.get_logger()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.isabs(default_ckpt):
            default_ckpt = os.path.join(script_dir, default_ckpt)
        robot_default_ckpt = "/home/agilex/agilex_ros2_ws/src/car_crashers/car_crashers/td3_gaplock_young-sweep-2.pt"
        if os.path.isfile(robot_default_ckpt):
            default_ckpt = robot_default_ckpt

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("primary_topic", "/vicon/Limo_04/Limo_04")
        self.declare_parameter("secondary_topic", "/vicon/Limo_02/Limo_02")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("ckpt", default_ckpt)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("use_safety", True)
        self.declare_parameter("hard_border", 1.0)
        self.declare_parameter("prevent_reverse", True)
        self.declare_parameter("prevent_reverse_min_speed", 0.0)
        self.declare_parameter("max_pose_age", 0.25)
        # LIMO cmd_vel mapping knobs (kept in sync with ros2/RainbowDQN.py)
        self.declare_parameter("steer_sign", -1.0)
        self.declare_parameter("lin_scale", 0.4)
        self.declare_parameter("ang_scale", 1.0)
        self.declare_parameter("max_steer_cmd", 1.0)
        self.declare_parameter("max_speed_cmd", 0.35)
        self.declare_parameter("max_reverse_speed_cmd", 0.0)

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
        self.steer_sign = float(self.get_parameter("steer_sign").value)
        self.lin_scale = float(self.get_parameter("lin_scale").value)
        self.ang_scale = float(self.get_parameter("ang_scale").value)
        self.max_steer_cmd = float(self.get_parameter("max_steer_cmd").value)
        self.max_speed_cmd = float(self.get_parameter("max_speed_cmd").value)
        self.max_reverse_speed_cmd = float(self.get_parameter("max_reverse_speed_cmd").value)

        self.last_scan: Optional[np.ndarray] = None
        self.primary_state = init_agent_state()
        self.secondary_state = init_agent_state()
        self._waiting_logged = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
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
            if not self._waiting_logged:
                self.logger.info("TD3 waiting for scan/primary/secondary info")
                self._waiting_logged = True
            return
        self._waiting_logged = False

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
            raw = self.actor(obs_t).cpu().numpy()[0]

        action = scale_continuous_action(raw)
        raw_steer = float(np.clip(action[0], ACTION_LOW[0], ACTION_HIGH[0]))
        raw_speed = float(np.clip(action[1], ACTION_LOW[1], ACTION_HIGH[1]))
        if self.prevent_reverse:
            raw_speed = max(raw_speed, self.min_throttle)

        steer_cmd = self.steer_sign * raw_steer * self.ang_scale
        speed_cmd = raw_speed * self.lin_scale

        steer_cmd = float(np.clip(steer_cmd, -self.max_steer_cmd, self.max_steer_cmd))
        if self.prevent_reverse:
            speed_cmd = float(np.clip(speed_cmd, 0.0, self.max_speed_cmd))
        else:
            speed_cmd = float(np.clip(speed_cmd, -self.max_reverse_speed_cmd, self.max_speed_cmd))

        cmd = Twist()
        cmd.angular.z = steer_cmd
        cmd.linear.x = speed_cmd
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
