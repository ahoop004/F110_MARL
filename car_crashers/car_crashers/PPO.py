#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS2 PPO actor node for the gaplock task.

Loads ppo_policy.pt (plain PyTorch weights extracted from ppo_model.zip)
and runs direct inference — no stable_baselines3 dependency on the robot.

Architecture (MlpPolicy, net_arch=[256, 256], confirmed from weight shapes):
  obs (65) -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Linear(2) -> clip

Topics (all configurable via ROS2 parameters):
  Subscribed:
    /scan               (sensor_msgs/LaserScan)
    /vicon/Limo_04/...  (geometry_msgs/TransformStamped) — attacker (ego)
    /vicon/Limo_02/...  (geometry_msgs/TransformStamped) — target (defender)
  Published:
    /cmd_vel            (geometry_msgs/Twist)
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan

from .gaplock_utils import (
    ACTION_LOW,
    ACTION_HIGH,
    OBS_DIM,
    build_observation,
    init_agent_state,
    update_agent_state,
)


class _PolicyNet(nn.Module):
    """Minimal PPO policy network matching SB3 MlpPolicy net_arch=[256,256]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.net(x))


def _load_policy(ckpt_path: str) -> _PolicyNet:
    state = torch.load(ckpt_path, map_location="cpu")
    obs_dim = state["mlp_extractor.policy_net.0.weight"].shape[1]
    act_dim = state["action_net.weight"].shape[0]
    net = _PolicyNet(obs_dim, act_dim)
    # Map flat state dict keys to our module structure
    mapped = {
        "net.0.weight": state["mlp_extractor.policy_net.0.weight"],
        "net.0.bias":   state["mlp_extractor.policy_net.0.bias"],
        "net.2.weight": state["mlp_extractor.policy_net.2.weight"],
        "net.2.bias":   state["mlp_extractor.policy_net.2.bias"],
        "action_head.weight": state["action_net.weight"],
        "action_head.bias":   state["action_net.bias"],
    }
    net.load_state_dict(mapped)
    return net.eval()


class RLActorNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_actor_ppo_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("primary_topic", "/vicon/Limo_04/Limo_04")
        self.declare_parameter("secondary_topic", "/vicon/Limo_02/Limo_02")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("use_safety", True)
        self.declare_parameter("hard_border", 1.0)
        self.declare_parameter("prevent_reverse", True)
        self.declare_parameter("prevent_reverse_min_speed", 0.01)
        self.declare_parameter("max_pose_age", 0.25)
        self.declare_parameter("throttle_scale", 0.5)  # multiply policy throttle by this
        self.declare_parameter("min_safe_distance", 0.35)  # metres — forward arc stop
        self.declare_parameter("safety_arc_deg", 45.0)     # ± degrees from front

        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_ckpt = os.path.join(script_dir, "ppo_policy.pt")
        self.declare_parameter("ckpt", default_ckpt)

        scan_topic = self.get_parameter("scan_topic").value
        primary_topic = self.get_parameter("primary_topic").value
        secondary_topic = self.get_parameter("secondary_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.use_safety = bool(self.get_parameter("use_safety").value)
        self.hard_border = float(self.get_parameter("hard_border").value)
        self.prevent_reverse = bool(self.get_parameter("prevent_reverse").value)
        self.min_throttle = float(self.get_parameter("prevent_reverse_min_speed").value)
        self.max_pose_age = float(self.get_parameter("max_pose_age").value)
        self.throttle_scale = float(self.get_parameter("throttle_scale").value)
        self.min_safe_dist  = float(self.get_parameter("min_safe_distance").value)
        self.safety_arc_rad = np.deg2rad(float(self.get_parameter("safety_arc_deg").value))
        self.ckpt_path: str = self.get_parameter("ckpt").value

        # ── State ─────────────────────────────────────────────────────────────
        self.last_scan: Optional[np.ndarray] = None
        self._last_scan_msg = None
        self.primary_state = init_agent_state()    # attacker (ego)
        self.secondary_state = init_agent_state()  # target (defender)
        self._tick_count = 0

        # ── Subscriptions / Publisher ──────────────────────────────────────────
        self.create_subscription(LaserScan, scan_topic, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(TransformStamped, primary_topic, self.on_primary, 1)
        self.create_subscription(TransformStamped, secondary_topic, self.on_secondary, 1)
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)

        # ── Load policy weights (pure PyTorch, no SB3) ────────────────────────
        if not os.path.isfile(self.ckpt_path):
            self.get_logger().fatal("Policy weights not found: %s" % self.ckpt_path)
            raise FileNotFoundError(self.ckpt_path)

        self.policy = _load_policy(self.ckpt_path)
        self.act_low = torch.tensor(ACTION_LOW)
        self.act_high = torch.tensor(ACTION_HIGH)
        self.get_logger().info(
            "PPO actor ready | obs_dim=%d | ckpt=%s" % (OBS_DIM, self.ckpt_path)
        )

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.rate_hz, self.on_tick)

    # ── Safety ────────────────────────────────────────────────────────────────

    def _forward_clear(self):
        """Return True if the forward arc has no obstacle closer than min_safe_dist."""
        if self.last_scan is None or len(self.last_scan) == 0:
            return True
        # Reconstruct angle array from the last stored LaserScan metadata
        msg = self._last_scan_msg
        if msg is None:
            return True
        n = len(msg.ranges)
        idx_lo = int((-self.safety_arc_rad - msg.angle_min) / msg.angle_increment)
        idx_hi = int(( self.safety_arc_rad - msg.angle_min) / msg.angle_increment)
        idx_lo = max(0, min(idx_lo, n - 1))
        idx_hi = max(0, min(idx_hi, n - 1))
        if idx_lo > idx_hi:
            idx_lo, idx_hi = idx_hi, idx_lo
        forward = np.array(msg.ranges[idx_lo: idx_hi + 1], dtype=np.float32)
        valid = forward[np.isfinite(forward) & (forward > 0.0)]
        if len(valid) == 0:
            return True
        return float(np.min(valid)) >= self.min_safe_dist

    # ── Callbacks ─────────────────────────────────────────────────────────────

    _TRAIN_FOV_RAD = np.deg2rad(270.0)

    def on_scan(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        # Crop to the central 270° that the policy was trained on.
        # If the physical lidar has a wider FOV (e.g. 360°) we strip the
        # outer beams symmetrically so the angular coverage matches training.
        scan_fov = msg.angle_max - msg.angle_min  # total FOV in radians
        if scan_fov > self._TRAIN_FOV_RAD + 0.01:
            n = len(ranges)
            beams_to_keep = int(round(n * self._TRAIN_FOV_RAD / scan_fov))
            start = (n - beams_to_keep) // 2
            ranges = ranges[start: start + beams_to_keep]
        self.last_scan = ranges
        self._last_scan_msg = msg  # keep full message for angle metadata

    def on_primary(self, msg):
        update_agent_state(self.primary_state, msg)
        # Use local clock for staleness check — avoids VICON/robot clock mismatch
        self.primary_state["stamp"] = self.get_clock().now().nanoseconds * 1e-9

    def on_secondary(self, msg):
        update_agent_state(self.secondary_state, msg)
        self.secondary_state["stamp"] = self.get_clock().now().nanoseconds * 1e-9

    def on_tick(self):
        self._tick_count += 1
        debug = (self._tick_count % 20 == 1)  # log every ~1 s at 20 Hz

        # ── Guard: wait for first data ─────────────────────────────────────
        if (
            self.last_scan is None
            or self.primary_state["pose"] is None
            or self.secondary_state["pose"] is None
        ):
            if debug:
                self.get_logger().info(
                    "Waiting for data — scan=%s primary=%s secondary=%s" % (
                        self.last_scan is not None,
                        self.primary_state["pose"] is not None,
                        self.secondary_state["pose"] is not None,
                    )
                )
            self.pub_cmd.publish(Twist())
            return

        # ── Guard: stale poses ─────────────────────────────────────────────
        now = self.get_clock().now().nanoseconds * 1e-9
        for label, state in (("primary", self.primary_state), ("secondary", self.secondary_state)):
            stamp = state["stamp"]
            if stamp is None or (now - stamp) > self.max_pose_age:
                self.get_logger().warn(
                    "%s pose stale -> zero command" % label,
                    throttle_duration_sec=2.0,
                )
                self.pub_cmd.publish(Twist())
                return

        # ── Guard: forward collision ───────────────────────────────────────
        if not self._forward_clear():
            self.get_logger().warn(
                "Obstacle within %.2fm — stopping" % self.min_safe_dist,
                throttle_duration_sec=1.0,
            )
            self.pub_cmd.publish(Twist())
            return

        # ── Inference ──────────────────────────────────────────────────────
        obs_vec = build_observation(self.last_scan, self.primary_state, self.secondary_state)
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0)
        with torch.no_grad():
            raw = self.policy(obs_t).squeeze(0)
        action = torch.clamp(raw, self.act_low, self.act_high)

        steer = float(action[0])
        throttle = float(action[1]) * self.throttle_scale
        if self.prevent_reverse:
            throttle = max(throttle, self.min_throttle)

        if debug:
            self.get_logger().info("action: steer=%.3f throttle=%.3f" % (steer, throttle))

        cmd = Twist()
        cmd.angular.z = steer
        cmd.linear.x = throttle
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RLActorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
