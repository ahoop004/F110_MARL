#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS1 Rainbow-DQN actor node aligned with scenarios/gaplock_rainbow_dqn.yaml.
Place alongside rainbow_gaplock.pt and gaplock_utils.py in your ROS package.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional

import numpy as np
import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan

from gaplock_utils import OBS_DIM, build_observation, init_agent_state, update_agent_state

DEFAULT_ACTION_SET = np.asarray(
    [
        [-0.35, 0.90],
        [-0.15, 0.80],
        [0.00, 0.80],
        [0.15, 0.80],
        [0.35, 0.90],
        [-0.20, 0.30],
        [0.00, 0.30],
        [0.20, 0.30],
        [0.00, 0.00],
        [0.00, -0.50],
    ],
    dtype=np.float32,
)


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)
        weight_shape = (self.out_features, self.in_features)
        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        self.weight_sigma = nn.Parameter(torch.empty(weight_shape))
        self.register_buffer("weight_epsilon", torch.zeros(weight_shape))
        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
        self.register_buffer("bias_epsilon", torch.zeros(self.out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        sigma_weight = self.sigma0 / math.sqrt(self.in_features)
        sigma_bias = self.sigma0 / math.sqrt(self.out_features)
        self.weight_sigma.data.fill_(sigma_weight)
        self.bias_sigma.data.fill_(sigma_bias)

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, device=self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int, *, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class RainbowQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: Iterable[int] = (512, 256, 128),
        *,
        atoms: int = 51,
        v_min: float = -50.0,
        v_max: float = 50.0,
        noisy: bool = True,
        sigma0: float = 0.4,
    ) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.atoms = int(atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.noisy = bool(noisy)
        self.sigma0 = float(sigma0)

        self.hidden_layers = nn.ModuleList()
        self._noisy_layers = []
        prev = input_dim
        for dim in hidden_dims:
            layer = self._make_linear(prev, int(dim))
            self.hidden_layers.append(layer)
            prev = int(dim)

        self.value_head = self._make_linear(prev, self.atoms)
        self.adv_head = self._make_linear(prev, self.n_actions * self.atoms)
        support = torch.linspace(self.v_min, self.v_max, self.atoms)
        self.register_buffer("support", support)

    def _make_linear(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.noisy:
            layer = NoisyLinear(in_dim, out_dim, sigma0=self.sigma0)
            self._noisy_layers.append(layer)
            return layer
        return nn.Linear(in_dim, out_dim)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for layer in self._noisy_layers:
            layer.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        value = self.value_head(x).view(-1, 1, self.atoms)
        adv = self.adv_head(x).view(-1, self.n_actions, self.atoms)
        adv = adv - adv.mean(dim=1, keepdim=True)
        return value + adv

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * self.support, dim=-1)


def load_rainbow_checkpoint(
    ckpt_path: str,
    model: RainbowQNetwork,
    device: str,
) -> tuple[RainbowQNetwork, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "q_net" not in ckpt:
        raise RuntimeError(f"Rainbow checkpoint missing q_net: keys={list(ckpt.keys())}")
    model.load_state_dict(ckpt["q_net"], strict=False)
    action_set = np.asarray(ckpt.get("action_set", DEFAULT_ACTION_SET), dtype=np.float32)
    return model, action_set


class RLActorNode:
    def __init__(self, default_ckpt: str = "rainbow_gaplock.pt") -> None:
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
        self.max_pose_age = float(rospy.get_param("~max_pose_age", 0.25))
        self.prevent_reverse = rospy.get_param("~prevent_reverse", True)
        self.min_throttle = float(rospy.get_param("~prevent_reverse_min_speed", 0.0))

        hidden_dims = rospy.get_param("~hidden_dims", [512, 256, 128])
        atoms = int(rospy.get_param("~atoms", 51))
        v_min = float(rospy.get_param("~v_min", -50.0))
        v_max = float(rospy.get_param("~v_max", 50.0))
        noisy = rospy.get_param("~noisy_layers", True)
        sigma0 = float(rospy.get_param("~noisy_sigma0", 0.4))

        self.last_scan: Optional[np.ndarray] = None
        self.primary_state = init_agent_state()
        self.secondary_state = init_agent_state()

        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)
        self.sub_primary = rospy.Subscriber(self.primary_topic, TransformStamped, self.on_primary, queue_size=1)
        self.sub_secondary = rospy.Subscriber(self.secondary_topic, TransformStamped, self.on_secondary, queue_size=1)
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)

        if not os.path.isfile(self.ckpt_path):
            rospy.logfatal("Rainbow checkpoint not found: %s", self.ckpt_path)
            raise FileNotFoundError(self.ckpt_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_net = RainbowQNetwork(
            OBS_DIM,
            DEFAULT_ACTION_SET.shape[0],
            hidden_dims=hidden_dims,
            atoms=atoms,
            v_min=v_min,
            v_max=v_max,
            noisy=noisy,
            sigma0=sigma0,
        ).to(self.device).eval()
        self.q_net, self.action_set = load_rainbow_checkpoint(self.ckpt_path, self.q_net, self.device)
        self.action_set = np.asarray(self.action_set, dtype=np.float32)
        self.q_net.eval()

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.on_tick)
        rospy.loginfo("Rainbow DQN ROS actor ready | ckpt=%s | device=%s", self.ckpt_path, self.device)

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
            q_values = self.q_net.q_values(obs_t)
            action_idx = int(torch.argmax(q_values, dim=-1).item())
        action = np.asarray(self.action_set[action_idx], dtype=np.float32)
        if self.prevent_reverse:
            action = action.copy()
            action[1] = max(action[1], self.min_throttle)

        cmd = Twist()
        cmd.angular.z = float(action[0])
        cmd.linear.x = float(action[1])
        self.pub_cmd.publish(cmd)


def main():
    rospy.init_node("rl_actor_rainbow_node", anonymous=False)
    RLActorNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
