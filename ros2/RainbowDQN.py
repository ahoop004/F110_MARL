#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 Rainbow-DQN actor node aligned with scenarios/gaplock_rainbow_dqn.yaml.
Place alongside rainbow_gaplock.pt and gaplock_utils.py in your ROS package.

UPDATE (LIMO cmd_vel fix):
- Adds steer_sign + scaling + command caps so the RL actor matches the steering sign
  that worked for your FTG node and keeps speeds sane on the LIMO.
"""

#from __future__ import annotations

import math
import os
import importlib
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        update_agent_state,
    )
except ImportError:  # pragma: no cover
    from gaplock_utils import (  # type: ignore
        ACTION_HIGH,
        ACTION_LOW,
        OBS_DIM,
        build_observation,
        init_agent_state,
        update_agent_state,
    )

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
#        [0.00, -0.50],
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
        hidden_dims: Iterable[int] = (256, 256),
        *,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy: bool = True,
        sigma0: float = 0.5,
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
        self.advantage_head = self._make_linear(prev, self.n_actions * self.atoms)
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
        advantage = self.advantage_head(x).view(-1, self.n_actions, self.atoms)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return value + advantage

    def dist(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        return torch.softmax(logits, dim=-1)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        probs = self.dist(obs)
        return torch.sum(probs * self.support, dim=-1)


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
    """Allow loading checkpoints pickled with numpy>=2 on numpy<2.

    Some pickled numpy objects reference internal modules under ``numpy._core``.
    Older numpy versions only expose ``numpy.core``; we alias the module paths so
    ``torch.load`` can unpickle these checkpoints.
    """

    if "numpy._core" in sys.modules:
        return

    try:
        core_mod = importlib.import_module("numpy.core")
    except Exception:
        return

    sys.modules.setdefault("numpy._core", core_mod)

    # Alias common submodules used by numpy pickles.
    for name in ("multiarray", "_multiarray_umath", "numeric", "umath", "overrides"):
        alias_name = f"numpy._core.{name}"
        if alias_name in sys.modules:
            continue
        try:
            sub_mod = importlib.import_module(f"numpy.core.{name}")
        except Exception:
            continue
        sys.modules.setdefault(alias_name, sub_mod)


def _infer_hidden_dims(state_dict: Mapping[str, torch.Tensor]) -> Tuple[int, ...]:
    dims = []
    idx = 0
    while True:
        tensor = state_dict.get(f"hidden_layers.{idx}.weight_mu")
        if tensor is None:
            tensor = state_dict.get(f"hidden_layers.{idx}.weight")
        if tensor is None:
            break
        dims.append(int(tensor.shape[0]))
        idx += 1
    return tuple(dims)


def _infer_atoms(state_dict: Mapping[str, torch.Tensor]) -> int:
    support = state_dict.get("support")
    if isinstance(support, torch.Tensor) and support.ndim == 1:
        return int(support.shape[0])
    tensor = state_dict.get("value_head.weight_mu")
    if tensor is None:
        tensor = state_dict.get("value_head.weight")
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
        return int(tensor.shape[0])
    raise RuntimeError("Unable to infer atoms from checkpoint/state_dict")


def _infer_n_actions(state_dict: Mapping[str, torch.Tensor], *, atoms: int) -> int:
    tensor = state_dict.get("advantage_head.weight_mu")
    if tensor is None:
        tensor = state_dict.get("advantage_head.weight")
    if tensor is None or tensor.ndim != 2:
        raise RuntimeError("Unable to infer action count from checkpoint/state_dict")
    out_dim = int(tensor.shape[0])
    if atoms <= 0 or out_dim % atoms != 0:
        raise RuntimeError(f"Invalid (out_dim={out_dim}, atoms={atoms}) for Rainbow head")
    return out_dim // atoms


def _infer_use_noisy(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(k.endswith("weight_mu") for k in state_dict.keys())


def _extract_rainbow_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(ckpt, nn.Module):
        return dict(ckpt.state_dict()), {}
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("q_net"), dict):
            return ckpt["q_net"], ckpt
        if isinstance(ckpt.get("state_dict"), dict):
            return ckpt["state_dict"], ckpt
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()) and all(
            isinstance(v, torch.Tensor) for v in ckpt.values()
        ):
            return ckpt, {}
        return {}, ckpt
    return {}, {}


def load_rainbow_checkpoint(ckpt_path: str, *, device: str) -> Tuple[RainbowQNetwork, np.ndarray]:
    _apply_numpy_pickle_compat()
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict, meta = _extract_rainbow_state_dict(ckpt)
    if not state_dict:
        raise RuntimeError(
            "Rainbow checkpoint format unsupported; expected dict with 'q_net' or a state_dict. "
            f"type={type(ckpt)!r} keys={list(meta.keys()) if isinstance(meta, dict) else None}"
        )

    action_set = np.asarray(meta.get("action_set", DEFAULT_ACTION_SET), dtype=np.float32)
    if action_set.ndim != 2 or action_set.shape[1] != 2:
        raise RuntimeError(f"Invalid action_set shape {action_set.shape}; expected (N, 2)")

    obs_dim = int(meta.get("obs_dim", OBS_DIM))
    if obs_dim != OBS_DIM:
        raise RuntimeError(f"Checkpoint obs_dim={obs_dim} does not match expected OBS_DIM={OBS_DIM}")

    atoms = int(meta.get("atoms", _infer_atoms(state_dict)))
    v_min = float(meta.get("v_min", -10.0))
    v_max = float(meta.get("v_max", 10.0))
    use_noisy = bool(meta.get("use_noisy", _infer_use_noisy(state_dict)))
    sigma0 = float(meta.get("noisy_sigma0", meta.get("sigma0", 0.5)))

    hidden_dims = _infer_hidden_dims(state_dict) or (256, 256)
    inferred_actions = _infer_n_actions(state_dict, atoms=atoms)
    if int(action_set.shape[0]) != inferred_actions:
        raise RuntimeError(
            "Checkpoint action_set length does not match network output: "
            f"action_set={int(action_set.shape[0])} inferred_n_actions={inferred_actions}"
        )

    model = RainbowQNetwork(
        obs_dim,
        inferred_actions,
        hidden_dims=hidden_dims,
        atoms=atoms,
        v_min=v_min,
        v_max=v_max,
        noisy=use_noisy,
        sigma0=sigma0,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, action_set


class RLActorNode(Node):
    def __init__(self, default_ckpt: str = "r_dqn.pt") -> None:
        super().__init__('rl_actor_dqn_node')
        self.logger = self.get_logger()
        self.scan_topic = "/scan"
        self.primary_topic = "/vicon/Limo_04/Limo_04"
        self.secondary_topic =  "/vicon/Limo_02/Limo_02"
        self.cmd_topic = "/cmd_vel"

        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_ckpt =  default_ckpt
        if not os.path.isabs(default_ckpt):
            default_ckpt = os.path.join(script_dir, default_ckpt)
        self.ckpt_path =  default_ckpt
        self.ckpt_path = "/home/agilex/agilex_ros2_ws/src/car_crashers/car_crashers/r_dqn.pt"

        self.rate_hz =  20.0
        self.use_safety =  True
        self.hard_border =  1.0
        self.max_pose_age = 0.25
        self.prevent_reverse =  True
        self.min_throttle = 0.0

        # Output mapping (sim-trained actions -> LIMO /cmd_vel)
        # steer_sign: use -1.0 if your base turns the "wrong way" for positive angular.z
        self.steer_sign =  -1.0
        # Scale raw policy outputs down to LIMO-safe speeds / turns
        self.lin_scale = 0.4
        self.ang_scale = 1.0
        # Final command caps
        self.max_steer_cmd = 1.0
        self.max_speed_cmd =  0.35
        self.max_reverse_speed_cmd =  0.0

        hidden_dims =  [512, 256]
        atoms =  51
        v_min =  -500.0
        v_max = 500.0
        noisy =  True
        sigma0 =  0.4

        self.last_scan: Optional[np.ndarray] = None
        self.primary_state = init_agent_state()
        self.secondary_state = init_agent_state()
        qos = QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos)

        
        self.sub_primary = self.create_subscription(TransformStamped, self.primary_topic, self.on_primary, 20)
        self.sub_secondary = self.create_subscription(TransformStamped, self.secondary_topic, self.on_secondary, 20)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 20)

        if not os.path.isfile(self.ckpt_path):
            self.logger.fatal("Rainbow checkpoint not found: %s" % self.ckpt_path)
            raise FileNotFoundError(self.ckpt_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_net, self.action_set = load_rainbow_checkpoint(self.ckpt_path, device=self.device)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_tick)
        self.logger.info("Rainbow DQN ROS actor ready | ckpt=%s | device=%s" % (self.ckpt_path, self.device))

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
            q_values = self.q_net.q_values(obs_t)
            action_idx = int(torch.argmax(q_values, dim=-1).item())

        action = np.asarray(self.action_set[action_idx], dtype=np.float32)
        raw_steer = float(action[0])
        raw_speed = float(action[1])

        # Optional: prevent reverse (keep speed >= min_throttle in the *raw* action space)
        if self.prevent_reverse:
            raw_speed = max(raw_speed, self.min_throttle)

        # Map / scale to LIMO /cmd_vel conventions
        steer_cmd = self.steer_sign * raw_steer * self.ang_scale
        speed_cmd = raw_speed * self.lin_scale

        # Clip to configured command limits
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
