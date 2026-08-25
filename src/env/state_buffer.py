"""State buffer helpers for `F110ParallelEnv`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.env.types import AgentRaceStatus


@dataclass(frozen=True)
class TerminalAgentConfig:
    """Deterministic physical behavior after an agent's trajectory ends."""

    remain_collidable: bool = True
    crashed_behavior: str = "stationary"
    finished_behavior: str = "coast_then_stop"
    finish_clearance_steps: int = 200

    @classmethod
    def from_mapping(cls, raw: object) -> "TerminalAgentConfig":
        cfg = raw if isinstance(raw, Mapping) else {}
        result = cls(
            remain_collidable=bool(cfg.get("remain_collidable", True)),
            crashed_behavior=str(cfg.get("crashed_behavior", "stationary")),
            finished_behavior=str(cfg.get("finished_behavior", "coast_then_stop")),
            finish_clearance_steps=int(cfg.get("finish_clearance_steps", 200)),
        )
        if not result.remain_collidable:
            raise ValueError("terminal_agents.remain_collidable must be true")
        if result.crashed_behavior != "stationary":
            raise ValueError("terminal_agents.crashed_behavior must be 'stationary'")
        if result.finished_behavior != "coast_then_stop":
            raise ValueError(
                "terminal_agents.finished_behavior must be 'coast_then_stop'"
            )
        if result.finish_clearance_steps < 0:
            raise ValueError("terminal_agents.finish_clearance_steps must be >= 0")
        return result


@dataclass
class TerminalVehicleState:
    status: AgentRaceStatus
    terminal_step: int
    last_action: np.ndarray
    last_vehicle_state: np.ndarray


class TerminalVehicleController:
    """Own post-terminal physical commands without creating decisions."""

    def __init__(
        self,
        agent_ids: Sequence[str],
        config: TerminalAgentConfig,
    ) -> None:
        self.agent_ids = tuple(agent_ids)
        self.config = config
        self.states: Dict[str, TerminalVehicleState] = {}

    def reset(self) -> None:
        self.states.clear()

    def capture(
        self,
        agent_id: str,
        *,
        status: AgentRaceStatus,
        terminal_step: int,
        action: np.ndarray,
        vehicle_state: np.ndarray,
    ) -> None:
        if agent_id in self.states:
            return
        self.states[agent_id] = TerminalVehicleState(
            status=status,
            terminal_step=int(terminal_step),
            last_action=np.asarray(action, dtype=np.float32).copy(),
            last_vehicle_state=np.asarray(vehicle_state, dtype=np.float64).copy(),
        )

    def apply(
        self,
        joint_actions: np.ndarray,
        *,
        agent_index: Mapping[str, int],
        simulator: Any,
        step: int,
    ) -> None:
        for agent_id, state in self.states.items():
            idx = agent_index[agent_id]
            if state.status in {AgentRaceStatus.CRASHED, AgentRaceStatus.TRUNCATED}:
                self._freeze(simulator, idx)
                joint_actions[idx] = (0.0, 0.0)
                continue

            # Linearly decay both the terminal steering command and requested
            # speed over the exact clearance window, then freeze all motion.
            clearance = self.config.finish_clearance_steps
            elapsed = max(0, int(step) - state.terminal_step)
            scale = max(0.0, 1.0 - elapsed / max(clearance, 1)) if clearance else 0.0
            joint_actions[idx, 0] = float(state.last_action[0]) * scale
            joint_actions[idx, 1] = max(float(state.last_action[1]), 0.0) * scale
            if scale <= 0.0:
                self._freeze(simulator, idx)

    @staticmethod
    def _freeze(simulator: Any, index: int) -> None:
        agent = simulator.agents[index]
        agent.state[3] = 0.0  # longitudinal velocity
        agent.state[5] = 0.0  # yaw rate
        agent.state[6] = 0.0  # slip angle


@dataclass
class StateBuffers:
    """Container tracking per-agent kinematics across simulator steps."""

    poses_x: np.ndarray
    poses_y: np.ndarray
    poses_theta: np.ndarray
    collisions: np.ndarray
    linear_vels_x_prev: np.ndarray
    linear_vels_y_prev: np.ndarray
    angular_vels_prev: np.ndarray
    linear_vels_x_curr: np.ndarray
    linear_vels_y_curr: np.ndarray
    angular_vels_curr: np.ndarray
    velocity_initialized: bool = False

    @classmethod
    def build(cls, n_agents: int) -> "StateBuffers":
        zeros = np.zeros((n_agents,), dtype=np.float32)
        return cls(
            poses_x=zeros.copy(),
            poses_y=zeros.copy(),
            poses_theta=zeros.copy(),
            collisions=zeros.copy(),
            linear_vels_x_prev=zeros.copy(),
            linear_vels_y_prev=zeros.copy(),
            angular_vels_prev=zeros.copy(),
            linear_vels_x_curr=zeros.copy(),
            linear_vels_y_curr=zeros.copy(),
            angular_vels_curr=zeros.copy(),
            velocity_initialized=False,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        for array in (
            self.poses_x,
            self.poses_y,
            self.poses_theta,
            self.collisions,
            self.linear_vels_x_prev,
            self.linear_vels_y_prev,
            self.angular_vels_prev,
            self.linear_vels_x_curr,
            self.linear_vels_y_curr,
            self.angular_vels_curr,
        ):
            array.fill(0.0)
        self.velocity_initialized = False

    def update(self, obs_dict: Dict[str, np.ndarray]) -> None:
        self._assign(self.poses_x, obs_dict.get("poses_x"))
        self._assign(self.poses_y, obs_dict.get("poses_y"))
        self._assign(self.poses_theta, obs_dict.get("poses_theta"))
        self._assign(self.collisions, obs_dict.get("collisions"))

        self.linear_vels_x_prev[:] = self.linear_vels_x_curr
        self.linear_vels_y_prev[:] = self.linear_vels_y_curr
        self.angular_vels_prev[:] = self.angular_vels_curr

        self._assign(self.linear_vels_x_curr, obs_dict.get("linear_vels_x"))
        self._assign(self.linear_vels_y_curr, obs_dict.get("linear_vels_y"))
        self._assign(self.angular_vels_curr, obs_dict.get("ang_vels_z"))

        self.velocity_initialized = True

    # ------------------------------------------------------------------
    @staticmethod
    def _assign(target: np.ndarray, source: np.ndarray | None) -> None:
        if source is None:
            target.fill(0.0)
            return
        arr = np.asarray(source, dtype=np.float32)
        count = min(target.shape[0], arr.shape[0])
        target[:count] = arr[:count]
        if count < target.shape[0]:
            target[count:] = 0.0
