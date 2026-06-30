"""Shared deployment math for Paper B Mini Cheetah policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


OBS_DIM = 142
ESTIMATOR_OUTPUT_DIM = 11
ACTOR_INPUT_DIM = OBS_DIM + ESTIMATOR_OUTPUT_DIM
ACTION_DIM = 12
FOOT_COUNT = 4

POLICY_DT = 0.01
ACTION_SCALE = 0.1
HIP_SCALE_REDUCTION = 1.0

KP = 17.0
KD = 0.4

DEFAULT_Q_POLICY = np.array(
    [
        0.1,
        -0.8,
        1.62,
        -0.1,
        -0.8,
        1.62,
        0.1,
        -0.8,
        1.62,
        -0.1,
        -0.8,
        1.62,
    ],
    dtype=np.float32,
)

# Robot controller order is FR, FL, RR, RL. This maps each policy index to the
# corresponding robot-order index.
POLICY_TO_ROBOT = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
ROBOT_TO_POLICY = POLICY_TO_ROBOT.copy()
FOOT_POLICY_TO_ROBOT = np.array([1, 0, 3, 2], dtype=np.int64)
FOOT_ROBOT_TO_POLICY = FOOT_POLICY_TO_ROBOT.copy()


def _array(values: Iterable[float], size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must have {size} values, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def robot_to_policy_order(values: Iterable[float]) -> np.ndarray:
    return _array(values, ACTION_DIM, "values")[ROBOT_TO_POLICY].copy()


def policy_to_robot_order(values: Iterable[float]) -> np.ndarray:
    policy = _array(values, ACTION_DIM, "values")
    robot = np.empty(ACTION_DIM, dtype=np.float32)
    robot[POLICY_TO_ROBOT] = policy
    return robot


def robot_foot_positions_to_policy_order(values: Iterable[float]) -> np.ndarray:
    foot_positions = _array(values, FOOT_COUNT * 3, "foot_positions_body").reshape(FOOT_COUNT, 3)
    return foot_positions[FOOT_ROBOT_TO_POLICY].reshape(-1).copy()


def policy_foot_positions_to_robot_order(values: Iterable[float]) -> np.ndarray:
    foot_positions = _array(values, FOOT_COUNT * 3, "foot_positions_body").reshape(FOOT_COUNT, 3)
    robot = np.empty((FOOT_COUNT, 3), dtype=np.float32)
    robot[FOOT_POLICY_TO_ROBOT] = foot_positions
    return robot.reshape(-1)


def action_to_target_q(action: Iterable[float]) -> np.ndarray:
    action = _array(action, ACTION_DIM, "action")
    return DEFAULT_Q_POLICY + ACTION_SCALE * action


def build_observation(
    base_quat: Iterable[float],
    base_ang_vel: Iterable[float],
    q_policy: Iterable[float],
    qd_policy: Iterable[float],
    previous_desired_joint_positions: Iterable[float],
    joint_position_error_history: Iterable[float],
    joint_velocity_history: Iterable[float],
    foot_positions_body: Iterable[float],
    command: Iterable[float],
) -> np.ndarray:
    base_quat = _array(base_quat, 4, "base_quat")
    base_ang_vel = _array(base_ang_vel, 3, "base_ang_vel")
    q_policy = _array(q_policy, ACTION_DIM, "q_policy")
    qd_policy = _array(qd_policy, ACTION_DIM, "qd_policy")
    previous_desired_joint_positions = _array(
        previous_desired_joint_positions, ACTION_DIM * 2, "previous_desired_joint_positions")
    joint_position_error_history = _array(
        joint_position_error_history, ACTION_DIM * 3, "joint_position_error_history")
    joint_velocity_history = _array(joint_velocity_history, ACTION_DIM * 3, "joint_velocity_history")
    foot_positions_body = _array(foot_positions_body, FOOT_COUNT * 3, "foot_positions_body")
    command = _array(command, 3, "command")

    obs = np.concatenate(
        [
            base_quat,
            base_ang_vel,
            q_policy,
            qd_policy,
            previous_desired_joint_positions,
            joint_position_error_history,
            joint_velocity_history,
            foot_positions_body,
            command,
        ]
    ).astype(np.float32)
    if obs.size != OBS_DIM:
        raise RuntimeError(f"observation size mismatch: {obs.size}")
    return obs


@dataclass
class ObservationHistory:
    previous_desired_joint_positions: np.ndarray = field(
        default_factory=lambda: np.tile(DEFAULT_Q_POLICY, 2).astype(np.float32))
    joint_position_error_history: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM * 3, dtype=np.float32))
    joint_velocity_history: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM * 3, dtype=np.float32))
    joint_history_delay_line_steps: int = 7
    joint_history_sparse_indices: tuple[int, int, int] = (4, 2, 0)
    _joint_position_error_delay_line: np.ndarray = field(init=False, repr=False)
    _joint_velocity_delay_line: np.ndarray = field(init=False, repr=False)
    _joint_state_initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._joint_position_error_delay_line = np.zeros(
            (self.joint_history_delay_line_steps, ACTION_DIM), dtype=np.float32)
        self._joint_velocity_delay_line = np.zeros_like(self._joint_position_error_delay_line)

    def reset(self) -> None:
        self.previous_desired_joint_positions[:] = np.tile(DEFAULT_Q_POLICY, 2)
        self.joint_position_error_history.fill(0.0)
        self.joint_velocity_history.fill(0.0)
        self._joint_position_error_delay_line.fill(0.0)
        self._joint_velocity_delay_line.fill(0.0)
        self._joint_state_initialized = False

    def update_joint_state(self, q_policy: Iterable[float], qd_policy: Iterable[float]) -> None:
        q_policy = _array(q_policy, ACTION_DIM, "q_policy")
        qd_policy = _array(qd_policy, ACTION_DIM, "qd_policy")
        joint_error = q_policy - DEFAULT_Q_POLICY

        if not self._joint_state_initialized:
            self._joint_position_error_delay_line[:] = joint_error
            self._joint_velocity_delay_line[:] = qd_policy
            self._joint_state_initialized = True
        else:
            self._joint_position_error_delay_line[:-1] = self._joint_position_error_delay_line[1:].copy()
            self._joint_position_error_delay_line[-1] = joint_error
            self._joint_velocity_delay_line[:-1] = self._joint_velocity_delay_line[1:].copy()
            self._joint_velocity_delay_line[-1] = qd_policy

        self.joint_position_error_history[:] = self._joint_position_error_delay_line[
            list(self.joint_history_sparse_indices)
        ].reshape(-1)
        self.joint_velocity_history[:] = self._joint_velocity_delay_line[
            list(self.joint_history_sparse_indices)
        ].reshape(-1)

    def update_desired_joint_positions(self, target_q: Iterable[float]) -> None:
        target_q = _array(target_q, ACTION_DIM, "target_q")
        self.previous_desired_joint_positions[ACTION_DIM:] = self.previous_desired_joint_positions[:ACTION_DIM]
        self.previous_desired_joint_positions[:ACTION_DIM] = target_q

    def build(
        self,
        base_quat: Iterable[float],
        base_ang_vel: Iterable[float],
        q_policy: Iterable[float],
        qd_policy: Iterable[float],
        foot_positions_body: Iterable[float],
        command: Iterable[float],
    ) -> np.ndarray:
        self.update_joint_state(q_policy, qd_policy)
        return build_observation(
            base_quat,
            base_ang_vel,
            q_policy,
            qd_policy,
            self.previous_desired_joint_positions,
            self.joint_position_error_history,
            self.joint_velocity_history,
            foot_positions_body,
            command,
        )
