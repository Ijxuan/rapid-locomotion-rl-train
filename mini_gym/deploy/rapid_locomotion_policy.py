"""Shared deployment math for rapid-locomotion Mini Cheetah policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


OBS_DIM = 42
HISTORY_LENGTH = 15
OBS_HISTORY_DIM = OBS_DIM * HISTORY_LENGTH
LATENT_DIM = 18
ACTOR_INPUT_DIM = OBS_DIM + LATENT_DIM
ACTION_DIM = 12

POLICY_DT = 0.02
ACTION_SCALE = 0.25
HIP_SCALE_REDUCTION = 0.5
KP = 20.0
KD = 0.5

LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05

COMMAND_SCALE = np.array([LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE], dtype=np.float32)

# Mini Cheetah Isaac Gym DOF order observed in the training environment:
# FL, FR, RL, RR; each leg is abad/hip, thigh, calf.
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
HIP_INDICES = np.array([0, 3, 6, 9], dtype=np.int64)


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


def action_to_target_q(action: Iterable[float]) -> np.ndarray:
    action = _array(action, ACTION_DIM, "action")
    scaled = action * ACTION_SCALE
    scaled[HIP_INDICES] *= HIP_SCALE_REDUCTION
    return DEFAULT_Q_POLICY + scaled


def build_observation(
    projected_gravity: Iterable[float],
    command: Iterable[float],
    q_policy: Iterable[float],
    qd_policy: Iterable[float],
    last_action: Iterable[float],
) -> np.ndarray:
    projected_gravity = _array(projected_gravity, 3, "projected_gravity")
    command = _array(command, 3, "command")
    q_policy = _array(q_policy, ACTION_DIM, "q_policy")
    qd_policy = _array(qd_policy, ACTION_DIM, "qd_policy")
    last_action = _array(last_action, ACTION_DIM, "last_action")
    obs = np.concatenate(
        [
            projected_gravity,
            command * COMMAND_SCALE,
            (q_policy - DEFAULT_Q_POLICY) * DOF_POS_SCALE,
            qd_policy * DOF_VEL_SCALE,
            last_action,
        ]
    ).astype(np.float32)
    if obs.size != OBS_DIM:
        raise RuntimeError(f"observation size mismatch: {obs.size}")
    return obs


@dataclass
class ObservationHistory:
    length: int = HISTORY_LENGTH
    obs_dim: int = OBS_DIM

    def __post_init__(self) -> None:
        self.buffer = np.zeros(self.length * self.obs_dim, dtype=np.float32)

    def reset(self) -> None:
        self.buffer.fill(0.0)

    def update(self, obs: Iterable[float]) -> np.ndarray:
        obs = _array(obs, self.obs_dim, "obs")
        self.buffer[:-self.obs_dim] = self.buffer[self.obs_dim:]
        self.buffer[-self.obs_dim:] = obs
        return self.buffer.copy()
