"""Paper B observation and estimator layout constants.

These helpers intentionally avoid Isaac Gym and params_proto imports so they
can be tested on machines that only edit code.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping


ACTION_DIM = 12
FOOT_COUNT = 4
XYZ_DIM = 3
QUAT_DIM = 4
COMMAND_DIM = 3
JOINT_HISTORY_STEPS = 3
ACTION_HISTORY_STEPS = 2

PAPER_B_POLICY_DT = 0.01
PAPER_B_JOINT_HISTORY_DT = 0.02
PAPER_B_ACTION_SCALE = 0.1
PAPER_B_KP = 17.0
PAPER_B_KD = 0.4


@dataclass(frozen=True)
class FieldSpec:
    name: str
    size: int


def _make_slices(fields: list[FieldSpec]) -> "OrderedDict[str, slice]":
    start = 0
    slices: "OrderedDict[str, slice]" = OrderedDict()
    for field in fields:
        slices[field.name] = slice(start, start + field.size)
        start += field.size
    return slices


OBSERVATION_FIELDS = [
    FieldSpec("base_quat", QUAT_DIM),
    FieldSpec("base_ang_vel", XYZ_DIM),
    FieldSpec("dof_pos", ACTION_DIM),
    FieldSpec("dof_vel", ACTION_DIM),
    FieldSpec("previous_desired_joint_positions", ACTION_DIM * ACTION_HISTORY_STEPS),
    FieldSpec("joint_position_error_history", ACTION_DIM * JOINT_HISTORY_STEPS),
    FieldSpec("joint_velocity_history", ACTION_DIM * JOINT_HISTORY_STEPS),
    FieldSpec("foot_positions_body", FOOT_COUNT * XYZ_DIM),
    FieldSpec("commands", COMMAND_DIM),
]

ESTIMATOR_TARGET_FIELDS = [
    FieldSpec("base_lin_vel", XYZ_DIM),
    FieldSpec("foot_height", FOOT_COUNT),
    FieldSpec("contact_probability", FOOT_COUNT),
]

OBSERVATION_SLICES = _make_slices(OBSERVATION_FIELDS)
ESTIMATOR_TARGET_SLICES = _make_slices(ESTIMATOR_TARGET_FIELDS)

OBS_DIM = sum(field.size for field in OBSERVATION_FIELDS)
ESTIMATOR_TARGET_DIM = sum(field.size for field in ESTIMATOR_TARGET_FIELDS)
ESTIMATOR_OUTPUT_DIM = ESTIMATOR_TARGET_DIM
ACTOR_INPUT_DIM = OBS_DIM + ESTIMATOR_OUTPUT_DIM


def layout_width(slices: Mapping[str, slice]) -> int:
    return max(part.stop for part in slices.values())


def observation_components(
    base_quat,
    base_ang_vel,
    dof_pos,
    dof_vel,
    previous_desired_joint_positions,
    joint_position_error_history,
    joint_velocity_history,
    foot_positions_body,
    commands,
):
    return (
        base_quat,
        base_ang_vel,
        dof_pos,
        dof_vel,
        previous_desired_joint_positions,
        joint_position_error_history,
        joint_velocity_history,
        foot_positions_body,
        commands,
    )


def estimator_target_components(base_lin_vel, foot_height, contact_probability):
    return base_lin_vel, foot_height, contact_probability


def action_to_desired_joint_positions(action, nominal_joint_positions):
    """Paper B action mapping: q_des = q_nominal + 0.1 * action."""
    return nominal_joint_positions + PAPER_B_ACTION_SCALE * action
