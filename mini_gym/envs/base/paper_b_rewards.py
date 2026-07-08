"""Paper B reward helpers that are independent from Isaac Gym."""

import math

PAPER_B_POSITIVE_REWARDS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "feet_air_time",
)

PAPER_B_NEGATIVE_REWARDS = (
    "feet_slip",
    "feet_clearance",
    "orientation",
    "torques",
    "dof_pos",
    "dof_vel",
    "dof_acc",
    "action_smoothness_1",
    "action_smoothness_2",
    "base_motion",
)

PAPER_B_DIRECT_REWARDS = (
    "moving_stand_still",
)


def paper_b_reward_gate(negative_reward, exponential_scale=0.02, gate_floor=0.05):
    exponent = exponential_scale * negative_reward
    min_exponent = math.log(gate_floor) if gate_floor and gate_floor > 0.0 else None
    if hasattr(exponent, "clamp"):
        if min_exponent is None:
            exponent = exponent.clamp(max=0.0)
        else:
            exponent = exponent.clamp(min=min_exponent, max=0.0)
        return exponent.exp()

    import numpy as np

    if min_exponent is None:
        exponent = np.minimum(exponent, 0.0)
    else:
        exponent = np.clip(exponent, min_exponent, 0.0)
    return np.exp(exponent)


def paper_b_total_reward(positive_reward, negative_reward, exponential_scale=0.02, gate_floor=0.05):
    gate = paper_b_reward_gate(negative_reward, exponential_scale, gate_floor)
    return positive_reward * gate


def paper_b_moving_speed_deficit(
    command_vx,
    base_vx,
    command_threshold=0.3,
    velocity_threshold=0.15,
    progress_fraction=0.2,
):
    """Return normalized speed deficit in the commanded direction.

    A value of 0 means enough progress. A value of 1 means no speed in the
    commanded direction. Moving opposite to the command can exceed 1.
    """
    if hasattr(command_vx, "clamp"):
        abs_command = command_vx.abs()
        moving_command = abs_command > command_threshold
        aligned_speed = command_vx.sign() * base_vx
        required_speed = (abs_command * progress_fraction).clamp(min=velocity_threshold)
        speed_deficit = (required_speed - aligned_speed).clamp(min=0.0)
        normalized_deficit = speed_deficit / required_speed.clamp(min=1.0e-6)
        return normalized_deficit * moving_command.float()

    import numpy as np

    command_vx = np.asarray(command_vx)
    base_vx = np.asarray(base_vx)
    abs_command = np.abs(command_vx)
    moving_command = abs_command > command_threshold
    aligned_speed = np.sign(command_vx) * base_vx
    required_speed = np.maximum(abs_command * progress_fraction, velocity_threshold)
    speed_deficit = np.maximum(required_speed - aligned_speed, 0.0)
    normalized_deficit = speed_deficit / np.maximum(required_speed, 1.0e-6)
    return normalized_deficit * moving_command.astype(float)


def paper_b_moving_speed_failure(*args, **kwargs):
    return paper_b_moving_speed_deficit(*args, **kwargs) > 0.0


def paper_b_airtime_piecewise(
    feet_air_time,
    first_contact,
    contact_filter,
    command_norm,
    stance_command_threshold=0.1,
    airtime_clip=0.3,
    airtime_max=0.25,
    airtime_cap=0.2,
    dt=0.01,
):
    if hasattr(feet_air_time, "clamp"):
        swing_bonus = (feet_air_time - airtime_clip).clamp(min=0.0, max=airtime_max)
        swing_bonus = swing_bonus.clamp(max=airtime_cap) * first_contact.float()
        stance_bonus = contact_filter.float() * dt
        moving = (command_norm > stance_command_threshold).unsqueeze(1)
        return swing_bonus * moving.float() + stance_bonus * (~moving).float()

    import numpy as np

    swing_bonus = np.clip(feet_air_time - airtime_clip, 0.0, airtime_max)
    swing_bonus = np.minimum(swing_bonus, airtime_cap) * first_contact.astype(float)
    stance_bonus = contact_filter.astype(float) * dt
    moving = np.expand_dims(command_norm > stance_command_threshold, axis=1)
    return np.where(moving, swing_bonus, stance_bonus)
