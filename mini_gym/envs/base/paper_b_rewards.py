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
